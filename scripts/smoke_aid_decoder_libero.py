#!/usr/bin/env python3
"""
End-to-end smoke test: tiny LIBERO cache -> 1 training step -> save -> eval -> single-token viz.

Requires:
  - Frozen VanillaVQDiffusion ckpt (encoder+VQ)
  - LIBERO RLDS pkl caches under ``data_root`` (same layout as encode_aid_cache.py)

Example:
  python scripts/smoke_aid_decoder_libero.py \\
      --backbone_ckpt ckpt/vanilla32/165000.pth \\
      --data_root data \\
      --work_dir outputs/smoke_aid_libero
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd: list[str], cwd: str, env: dict | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def quick_eval(
    decoder_ckpt: str,
    zq_npy: str,
    actions_npy: str,
    device: str,
    n_eval: int,
    seed: int,
    K: int,
) -> dict:
    import numpy as np
    import torch

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if _script_dir not in sys.path:
        sys.path.insert(0, _script_dir)

    from aid_decoder import AidDecoder, prefix_mask, single_token_mask

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    ck = torch.load(decoder_ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    dec = AidDecoder(
        horizon=cfg["horizon"],
        action_dim=cfg["action_dim"],
        K=cfg["K"],
        d_enc=cfg["d_enc"],
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
    )
    dec.load_state_dict(ck["decoder_state_dict"])
    dec.to(dev).eval()

    z_mm = np.load(zq_npy, mmap_mode="r")
    x_mm = np.load(actions_npy, mmap_mode="r")
    N = len(z_mm)
    rng = np.random.default_rng(seed)
    n_ev = min(n_eval, N)
    idx = rng.choice(N, size=n_ev, replace=False)
    z_e = torch.from_numpy(z_mm[idx].astype(np.float32)).to(dev)
    x_e = torch.from_numpy(x_mm[idx].astype(np.float32)).to(dev)

    def batch_l1(zq, x, mask_fn):
        bs = 64
        l1s = []
        for i in range(0, len(zq), bs):
            z = zq[i : i + bs]
            gt = x[i : i + bs]
            B = z.shape[0]
            mask = mask_fn(B)
            with torch.no_grad():
                recon = dec(z, mask)
            l1s.append(torch.mean(torch.abs(recon - gt)).item())
        return float(sum(l1s) / len(l1s))

    out = {}
    out["all_tokens"] = batch_l1(z_e, x_e, lambda B: torch.ones(B, K, dtype=torch.bool, device=dev))
    out["single_token_0"] = batch_l1(
        z_e, x_e, lambda B: single_token_mask(0, K, B, dev)
    )
    out["prefix_1"] = batch_l1(z_e, x_e, lambda B: prefix_mask(1, K, B, dev))
    return {"mean_l1": out, "n_eval": n_ev}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone_ckpt", required=True)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--work_dir", default="outputs/smoke_aid_libero")
    ap.add_argument("--max_chunks", type=int, default=256)
    ap.add_argument("--encode_batch_size", type=int, default=64)
    ap.add_argument("--train_batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--token_idx_viz", type=int, default=0)
    args = ap.parse_args()

    root = _repo_root()
    cache_dir = os.path.join(args.work_dir, "cache")
    ckpt_dir = os.path.join(args.work_dir, "ckpt")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    py = sys.executable
    encode_cmd = [
        py,
        os.path.join(root, "scripts", "encode_aid_cache.py"),
        "--backbone_ckpt",
        args.backbone_ckpt,
        "--data_root",
        args.data_root,
        "--batch_size",
        str(args.encode_batch_size),
        "--device",
        args.device,
        "--max_chunks",
        str(args.max_chunks),
        "--numpy_out_dir",
        cache_dir,
        "--skip_pt",
    ]
    run(encode_cmd, cwd=root)

    zq_path = os.path.join(cache_dir, "aid_zq.npy")
    act_path = os.path.join(cache_dir, "aid_actions.npy")

    tb = min(args.train_batch_size, args.max_chunks)
    train_cmd = [
        py,
        os.path.join(root, "scripts", "train_aid_decoder.py"),
        "--zq_npy",
        zq_path,
        "--actions_npy",
        act_path,
        "--save_dir",
        ckpt_dir,
        "--batch_size",
        str(tb),
        "--epochs",
        "1",
        "--max_steps",
        "1",
        "--save_every",
        "1",
        "--log_every",
        "1",
        "--d_model",
        "128",
        "--num_layers",
        "2",
        "--num_heads",
        "4",
    ]
    run(train_cmd, cwd=root)

    smoke_ckpt = os.path.join(ckpt_dir, "smoke_000001.pth")
    if not os.path.isfile(smoke_ckpt):
        raise SystemExit(f"Expected checkpoint missing: {smoke_ckpt}")

    import torch

    ck_meta = torch.load(smoke_ckpt, map_location="cpu", weights_only=False)
    K = ck_meta["cfg"]["K"]

    metrics = quick_eval(
        smoke_ckpt,
        zq_path,
        act_path,
        args.device,
        n_eval=min(128, args.max_chunks),
        seed=0,
        K=K,
    )
    eval_path = os.path.join(args.work_dir, "smoke_eval.json")
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved {eval_path}")
    print("mean_l1:", metrics["mean_l1"])

    viz_dir = os.path.join(args.work_dir, "viz_single_token")
    viz_cmd = [
        py,
        os.path.join(root, "scripts", "visualize_aid_single_token.py"),
        "--decoder_ckpt",
        smoke_ckpt,
        "--zq_npy",
        zq_path,
        "--actions_npy",
        act_path,
        "--token_idx",
        str(args.token_idx_viz),
        "--n_samples",
        "2",
        "--device",
        args.device,
        "--out_dir",
        viz_dir,
    ]
    run(viz_cmd, cwd=root)
    print(f"Smoke OK. Artifacts under {args.work_dir}")


if __name__ == "__main__":
    main()
