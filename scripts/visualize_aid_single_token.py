"""
Visualize AidDecoder (cross-attn / query-style decoder) when exactly ONE VQ token is visible.

Masks z_q so only ``token_idx`` is passed to the decoder; plots GT vs reconstruction per action dim.

Usage:
  python scripts/visualize_aid_single_token.py \\
      --decoder_ckpt outputs/smoke_aid_libero/ckpt/smoke_000001.pth \\
      --zq_npy outputs/smoke_aid_libero/cache/aid_zq.npy \\
      --actions_npy outputs/smoke_aid_libero/cache/aid_actions.npy \\
      --token_idx 0 --n_samples 3 --out_dir outputs/smoke_aid_libero/viz
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
for _p in (_SCRIPT_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from aid_decoder import AidDecoder, single_token_mask  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--decoder_ckpt", required=True)
    p.add_argument("--zq_npy", required=True)
    p.add_argument("--actions_npy", required=True)
    p.add_argument("--token_idx", type=int, default=0)
    p.add_argument("--n_samples", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_dir", default="outputs/aid_single_token_viz")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.decoder_ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    K = cfg["K"]
    if args.token_idx < 0 or args.token_idx >= K:
        raise SystemExit(f"token_idx must be in [0, {K - 1}], got {args.token_idx}")

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
    dec.to(device).eval()

    z_mm = np.load(args.zq_npy, mmap_mode="r")
    x_mm = np.load(args.actions_npy, mmap_mode="r")
    N = len(z_mm)
    rng = np.random.default_rng(args.seed)
    n_plot = min(args.n_samples, N)
    idx = rng.choice(N, size=n_plot, replace=False)

    os.makedirs(args.out_dir, exist_ok=True)
    H, A = cfg["horizon"], cfg["action_dim"]

    for j, sample_i in enumerate(idx):
        z = torch.from_numpy(z_mm[sample_i : sample_i + 1].astype(np.float32)).to(device)
        gt = x_mm[sample_i].astype(np.float32)  # (H, A)
        B = 1
        mask = single_token_mask(args.token_idx, K, B, device)
        recon = dec(z, mask).squeeze(0).cpu().numpy()

        fig, axes = plt.subplots(A, 1, figsize=(10, 1.8 * A), sharex=True)
        if A == 1:
            axes = [axes]
        t_ax = np.arange(H)
        for a in range(A):
            ax = axes[a]
            ax.plot(t_ax, gt[:, a], color="black", linewidth=2.0, label="GT")
            ax.plot(t_ax, recon[:, a], color="tab:orange", linewidth=1.2, label="1-token recon")
            ax.set_ylabel(f"a{a}")
            ax.grid(True, alpha=0.3)
        axes[0].set_title(
            f"AidDecoder single-token mask | sample={int(sample_i)} | token_idx={args.token_idx}"
        )
        axes[-1].set_xlabel("horizon")
        axes[0].legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        out_png = os.path.join(args.out_dir, f"sample{j:02d}_idx{sample_i}_tok{args.token_idx}.png")
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(out_png)

    meta = {
        "decoder_ckpt": args.decoder_ckpt,
        "zq_npy": args.zq_npy,
        "actions_npy": args.actions_npy,
        "token_idx": args.token_idx,
        "sample_indices": [int(i) for i in idx],
        "K": K,
        "horizon": H,
        "action_dim": A,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
