"""
Offline encoder: encode LIBERO action chunks through frozen backbone encoder+VQ.

Default output: a torch ``.pt`` bundle (``--out``).

For AidDecoder training via numpy memmaps (see ``train_aid_decoder.py``), pass
``--numpy_out_dir DIR`` to write ``aid_zq.npy`` and ``aid_actions.npy`` (float16).
Use ``--max_chunks N`` for small caches / smoke tests, and ``--skip_pt`` to skip the ``.pt`` file.

Example (smoke-sized numpy cache):

  python scripts/encode_aid_cache.py \\
    --backbone_ckpt ckpt/vanilla32/165000.pth \\
    --data_root data --max_chunks 256 \\
    --numpy_out_dir outputs/smoke/cache --skip_pt
"""

import os
import sys
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_normalized_actions(data_root: str, max_chunks: int = 0) -> torch.Tensor:
    """Load LIBERO RLDS action chunks (normalized). If ``max_chunks`` > 0, stop after that many rows."""
    from catok.training.rlds_dataset import make_rlds_dataloader

    libero_datasets = [
        "Libero_RLDS/libero_spatial_no_noops",
        "Libero_RLDS/libero_goal_no_noops",
        "Libero_RLDS/libero_object_no_noops",
        "Libero_RLDS/libero_10_no_noops",
    ]
    dataset_specs = [(ds, 1.0) for ds in libero_datasets]
    stats_path_map = {ds: f"{data_root}/{ds}/stats.json" for ds in libero_datasets}
    action_dim_map = {ds: 7 for ds in libero_datasets}

    _, loader, _ = make_rlds_dataloader(
        dataset_specs=dataset_specs,
        data_root=data_root,
        horizon=20,
        batch_size=4096,
        action_only=True,
        num_workers=0,
        target_action_dim=7,
        stats_path_map=stats_path_map,
        normalizer_method="qq",
        normalizer_config={"clip": True, "action_dim": 7},
        action_dim_map=action_dim_map,
        drop_last=False,
        eval_mode=True,   # sequential, no shuffle
    )

    chunks = []
    total = 0
    for batch in loader:
        chunks.append(batch)
        total += batch.shape[0]
        if max_chunks > 0 and total >= max_chunks:
            break
    out = torch.cat(chunks, dim=0).float()
    if max_chunks > 0:
        out = out[:max_chunks]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_ckpt", default="ckpt/vanilla32/165000.pth")
    parser.add_argument("--data_root",     default="data")
    parser.add_argument("--out",           default="data/aid_zq_cache.pt")
    parser.add_argument("--batch_size",    type=int, default=512)
    parser.add_argument("--device",        default="cuda")
    parser.add_argument(
        "--max_chunks",
        type=int,
        default=0,
        help="If >0, only load/encode this many action chunks (first sequential batches).",
    )
    parser.add_argument(
        "--numpy_out_dir",
        default=None,
        help="If set, save aid_zq.npy + aid_actions.npy (float16) here for train_aid_decoder.py.",
    )
    parser.add_argument(
        "--skip_pt",
        action="store_true",
        help="Do not write the default torch .pt payload (use with --numpy_out_dir).",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.skip_pt and not args.numpy_out_dir:
        parser.error("--skip_pt requires --numpy_out_dir")

    # Load backbone (encoder + VQ only, frozen)
    print("Loading backbone ...")
    from catok.models.catok_ddt.vanilla_utils import load_checkpoint
    backbone, _ = load_checkpoint(args.backbone_ckpt, device=device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    # Read config for metadata
    import torch as _t
    raw = _t.load(args.backbone_ckpt, map_location="cpu", weights_only=False)
    cfg = raw["config"]["tokenizer"]
    K = cfg["encoder"]["num_tokens"]
    d_enc = cfg["encoder"]["d_model"]
    horizon = cfg["basic"]["horizon"]
    action_dim = cfg["basic"]["action_dim"]
    print(f"  K={K}, d_enc={d_enc}, horizon={horizon}, action_dim={action_dim}")

    # Load normalized actions
    print("Loading normalized action chunks from pkl cache ...")
    actions = load_normalized_actions(args.data_root, max_chunks=args.max_chunks)
    N = actions.shape[0]
    print(f"  {N} chunks loaded, shape={tuple(actions.shape)}")

    # Encode in batches
    print(f"Encoding {N} chunks on {device} (batch_size={args.batch_size}) ...")
    z_q_list = []
    loader = DataLoader(TensorDataset(actions), batch_size=args.batch_size, shuffle=False)
    for i, (batch,) in enumerate(loader):
        x = batch.to(device)
        with torch.no_grad():
            z = backbone.encoder(x)
            _, z_q, _, _ = backbone.vq(z)
        z_q_list.append(z_q.cpu().half())
        if (i + 1) % 50 == 0:
            print(f"  {(i+1)*args.batch_size}/{N} ...")

    z_q_all = torch.cat(z_q_list, dim=0)   # (N, K, d_enc) float16
    print(f"z_q shape: {tuple(z_q_all.shape)}, dtype={z_q_all.dtype}")

    if args.numpy_out_dir:
        os.makedirs(args.numpy_out_dir, exist_ok=True)
        z_np = os.path.join(args.numpy_out_dir, "aid_zq.npy")
        a_np = os.path.join(args.numpy_out_dir, "aid_actions.npy")
        np.save(z_np, z_q_all.numpy())
        np.save(a_np, actions.numpy().astype(np.float16))
        print(f"Saved numpy caches: {z_np}, {a_np}")

    if not args.skip_pt:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        payload = {
            "z_q":     z_q_all,
            "actions": actions.half(),
            "cfg": {"K": K, "d_enc": d_enc, "horizon": horizon, "action_dim": action_dim},
        }
        torch.save(payload, args.out)
        size_gb = os.path.getsize(args.out) / 1e9
        print(f"Saved {args.out} ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
