"""
Training script for the Aid Decoder.

Usage (single GPU, no Hydra):
  # Step 1: pre-encode (once)
  python scripts/encode_aid_cache.py --backbone_ckpt ckpt/vanilla32/165000.pth

  # Step 2: train (backbone NOT loaded — uses pre-encoded z_q numpy cache via memmap)
  python scripts/train_aid_decoder.py

Key design choices:
  - Pre-encoded z_q/actions numpy memmap avoids loading heavy data into cgroup RAM.
  - AidDecoder is instantiated without AidDecoderModel (no backbone) to save RAM.
  - Optimizer only receives decoder.parameters() — encoder/VQ are never touched.
"""

import os
import sys
import math
import random
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aid_decoder import AidDecoder, sample_tail_mask, _dct_loss
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CFG = {
    # Paths — numpy memmap files produced by encode_aid_cache.py
    "zq_npy":    "data/aid_zq.npy",
    "actions_npy": "data/aid_actions.npy",
    "save_dir":  "ckpt/aid_decoder",
    # AidDecoder arch
    "d_model":    256,
    "num_layers": 4,
    "num_heads":  4,
    # Training
    "batch_size":    256,
    "lr":            3e-4,
    "weight_decay":  1e-2,
    "epochs":        200,
    "warmup_ratio":  0.01,
    "grad_clip":     1.0,
    "seed":          42,
    # Logging / saving
    "save_every":    5000,
    "log_every":     20,
    "use_wandb":     False,
    "wandb_project": "aid_decoder",
    "wandb_name":    "aid_decoder_libero",
    # Smoke test: set > 0 to stop after N steps
    "max_steps":     0,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zq_npy",      default=None, help="Path to z_q numpy file (data/aid_zq.npy)")
    p.add_argument("--actions_npy", default=None, help="Path to actions numpy file (data/aid_actions.npy)")
    p.add_argument("--save_dir",    default=None)
    p.add_argument("--batch_size",  type=int, default=None)
    p.add_argument("--epochs",      type=int, default=None)
    p.add_argument("--lr",          type=float, default=None)
    p.add_argument("--max_steps",   type=int, default=None,
                   help="Stop after N steps (smoke test). 0 = run to completion.")
    p.add_argument("--resume_ckpt", default=None,
                   help="Resume training from this checkpoint path (skips to saved step).")
    p.add_argument("--use_wandb",   action="store_true")
    p.add_argument("--d_model",     type=int, default=None)
    p.add_argument("--num_layers",  type=int, default=None)
    p.add_argument("--num_heads",   type=int, default=None)
    p.add_argument("--save_every",  type=int, default=None)
    p.add_argument("--log_every",   type=int, default=None)
    return p.parse_args()


def apply_args(cfg, args):
    for k, v in vars(args).items():
        if v is not None:
            cfg[k] = v
    if args.use_wandb:
        cfg["use_wandb"] = True
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg  = apply_args(dict(DEFAULT_CFG), args)

    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    mask_rng = torch.Generator()
    mask_rng.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Load pre-encoded z_q cache via numpy memmap (low RAM footprint)
    # ------------------------------------------------------------------
    print(f"Loading z_q memmap from {cfg['zq_npy']} ...")
    z_q_mm = np.load(cfg["zq_npy"],    mmap_mode="r")   # (N, K, d_enc) float16
    x_mm   = np.load(cfg["actions_npy"], mmap_mode="r") # (N, H, A)     float16
    N, K, d_enc = z_q_mm.shape
    _, horizon, action_dim = x_mm.shape
    print(f"  {N} samples, K={K}, d_enc={d_enc}, horizon={horizon}, action_dim={action_dim}")

    class MemmapDataset(torch.utils.data.Dataset):
        def __init__(self, zq, actions):
            self.zq = zq
            self.actions = actions
        def __len__(self):
            return len(self.zq)
        def __getitem__(self, idx):
            return (torch.from_numpy(self.zq[idx].astype(np.float32)),
                    torch.from_numpy(self.actions[idx].astype(np.float32)))

    dataset = MemmapDataset(z_q_mm, x_mm)
    drop_last = len(dataset) > cfg["batch_size"]
    train_loader = DataLoader(
        dataset,
        batch_size=min(cfg["batch_size"], max(1, len(dataset))),
        shuffle=True,
        drop_last=drop_last,
        num_workers=0,   # CephFS memmap + workers = deadlock; keep 0
        pin_memory=False,
    )
    print(f"DataLoader batches/epoch: {len(train_loader)}")

    # ------------------------------------------------------------------
    # Model — AidDecoder only (no backbone)
    # ------------------------------------------------------------------
    decoder = AidDecoder(
        horizon=horizon,
        action_dim=action_dim,
        K=K,
        d_enc=d_enc,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
    ).to(device)

    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"AidDecoder trainable params: {n_params:,}")

    # ------------------------------------------------------------------
    # Optimizer + scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    total_steps  = cfg["epochs"] * len(train_loader)
    warmup_steps = max(1, int(total_steps * cfg["warmup_ratio"]))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * t))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # Wandb
    # ------------------------------------------------------------------
    if cfg["use_wandb"]:
        import wandb
        wandb.init(project=cfg["wandb_project"], name=cfg["wandb_name"], config=cfg)

    # ------------------------------------------------------------------
    # Save dir + config
    # ------------------------------------------------------------------
    os.makedirs(cfg["save_dir"], exist_ok=True)
    cfg_to_save = {**cfg, "K": int(K), "d_enc": int(d_enc), "horizon": int(horizon),
                   "action_dim": int(action_dim), "d_model": cfg["d_model"],
                   "num_layers": cfg["num_layers"], "num_heads": cfg["num_heads"]}
    with open(os.path.join(cfg["save_dir"], "config.json"), "w") as f:
        json.dump(cfg_to_save, f, indent=2)

    # ------------------------------------------------------------------
    # Resume from checkpoint (optional)
    # ------------------------------------------------------------------
    global_step = 0
    if cfg.get("resume_ckpt"):
        print(f"Resuming from {cfg['resume_ckpt']} ...")
        resume = torch.load(cfg["resume_ckpt"], map_location="cpu", weights_only=False)
        decoder.load_state_dict(resume["decoder_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        global_step = resume["step"]
        # fast-forward scheduler without N individual steps
        scheduler.last_epoch = global_step
        scheduler._step_count = global_step + 1
        print(f"  Resumed at step {global_step}, lr={scheduler.get_last_lr()[0]:.2e}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    decoder.train()
    steps_per_epoch = len(train_loader)

    start_epoch = global_step // steps_per_epoch
    for epoch in range(start_epoch, cfg["epochs"]):
        for z_q_batch, x_batch in train_loader:
            # skip steps already done in the resume epoch
            if global_step >= (epoch + 1) * steps_per_epoch:
                continue
            global_step += 1

            z_q = z_q_batch.to(device)   # (B, K, d_enc)
            x   = x_batch.to(device)     # (B, H, A)

            B = z_q.shape[0]
            mask = sample_tail_mask(B, K, device, rng=mask_rng)   # (B, K) bool

            optimizer.zero_grad()
            recon = decoder(z_q, mask)                             # (B, H, A)
            tgt   = x[:, :horizon, :action_dim]
            l1    = F.l1_loss(recon, tgt)
            dct   = _dct_loss(recon, tgt)
            loss  = l1 + dct
            loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), cfg["grad_clip"])
            optimizer.step()
            scheduler.step()

            if global_step % cfg["log_every"] == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"step={global_step:06d}  "
                    f"loss={loss.item():.4f}  "
                    f"recon={l1.item():.4f}  "
                    f"dct={dct.item():.4f}  "
                    f"lr={lr_now:.2e}"
                )
                if cfg["use_wandb"]:
                    import wandb
                    wandb.log({"loss": loss.item(), "recon": l1.item(),
                               "dct": dct.item(), "lr": lr_now}, step=global_step)

            if global_step % cfg["save_every"] == 0:
                save_path = os.path.join(cfg["save_dir"], f"{global_step:06d}.pth")
                torch.save({
                    "step": global_step,
                    "decoder_state_dict": decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "cfg": cfg_to_save,
                }, save_path)
                print(f"Saved checkpoint: {save_path}")

            if cfg["max_steps"] > 0 and global_step >= cfg["max_steps"]:
                print(f"Smoke test done at step {global_step}. Loss={loss.item():.4f}")
                save_path = os.path.join(cfg["save_dir"], f"smoke_{global_step:06d}.pth")
                torch.save({
                    "step": global_step,
                    "decoder_state_dict": decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "cfg": cfg_to_save,
                }, save_path)
                if cfg["use_wandb"]:
                    import wandb
                    wandb.finish()
                return

    if cfg["use_wandb"]:
        import wandb
        wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    main()
