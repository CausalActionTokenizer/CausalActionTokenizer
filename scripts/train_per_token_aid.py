"""
Training script for PerTokenAidDecoder.

Pre-condition: /dev/shm/aid_zq.npy and /dev/shm/aid_actions.npy exist
(produced by scripts/encode_aid_cache.py).

Usage:
  python scripts/train_per_token_aid.py \
      --zq_npy /dev/shm/aid_zq.npy \
      --actions_npy /dev/shm/aid_actions.npy \
      --save_dir ckpt/per_token_aid \
      --batch_size 512 \
      --epochs 100

Each training step randomly picks ONE token index per sample, passes
(z_q[:, i, :], i) → decoder → reconstructed action.
"""

import os, sys, math, random, argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from per_token_aid_decoder import PerTokenAidDecoder, _dct_loss

# ---------------------------------------------------------------------------
DEFAULT_CFG = {
    "zq_npy":      "/dev/shm/aid_zq.npy",
    "actions_npy": "/dev/shm/aid_actions.npy",
    "save_dir":    "ckpt/per_token_aid",
    "d_model":     256,
    "num_layers":  4,
    "batch_size":  512,
    "lr":          3e-4,
    "weight_decay": 1e-2,
    "epochs":      100,
    "warmup_ratio": 0.01,
    "grad_clip":   1.0,
    "seed":        42,
    "save_every":  1000,
    "log_every":   20,
    "max_steps":   0,
}


def parse_args():
    p = argparse.ArgumentParser()
    for k, v in DEFAULT_CFG.items():
        t = type(v) if v is not None else str
        if t == bool:
            p.add_argument(f"--{k}", action="store_true")
        else:
            p.add_argument(f"--{k}", type=t, default=None)
    p.add_argument("--resume_ckpt", default=None)
    return p.parse_args()


class MemmapDataset(Dataset):
    def __init__(self, zq, actions):
        self.zq = zq
        self.actions = actions
    def __len__(self): return len(self.zq)
    def __getitem__(self, idx):
        return (torch.from_numpy(self.zq[idx].astype(np.float32)),
                torch.from_numpy(self.actions[idx].astype(np.float32)))


def main():
    args = parse_args()
    cfg = dict(DEFAULT_CFG)
    for k, v in vars(args).items():
        if v is not None and k != "resume_ckpt":
            cfg[k] = v

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    idx_rng = torch.Generator()
    idx_rng.manual_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading memmap caches ...")
    zq_mm = np.load(cfg["zq_npy"],      mmap_mode="r")
    x_mm  = np.load(cfg["actions_npy"], mmap_mode="r")
    N, K, d_enc = zq_mm.shape
    _, horizon, action_dim = x_mm.shape
    print(f"  N={N}, K={K}, d_enc={d_enc}, horizon={horizon}, action_dim={action_dim}")

    dataset = MemmapDataset(zq_mm, x_mm)
    loader  = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True,
                         drop_last=True, num_workers=0, pin_memory=False)
    print(f"  steps/epoch: {len(loader)}")

    model = PerTokenAidDecoder(
        horizon=horizon, action_dim=action_dim, K=K, d_enc=d_enc,
        d_model=cfg["d_model"], num_layers=cfg["num_layers"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PerTokenAidDecoder params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total_steps  = cfg["epochs"] * len(loader)
    warmup_steps = max(1, int(total_steps * cfg["warmup_ratio"]))

    def lr_lambda(s):
        if s < warmup_steps: return s / warmup_steps
        t = (s - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * t))

    scheduler = LambdaLR(optimizer, lr_lambda)

    os.makedirs(cfg["save_dir"], exist_ok=True)
    cfg_save = {**cfg, "K": K, "d_enc": d_enc, "horizon": horizon, "action_dim": action_dim}
    with open(os.path.join(cfg["save_dir"], "config.json"), "w") as f:
        json.dump(cfg_save, f, indent=2)

    global_step = 0
    if args.resume_ckpt:
        print(f"Resuming from {args.resume_ckpt} ...")
        ckpt = torch.load(args.resume_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        global_step = ckpt["step"]
        scheduler.last_epoch = global_step
        scheduler._step_count = global_step + 1
        print(f"  Resumed at step {global_step}")

    model.train()
    steps_per_epoch = len(loader)
    start_epoch = global_step // steps_per_epoch

    for epoch in range(start_epoch, cfg["epochs"]):
        for zq_batch, x_batch in loader:
            if global_step >= (epoch + 1) * steps_per_epoch:
                continue
            global_step += 1

            zq = zq_batch.to(device)   # (B, K, d_enc)
            x  = x_batch.to(device)    # (B, H, A)
            B  = zq.shape[0]

            # Randomly pick ONE token index per sample
            tok_idx = torch.randint(0, K, (B,), generator=idx_rng).to(device)  # (B,)
            tok_emb = zq[torch.arange(B, device=device), tok_idx]               # (B, d_enc)

            optimizer.zero_grad()
            recon = model(tok_emb, tok_idx)   # (B, H, A)
            tgt   = x[:, :horizon, :action_dim]
            l1    = F.l1_loss(recon, tgt)
            dct   = _dct_loss(recon, tgt)
            loss  = l1 + dct
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            scheduler.step()

            if global_step % cfg["log_every"] == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"step={global_step:06d}  loss={loss.item():.4f}  "
                      f"recon={l1.item():.4f}  dct={dct.item():.4f}  lr={lr_now:.2e}")

            if global_step % cfg["save_every"] == 0:
                save_path = os.path.join(cfg["save_dir"], f"{global_step:06d}.pth")
                torch.save({
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "cfg": cfg_save,
                }, save_path)
                print(f"Saved: {save_path}")

            if cfg["max_steps"] > 0 and global_step >= cfg["max_steps"]:
                print(f"Smoke test done at step {global_step}.")
                return

    print("Training complete.")


if __name__ == "__main__":
    main()
