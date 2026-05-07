"""
Training script for Aid Decoder — sequential-read optimized for CephFS.

Key insight: CephFS sequential reads are 1000x faster than random reads.
Each epoch: shuffle indices in RAM, write one contiguous permuted numpy file
to CephFS, then read it back sequentially for training. One sequential write
+ one sequential read per epoch >> 942 random reads per epoch.

Memory: only mmap overhead during read — fits within cgroup.
"""

import os, sys, math, random, argparse, json, tempfile
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from aid_decoder import AidDecoder, sample_tail_mask, _dct_loss
import torch.nn.functional as F

DEFAULT_CFG = {
    "zq_npy":      "data/aid_zq.npy",
    "actions_npy": "data/aid_actions.npy",
    "save_dir":    "ckpt/aid_decoder",
    "shuffle_dir": "data/aid_shuffle",   # where pre-shuffled epoch files go
    "d_model":     256,
    "num_layers":  4,
    "num_heads":   4,
    "batch_size":  256,
    "lr":          3e-4,
    "weight_decay":1e-2,
    "epochs":      200,
    "warmup_ratio":0.01,
    "grad_clip":   1.0,
    "seed":        42,
    "save_every":  5000,
    "log_every":   20,
    "use_wandb":   False,
    "wandb_project":"aid_decoder",
    "wandb_name":  "aid_decoder_libero",
    "max_steps":   0,
}

def parse_args():
    p = argparse.ArgumentParser()
    for k in ["zq_npy","actions_npy","save_dir","shuffle_dir","resume_ckpt"]:
        p.add_argument(f"--{k}", default=None)
    for k in ["batch_size","epochs","max_steps","d_model","num_layers","num_heads"]:
        p.add_argument(f"--{k}", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--use_wandb", action="store_true")
    return p.parse_args()

def apply_args(cfg, args):
    for k, v in vars(args).items():
        if v is not None:
            cfg[k] = v
    if args.use_wandb:
        cfg["use_wandb"] = True
    return cfg

def write_shuffled_epoch(z_mm, x_mm, perm, shuffle_dir, epoch):
    """Write one permuted epoch as two contiguous .npy files."""
    os.makedirs(shuffle_dir, exist_ok=True)
    zp = os.path.join(shuffle_dir, f"epoch_{epoch:04d}_zq.npy")
    xp = os.path.join(shuffle_dir, f"epoch_{epoch:04d}_x.npy")
    # Sort perm for sequential mmap read, then reorder — still random order
    # Actually write in perm order but do it in chunks to avoid peak memory
    N = len(perm)
    chunk = 8192
    z_out = np.lib.format.open_memmap(zp, mode='w+', dtype=np.float16, shape=z_mm.shape)
    x_out = np.lib.format.open_memmap(xp, mode='w+', dtype=np.float16, shape=x_mm.shape)
    for start in range(0, N, chunk):
        idx = np.sort(perm[start:start+chunk])   # sort each chunk → sequential src read
        dst = np.arange(start, min(start+chunk, N))
        z_out[dst] = z_mm[idx]
        x_out[dst] = x_mm[idx]
    del z_out, x_out
    return zp, xp

def main():
    args = parse_args()
    cfg  = apply_args(dict(DEFAULT_CFG), args)

    seed = cfg["seed"]
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    mask_rng = torch.Generator(); mask_rng.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Opening source mmap: {cfg['zq_npy']} ...")
    z_src = np.load(cfg["zq_npy"],      mmap_mode="r")
    x_src = np.load(cfg["actions_npy"], mmap_mode="r")
    N, K, d_enc = z_src.shape
    _, horizon, action_dim = x_src.shape
    print(f"  {N} samples | K={K} d_enc={d_enc} horizon={horizon} action_dim={action_dim}")

    batch_size      = cfg["batch_size"]
    steps_per_epoch = N // batch_size
    print(f"Steps per epoch: {steps_per_epoch}")

    decoder = AidDecoder(
        horizon=horizon, action_dim=action_dim, K=K, d_enc=d_enc,
        d_model=cfg["d_model"], num_layers=cfg["num_layers"], num_heads=cfg["num_heads"],
    ).to(device)
    print(f"AidDecoder params: {sum(p.numel() for p in decoder.parameters()):,}")

    optimizer    = torch.optim.AdamW(decoder.parameters(), lr=cfg["lr"],
                                     weight_decay=cfg["weight_decay"])
    total_steps  = cfg["epochs"] * steps_per_epoch
    warmup_steps = max(1, int(total_steps * cfg["warmup_ratio"]))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * t))

    scheduler = LambdaLR(optimizer, lr_lambda)

    if cfg["use_wandb"]:
        import wandb
        wandb.init(project=cfg["wandb_project"], name=cfg["wandb_name"], config=cfg)

    os.makedirs(cfg["save_dir"], exist_ok=True)
    cfg_to_save = {**cfg, "K":int(K), "d_enc":int(d_enc),
                   "horizon":int(horizon), "action_dim":int(action_dim)}
    with open(os.path.join(cfg["save_dir"], "config.json"), "w") as f:
        json.dump(cfg_to_save, f, indent=2)

    global_step = 0
    if cfg.get("resume_ckpt"):
        print(f"Resuming from {cfg['resume_ckpt']} ...")
        ck = torch.load(cfg["resume_ckpt"], map_location="cpu", weights_only=False)
        decoder.load_state_dict(ck["decoder_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        global_step = ck["step"]
        scheduler.last_epoch   = global_step
        scheduler._step_count  = global_step + 1
        print(f"  Resumed step={global_step}, lr={scheduler.get_last_lr()[0]:.2e}")

    start_epoch      = (global_step + steps_per_epoch - 1) // steps_per_epoch
    steps_to_advance = start_epoch * steps_per_epoch - global_step
    if steps_to_advance > 0:
        print(f"  Advancing {steps_to_advance} steps → epoch {start_epoch}")
        global_step           += steps_to_advance
        scheduler.last_epoch   = global_step
        scheduler._step_count  = global_step + 1

    decoder.train()
    epoch_rng = np.random.default_rng(seed + start_epoch)
    prev_zp = prev_xp = None

    for epoch in range(start_epoch, cfg["epochs"]):
        perm = epoch_rng.permutation(N)

        # Write shuffled epoch to CephFS as contiguous file (sequential write)
        print(f"Epoch {epoch}: writing shuffled data ...", flush=True)
        import time; t0 = time.time()
        zp, xp = write_shuffled_epoch(z_src, x_src, perm, cfg["shuffle_dir"], epoch)
        print(f"  Written in {time.time()-t0:.1f}s, reading sequentially ...", flush=True)

        # Read the shuffled epoch sequentially — fast on CephFS
        z_ep = np.load(zp, mmap_mode="r")   # (N, K, d_enc) sequential layout
        x_ep = np.load(xp, mmap_mode="r")   # (N, H, A)

        for bi in range(steps_per_epoch):
            s = bi * batch_size
            e = s + batch_size
            z_q = torch.from_numpy(z_ep[s:e].astype(np.float32)).to(device)
            x   = torch.from_numpy(x_ep[s:e].astype(np.float32)).to(device)
            global_step += 1

            B    = z_q.shape[0]
            mask = sample_tail_mask(B, K, device, rng=mask_rng)
            optimizer.zero_grad()
            recon = decoder(z_q, mask)
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
                print(f"step={global_step:06d}  loss={loss.item():.4f}  "
                      f"recon={l1.item():.4f}  dct={dct.item():.4f}  lr={lr_now:.2e}", flush=True)
                if cfg["use_wandb"]:
                    import wandb
                    wandb.log({"loss":loss.item(),"recon":l1.item(),
                               "dct":dct.item(),"lr":lr_now}, step=global_step)

            if global_step % cfg["save_every"] == 0:
                sp = os.path.join(cfg["save_dir"], f"{global_step:06d}.pth")
                torch.save({"step":global_step,
                            "decoder_state_dict":decoder.state_dict(),
                            "optimizer_state_dict":optimizer.state_dict(),
                            "cfg":cfg_to_save}, sp)
                print(f"Saved: {sp}", flush=True)

            if cfg["max_steps"] > 0 and global_step >= cfg["max_steps"]:
                sp = os.path.join(cfg["save_dir"], f"smoke_{global_step:06d}.pth")
                torch.save({"step":global_step, "decoder_state_dict":decoder.state_dict(),
                            "optimizer_state_dict":optimizer.state_dict(),
                            "cfg":cfg_to_save}, sp)
                print(f"Done at step {global_step}.")
                if cfg["use_wandb"]:
                    import wandb; wandb.finish()
                return

        # Clean up epoch file to save CephFS space
        del z_ep, x_ep
        try:
            os.remove(zp); os.remove(xp)
        except:
            pass

    if cfg["use_wandb"]:
        import wandb; wandb.finish()
    print("Training complete.")

if __name__ == "__main__":
    main()
