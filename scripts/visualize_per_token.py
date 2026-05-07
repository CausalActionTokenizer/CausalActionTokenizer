"""
Visualization script for PerTokenAidDecoder.

For each of K=32 tokens, shows:
  (A) Time-domain reconstruction vs GT (all action dims)
  (B) Frequency-domain (rFFT amplitude) reconstruction vs GT

Produces per-chunk figures:
  outputs/per_token_viz/<ckpt_step>/chunk_<NN>_timedomain.png
  outputs/per_token_viz/<ckpt_step>/chunk_<NN>_spectrum.png
  outputs/per_token_viz/<ckpt_step>/meta.json

Usage:
  python scripts/visualize_per_token.py \
      --decoder_ckpt ckpt/per_token_aid/094200.pth \
      --zq_npy /dev/shm/aid_zq.npy \
      --actions_npy /dev/shm/aid_actions.npy \
      --n_chunks 10 \
      --seed 42 \
      --device cuda
"""

import os, sys, argparse, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from per_token_aid_decoder import PerTokenAidDecoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--decoder_ckpt", required=True)
    p.add_argument("--zq_npy",       default="/dev/shm/aid_zq.npy")
    p.add_argument("--actions_npy",  default="/dev/shm/aid_actions.npy")
    p.add_argument("--n_chunks",     type=int, default=10)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--device",       default="cuda")
    p.add_argument("--out_dir",      default=None)
    return p.parse_args()


@torch.no_grad()
def decode_all_tokens(model, zq_chunk, device):
    """
    Returns reconstructions for each token index.
    zq_chunk: (K, d_enc) numpy float16
    Returns: (K, horizon, action_dim) numpy float32
    """
    K = zq_chunk.shape[0]
    zq_t = torch.from_numpy(zq_chunk.astype(np.float32)).to(device)  # (K, d_enc)
    idx  = torch.arange(K, device=device)                              # (K,)
    recon = model(zq_t, idx)                                           # (K, H, A)
    return recon.cpu().numpy()


def plot_timedomain(recons, gt, out_path, chunk_idx):
    """
    recons: (K, H, A)
    gt:     (H, A)
    """
    K, H, A = recons.shape
    cmap = cm.plasma(np.linspace(0, 1, K))

    fig, axes = plt.subplots(A, 1, figsize=(12, 2 * A), sharex=True)
    if A == 1:
        axes = [axes]
    for a in range(A):
        ax = axes[a]
        ax.plot(gt[:, a], color="black", linewidth=2, label="GT", zorder=K + 1)
        for i in range(K):
            ax.plot(recons[i, :, a], color=cmap[i], linewidth=0.8, alpha=0.7,
                    label=f"tok {i}" if a == 0 else None)
        ax.set_ylabel(f"dim {a}", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_title(f"Chunk {chunk_idx:02d} — Per-Token Time-Domain Reconstruction\n"
                      f"(color: token 0=dark → {K-1}=bright, black=GT)")
    axes[-1].set_xlabel("Horizon step")

    # Colorbar as legend proxy
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, K - 1))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="Token index", fraction=0.015, pad=0.01)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_spectrum(recons, gt, out_path, chunk_idx):
    """
    rFFT amplitude spectrum, averaged over action dims.
    recons: (K, H, A)
    gt:     (H, A)
    """
    K, H, A = recons.shape
    cmap = cm.plasma(np.linspace(0, 1, K))

    # Compute rFFT amplitudes averaged over action dims
    def rfft_amp(x):  # x: (H, A) or (K, H, A)
        if x.ndim == 2:
            return np.abs(np.fft.rfft(x, axis=0)).mean(axis=1)  # (H//2+1,)
        else:
            return np.abs(np.fft.rfft(x, axis=1)).mean(axis=2)  # (K, H//2+1)

    gt_amp    = rfft_amp(gt)           # (F,)
    recon_amp = rfft_amp(recons)       # (K, F)
    freqs = np.arange(gt_amp.shape[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: amplitude spectra
    ax1.plot(freqs, gt_amp, color="black", linewidth=2.5, label="GT", zorder=K + 1)
    for i in range(K):
        ax1.plot(freqs, recon_amp[i], color=cmap[i], linewidth=0.8, alpha=0.7)
    ax1.set_xlabel("Frequency bin")
    ax1.set_ylabel("rFFT amplitude (mean over action dims)")
    ax1.set_title(f"Chunk {chunk_idx:02d} — Spectrum per token")
    ax1.grid(True, alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, K - 1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax1, label="Token index")

    # Right: spectral difference heatmap  (K, F)
    diff = recon_amp - gt_amp[None, :]
    im = ax2.imshow(diff, aspect="auto", origin="upper",
                    cmap="RdBu_r", vmin=-np.abs(diff).max(), vmax=np.abs(diff).max(),
                    extent=[freqs[0], freqs[-1], K - 0.5, -0.5])
    ax2.set_xlabel("Frequency bin")
    ax2.set_ylabel("Token index")
    ax2.set_title(f"Chunk {chunk_idx:02d} — Spectral diff (recon − GT)")
    fig.colorbar(im, ax=ax2, label="|FFT(recon)| − |FFT(GT)|")

    fig.suptitle("rFFT amplitude spectrum — PerTokenAidDecoder\n"
                 "(each token reconstructs action independently)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    print(f"Loading PerTokenAidDecoder from {args.decoder_ckpt} ...")
    ckpt = torch.load(args.decoder_ckpt, map_location="cpu", weights_only=False)
    cfg  = ckpt["cfg"]
    model = PerTokenAidDecoder(
        horizon=cfg["horizon"], action_dim=cfg["action_dim"],
        K=cfg["K"], d_enc=cfg["d_enc"],
        d_model=cfg["d_model"], num_layers=cfg["num_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    step = ckpt["step"]
    print(f"  K={cfg['K']}, horizon={cfg['horizon']}, action_dim={cfg['action_dim']}, step={step}")

    # Load caches
    print(f"Loading z_q cache ({args.zq_npy}) ...")
    zq_mm = np.load(args.zq_npy,      mmap_mode="r")   # (N, K, d_enc)
    x_mm  = np.load(args.actions_npy, mmap_mode="r")   # (N, H, A)
    N = len(zq_mm)
    print(f"  {N} samples available")

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(N, size=args.n_chunks, replace=False)
    print(f"  Sampled chunk indices (seed={args.seed}): {indices.tolist()}")

    # Output dir
    out_dir = args.out_dir or os.path.join(
        "outputs", "per_token_viz", str(step)
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")

    meta = {"ckpt": args.decoder_ckpt, "step": step, "seed": args.seed,
            "chunk_indices": indices.tolist(),
            "spectrum": "rFFT amplitude averaged over action dims",
            "note": "each token reconstructs action independently; color = token index (plasma colormap)"}

    for ci, cache_idx in enumerate(indices):
        zq_chunk = zq_mm[cache_idx]    # (K, d_enc) float16
        gt       = x_mm[cache_idx].astype(np.float32)   # (H, A)

        recons = decode_all_tokens(model, zq_chunk, device)  # (K, H, A)

        td_path  = os.path.join(out_dir, f"chunk_{ci:02d}_timedomain.png")
        sp_path  = os.path.join(out_dir, f"chunk_{ci:02d}_spectrum.png")
        plot_timedomain(recons, gt, td_path,  ci)
        plot_spectrum  (recons, gt, sp_path,  ci)
        print(f"  chunk {ci:02d} (cache idx={cache_idx}): saved")

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
