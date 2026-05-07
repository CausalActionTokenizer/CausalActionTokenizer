"""
Spectrum visualization for Aid Decoder (task.md §4.1).

For each of 10 randomly sampled action chunks (fixed seed=42), reconstruct
using prefix tokens n=1..K, then compare the frequency spectrum of each
reconstruction against the ground truth.

Spectrum definition: real-valued rFFT along the horizon dimension, amplitude
spectrum |FFT(a)| averaged across action dimensions.

Two plot types per chunk (both produced by default):
  (a) Spectral difference: |FFT(hat_a_n)| - |FFT(a)|  as a heat-map over
      (n, frequency_bin).
  (b) Side-by-side amplitude spectra: one subplot per n, GT overlaid in black.

Output: outputs/aid_decoder_spectrum/<decoder_ckpt_name>/
        chunk_{i:02d}_spectrum_diff.png
        chunk_{i:02d}_spectrum_lines.png

Usage:
    python scripts/visualize_spectrum.py \
        --decoder_ckpt ckpt/aid_decoder/055000.pth \
        --zq_npy       data/aid_zq.npy \
        --actions_npy  data/aid_actions.npy \
        --n_chunks 10 \
        --seed 42 \
        --device cuda

Data source: pre-encoded z_q cache produced by encode_aid_cache.py.
No backbone is loaded during visualization (low memory footprint).
"""

import os
import sys
import argparse
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import cm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aid_decoder import AidDecoder, prefix_mask


# ---------------------------------------------------------------------------
# Spectrum helpers
# ---------------------------------------------------------------------------

def rfft_amplitude(action: np.ndarray) -> np.ndarray:
    """action: (T, A) → (F,) amplitude averaged over action dims, F = T//2+1."""
    spec = np.fft.rfft(action, axis=0)   # (F, A)
    return np.abs(spec).mean(axis=1)     # (F,)


def rfft_amplitude_per_dim(action: np.ndarray) -> np.ndarray:
    """Returns (F, A) amplitude spectrum."""
    return np.abs(np.fft.rfft(action, axis=0))


# ---------------------------------------------------------------------------
# Load decoder (no backbone — uses pre-encoded z_q)
# ---------------------------------------------------------------------------

def load_decoder(decoder_ckpt: str, device: torch.device) -> tuple:
    """Load AidDecoder from checkpoint. Returns (decoder, cfg_dict)."""
    ckpt = torch.load(decoder_ckpt, map_location="cpu", weights_only=False)
    cfg  = ckpt["cfg"]

    decoder = AidDecoder(
        horizon=cfg["horizon"],
        action_dim=cfg["action_dim"],
        K=cfg["K"],
        d_enc=cfg["d_enc"],
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
    )
    decoder.load_state_dict(ckpt["decoder_state_dict"])
    decoder.to(device)
    decoder.eval()
    return decoder, cfg


# ---------------------------------------------------------------------------
# Decode with first-n prefix tokens (zeros for the rest)
# ---------------------------------------------------------------------------

@torch.no_grad()
def decode_prefix(decoder: AidDecoder, z_q: torch.Tensor, n: int) -> torch.Tensor:
    """
    z_q: (1, K, d_enc)
    n:   number of prefix tokens to use
    Returns: (1, horizon, action_dim)
    """
    K = z_q.shape[1]
    B = z_q.shape[0]
    mask = prefix_mask(n, K, B, z_q.device)   # (1, K) bool
    return decoder(z_q, mask)                  # (1, H, A)


# ---------------------------------------------------------------------------
# Plot (a): spectral difference heat-map + overlay
# ---------------------------------------------------------------------------

def plot_spectrum_diff(gt: np.ndarray, preds: list, K: int,
                       chunk_idx: int, save_path: str):
    """
    gt:    (T, A)
    preds: list of (T, A) arrays, length K, preds[i] = reconstruction with n=i+1 tokens
    """
    gt_amp    = rfft_amplitude(gt)
    F         = len(gt_amp)
    freq_bins = np.arange(F)

    diff_mat = np.stack([rfft_amplitude(p) - gt_amp for p in preds], axis=0)  # (K, F)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax = axes[0]
    vmax = np.abs(diff_mat).max() * 0.8 + 1e-6
    im = ax.imshow(
        diff_mat, aspect="auto", origin="lower", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        extent=[-0.5, F - 0.5, 0.5, K + 0.5],
        interpolation="nearest",
    )
    ax.set_xlabel("Frequency bin")
    ax.set_ylabel("n (# prefix tokens)")
    ax.set_title(f"Chunk {chunk_idx}: |FFT(â_n)| − |FFT(a)|  (mean over dims)")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.colorbar(im, ax=ax, label="Amplitude diff")

    ax2 = axes[1]
    ax2.plot(freq_bins, gt_amp, "k-", linewidth=2, label="GT")
    cmap_lines = cm.get_cmap("plasma", K)
    for i, pred in enumerate(preds):
        ax2.plot(freq_bins, rfft_amplitude(pred),
                 color=cmap_lines(i / max(K - 1, 1)), alpha=0.7, linewidth=1)
    sm = cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(1, K))
    sm.set_array([])
    plt.colorbar(sm, ax=ax2, label="n (# prefix tokens)")
    ax2.set_xlabel("Frequency bin")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Amplitude spectra (mean over dims)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"rFFT spectrum comparison — chunk {chunk_idx}\n"
        f"spectrum: rFFT magnitude (np.fft.rfft), averaged over {gt.shape[1]} action dims",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot (b): per-n × per-dim spectra grid
# ---------------------------------------------------------------------------

def plot_spectrum_lines(gt: np.ndarray, preds: list, K: int,
                        chunk_idx: int, save_path: str):
    A = gt.shape[1]
    gt_per_dim = rfft_amplitude_per_dim(gt)   # (F, A)
    F          = gt_per_dim.shape[0]
    freq_bins  = np.arange(F)

    ncols = min(4, K)
    nrows = (K + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4 * ncols, 3 * nrows),
                             sharex=True, sharey=True)
    axes_flat = np.array(axes).flatten()
    dim_colors = cm.get_cmap("tab10", min(A, 10))

    for i, pred in enumerate(preds):
        ax = axes_flat[i]
        pred_per_dim = rfft_amplitude_per_dim(pred)
        for d in range(A):
            c = dim_colors(d % 10)
            ax.plot(freq_bins, gt_per_dim[:, d],   color=c, linestyle="--", linewidth=1, alpha=0.5)
            ax.plot(freq_bins, pred_per_dim[:, d],  color=c, linestyle="-",  linewidth=1)
        ax.set_title(f"n={i+1}", fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(bottom=0)

    for j in range(K, len(axes_flat)):
        axes_flat[j].set_visible(False)
    for ax in axes_flat[:K]:
        ax.set_xlabel("Freq bin", fontsize=7)
    fig.text(0.04, 0.5, "Amplitude", va="center", rotation="vertical", fontsize=9)

    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([0], [0], linestyle="--", color="gray", label="GT"),
        Line2D([0], [0], linestyle="-",  color="gray", label="Recon"),
    ], loc="upper right", fontsize=8)

    fig.suptitle(
        f"Per-dim rFFT amplitude — chunk {chunk_idx}, n=1..{K}\n"
        f"(dashed=GT, solid=recon; spectrum: rFFT magnitude per action dim, seed=42)",
        fontsize=10,
    )
    plt.tight_layout(rect=[0.05, 0, 1, 0.95])
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder_ckpt", required=True,
                        help="Path to AidDecoder checkpoint (.pth from train_aid_decoder.py)")
    parser.add_argument("--zq_npy",     default="data/aid_zq.npy",
                        help="Pre-encoded z_q numpy file (N, K, d_enc) float16")
    parser.add_argument("--actions_npy", default="data/aid_actions.npy",
                        help="Normalized action numpy file (N, H, A) float16")
    parser.add_argument("--n_chunks",  type=int, default=10)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--device",    default="cuda")
    parser.add_argument("--out_dir",   default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    decoder_stem = os.path.splitext(os.path.basename(args.decoder_ckpt))[0]
    out_dir = args.out_dir or os.path.join("outputs", "aid_decoder_spectrum", decoder_stem)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # Load decoder (no backbone needed)
    print("Loading AidDecoder ...")
    decoder, cfg = load_decoder(args.decoder_ckpt, device)
    K          = cfg["K"]
    horizon    = cfg["horizon"]
    action_dim = cfg["action_dim"]
    print(f"  K={K}, horizon={horizon}, action_dim={action_dim}")

    # Load z_q + actions via memmap (low RAM)
    print(f"Loading z_q cache ({args.zq_npy}) ...")
    z_q_mm  = np.load(args.zq_npy,      mmap_mode="r")   # (N, K, d_enc) float16
    x_mm    = np.load(args.actions_npy,  mmap_mode="r")   # (N, H, A)     float16
    N = z_q_mm.shape[0]
    print(f"  {N} samples available")

    # Fixed random sample of n_chunks indices
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(N, size=args.n_chunks, replace=False)
    print(f"  Sampled chunk indices (seed={args.seed}): {indices.tolist()}")

    # For each chunk, decode with n=1..K prefix tokens
    print("Generating reconstructions for n=1..K ...")
    for ci, idx in enumerate(indices):
        z_q = torch.from_numpy(z_q_mm[idx].astype(np.float32)).unsqueeze(0).to(device)  # (1,K,d_enc)
        gt_np = x_mm[idx].astype(np.float32)[:horizon, :action_dim]                     # (H, A)

        preds_np = []
        for n in range(1, K + 1):
            recon = decode_prefix(decoder, z_q, n)   # (1, H, A)
            preds_np.append(recon[0].cpu().numpy()[:horizon, :action_dim])

        diff_path  = os.path.join(out_dir, f"chunk_{ci:02d}_spectrum_diff.png")
        lines_path = os.path.join(out_dir, f"chunk_{ci:02d}_spectrum_lines.png")
        plot_spectrum_diff(gt_np, preds_np, K, ci, diff_path)
        plot_spectrum_lines(gt_np, preds_np, K, ci, lines_path)
        print(f"  chunk {ci:02d} (cache idx={idx}): saved")

    meta = {
        "decoder_ckpt":    args.decoder_ckpt,
        "zq_npy":          args.zq_npy,
        "actions_npy":     args.actions_npy,
        "n_chunks":        args.n_chunks,
        "seed":            args.seed,
        "K":               K,
        "horizon":         horizon,
        "action_dim":      action_dim,
        "original_indices": indices.tolist(),
        "spectrum_def":    "rFFT magnitude (np.fft.rfft) along horizon dim, then mean over action dims",
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
