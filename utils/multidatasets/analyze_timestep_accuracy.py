"""
Analyze velocity prediction accuracy of VanillaFlow at different timesteps t.

For each fixed t in [0, 1]:
  - Interpolate x_t = t * x_1 + (1-t) * x_0
  - Compute target velocity: v = x_1 - x_0
  - Query model: pred_v = model(x_t, t, context)
  - Measure L1(pred_v, v)  and  L1(recon_x, x_1)   where recon_x = x_t + (1-t)*pred_v

This reveals where the model is under-trained (useful for comparing logit-normal vs uniform sampling).

Usage:
    python utils/multidatasets/analyze_timestep_accuracy.py -m ckpt/best.pth
    python utils/multidatasets/analyze_timestep_accuracy.py -m ckpt/best.pth --t_steps 50 --n_batches 20
    python utils/multidatasets/analyze_timestep_accuracy.py -m a.pth -m2 b.pth --label1 uniform --label2 lognorm
"""

import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
from pathlib import Path

from catok.models.catok_ddt.vanilla_utils import load_checkpoint, construct_dataloader


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def measure_accuracy_per_t(model, loader, t_grid, device, n_batches, noise_level):
    """
    Returns two arrays of shape (len(t_grid),):
      - vel_l1:   mean L1 between predicted and target velocity
      - recon_l1: mean L1 between one-step reconstruction and ground truth x_1
    """
    model.eval()
    vel_l1_sum   = np.zeros(len(t_grid))
    recon_l1_sum = np.zeros(len(t_grid))
    counts       = np.zeros(len(t_grid))

    for batch_idx, action_chunks in enumerate(tqdm(loader, total=n_batches, desc="batches")):
        if batch_idx >= n_batches:
            break

        x_1 = action_chunks.to(device=device, dtype=torch.float32)
        b   = x_1.shape[0]

        # Encode once per batch (shared across all t values)
        z   = model.encoder(x_1)
        if model.vq_mode is not None:
            _, z_q, _, _ = model.vq(z)
        else:
            z_q = z

        x_0 = torch.randn_like(x_1) * noise_level   # fixed noise for this batch

        for i, t_val in enumerate(t_grid):
            t_img   = torch.full((b, 1, 1), t_val, device=device, dtype=torch.float32)
            t_batch = torch.full((b,),      t_val, device=device, dtype=torch.float32)

            x_t      = t_img * x_1 + (1.0 - t_img) * x_0
            target_v = x_1 - x_0

            pred_v, _ = model.decoder(
                x_t,
                t_batch,
                encoder_hidden_states=z_q,
                context_see_xt=model.context_see_xt,
                mask=model.flow._get_cond_mask(t_batch, n_tokens=z_q.shape[1]),
            )

            recon_x = x_t + (1.0 - t_img) * pred_v

            vel_l1_sum[i]   += F.l1_loss(pred_v, target_v).item() * b
            recon_l1_sum[i] += F.l1_loss(recon_x, x_1).item() * b
            counts[i]       += b

    vel_l1   = vel_l1_sum   / counts
    recon_l1 = recon_l1_sum / counts
    return vel_l1, recon_l1


def run(args):
    set_seed(42)
    device = args.device

    t_grid = np.linspace(args.t_min, args.t_max, args.t_steps)

    # ── Load models ──────────────────────────────────────────────────────────
    results = []
    for ckpt_path, label in zip(args.model_paths, args.labels):
        print(f"\n[{label}] Loading {ckpt_path}")
        model, cfg = load_checkpoint(ckpt_path, device)
        model.eval()

        noise_level = getattr(model.flow, "noise_level", 1.0)
        print(f"  noise_level = {noise_level}")

        _, loader, _ = construct_dataloader(
            cfg,
            batch_size=args.batch_size,
            num_workers=4,
            eval_mode=True,
            dataset_names=args.dataset if args.dataset else None,
        )

        vel_l1, recon_l1 = measure_accuracy_per_t(
            model, loader, t_grid, device, args.n_batches, noise_level
        )
        results.append(dict(label=label, vel_l1=vel_l1, recon_l1=recon_l1))

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for r in results:
        axes[0].plot(t_grid, r["vel_l1"],   label=r["label"], marker="o", markersize=3)
        axes[1].plot(t_grid, r["recon_l1"], label=r["label"], marker="o", markersize=3)

    axes[0].set_title("Velocity prediction L1  vs  t")
    axes[0].set_xlabel("Timestep t")
    axes[0].set_ylabel("L1 loss")
    axes[0].legend()
    axes[0].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].set_title("One-step reconstruction L1  vs  t")
    axes[1].set_xlabel("Timestep t")
    axes[1].set_ylabel("L1 loss")
    axes[1].legend()
    axes[1].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    axes[1].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot → {out_path}")
    plt.close()

    # ── Print table ───────────────────────────────────────────────────────────
    print("\nt       ", end="")
    for r in results:
        print(f"  vel_l1[{r['label']}]  recon_l1[{r['label']}]", end="")
    print()
    for i, t_val in enumerate(t_grid):
        print(f"t={t_val:.3f}", end="")
        for r in results:
            print(f"  {r['vel_l1'][i]:.5f}          {r['recon_l1'][i]:.5f}      ", end="")
        print()


def main():
    parser = argparse.ArgumentParser(description="Analyze per-timestep prediction accuracy")
    parser.add_argument("-m",  "--model_paths", dest="model_paths", action="append",
                        required=True, metavar="CKPT",
                        help="Checkpoint path(s). Use -m ckpt1.pth -m ckpt2.pth for comparison.")
    parser.add_argument("--label", dest="labels", action="append", default=None,
                        metavar="NAME",
                        help="Legend label for each model (same order as -m). "
                             "Defaults to checkpoint filename.")
    parser.add_argument("--t_steps",  type=int,   default=20,     help="Number of t grid points (default 20)")
    parser.add_argument("--t_min",    type=float, default=0.0,  help="Minimum t (default 0.025)")
    parser.add_argument("--t_max",    type=float, default=1.0,  help="Maximum t (default 0.975)")
    parser.add_argument("--n_batches",type=int,   default=10,     help="Number of batches to average (default 10)")
    parser.add_argument("--batch_size",type=int,  default=512,    help="Batch size (default 512)")
    parser.add_argument("--dataset",  type=str,   default=None,   help="Evaluate on a single dataset by name")
    parser.add_argument("--device",   type=str,   default="cuda")
    parser.add_argument("-o", "--output", type=str,
                        default="eval_timestep_accuracy.png",
                        help="Output plot path (default: eval_timestep_accuracy.png)")
    args = parser.parse_args()

    # Fill in default labels from filenames
    if args.labels is None:
        args.labels = [Path(p).stem for p in args.model_paths]
    while len(args.labels) < len(args.model_paths):
        args.labels.append(Path(args.model_paths[len(args.labels)]).stem)

    run(args)


if __name__ == "__main__":
    main()
