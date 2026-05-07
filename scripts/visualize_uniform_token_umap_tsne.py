"""
Load latest (or specified) vanilla checkpoint, sample N action chunks from the
training mixture, encode to discrete codes + codebook vectors (B, num_tokens, D), and
plot t-SNE / UMAP for a subset of token positions (default 0,4,8,12).

Example:
  cd /path/to/CausalActionTokenizer
  python scripts/visualize_uniform_token_umap_tsne.py --ckpt_dir ckpt/vanilla16
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.manifold import TSNE
from tqdm import tqdm

from catok.models.catok_ddt.vanilla_utils import construct_dataloader, load_checkpoint


def find_last_ckpt(ckpt_dir: str | Path) -> Path:
    paths = list(Path(ckpt_dir).glob("*.pth"))
    if not paths:
        raise FileNotFoundError(f"No .pth in {ckpt_dir}")

    def step(p: Path) -> int:
        m = re.search(r"(\d+)\.pth$", p.name)
        return int(m.group(1)) if m else -1

    return max(paths, key=step)


@torch.no_grad()
def encode_codebook_tokens(model, actions: torch.Tensor, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Discrete codes (B, K) and codebook vectors (B, K, D).

    D equals ``vq.vq_dim`` (often 128 when ``add_vq_latent=False``, even if yaml lists ``d_vq``).
    """
    device = next(model.parameters()).device
    latents = []
    indices_list = []
    for i in range(0, len(actions), batch_size):
        batch = actions[i : i + batch_size].to(device=device, dtype=torch.float32)
        z = model.encoder(batch)
        if model.vq_mode is None:
            raise ValueError("Model has no VQ.")
        _, _, indices, _ = model.vq(z)
        emb = F.embedding(indices.long(), model.vq.embedding)
        latents.append(emb.cpu())
        indices_list.append(indices.cpu().to(torch.int32))

    token_emb = torch.cat(latents, dim=0).numpy()
    token_ids = torch.cat(indices_list, dim=0).numpy()
    return token_emb, token_ids


def _scatter_by_label(
    coords: np.ndarray,
    labels: np.ndarray,
    token_indices: list[int],
    title: str,
    save_path: Path,
    colors_rgba: np.ndarray,
):
    fig, ax = plt.subplots(figsize=(10, 8))
    for k, tok in enumerate(token_indices):
        mask = labels == tok
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[colors_rgba[k]],
            s=12,
            alpha=0.88,
            label=f"pos {tok}",
            edgecolors="none",
        )
    ax.legend(markerscale=2, fontsize=10, loc="best")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default="ckpt/vanilla16")
    parser.add_argument("--ckpt", default=None, help="Override .pth path")
    parser.add_argument("--n_samples", type=int, default=4028)
    parser.add_argument("--token_indices", type=int, nargs="+", default=[0, 4, 8, 12])
    parser.add_argument("--out_dir", default="visualize")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tsne_perplexity", type=float, default=40)
    parser.add_argument("--umap_neighbors", type=int, default=50)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help="Optional tag appended to output filenames to avoid overwriting prior runs.",
    )
    args = parser.parse_args()

    suffix = f"_{args.output_suffix}" if args.output_suffix else ""

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt_path = Path(args.ckpt) if args.ckpt else find_last_ckpt(args.ckpt_dir)
    print(f"Checkpoint: {ckpt_path.resolve()}")

    device = args.device if torch.cuda.is_available() else "cpu"
    model, cfg = load_checkpoint(str(ckpt_path), device)
    model.eval()

    target_n = args.n_samples
    _, _, concat_ds = construct_dataloader(
        cfg,
        batch_size=min(args.batch_size, target_n),
        num_workers=args.num_workers,
        eval_mode=True,
    )

    n_ds = len(concat_ds)
    if n_ds < target_n:
        raise RuntimeError(f"Dataset size {n_ds} < required {target_n}")

    subset_idx = rng.choice(n_ds, size=target_n, replace=False)
    sampler = SubsetRandomSampler(subset_idx.tolist())
    bs = min(args.batch_size, target_n)
    loader = DataLoader(
        concat_ds,
        batch_size=bs,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    chunks: list[torch.Tensor] = []
    for batch in tqdm(loader, desc="sample action chunks", total=(target_n + bs - 1) // bs):
        chunks.append(batch)

    if not chunks:
        raise RuntimeError("DataLoader returned no batches (missing data paths?)")

    actions = torch.cat(chunks, dim=0)
    assert actions.shape[0] == target_n, (actions.shape[0], target_n)

    token_emb, token_ids = encode_codebook_tokens(model, actions, batch_size=256)
    n, k_tok, d_emb = token_emb.shape
    print(f"codebook token_emb {token_emb.shape}; discrete token_ids {token_ids.shape}")

    token_indices = list(args.token_indices)
    for t in token_indices:
        if t < 0 or t >= k_tok:
            raise ValueError(f"token index {t} out of range [0, {k_tok})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = ckpt_path.stem
    tag = "_".join(map(str, token_indices))
    npz_path = out_dir / f"{stem}_n{target_n}_tokens_{tag}{suffix}.npz"
    np.savez_compressed(
        npz_path,
        token_emb=token_emb.astype(np.float32),
        token_ids=token_ids.astype(np.int32),
        token_positions=np.array(token_indices, dtype=np.int32),
        seed=args.seed,
    )
    print(f"Saved embeddings: {npz_path}")

    sub = token_emb[:, token_indices, :]
    flat = sub.reshape(-1, d_emb)
    labels = np.tile(np.array(token_indices, dtype=np.int64), n)

    n_pts = flat.shape[0]
    perp = min(args.tsne_perplexity, max(5, n_pts // 4))
    print(f"t-SNE: n={n_pts}, perplexity={perp}")
    try:
        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            max_iter=1000,
            random_state=args.seed,
            init="pca",
        )
    except TypeError:
        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            n_iter=1000,
            random_state=args.seed,
            init="pca",
        )
    coords_tsne = tsne.fit_transform(flat)

    cmap = plt.colormaps["tab10"]
    colors = cmap(np.linspace(0, 0.9, len(token_indices)))

    title_tsne = (
        f'{cfg.get("exp_name", "model")} ckpt {stem} — t-SNE '
        f"(N={n}, dim={d_emb}, pos={token_indices}, seed={args.seed}, perp={perp})"
    )
    tsne_png = out_dir / f"{stem}_n{n}_{tag}{suffix}_tsne.png"
    _scatter_by_label(coords_tsne, labels, token_indices, title_tsne, tsne_png, colors)
    print(f"Saved {tsne_png}")

    import umap

    n_n = min(args.umap_neighbors, max(2, n_pts - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_n,
        min_dist=args.umap_min_dist,
        random_state=args.seed,
    )
    print(f"UMAP: n_neighbors={n_n}, min_dist={args.umap_min_dist}")
    coords_umap = reducer.fit_transform(flat)
    title_umap = (
        f'{cfg.get("exp_name", "model")} ckpt {stem} — UMAP '
        f"(N={n}, dim={d_emb}, pos={token_indices}, seed={args.seed}, "
        f"n_neighbors={n_n}, min_dist={args.umap_min_dist})"
    )
    umap_png = out_dir / f"{stem}_n{n}_{tag}{suffix}_umap.png"
    _scatter_by_label(coords_umap, labels, token_indices, title_umap, umap_png, colors)
    print(f"Saved {umap_png}")


if __name__ == "__main__":
    main()
