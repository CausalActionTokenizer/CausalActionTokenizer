"""
Aid Decoder for token-level visualization of CausalActionTokenizer.

Standalone file — no coupling with existing library modules.
The encoder + VQ from a pretrained VanillaVQDiffusion checkpoint are loaded
and frozen. Only the AidDecoder parameters are trained.

Architecture:
  K VQ token embeddings (frozen)  +  tail-causal mask  +  learnable action queries
  ──> N layers of cross-attention (tokens → action queries)  +  self-attention
  ──> linear head  ──> (B, horizon, action_dim)

Tail-causal mask strategy (training):
  Uniformly sample k ~ Uniform(1, K), then mask the last k tokens.
  Pattern: [o o o x x x] where x = masked (replaced by zero).
  At visualization time: use first-n-prefix mode (all tokens after position n are zeroed).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_dct import dct as torch_dct_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dct_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """DCT MSE loss along the horizon dimension."""
    # pred/target: (B, T, A)
    x = target.permute(0, 2, 1)   # (B, A, T)
    y = pred.permute(0, 2, 1)
    return F.mse_loss(torch_dct_fn(x, norm="ortho"), torch_dct_fn(y, norm="ortho"))


def sample_tail_mask(B: int, K: int, device: torch.device, rng: torch.Generator = None) -> torch.Tensor:
    """
    For each sample in batch, uniformly draw k in [1, K] and mask the last k tokens.
    Returns bool tensor (B, K): True = token is visible, False = masked.
    """
    # rng must live on CPU; generate on CPU then move to target device
    if rng is not None:
        k_vals = torch.randint(1, K + 1, (B,), generator=rng).to(device)
    else:
        k_vals = torch.randint(1, K + 1, (B,), device=device)
    # position indices 0..K-1; token i is masked if i >= K - k
    idx = torch.arange(K, device=device).unsqueeze(0)   # (1, K)
    threshold = (K - k_vals).unsqueeze(1)                # (B, 1)
    visible = idx < threshold                            # (B, K)
    return visible


def prefix_mask(n: int, K: int, B: int, device: torch.device) -> torch.Tensor:
    """Mask for visualization: only first n tokens are visible."""
    mask = torch.zeros(B, K, dtype=torch.bool, device=device)
    mask[:, :n] = True
    return mask


def single_token_mask(token_idx: int, K: int, B: int, device: torch.device) -> torch.Tensor:
    """Only VQ position ``token_idx`` is visible (AidDecoder cross-attn sees one token)."""
    mask = torch.zeros(B, K, dtype=torch.bool, device=device)
    mask[:, token_idx] = True
    return mask


# ---------------------------------------------------------------------------
# Cross-attention block: tokens (context) -> action queries
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

        self.norm_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

        self.norm_ff = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # cross-attention: q attends to kv (visible tokens)
        q2, _ = self.cross_attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv))
        q = q + q2
        # self-attention among action queries
        q2, _ = self.self_attn(self.norm_self(q), self.norm_self(q), self.norm_self(q))
        q = q + q2
        # feed-forward
        q = q + self.ff(self.norm_ff(q))
        return q


# ---------------------------------------------------------------------------
# AidDecoder
# ---------------------------------------------------------------------------

class AidDecoder(nn.Module):
    """
    Auxiliary decoder trained on top of frozen encoder + VQ.

    Args:
        horizon:    action chunk length (T)
        action_dim: number of action dimensions (A)
        K:          number of VQ tokens per chunk
        d_enc:      embedding dimension from the frozen encoder / VQ
        d_model:    internal hidden size of the aid decoder
        num_layers: number of CrossAttentionBlock layers
        num_heads:  attention heads (must divide d_model)
    """
    def __init__(
        self,
        horizon: int = 20,
        action_dim: int = 7,
        K: int = 32,
        d_enc: int = 128,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()
        self.K = K
        self.horizon = horizon
        self.action_dim = action_dim

        # Project VQ embeddings into d_model space
        self.token_proj = nn.Linear(d_enc, d_model)

        # Learnable action queries: one per time step
        self.action_queries = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)

        # Positional embeddings for action queries
        self.action_pos = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)

        # Cross-attention layers
        self.layers = nn.ModuleList([
            CrossAttentionBlock(d_model, num_heads)
            for _ in range(num_layers)
        ])

        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, action_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, z_q: torch.Tensor, visible_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_q:          (B, K, d_enc) — VQ embeddings (from frozen VQ)
            visible_mask: (B, K) bool — True = token is passed to decoder

        Returns:
            (B, horizon, action_dim)
        """
        B = z_q.shape[0]

        # Zero out masked tokens so they carry no information
        tokens = z_q * visible_mask.unsqueeze(-1).float()   # (B, K, d_enc)
        tokens = self.token_proj(tokens)                     # (B, K, d_model)

        # Build action queries
        q = self.action_queries.expand(B, -1, -1) + self.action_pos.expand(B, -1, -1)

        for layer in self.layers:
            q = layer(q, tokens)

        q = self.norm_out(q)
        out = self.head(q)   # (B, horizon, action_dim)
        return out


# ---------------------------------------------------------------------------
# Full model: frozen backbone + trainable AidDecoder
# ---------------------------------------------------------------------------

class AidDecoderModel(nn.Module):
    """
    Wraps frozen VanillaVQDiffusion (encoder + VQ) and a trainable AidDecoder.
    """
    def __init__(
        self,
        ckpt_path: str,
        device: torch.device,
        # AidDecoder hyper-params
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()

        # --- Load frozen backbone ---
        from catok.models.catok_ddt.vanilla_utils import load_checkpoint
        backbone, cfg = load_checkpoint(ckpt_path, device=device)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        # Also freeze buffers by keeping eval mode; we store as non-module attribute
        # so optimizer never touches it.
        self._backbone = backbone  # not registered as nn.Module submodule
        self._register_backbone_buffers(backbone)

        tok_cfg = cfg['tokenizer']
        horizon    = tok_cfg['basic']['horizon']
        action_dim = tok_cfg['basic']['action_dim']
        K          = tok_cfg['encoder']['num_tokens']
        d_enc      = tok_cfg['encoder']['d_model']

        self.horizon    = horizon
        self.action_dim = action_dim
        self.K          = K

        # --- Trainable AidDecoder ---
        self.decoder = AidDecoder(
            horizon=horizon,
            action_dim=action_dim,
            K=K,
            d_enc=d_enc,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
        )

        self._device = device

    def _register_backbone_buffers(self, backbone):
        """Move backbone to a plain attribute so DDP / optimizer ignore it."""
        # Store backbone directly; call .to(device) manually.
        pass

    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, x: torch.Tensor):
        """Run frozen encoder + VQ. Returns z_q (B, K, d_enc)."""
        self._backbone.eval()
        z = self._backbone.encoder(x)
        _, z_q, _, _ = self._backbone.vq(z)
        return z_q

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        rng: torch.Generator = None,
    ):
        """
        Training forward. Samples a tail-causal mask, encodes x, decodes.

        Returns:
            recon:     (B, horizon, action_dim)
            loss:      scalar
            loss_dict: dict with 'recon' and 'dct' keys
        """
        B = x.shape[0]
        z_q = self.encode(x)                                        # (B, K, d_enc)
        mask = sample_tail_mask(B, self.K, x.device, rng=rng)      # (B, K) bool

        recon = self.decoder(z_q, mask)                             # (B, H, A)

        # Trim to actual action_dim if needed (x may be padded)
        tgt = x[:, :self.horizon, :self.action_dim]

        l1   = F.l1_loss(recon, tgt)
        dct  = _dct_loss(recon, tgt)
        loss = l1 + dct

        return recon, loss, {'recon': l1.item(), 'dct': dct.item()}

    @torch.no_grad()
    def decode_prefix(self, x: torch.Tensor, n: int) -> torch.Tensor:
        """
        Visualization: decode using only the first n tokens (prefix).

        Args:
            x: (B, horizon, action_dim) normalized action chunk
            n: number of prefix tokens to use (1 <= n <= K)

        Returns:
            (B, horizon, action_dim) reconstructed action
        """
        assert 1 <= n <= self.K, f"n must be in [1, {self.K}], got {n}"
        B = x.shape[0]
        z_q = self.encode(x)
        mask = prefix_mask(n, self.K, B, x.device)
        return self.decoder(z_q, mask)
