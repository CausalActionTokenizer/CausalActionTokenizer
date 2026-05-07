"""
Per-Token Aid Decoder — standalone model definition.

Architecture:
  Single token embedding (d_enc) + token index embedding
  → MLP decoder → (horizon, action_dim)

One decoder is conditioned on WHICH token it receives (via learned index embedding),
so a single model handles all K=32 tokens.

No coupling with the existing library.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_dct import dct as torch_dct_fn


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def _dct_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """DCT MSE loss along the horizon dimension. pred/target: (B, T, A)."""
    x = target.permute(0, 2, 1)
    y = pred.permute(0, 2, 1)
    return F.mse_loss(torch_dct_fn(x, norm="ortho"), torch_dct_fn(y, norm="ortho"))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PerTokenAidDecoder(nn.Module):
    """
    Decoder conditioned on a single token (index i, embedding z_q[:, i, :]).

    Forward:
        token_emb: (B, d_enc)  — the single token's VQ embedding
        token_idx: (B,) int    — which token index this is (0..K-1)
    Returns:
        (B, horizon, action_dim)

    At visualization time, call forward for each i in 0..K-1 with the
    corresponding token embedding, to get per-token reconstructions.
    """

    def __init__(
        self,
        horizon: int = 20,
        action_dim: int = 7,
        K: int = 32,
        d_enc: int = 128,
        d_model: int = 256,
        num_layers: int = 4,
    ):
        super().__init__()
        self.K = K
        self.horizon = horizon
        self.action_dim = action_dim

        # Token index embedding — tells the decoder which position this token is
        self.idx_emb = nn.Embedding(K, d_model)

        # Project VQ embedding → d_model
        self.token_proj = nn.Linear(d_enc, d_model)

        # MLP decoder: (fused token + index) → flat action
        layers = []
        in_dim = d_model
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, d_model), nn.GELU()]
            in_dim = d_model
        layers.append(nn.Linear(d_model, horizon * action_dim))
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, token_emb: torch.Tensor, token_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_emb: (B, d_enc)
            token_idx: (B,) long

        Returns:
            (B, horizon, action_dim)
        """
        B = token_emb.shape[0]
        t = self.token_proj(token_emb)            # (B, d_model)
        t = t + self.idx_emb(token_idx)           # (B, d_model)
        out = self.mlp(t)                         # (B, horizon * action_dim)
        return out.view(B, self.horizon, self.action_dim)
