# Copyright (C) 2025. All rights reserved.
# Action VAE for converting (B, T, D) action sequences to continuous latents

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
from scipy.fft import dct, idct

from catok.training.statistics import build_action_normalizer


class CausalConv1d(nn.Module):
    """Causal 1D convolution that only looks at past/current timesteps."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
    
    def forward(self, x):
        # x: (B, C, T)
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]  # Remove future padding
        return x


class ResidualBlock(nn.Module):
    """Residual block with optional downsampling/upsampling."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        downsample: bool = False,
        upsample: bool = False,
        use_causal: bool = False,
    ):
        super().__init__()
        self.downsample = downsample
        self.upsample = upsample
        
        Conv1d = CausalConv1d if use_causal else lambda ic, oc, ks, **kw: nn.Conv1d(ic, oc, ks, padding=ks//2, **kw)
        
        self.norm1 = nn.GroupNorm(min(8, in_channels), in_channels)
        self.conv1 = Conv1d(in_channels, out_channels, kernel_size)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = Conv1d(out_channels, out_channels, kernel_size)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
        
        self.act = nn.SiLU()
    
    def forward(self, x):
        # x: (B, C, T)
        h = self.norm1(x)
        h = self.act(h)
        
        if self.upsample:
            h = F.interpolate(h, scale_factor=2, mode='linear', align_corners=False)
            x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
        
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        
        if self.downsample:
            h = F.avg_pool1d(h, 2)
            x = F.avg_pool1d(x, 2)
        
        return self.skip(x) + h


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and MLP."""
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        # x: (B, T, D)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class Encoder(nn.Module):
    """VAE Encoder: (B, T, D) -> (B, latent_dim) for mu and logvar.
    
    Transformer mode: uses max(hidden_dims) as hidden size, then projects to latent.
    Conv mode: progressively increases dimensions through hidden_dims.
    
    Optionally includes magnitude embedding to preserve scale information.
    """
    def __init__(
        self,
        action_dim: int,
        action_T: int,
        hidden_dims: list = [64, 128, 256],
        latent_dim: int = 32,
        use_transformer: bool = True,
        num_transformer_layers: int = 2,
        num_heads: int = 4,
        use_magnitude_embedding: bool = False,
        magnitude_embed_dim: int = 16,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_T = action_T
        self.latent_dim = latent_dim
        self.use_transformer = use_transformer
        self.hidden_dims = hidden_dims
        self.use_magnitude_embedding = use_magnitude_embedding
        self.magnitude_embed_dim = magnitude_embed_dim
        
        # Transformer uses max hidden dim
        self.transformer_hidden = max(hidden_dims)
        
        if use_transformer:
            # Input projection: (B, T, D) -> (B, T, max_hidden)
            self.input_proj = nn.Linear(action_dim, self.transformer_hidden)
            
            # Magnitude embedding: embed per-dimension magnitude -> transformer_hidden
            # Use magnitude_embed_dim as intermediate dimension
            if use_magnitude_embedding:
                self.magnitude_embed = nn.Sequential(
                    nn.Linear(action_dim, magnitude_embed_dim),
                    nn.SiLU(),
                    nn.Linear(magnitude_embed_dim, self.transformer_hidden),  # Output matches transformer dim
                )
            
            # Transformer-based encoder with max hidden dim
            self.pos_embed = nn.Parameter(torch.randn(1, action_T, self.transformer_hidden) * 0.02)
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(self.transformer_hidden, num_heads)
                for _ in range(num_transformer_layers)
            ])
            self.pool = nn.AdaptiveAvgPool1d(1)
            final_dim = self.transformer_hidden
        else:
            # Conv-based encoder: progressively increase dimensions
            self.input_proj = nn.Linear(action_dim, hidden_dims[0])
            
            # Magnitude embedding for conv mode
            if use_magnitude_embedding:
                self.magnitude_embed = nn.Sequential(
                    nn.Linear(action_dim, magnitude_embed_dim),
                    nn.SiLU(),
                    nn.Linear(magnitude_embed_dim, hidden_dims[0]),
                )
            
            layers = []
            in_dim = hidden_dims[0]
            for out_dim in hidden_dims[1:]:
                layers.append(ResidualBlock(in_dim, out_dim, downsample=True))
                in_dim = out_dim
            self.conv_layers = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool1d(1)
            final_dim = hidden_dims[-1]
        
        # Output projection to mu and logvar
        self.fc_mu = nn.Linear(final_dim, latent_dim)
        self.fc_logvar = nn.Linear(final_dim, latent_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        magnitude: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D) action sequences (normalized)
            magnitude: (B, D) per-dimension magnitude (pre-normalization), optional
        Returns:
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
        """
        B, T, D = x.shape
        
        # Input projection
        h = self.input_proj(x)  # (B, T, hidden)
        
        # Add magnitude embedding if enabled (directly outputs transformer_hidden dim)
        if self.use_magnitude_embedding and magnitude is not None:
            mag_hidden = self.magnitude_embed(magnitude)  # (B, transformer_hidden)
            h = h + mag_hidden.unsqueeze(1)  # Add to all timesteps
        
        if self.use_transformer:
            # Add positional embedding
            h = h + self.pos_embed[:, :T, :]
            
            # Transformer blocks
            for block in self.transformer_blocks:
                h = block(h)
            
            # Pool across time: (B, T, hidden) -> (B, hidden)
            h = h.transpose(1, 2)  # (B, hidden, T)
            h = self.pool(h).squeeze(-1)  # (B, hidden)
        else:
            # Conv processing
            h = h.transpose(1, 2)  # (B, hidden, T)
            h = self.conv_layers(h)
            h = self.pool(h).squeeze(-1)  # (B, hidden)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class Decoder(nn.Module):
    """VAE Decoder: (B, latent_dim) -> (B, T, D).
    
    Transformer mode: uses max(hidden_dims) for transformer, then progressively 
    downsamples through hidden_dims (sorted descending) to action_dim.
    Conv mode: progressively decreases dimensions through hidden_dims.
    
    Optionally predicts magnitude to help scale reconstruction.
    """
    def __init__(
        self,
        action_dim: int,
        action_T: int,
        hidden_dims: list = [256, 128, 64],
        latent_dim: int = 32,
        use_transformer: bool = True,
        num_transformer_layers: int = 2,
        num_heads: int = 4,
        use_magnitude_embedding: bool = False,
        magnitude_embed_dim: int = 16,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_T = action_T
        self.latent_dim = latent_dim
        self.use_transformer = use_transformer
        self.hidden_dims = hidden_dims
        self.use_magnitude_embedding = use_magnitude_embedding
        self.magnitude_embed_dim = magnitude_embed_dim
        
        # Transformer uses max hidden dim
        self.transformer_hidden = max(hidden_dims)
        
        # Magnitude prediction and conditioning
        if use_magnitude_embedding:
            # Predict magnitude from latent: latent_dim -> magnitude_embed_dim -> action_dim
            self.magnitude_predictor = nn.Sequential(
                nn.Linear(latent_dim, magnitude_embed_dim),
                nn.SiLU(),
                nn.Linear(magnitude_embed_dim, action_dim),  # Predict per-dimension magnitude
            )
            # Embed predicted magnitude back to transformer_hidden for conditioning
            self.magnitude_embed = nn.Sequential(
                nn.Linear(action_dim, magnitude_embed_dim),
                nn.SiLU(),
                nn.Linear(magnitude_embed_dim, self.transformer_hidden),
            )
        
        if use_transformer:
            # Transformer-based decoder with max hidden dim
            self.latent_proj = nn.Linear(latent_dim, self.transformer_hidden)
            self.pos_embed = nn.Parameter(torch.randn(1, action_T, self.transformer_hidden) * 0.02)
            
            # Learnable query tokens for each timestep
            self.query_embed = nn.Parameter(torch.randn(1, action_T, self.transformer_hidden) * 0.02)
            
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(self.transformer_hidden, num_heads)
                for _ in range(num_transformer_layers)
            ])
            
            # Progressive downsampling: max_hidden -> hidden_dims (descending) -> action_dim
            # Sort hidden_dims in descending order for downsampling
            sorted_dims = sorted(hidden_dims, reverse=True)
            
            # Build downsampling MLP: transformer_hidden -> sorted_dims[1:] -> action_dim
            downsample_layers = []
            in_dim = self.transformer_hidden
            for out_dim in sorted_dims[1:]:  # Skip the first (max) since transformer already outputs it
                downsample_layers.append(nn.Linear(in_dim, out_dim))
                downsample_layers.append(nn.SiLU())
                in_dim = out_dim
            downsample_layers.append(nn.Linear(in_dim, action_dim))
            self.output_proj = nn.Sequential(*downsample_layers)
        else:
            # Conv-based decoder
            self.latent_proj = nn.Linear(latent_dim, hidden_dims[0] * (action_T // (2 ** (len(hidden_dims) - 1))))
            
            layers = []
            in_dim = hidden_dims[0]
            for i, out_dim in enumerate(hidden_dims[1:]):
                layers.append(ResidualBlock(in_dim, out_dim, upsample=True))
                in_dim = out_dim
            self.conv_layers = nn.Sequential(*layers)
            
            self.output_proj = nn.Conv1d(hidden_dims[-1], action_dim, 3, padding=1)
            self.target_T = action_T
    
    def forward(
        self, 
        z: torch.Tensor,
        return_magnitude: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            z: (B, latent_dim) latent codes
            return_magnitude: whether to return predicted magnitude
        Returns:
            x_recon: (B, T, D) reconstructed action sequences
            mag_pred: (B, D) predicted magnitude (if return_magnitude=True and use_magnitude_embedding)
        """
        B = z.shape[0]
        mag_pred = None
        
        # Predict magnitude from latent
        if self.use_magnitude_embedding:
            mag_pred = self.magnitude_predictor(z)  # (B, action_dim)
        
        if self.use_transformer:
            # Project latent and broadcast to sequence
            h = self.latent_proj(z)  # (B, transformer_hidden)
            
            # Add magnitude conditioning (embed predicted magnitude to transformer_hidden)
            if self.use_magnitude_embedding and mag_pred is not None:
                mag_hidden = self.magnitude_embed(mag_pred)  # (B, transformer_hidden)
                h = h + mag_hidden
            
            h = h.unsqueeze(1)  # (B, 1, transformer_hidden)
            
            # Combine with query embeddings
            queries = self.query_embed.expand(B, -1, -1)  # (B, T, transformer_hidden)
            h = queries + h + self.pos_embed  # (B, T, transformer_hidden)
            
            # Transformer blocks
            for block in self.transformer_blocks:
                h = block(h)
            
            # Progressive downsampling to action dim
            x_recon = self.output_proj(h)  # (B, T, action_dim)
        else:
            # Conv processing
            h = self.latent_proj(z)  # (B, hidden * T_small)
            T_small = self.action_T // (2 ** (len(self.conv_layers) + 1))
            h = h.view(B, -1, T_small)  # (B, hidden, T_small)
            
            h = self.conv_layers(h)
            
            # Interpolate to target length if needed
            if h.shape[-1] != self.target_T:
                h = F.interpolate(h, size=self.target_T, mode='linear', align_corners=False)
            
            x_recon = self.output_proj(h)  # (B, D, T)
            x_recon = x_recon.transpose(1, 2)  # (B, T, D)
        
        if return_magnitude:
            return x_recon, mag_pred
        return x_recon


class ActionVAE(nn.Module):
    """
    Variational Autoencoder for action sequences.
    
    Transforms (B, T, D) action sequences to continuous latent representations
    and reconstructs them back.
    
    Optionally uses magnitude embedding to preserve scale information before normalization.
    """
    def __init__(
        self,
        action_dim: int = 7,
        action_T: int = 8,
        hidden_dims: list = [64, 128, 256],
        latent_dim: int = 32,
        use_transformer: bool = True,
        num_transformer_layers: int = 2,
        num_heads: int = 4,
        kl_weight: float = 1e-3,
        normalizer: Optional[str] = None,
        normalizer_config: Optional[dict] = None,
        encoding_mode: Optional[str] = None,  # 'dct' or None
        dct_scale: float = 1.0,
        # Magnitude embedding options
        use_magnitude_embedding: bool = False,
        magnitude_embed_dim: int = 16,
        magnitude_loss_weight: float = 0.1,
        magnitude_type: str = 'rms',  # 'rms', 'max', 'mean'
        # Loss options
        recon_l1_weight: float = 0.0,  # Weight for L1 reconstruction loss
        use_huber_loss: bool = False,  # Use Huber loss instead of MSE
        huber_delta: float = 1.0,  # Delta parameter for Huber loss
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.action_T = action_T
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.encoding_mode = encoding_mode
        self.dct_scale = dct_scale
        
        # Loss settings
        self.recon_l1_weight = recon_l1_weight
        self.use_huber_loss = use_huber_loss
        self.huber_delta = huber_delta
        
        # Magnitude embedding settings
        self.use_magnitude_embedding = use_magnitude_embedding
        self.magnitude_embed_dim = magnitude_embed_dim
        self.magnitude_loss_weight = magnitude_loss_weight
        self.magnitude_type = magnitude_type
        
        # Build encoder and decoder
        # Encoder: projects to max(hidden_dims) for transformer
        self.encoder = Encoder(
            action_dim=action_dim,
            action_T=action_T,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            use_transformer=use_transformer,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
            use_magnitude_embedding=use_magnitude_embedding,
            magnitude_embed_dim=magnitude_embed_dim,
        )
        
        # Decoder: uses max(hidden_dims) for transformer, then downsamples
        # For conv mode, reverse the hidden_dims for upsampling
        decoder_hidden_dims = hidden_dims[::-1]
        self.decoder = Decoder(
            action_dim=action_dim,
            action_T=action_T,
            hidden_dims=decoder_hidden_dims,
            latent_dim=latent_dim,
            use_transformer=use_transformer,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
            use_magnitude_embedding=use_magnitude_embedding,
            magnitude_embed_dim=magnitude_embed_dim,
        )
        
        # Build normalizer
        self.normalizer = build_action_normalizer(normalizer, normalizer_config)
    
    def compute_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-dimension magnitude statistics before normalization.
        
        Args:
            x: (B, T, D) raw action sequences
        Returns:
            magnitude: (B, D) per-dimension magnitude
        """
        if self.magnitude_type == 'rms':
            # Root Mean Square per dimension
            magnitude = torch.sqrt(torch.mean(x ** 2, dim=1))  # (B, D)
        elif self.magnitude_type == 'max':
            # Max absolute value per dimension
            magnitude = torch.max(torch.abs(x), dim=1)[0]  # (B, D)
        elif self.magnitude_type == 'mean':
            # Mean absolute value per dimension
            magnitude = torch.mean(torch.abs(x), dim=1)  # (B, D)
        else:
            raise ValueError(f"Unknown magnitude_type: {self.magnitude_type}")
        return magnitude
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * eps."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def prepare_data(self, action_chunks: torch.Tensor) -> torch.Tensor:
        """Preprocess action data (normalize and optionally apply DCT)."""
        to_device = action_chunks.device
        to_dtype = action_chunks.dtype
        transformed = action_chunks

        if self.normalizer is not None:
            transformed = transformed.cpu().detach().numpy()
            transformed = self.normalizer.normalize(transformed)
            transformed = torch.from_numpy(transformed).to(device=to_device, dtype=to_dtype)
        
        if self.encoding_mode == 'dct':
            transformed = transformed.cpu().detach().numpy()
            transformed = dct(transformed, axis=1, norm="ortho") * self.dct_scale
            transformed = torch.from_numpy(transformed).to(device=to_device, dtype=to_dtype)
        
        return transformed
    
    def post_processing(self, action_chunks: torch.Tensor) -> torch.Tensor:
        """Postprocess reconstructed actions (denormalize and optionally apply IDCT)."""
        to_device = action_chunks.device
        to_dtype = action_chunks.dtype
        transformed = action_chunks
        
        if self.encoding_mode == 'dct':
            transformed = transformed.cpu().detach().numpy()
            transformed = idct(transformed, axis=1, norm="ortho") / self.dct_scale
            transformed = torch.from_numpy(transformed).to(device=to_device, dtype=to_dtype)
        
        if self.normalizer is not None:
            transformed = transformed.cpu().detach().numpy()
            transformed = self.normalizer.denormalize(transformed)
            transformed = torch.from_numpy(transformed).to(device=to_device, dtype=to_dtype)
        
        transformed = torch.clamp(transformed, min=-1.0, max=1.0)
        return transformed
    
    @property
    def device(self):
        """Get the device of the model."""
        return next(self.parameters()).device
    
    @property
    def dtype(self):
        """Get the dtype of the model."""
        return next(self.parameters()).dtype
    
    def encode(
        self, 
        x: torch.Tensor,
        return_magnitude: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode action sequences to latent space.
        
        Args:
            x: (B, T, D) action sequences (raw, unnormalized)
            return_magnitude: whether to return computed magnitude
        Returns:
            z: (B, latent_dim) sampled latent codes
            mu: (B, latent_dim) mean
            logvar: (B, latent_dim) log variance
            (optional) magnitude: (B, D) if return_magnitude=True
        """
        x = x.to(device=self.device, dtype=self.dtype)
        
        # Compute magnitude before normalization
        magnitude = None
        if self.use_magnitude_embedding:
            magnitude = self.compute_magnitude(x)
        
        x_normalized = self.prepare_data(x)
        mu, logvar = self.encoder(x_normalized, magnitude=magnitude)
        z = self.reparameterize(mu, logvar)
        
        if return_magnitude:
            return z, mu, logvar, magnitude
        return z, mu, logvar
    
    def decode(
        self, 
        z: torch.Tensor, 
        post_process: bool = True,
        return_magnitude: bool = False,
    ) -> torch.Tensor:
        """
        Decode latent codes to action sequences.
        
        Args:
            z: (B, latent_dim) latent codes
            post_process: whether to apply denormalization
            return_magnitude: whether to return predicted magnitude
        Returns:
            x_recon: (B, T, D) reconstructed action sequences
            (optional) mag_pred: (B, D) predicted magnitude
        """
        if self.use_magnitude_embedding and return_magnitude:
            x_recon, mag_pred = self.decoder(z, return_magnitude=True)
        else:
            x_recon = self.decoder(z, return_magnitude=False)
            mag_pred = None
        
        if post_process:
            x_recon = self.post_processing(x_recon)
        
        if return_magnitude:
            return x_recon, mag_pred
        return x_recon
    
    def forward(
        self,
        x: torch.Tensor,
        return_loss: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass: encode, reparameterize, decode, compute loss.
        
        Args:
            x: (B, T, D) action sequences (raw, unnormalized)
            return_loss: whether to return loss (training) or just reconstruction
        Returns:
            If return_loss:
                loss: total loss
                loss_dict: dict with individual loss components
            Else:
                x_recon: reconstructed actions
                loss_dict: empty dict
        """
        x = x.to(device=self.device, dtype=self.dtype)
        
        # Compute magnitude before normalization (if enabled)
        magnitude = None
        if self.use_magnitude_embedding:
            magnitude = self.compute_magnitude(x)
        
        x_normalized = self.prepare_data(x)
        
        # Encode (with magnitude if enabled)
        mu, logvar = self.encoder(x_normalized, magnitude=magnitude)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode (with magnitude prediction if enabled)
        if self.use_magnitude_embedding:
            x_recon, mag_pred = self.decoder(z, return_magnitude=True)
        else:
            x_recon = self.decoder(z, return_magnitude=False)
            mag_pred = None
        
        if not return_loss:
            x_recon_denorm = self.post_processing(x_recon)
            return x_recon_denorm, {}
        
        # Compute losses
        # KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0, 1)
        # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Reconstruction loss
        if self.use_huber_loss:
            # Use only Huber loss when enabled
            huber_loss = F.huber_loss(x_recon, x_normalized, reduction='mean', delta=self.huber_delta)
            recon_loss = torch.tensor(0.0, device=x.device)  # MSE not used
            recon_l1_loss = torch.tensor(0.0, device=x.device)  # L1 not used
            total_loss = huber_loss + self.kl_weight * kl_loss
        else:
            # Use MSE + weighted L1
            recon_loss = F.mse_loss(x_recon, x_normalized, reduction='mean')
            recon_l1_loss = F.l1_loss(x_recon, x_normalized, reduction='mean')
            huber_loss = torch.tensor(0.0, device=x.device)  # Huber not used
            total_loss = recon_loss + self.recon_l1_weight * recon_l1_loss + self.kl_weight * kl_loss
        
        # Magnitude prediction loss (if enabled)
        mag_loss = torch.tensor(0.0, device=x.device)
        if self.use_magnitude_embedding and mag_pred is not None and magnitude is not None:
            mag_loss = F.mse_loss(mag_pred, magnitude, reduction='mean')
            total_loss = total_loss + self.magnitude_loss_weight * mag_loss
        
        # Compute metrics in original space for monitoring
        with torch.no_grad():
            x_recon_denorm = self.post_processing(x_recon.detach())
            recon_l1_orig = F.l1_loss(x_recon_denorm, x, reduction='mean')
            recon_l2_orig = F.mse_loss(x_recon_denorm, x, reduction='mean')
        
        loss_dict = {
            'loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'recon_l1_loss': recon_l1_loss.item(),
            'recon_l1_loss_weighted': (self.recon_l1_weight * recon_l1_loss).item(),
            'huber_loss': huber_loss.item(),
            'use_huber_loss': self.use_huber_loss,
            'kl_loss': kl_loss.item(),
            'kl_loss_weighted': (self.kl_weight * kl_loss).item(),
            'recon_l1_orig': recon_l1_orig.item(),
            'recon_l2_orig': recon_l2_orig.item(),
            'mu_mean': mu.mean().item(),
            'mu_std': mu.std().item(),
            'logvar_mean': logvar.mean().item(),
            'z_mean': z.mean().item(),
            'z_std': z.std().item(),
        }
        
        # Add magnitude loss if enabled
        if self.use_magnitude_embedding:
            loss_dict['mag_loss'] = mag_loss.item()
            loss_dict['mag_loss_weighted'] = (self.magnitude_loss_weight * mag_loss).item()
            if magnitude is not None:
                loss_dict['mag_mean'] = magnitude.mean().item()
                loss_dict['mag_std'] = magnitude.std().item()
            if mag_pred is not None:
                loss_dict['mag_pred_mean'] = mag_pred.mean().item()
                loss_dict['mag_pred_std'] = mag_pred.std().item()
        
        return total_loss, loss_dict
    
    @torch.no_grad()
    def sample(self, num_samples: int = 1) -> torch.Tensor:
        """
        Sample action sequences from the prior p(z) = N(0, 1).
        
        Args:
            num_samples: number of samples to generate
        Returns:
            samples: (num_samples, T, D) generated action sequences
        """
        z = torch.randn(num_samples, self.latent_dim, device=self.device, dtype=self.dtype)
        samples = self.decode(z, post_process=True)
        return samples
    
    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct action sequences.
        
        Args:
            x: (B, T, D) action sequences
        Returns:
            x_recon: (B, T, D) reconstructed action sequences
        """
        x_recon, _ = self.forward(x, return_loss=False)
        return x_recon


if __name__ == "__main__":
    # Test the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = ActionVAE(
        action_dim=7,
        action_T=8,
        hidden_dims=[64, 128, 256],
        latent_dim=32,
        use_transformer=True,
        num_transformer_layers=2,
        num_heads=4,
        kl_weight=1e-3,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(4, 8, 7).to(device)  # (B, T, D)
    loss, loss_dict = model(x)
    print(f"Loss: {loss.item():.4f}")
    print(f"Loss dict: {loss_dict}")
    
    # Test encode/decode
    z, mu, logvar = model.encode(x)
    print(f"Latent shape: {z.shape}")  # (B, latent_dim)
    
    x_recon = model.decode(z)
    print(f"Reconstruction shape: {x_recon.shape}")  # (B, T, D)
    
    # Test sampling
    samples = model.sample(num_samples=2)
    print(f"Samples shape: {samples.shape}")  # (num_samples, T, D)
