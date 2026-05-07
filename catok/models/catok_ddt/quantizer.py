from catok.models.catok_ddt.vector_quantize_pytorch import VectorQuantize as VectorQuantize_EMA
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal, Optional


# ============================================================================
# Residual Vector Quantization Components
# ============================================================================

class VectorQuantizerEMA(nn.Module):
    """
    Vector Quantizer with Exponential Moving Average (EMA) updates.
    This is a simpler implementation compared to VectorQuantize_EMA.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # EMA parameters
        self.decay = decay
        self.epsilon = epsilon
        
        # Initialize Embedding (register_buffer for non-gradient update)
        embedding = torch.randn(self.num_embeddings, self.embedding_dim)
        self.register_buffer('embedding', embedding)
        
        # Auxiliary variables for EMA
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', embedding.clone())

    def forward(self, inputs):
        # inputs: (Batch, Seq_len, Dim) -> (Batch * Seq_len, Dim)
        flat_input = inputs.reshape(-1, self.embedding_dim)
        
        # Get nearest codes using L2 distance
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # (N, 1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)  # (N, Num_Embeddings)
        
        # EMA update during training
        if self.training:
            # Get code usage count in batch
            encodings_sum = encodings.sum(0)
            # (Num_Embeddings, N) @ (N, Dim) -> (Num_Embeddings, Dim)
            dw = torch.matmul(encodings.t(), flat_input)
            
            # Update cluster size
            self.ema_cluster_size.data.mul_(self.decay).add_(encodings_sum, alpha=1 - self.decay)
            
            # Update embeddings sum
            self.ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)
            
            # Laplace Smoothing to avoid division by zero
            n = self.ema_cluster_size.sum()
            cluster_size = (self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
            
            # Update Codebook
            self.embedding.data.copy_(self.ema_w / cluster_size.unsqueeze(1))
        
        # Quantize
        quantized = torch.matmul(encodings, self.embedding).reshape(inputs.shape)
        
        # Only Commitment Loss (EMA handles codebook update)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Calculate Perplexity (codebook usage rate)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return loss, quantized, encoding_indices, perplexity


class ResidualVQEMA(nn.Module):
    """
    Residual Vector Quantization with EMA updates.
    Uses multiple VQ layers sequentially, where each layer quantizes the residual
    from the previous layer.
    """
    def __init__(self, num_quantizers, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.layers = nn.ModuleList([
            VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
            for _ in range(num_quantizers)
        ])
        
    def forward(self, inputs):
        # inputs: (Batch, Seq, Dim)
        quantized_out = 0.0
        residual = inputs
        
        all_losses = 0.0
        all_perplexities = []
        all_indices = []
        
        for layer in self.layers:
            loss, quantized, indices, perplexity = layer(residual)
            
            quantized_out = quantized_out + quantized
            residual = inputs - quantized_out
            
            all_losses += loss
            all_perplexities.append(perplexity)
            all_indices.append(indices)
        
        return all_losses, quantized_out, all_indices, all_perplexities


class ResidualVQWrapper(nn.Module):
    """
    Wrapper for ResidualVQEMA that provides the same interface as VectorQuantize.
    This allows RVQ to be used as a drop-in replacement for VQ in the encoder.
    
    Interface: forward(x) -> (quantize, embed_ind, loss, log_dict)
    """
    def __init__(
        self,
        dim,
        output_dim,
        codebook_dim,
        codebook_size,
        num_quantizers=4,
        commitment_cost=0.25,
        decay=0.99,
        **kwargs  # Accept and ignore other VQ-specific parameters
    ):
        super().__init__()
        self.dim = dim
        self.codebook_dim = codebook_dim
        self.output_dim = output_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        
        # Project from input dim to codebook dim
        self.project_in = nn.Linear(dim, codebook_dim) if dim != codebook_dim else nn.Identity()
        
        # Residual VQ
        self.rvq = ResidualVQEMA(
            num_quantizers=num_quantizers,
            num_embeddings=codebook_size,
            embedding_dim=codebook_dim,
            commitment_cost=commitment_cost,
            decay=decay
        )
        
        # Project from codebook dim to output dim
        self.project_out = nn.Linear(codebook_dim, output_dim) if codebook_dim != output_dim else nn.Identity()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, **kwargs):
        """
        Forward pass with VectorQuantize-compatible interface.
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            **kwargs: Ignored (for compatibility with VQ interface)
        
        Returns:
            quantize: Quantized embeddings (batch, seq_len, output_dim)
            embed_ind: Embedding indices (batch, seq_len, num_quantizers)
            loss: Commitment loss
            log_dict: Dictionary with logging information
        """
        # Project to codebook dimension
        h = self.project_in(x)
        
        # Apply Residual VQ
        loss, quantized, all_indices, all_perplexities = self.rvq(h)
        
        # Project to output dimension
        out = self.project_out(quantized)
        
        # Stack indices from all quantizer layers: (batch*seq, 1) * num_q -> (batch, seq, num_q)
        batch_size, seq_len = x.shape[:2]
        embed_ind = torch.cat(all_indices, dim=-1)  # (batch*seq, num_quantizers)
        embed_ind = embed_ind.view(batch_size, seq_len, self.num_quantizers)
        
        # Average perplexity across all quantizer layers
        avg_perplexity = sum([p.item() for p in all_perplexities]) / len(all_perplexities)
        
        # Create log dict compatible with VQ interface
        log_dict = {
            "n_reactive": 0,
            "commit_loss": loss.item() if isinstance(loss, torch.Tensor) else loss,
            "diversity_entropy": 0.0,
            "deterministic_entropy": 0.0,
            "perplexity": avg_perplexity,
            "delta_embed": 0.0,
            "rvq_mode": True,
            "num_quantizers": self.num_quantizers,
            # Per-layer perplexities
            "rvq_perplexities": [p.item() for p in all_perplexities],
        }
        
        return out, embed_ind, loss, log_dict
    
    def get_codes_from_indices(self, indices):
        """
        Get codebook embeddings from indices for RVQ.
        
        Args:
            indices: Token indices of shape (batch, seq_len, num_quantizers)
                     or (batch * seq_len, num_quantizers)
        
        Returns:
            codes: Summed codebook embeddings of shape (batch, seq_len, codebook_dim)
        """
        # Handle different input shapes
        if indices.dim() == 2:
            # (batch * seq_len, num_quantizers)
            flat_indices = indices
            batch_seq = indices.shape[0]
            reshape_output = False
        else:
            # (batch, seq_len, num_quantizers)
            batch_size, seq_len, num_q = indices.shape
            flat_indices = indices.view(-1, num_q)  # (batch * seq_len, num_quantizers)
            reshape_output = True
        
        # Sum embeddings from all quantizer layers (residual fashion)
        quantized_sum = torch.zeros(flat_indices.shape[0], self.codebook_dim, device=indices.device)
        
        for i, layer in enumerate(self.rvq.layers):
            layer_indices = flat_indices[:, i]  # (batch * seq_len,)
            # Get embeddings from this layer's codebook
            layer_embeddings = layer.embedding[layer_indices]  # (batch * seq_len, codebook_dim)
            quantized_sum = quantized_sum + layer_embeddings
        
        if reshape_output:
            quantized_sum = quantized_sum.view(batch_size, seq_len, self.codebook_dim)
        
        return quantized_sum
    
    def get_output_from_indices(self, indices):
        """
        Get output embeddings from indices for RVQ (compatible with VectorQuantize interface).
        
        Args:
            indices: Token indices of shape (batch, seq_len, num_quantizers)
                     or (batch * seq_len, num_quantizers)
        
        Returns:
            output: Projected embeddings of shape (batch, seq_len, output_dim)
        """
        codes = self.get_codes_from_indices(indices)
        return self.project_out(codes)


def construct_residual_vq(
    latent_dim,
    code_dim,
    output_dim,
    codebook_size,
    num_quantizers=4,
    commitment_cost=0.25,
    decay=0.99,
    **kwargs
):
    """
    Construct a ResidualVQWrapper module.
    
    Args:
        latent_dim: Input dimension from encoder
        code_dim: Codebook embedding dimension
        output_dim: Output dimension for decoder input
        codebook_size: Number of codes in each codebook
        num_quantizers: Number of residual VQ layers
        commitment_cost: Weight for commitment loss
        decay: EMA decay rate
        **kwargs: Additional arguments (ignored for compatibility)
    
    Returns:
        ResidualVQWrapper module
    """
    return ResidualVQWrapper(
        dim=latent_dim,
        output_dim=output_dim,
        codebook_dim=code_dim,
        codebook_size=codebook_size,
        num_quantizers=num_quantizers,
        commitment_cost=commitment_cost,
        decay=decay,
    )


class ContinuousProjector(nn.Module):
    """
    A continuous projector that bypasses vector quantization.
    Instead of discretizing embeddings, it directly projects continuous query embeddings
    to the decoder input dimension using a linear projection.
    
    This provides an alternative to VQ-based action tokenization where the latent space
    remains continuous, potentially preserving more fine-grained action information.
    """
    def __init__(
        self,
        dim,
        codebook_dim,
        output_dim,
        use_layernorm=True,
        **kwargs  # Accept and ignore VQ-specific parameters
    ):
        super().__init__()
        self.dim = dim
        self.codebook_dim = codebook_dim
        self.output_dim = output_dim
        
        # Project from input dim to codebook dim (intermediate representation)
        self.project_in = nn.Linear(dim, codebook_dim) if dim != codebook_dim else nn.Identity()
        
        # Optional layer normalization for stable training
        self.layernorm = nn.LayerNorm(codebook_dim) if use_layernorm else nn.Identity()
        
        # Project from codebook dim to output dim (decoder input)
        self.project_out = nn.Linear(codebook_dim, output_dim) if codebook_dim != output_dim else nn.Identity()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, **kwargs):
        """
        Forward pass that projects continuous embeddings without quantization.
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            **kwargs: Ignored (for compatibility with VQ interface)
        
        Returns:
            quantize: Projected embeddings (batch, seq_len, output_dim)
            embed_ind: Dummy indices (zeros) for compatibility
            loss: Zero loss tensor
            log_dict: Dictionary with logging information
        """
        # Project to intermediate dimension
        h = self.project_in(x)
        
        # Apply layer normalization
        h = self.layernorm(h)
        
        # Project to output dimension
        out = self.project_out(h)
        
        # Create dummy indices (all zeros) for compatibility with VQ interface
        batch_size, seq_len = x.shape[:2]
        embed_ind = torch.zeros(batch_size, seq_len, dtype=torch.long, device=x.device)
        
        # Zero loss since there's no quantization
        loss = torch.tensor(0.0, device=x.device, requires_grad=False)
        
        # Log dict with continuous-specific metrics
        log_dict = {
            "n_reactive": 0,
            "commit_loss": 0.0,
            "diversity_entropy": 0.0,
            "deterministic_entropy": 0.0,
            "perplexity": 0.0,
            "delta_embed": 0.0,
            "continuous_mode": True,
            # Compute embedding statistics
            "input_l2_norm": torch.norm(x, dim=-1).mean().item(),
            "output_l2_norm": torch.norm(out, dim=-1).mean().item(),
        }
        
        return out, embed_ind, loss, log_dict


def construct_continuous_projector(
        latent_dim, code_dim, output_dim,
        use_layernorm=True, **kwargs):
    """
    Construct a ContinuousProjector module that bypasses vector quantization.
    
    Args:
        latent_dim: Input dimension from encoder
        code_dim: Intermediate dimension (analogous to codebook_dim in VQ)
        output_dim: Output dimension for decoder input
        use_layernorm: Whether to use layer normalization
        **kwargs: Additional arguments (ignored for compatibility)
    
    Returns:
        ContinuousProjector module
    """
    return ContinuousProjector(
        dim=latent_dim,
        codebook_dim=code_dim,
        output_dim=output_dim,
        use_layernorm=use_layernorm,
    )


def construct_quantizer(
        latent_dim, code_dim, output_dim, codebook_size, K,
        w_diversity, w_commit, dead_code_threshold=0.0, decay=0.99,
        smart_re_K=0, continuous=False, reg=[1/4., 1/2.],
        reset_cluster_size=None, ema_entropy_ratio=0.7, frozen_embed=None,
        ema_update=True,):
    """
    Construct a VectorQuantize module with configurable EMA update.
    
    Args:
        ema_update: Whether to use EMA (Exponential Moving Average) to update 
                    the codebook embeddings. Default is True.
                    When True, codebook is updated via EMA during training.
                    When False, codebook can be updated via gradients (learnable_codebook).
    """
    args = dict(
        dim=latent_dim,
        output_dim=output_dim,
        codebook_dim=code_dim,
        codebook_size=codebook_size,
        ema_update=ema_update,
        decay=decay,
        kmeans_init=True,
        kmeans_iters=10,
        threshold_ema_dead_code=dead_code_threshold,
        use_cosine_sim=True,
        commitment_weight=w_commit,
        diversity_weight=w_diversity,
        smart_re_K=smart_re_K,
        continuous=continuous,
        reg=reg,
        reset_cluster_size=reset_cluster_size,
        ema_entropy_ratio=ema_entropy_ratio,
        frozen_embed=frozen_embed,
    )

    constructor = VectorQuantize_EMA
    
    return constructor(**args)


def construct_quantizer_by_mode(
    vq_mode: Optional[Literal['VQ', 'VQ_EMA', 'RVQ', 'RVQ_EMA', None]],
    latent_dim: int,
    code_dim: int,
    output_dim: int,
    quantizer_config: dict,
):
    """
    Unified function to construct different types of quantizers based on vq_mode.
    
    Args:
        vq_mode: Type of quantizer to use:
            - None or 'none': ContinuousProjector (no quantization)
            - 'VQ' or 'VQ_EMA': Standard Vector Quantization with EMA updates
            - 'RVQ' or 'RVQ_EMA': Residual Vector Quantization with EMA updates
        latent_dim: Input dimension from encoder
        code_dim: Codebook embedding dimension
        output_dim: Output dimension for decoder input
        quantizer_config: Dictionary containing quantizer-specific parameters:
            - codebook_size: Number of codes in codebook
            - w_commit: Commitment loss weight
            - w_diversity: Diversity loss weight  
            - decay: EMA decay rate
            - dead_code_threshold: Threshold for dead code reactivation
            - num_quantizers: Number of RVQ layers (only for RVQ mode)
            - ... other VQ-specific parameters
    
    Returns:
        Quantizer module with unified interface: forward(x) -> (quantize, indices, loss, log_dict)
    """
    # Normalize vq_mode
    if vq_mode is not None:
        vq_mode = vq_mode.upper()
    
    if vq_mode is None or vq_mode == 'NONE':
        # Continuous projector (no quantization)
        return construct_continuous_projector(
            latent_dim=latent_dim,
            code_dim=code_dim,
            output_dim=output_dim,
            use_layernorm=quantizer_config.get('use_layernorm', True),
        )
    
    elif vq_mode in ['RVQ', 'RVQ_EMA']:
        # Residual Vector Quantization
        return construct_residual_vq(
            latent_dim=latent_dim,
            code_dim=code_dim,
            output_dim=output_dim,
            codebook_size=quantizer_config.get('codebook_size', 4096),
            num_quantizers=quantizer_config.get('num_quantizers', 4),
            commitment_cost=quantizer_config.get('w_commit', 0.25),
            decay=quantizer_config.get('decay', 0.99),
        )
    
    elif vq_mode in ['VQ', 'VQ_EMA']:
        # Standard Vector Quantization with EMA
        return construct_quantizer(
            latent_dim=latent_dim,
            code_dim=code_dim,
            output_dim=output_dim,
            codebook_size=quantizer_config.get('codebook_size', 4096),
            K=quantizer_config.get('K', 16),
            w_diversity=quantizer_config.get('w_diversity', 0.0),
            w_commit=quantizer_config.get('w_commit', 1.0),
            dead_code_threshold=quantizer_config.get('dead_code_threshold', 0.0),
            decay=quantizer_config.get('decay', 0.99),
            smart_re_K=quantizer_config.get('smart_re_K', 0),
            continuous=quantizer_config.get('continuous', False),
            reg=quantizer_config.get('reg', [1/4., 1/2.]),
            reset_cluster_size=quantizer_config.get('reset_cluster_size', None),
            ema_entropy_ratio=quantizer_config.get('ema_entropy_ratio', 0.7),
            frozen_embed=quantizer_config.get('frozen_embed', None),
            ema_update=quantizer_config.get('ema_update', True),
        )
    
    else:
        raise ValueError(
            f"Unknown vq_mode: {vq_mode}. "
            f"Supported modes: None, 'VQ', 'VQ_EMA', 'RVQ', 'RVQ_EMA'"
        )

