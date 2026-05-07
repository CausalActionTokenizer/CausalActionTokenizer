import torch
import torch.nn as nn
import numpy as np
import math
from catok.models.catok_ddt.sd3.mmdit import ActionEmbedder, get_1d_sincos_pos_embed_from_grid, get_2d_sincos_pos_embed_from_grid, CustomLayerNorm, WaveActionEmbedder, ActionPatchEmbedder
import torch.nn.functional as F
from catok.models.catok_ddt.quantizer import construct_quantizer, construct_continuous_projector, construct_residual_vq, construct_quantizer_by_mode
from catok.models.catok_ddt.modules import DiTCrossAttnBlock, ViTBlock, QFormer, DualBlock, ConcatBlock, DiTDualBlock, DualBlockMultiRes
from einops import rearrange
import torch.distributed as dist
import random

from torch.utils.checkpoint import checkpoint
def ckpt_wrapper(module):
    def ckpt_forward(*inputs):
        outputs = module(*inputs)
        return outputs
    return ckpt_forward

def get_1d_sincos_pos_embed(embed_dim, action_T):
    pos = np.arange(action_T, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(
        embed_dim, pos)
    return pos_embed

def get_2d_sincos_pos_embed_rect(embed_dim, grid_h, grid_w):
    """
    Generate 2D sincos positional embedding for a rectangular grid.
    Args:
        embed_dim: output dimension for each position
        grid_h: height of the grid (e.g. action_T)
        grid_w: width of the grid (e.g. action_dim)
    Returns:
        pos_embed: (grid_h * grid_w, embed_dim) numpy array
    """
    gh = np.arange(grid_h, dtype=np.float32)
    gw = np.arange(grid_w, dtype=np.float32)
    grid = np.meshgrid(gw, gh)  # w first, then h
    grid = np.stack(grid, axis=0)  # (2, grid_h, grid_w)
    grid = grid.reshape([2, 1, grid_h, grid_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)  # (grid_h * grid_w, embed_dim)
    return pos_embed

class Encoder(nn.Module):
    def __init__(
        self, K, 
        # input_size=32, 
        encoder_hidden_size=256, 
        # patch_size=8, in_channels=4,
        action_dim=7, action_T=8, 
        hidden_size=256, depth=None, num_heads=4, mlp_ratio=4.0,
        pre_norm=False, post_norm=True, encoder_out_dim=None, apply_losses_together=False,
        gradient_checkpointing=False, pos_embed_max_size=None, quantizer_config=None, attn_mask=False, single_token=False,
        layernorm_elementwise_affine=True, layernorm_bias=True, linear_bias=True, 
        embedder_type='linear', pos_embed_scale=None, no_vq=False, learnable_pos_embed=False, **kwargs
    ):
        super().__init__()
        self.K = K
        
        # Determine vq_mode early for n_e calculation
        # Priority: vq_mode in config > no_vq flag > use_rvq flag > default VQ
        if 'vq_mode' in quantizer_config:
            _vq_mode = quantizer_config['vq_mode']
        elif no_vq:
            _vq_mode = None
        elif quantizer_config.get('use_rvq', False):
            _vq_mode = 'RVQ'
        else:
            _vq_mode = 'VQ'
        
        # Backward compatibility: derive no_vq from vq_mode
        self.no_vq = (_vq_mode is None)
        self.n_e = quantizer_config.get('codebook_size', 0) if _vq_mode is not None else 0
        
        # action
        self.action_T = action_T
        self.action_dim = action_dim
        
        # image 
        # self.in_channels = in_channels
        # self.out_channels = in_channels
        # self.patch_size = patch_size
        
        self.num_heads = num_heads
        depth = depth or self.K
        self.depth = depth
        self.hidden_size = hidden_size
        self.pre_norm = pre_norm
        self.post_norm = post_norm
        self.pos_embed_max_size = pos_embed_max_size
        encoder_out_dim = encoder_out_dim or hidden_size
        self.gradient_checkpointing = gradient_checkpointing
        self.code_dim = quantizer_config['code_dim']
        # TODO: check tokens number
        # self.n_tokens = K * (input_size // patch_size) ** 2
        self.n_tokens = K * action_T
        self.apply_losses_together = apply_losses_together
        self.attn_mask = attn_mask
        self.single_token = single_token
        # models
        if embedder_type == 'linear':
            self.x_embedder = ActionEmbedder(
                action_dim=action_dim,
                embed_dim=hidden_size,
                bias=linear_bias,
            )
        elif embedder_type == 'wave':
            self.x_embedder = WaveActionEmbedder(action_dim, hidden_size)
        elif embedder_type == 'patch':
            self.x_embedder = ActionPatchEmbedder(action_dim, hidden_size)
        else:
            raise ValueError(f"Invalid embedder type: {embedder_type}")
        '''
        if pos_embed_max_size is not None:
            num_patches = pos_embed_max_size * pos_embed_max_size
            self.x_embedder.num_patches = pos_embed_max_size * pos_embed_max_size
        else:
            num_patches = self.x_embedder.num_patches
        '''
            
        # Store embedder type for dynamic pos_embed generation
        self.embedder_type = embedder_type
        
        # For linear and wave embedders, output length equals action_T
        # For patch embedder, output length depends on convolution output
        # We'll create pos_embed dynamically in forward based on actual embedding length
        # Initialize with a placeholder - will be created dynamically
        self.pos_embed = None
        self.pos_embed_max_length = pos_embed_max_size if pos_embed_max_size is not None else action_T * action_dim
        
        # pos_embed_scale: scale factor for positional embedding
        # If None, will be computed automatically to match action embedding magnitude
        # For action values in [-1, 1], typical scale is around 0.02 (1/sqrt(hidden_size/2))
        self.pos_embed_scale = pos_embed_scale
        
        # Learnable positional embedding option
        self.learnable_pos_embed = learnable_pos_embed
        if learnable_pos_embed:
            # Initialize with sincos values, then make it a learnable parameter
            if embedder_type == 'patch':
                # Use 2D sincos for patch embedder (grid_h=action_T, grid_w=action_dim)
                sincos_embed = get_2d_sincos_pos_embed_rect(hidden_size, action_T, action_dim)
            else:
                sincos_embed = get_1d_sincos_pos_embed(hidden_size, action_T)
            sincos_embed = torch.from_numpy(sincos_embed).float().unsqueeze(0)  # (1, seq_len, hidden_size)
            if pos_embed_scale is not None:
                sincos_embed = sincos_embed * pos_embed_scale
            self.learnable_pos_embed_param = nn.Parameter(sincos_embed)
        
        self.blocks = nn.ModuleList([
            ViTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        if layernorm_elementwise_affine:
            self.final_layer_norm = CustomLayerNorm(encoder_out_dim, elementwise_affine=True, bias=layernorm_bias, eps=1e-6)
            self.final_layer_norm2 = CustomLayerNorm(self.code_dim, elementwise_affine=True, bias=layernorm_bias, eps=1e-6)
            self.final_layer_norm3 = CustomLayerNorm(encoder_hidden_size, elementwise_affine=True, bias=layernorm_bias, eps=1e-6)
        else:
            self.final_layer_norm = nn.LayerNorm(encoder_out_dim, elementwise_affine=False, eps=1e-6)
            self.final_layer_norm2 = nn.LayerNorm(self.code_dim, elementwise_affine=False, eps=1e-6)
            self.final_layer_norm3 = nn.LayerNorm(encoder_hidden_size, elementwise_affine=False, eps=1e-6) 

        # Determine vq_mode from config
        # Priority: vq_mode > no_vq flag > use_rvq flag > default VQ
        # This provides backward compatibility with old configs
        if 'vq_mode' in quantizer_config:
            self.vq_mode = quantizer_config['vq_mode']
        elif no_vq:
            self.vq_mode = None  # Continuous (no quantization)
        elif quantizer_config.get('use_rvq', False):
            self.vq_mode = 'RVQ'
        else:
            self.vq_mode = 'VQ'  # Default to standard VQ
        
        # For backward compatibility: set flags based on vq_mode
        self.use_rvq = self.vq_mode in ['RVQ', 'RVQ_EMA'] if self.vq_mode else False
        
        # Construct quantizer using unified function
        self.quantizer = construct_quantizer_by_mode(
            vq_mode=self.vq_mode,
            latent_dim=encoder_out_dim,
            code_dim=self.code_dim,
            output_dim=encoder_hidden_size,
            quantizer_config=quantizer_config,
        )
        self.initialize_weights()
        
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
    
    def get_pos_embed(self, seq_length, device, dtype):
        """
        Generate positional embedding for given sequence length.
        Args:
            seq_length: actual sequence length from embedder output
            device: device to create pos_embed on
            dtype: dtype for pos_embed
        Returns:
            pos_embed: (1, seq_length, hidden_size) tensor
        """
        if self.learnable_pos_embed:
            # Use learnable positional embedding (slice if seq_length differs)
            pos_embed = self.learnable_pos_embed_param[:, :seq_length, :]
            return pos_embed.to(dtype=dtype)
        
        if self.embedder_type == 'patch':
            # Use 2D sincos for patch embedder (grid_h=action_T, grid_w=action_dim)
            pos_embed = get_2d_sincos_pos_embed_rect(self.hidden_size, self.action_T, self.action_dim)
        else:
            pos_embed = get_1d_sincos_pos_embed(
                embed_dim=self.hidden_size,
                action_T=seq_length
            )
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).to(device=device, dtype=dtype)
        
        # Apply pos_embed_scale if specified
        if self.pos_embed_scale is not None:
            pos_embed = pos_embed * self.pos_embed_scale
        
        return pos_embed

    def forward_quantizer(self, quantizer, x):
        outs_q, indices, loss, log_dict = quantizer(x)
        return outs_q, indices, loss, log_dict
    
    def get_encoder_outs(self, x, kwargs=None):
        outs = []
        embedding_stats = {}
        
        for i, block in enumerate(self.blocks):
            if self.gradient_checkpointing:
                x = checkpoint(ckpt_wrapper(block), x, use_reentrant=False)
            else:
                x = block(x)
            
            # Collect per-layer action embedding statistics
            with torch.no_grad():
                x_l2 = torch.norm(x, dim=-1)  # (B, T)
                embedding_stats[f'layer_{i}_action_l2_mean'] = x_l2.mean().item()
                embedding_stats[f'layer_{i}_action_l2_std'] = x_l2.std().item()
            
            if i >= self.depth - self.K:
                outs.append(x)
    
        assert len(outs) == self.K
        outs = torch.cat(outs, dim=1)
        return outs, embedding_stats

    def get_encoder_mask(self, x, d):
        B, N, P = x.shape[0], self.K, x.shape[1]
        enc_mask = torch.arange(self.K).repeat_interleave(P)[None, ...].expand(B,N).to(d.device)
        return (enc_mask <= d.unsqueeze(1))
    
    def calc_entropy(self, p):
        ap = p.mean(dim=0)
        p_log_p = ap * torch.log(ap)
        entropy_to_max = -p_log_p.sum(dim=-1)
        # E(H(p))
        p_log_p = p * torch.log(p)
        entropy_to_min = -p_log_p.sum(dim=-1)
        entropy_to_min = entropy_to_min
        return entropy_to_min
    
    def get_perplexity_list(self, log_dict, chunks=50):
        # For continuous mode (no quantization), return empty lists
        if self.vq_mode is None:
            return [], []
        
        # For RVQ mode, return per-layer perplexities if available
        if self.vq_mode in ['RVQ', 'RVQ_EMA']:
            if 'rvq_perplexities' in log_dict:
                # Return RVQ per-layer perplexities as the list
                return log_dict['rvq_perplexities'], [0.0] * len(log_dict['rvq_perplexities'])
            return [], []
        
        if 'perplexity_list' in log_dict:
            # separate codebook
            perplexity_list = torch.tensor(log_dict['perplexity_list'])
            perplexity_list = torch.stack([t.mean(dim=0) for t in perplexity_list.tensor_split(chunks, dim=0)],dim=0).float()
            deter_list = torch.tensor(log_dict['deter_list']).float()
            deter_list = torch.stack([t.mean(dim=0) for t in deter_list.tensor_split(chunks, dim=0)],dim=0).float()
            return perplexity_list.tolist(), deter_list.tolist()

        probs = self.quantizer._codebook.timestep_p_over_c.mean(dim=0)
        chunk_probs = torch.stack([t.mean(dim=0) for t in probs.tensor_split(chunks, dim=0)],dim=0).float()
        ap = chunk_probs
        perplexity_list = torch.exp(-torch.sum(ap * torch.log(ap + 1e-10), dim=1)).tolist()
        deterministic_list = self.calc_entropy(ap).tolist()
        return perplexity_list, deterministic_list
    
    def forward(self, x=None, hidden_states=None, d=None, kwargs=None):
        """
        Forward pass of feature encoder.
        x: (N, T, A) tensor of action inputs.
        d: N, the depth for each sample
        hidden_states: Optional pre-computed hidden states. If provided, skip x_embedder
                       and use hidden_states directly for mask computation.
        """
        embedding_stats = {}
        
        if hidden_states is None:
            # Normal forward: process input x through embedder
            # Get embedding from embedder
            x = self.x_embedder(x)  # (B, Length, hidden_size)
            
            # Collect action embedding statistics (after x_embedder, before pos_embed)
            with torch.no_grad():
                action_emb_l2 = torch.norm(x, dim=-1)  # (B, T)
                embedding_stats['action_emb_l2_mean'] = action_emb_l2.mean().item()
                embedding_stats['action_emb_l2_std'] = action_emb_l2.std().item()
            
            # Generate pos_embed based on actual embedding length
            seq_length = x.shape[1]
            pos_embed = self.get_pos_embed(seq_length, device=x.device, dtype=x.dtype)
            
            # Collect pos_embed statistics
            with torch.no_grad():
                pos_emb_l2 = torch.norm(pos_embed, dim=-1)  # (1, T)
                embedding_stats['pos_emb_l2_mean'] = pos_emb_l2.mean().item()
                embedding_stats['pos_emb_l2_std'] = pos_emb_l2.std().item()
            
            # Add positional embedding
            x = x + pos_embed
            
            # Collect combined embedding statistics (after adding pos_embed)
            with torch.no_grad():
                combined_emb_l2 = torch.norm(x, dim=-1)  # (B, T)
                embedding_stats['combined_emb_l2_mean'] = combined_emb_l2.mean().item()
                embedding_stats['combined_emb_l2_std'] = combined_emb_l2.std().item()
            
            outs, layer_stats = self.get_encoder_outs(x, kwargs=kwargs) #torch.Size([4, 512, 512])
            embedding_stats.update(layer_stats)
            
            if self.pre_norm:
                outs = self.final_layer_norm(outs) 
            to_quantizer_features = outs
            perplexity_list = []
            deterministic_list = []
            if self.apply_losses_together:  # False
                enc_mask = self.get_encoder_mask(x, d)
                grad_mask = enc_mask[..., None].expand_as(to_quantizer_features).float()
                to_quantizer_features = to_quantizer_features * grad_mask + \
                    to_quantizer_features.detach() * (1-grad_mask)
            
            outs_q, indices, loss, log_dict = \
                self.forward_quantizer(self.quantizer, to_quantizer_features)
            
            # prepare logs
            perplexity_list, deterministic_list = self.get_perplexity_list(log_dict)
            log_dict.update({
                "perplexity_list": perplexity_list,
                "deter_list": deterministic_list,
            })
            # Add embedding statistics to log_dict
            log_dict.update(embedding_stats)

            if self.post_norm:
                outs_q = self.final_layer_norm3(outs_q)
                
            # x is the embedded input for mask computation
            x_for_mask = x
        else:
            # hidden_states provided: skip embedder, use hidden_states directly
            outs_q = hidden_states
            loss = 0.0
            log_dict = {}
            to_quantizer_features = None
            indices = None
            # Use hidden_states for mask computation (it has the right shape: B, num_tokens, hidden_dim)
            x_for_mask = hidden_states
            
        if d is None:
            return outs_q, indices
        
        enc_mask = self.get_encoder_mask(x_for_mask, d)
        # fix mask for the attn mask
        # attn_mask = torch.logical_not(enc_mask)
        attn_mask = enc_mask
        mask_v = enc_mask[..., None].expand_as(outs_q)
        # encoder_hidden_states = outs_q * mask_v
        encoder_hidden_states = outs_q
        return encoder_hidden_states, to_quantizer_features, outs_q, attn_mask, loss, log_dict, indices
    
class QformerEncoder(Encoder):
    def __init__(
        self, K, 
        # input_size=32, 
        encoder_hidden_size=256, 
        action_dim = 7, action_T = 8,
        # patch_size=8, in_channels=4,
        hidden_size=256, depth=None, num_heads=4, mlp_ratio=4.0,
        pre_norm=False, post_norm=True, qformer_mode='qformer',
        gradient_checkpointing=False, pos_embed_max_size=None, apply_losses_together=False,
        xavier_init=False, diti=None, quantizer_config=None, attn_mask=False, single_token=False, 
        no_vq=False, **kwargs
    ):
        super().__init__(
            K, 
            # input_size, 
            encoder_hidden_size, 
            # patch_size, in_channels, 
            action_dim, action_T,
            hidden_size, depth, num_heads,
            mlp_ratio, pre_norm, post_norm, encoder_out_dim=kwargs['query_dim'],
            gradient_checkpointing=gradient_checkpointing, apply_losses_together=apply_losses_together,
            pos_embed_max_size=pos_embed_max_size, quantizer_config=quantizer_config, attn_mask=attn_mask, 
            single_token=single_token, no_vq=no_vq, **kwargs
        )
        qformer_depth = depth
        self.num_query_token = K # num_query_token
        query_dim = kwargs['query_dim']
        self.query_tokens = nn.Parameter(torch.zeros(1, self.num_query_token, query_dim))
        self.query_tokens.data.normal_(mean=0.0, std=0.02) #initialization
        self.mode = qformer_mode
        self.diti = diti
        self.attn_mask = attn_mask
        self.single_token = single_token
        if diti:
            kwargs["diti"] = diti
        # Remove embedder_type, pos_embed_scale, learnable_pos_embed from kwargs as they're only used in Encoder.__init__
        kwargs.pop('embedder_type', None)
        kwargs.pop('pos_embed_scale', None)
        kwargs.pop('learnable_pos_embed', None)
        if self.mode == 'qformer':
            self.qformer = QFormer(
                self.num_query_token, hidden_size, query_dim, num_heads, qformer_depth, mlp_ratio=mlp_ratio
            )
            self.blocks = nn.Identity()
        elif self.mode == 'dual':
            self.blocks = nn.ModuleList([
                DualBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **kwargs) for _ in range(depth)
            ])
        elif self.mode == 'concat':
            self.blocks = nn.ModuleList([
                ConcatBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **kwargs) for _ in range(depth)
            ])

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        if xavier_init:
            self.apply(_basic_init)

    def get_encoder_outs(self, x, kwargs=None):
        query_tokens = self.query_tokens.expand(x.shape[0], -1, -1)
        embedding_stats = {}
        
        # Collect initial query token statistics
        with torch.no_grad():
            query_l2 = torch.norm(query_tokens, dim=-1)  # (B, K)
            embedding_stats['query_init_l2_mean'] = query_l2.mean().item()
            embedding_stats['query_init_l2_std'] = query_l2.std().item()
        
        if self.mode == 'qformer':
            query_tokens = self.qformer(x, query_tokens) # [B, L, C]
        elif self.mode == 'concat':
            for i, block in enumerate(self.blocks):
                if self.gradient_checkpointing:
                    x, query_tokens = checkpoint(ckpt_wrapper(block), x, query_tokens, use_reentrant=False)
                else:
                    x, query_tokens = block(x, query_tokens)
                
                # Collect per-layer statistics
                with torch.no_grad():
                    x_l2 = torch.norm(x, dim=-1)
                    query_l2 = torch.norm(query_tokens, dim=-1)
                    embedding_stats[f'layer_{i}_action_l2_mean'] = x_l2.mean().item()
                    embedding_stats[f'layer_{i}_action_l2_std'] = x_l2.std().item()
                    embedding_stats[f'layer_{i}_query_l2_mean'] = query_l2.mean().item()
                    embedding_stats[f'layer_{i}_query_l2_std'] = query_l2.std().item()
                    
                    # Collect attention pos_embed (t_emb) statistics if time_adaln is enabled
                    if hasattr(block, 'time_adaln') and block.time_adaln:
                        K = query_tokens.shape[1]
                        if block.diti is not None:
                            pos_embed_input = block.diti.get_position(torch.arange(K).to(x.device))
                        else:
                            pos_embed_input = torch.arange(K).to(x.device).float()
                        t_emb = block.t_embedder(pos_embed_input)
                        t_emb_l2 = torch.norm(t_emb, dim=-1)
                        embedding_stats[f'layer_{i}_attn_pos_emb_l2_mean'] = t_emb_l2.mean().item()
                        embedding_stats[f'layer_{i}_attn_pos_emb_l2_std'] = t_emb_l2.std().item()
                    query_diversity = torch.mean(query_tokens, dim=1).std(dim=0).mean().item()
                    embedding_stats[f'query_diversity'] = query_diversity
                    
        elif self.mode == 'dual':
            # attn mask
            
            if self.attn_mask: #False
                mask = mask = torch.ones(self.K, self.K).tril().bool().cuda()
                x_mask = torch.ones((self.K, x.shape[1])).cuda()
                mask = torch.cat((x_mask, mask), dim=1).bool()
                mask = mask.unsqueeze(0).unsqueeze(1).repeat(x.shape[0],1,1,1)
            else:
                mask = None

            for i, block in enumerate(self.blocks):
                if self.gradient_checkpointing:
                    x, query_tokens = checkpoint(ckpt_wrapper(block), x, query_tokens, mask, use_reentrant=False)
                else:
                    x, query_tokens = block(x, query_tokens, mask=mask)
                
                # Collect per-layer statistics
                with torch.no_grad():
                    x_l2 = torch.norm(x, dim=-1)
                    query_l2 = torch.norm(query_tokens, dim=-1)
                    embedding_stats[f'layer_{i}_action_l2_mean'] = x_l2.mean().item()
                    embedding_stats[f'layer_{i}_action_l2_std'] = x_l2.std().item()
                    embedding_stats[f'layer_{i}_query_l2_mean'] = query_l2.mean().item()
                    embedding_stats[f'layer_{i}_query_l2_std'] = query_l2.std().item()
                    
                    # Collect attention pos_embed (t_emb) statistics if time_adaln is enabled
                    if hasattr(block, 'time_adaln') and block.time_adaln:
                        K = query_tokens.shape[1]
                        if block.diti is not None:
                            pos_embed_input = block.diti.get_position(torch.arange(K).to(x.device))
                        else:
                            pos_embed_input = torch.arange(K).to(x.device).float()
                        t_emb = block.t_embedder(pos_embed_input)
                        t_emb_l2 = torch.norm(t_emb, dim=-1)
                        embedding_stats[f'layer_{i}_attn_pos_emb_l2_mean'] = t_emb_l2.mean().item()
                        embedding_stats[f'layer_{i}_attn_pos_emb_l2_std'] = t_emb_l2.std().item()
        else:
            raise ValueError("Unknown mode to QFormerEncoder.")
        return query_tokens, embedding_stats
    
    def get_encoder_mask(self, x, d, single_token=False):
        # no spatial token, so num patches is essentially 1
        B, N = x.shape[0], self.K
        enc_mask = torch.arange(self.K).repeat_interleave(1)[None, ...].expand(B,N).to(d.device)
        
        if single_token:
            return (enc_mask == d.unsqueeze(1))
        else:
            return (enc_mask <= d.unsqueeze(1))
        
if __name__ == "__main__":
    # python catok/models/catok_ddt/encoders.py    
    import torch.distributed as dist
    dist.init_process_group(backend='nccl',init_method='env://')
    
    print("ok")
    k = 512
    full_tokens = True
    t2k = 1.0
    quantizer_config = {
        "codebook_size": 32768,
        "code_dim": 16,
        "w_diversity": 1.0,
        "ema_entropy_ratio": 0.8,
        "w_commit": 1.0,
        "decay": 0.99,
        "dead_code_threshold": 0.2,
        "reset_cluster_size": 0.2,
        "smart_re_K": k,
        "continuous": False,
        "reg": [0.1, 0.3],
        "K": k
        }
    
    device = 'cuda'
    
    encoder = QformerEncoder(
        K=k,
        encoder_hidden_size=16,
        quantizer_config=quantizer_config,
        query_dim=256,
        depth=4
    ).to(device)
    print(encoder)
    
    from catok.models.catok_ddt.diti_utils import DiTi_cont
    diti = DiTi_cont(
        1000, k, stages='1000', k_per_stage='512'
    )
    
    
    x = torch.rand(4, 8, 7).cuda()  # (B, T, A)
    t = torch.rand(x.shape[0]).cuda()
    
    
    
    if full_tokens:
        k_batch = diti.to_indices(torch.ones_like(t) * 1000.0)
    else:
        t_tmp = (t2k * t).clamp(0, 1.0)
        k_batch = diti.to_indices(t_tmp * 1000.0)
    

    
    with torch.no_grad():
        # encoder_hidden_states, to_quantizer_features, outs_q, attn_mask, loss, log_dict, indices
        encoder_hidden_states, to_quantizer_features, outs_q, attn_mask, quan_loss, log_dict, indices = encoder(x=x, d=k_batch, kwargs=None)
        to_quantizer_features_ema = None
    
    print(f"Encoder hidden states shape: {encoder_hidden_states.shape}")
    print(f"To quantizer features shape: {to_quantizer_features.shape if to_quantizer_features is not None else None}")
    print(f"Outs_q shape: {outs_q.shape}")
    print(f"indices", indices.shape, indices.max())
    '''
    common:
        output_path: 'output'
        log_path: '/cache/logs'
        tb_path: './outputs/catok_enc_tb/v4'
        val_url: './outputs/catok_enc_tb/v4'
        save_per_epochs: 1.0
        eval_per_epochs: 1.0
        eval_first: 0
        use_fp16: 0
        use_bf16: 1
        use_zero: 0
        use_fsdp: 0
        use_2d_rope: 0
        use_deepspeed: 0
        random_seed: 123
        log_interval: 50
        machines: 1
        task: 'catokenc'
        experiment_index: 0
        delete_after_upload: True
        log_recon_interval: 100
        val_interval: 0
        ckpt_interval: 1000
        vae_path: '/cache/data/sd3_medium.ckpt'
        resume_exclude_opt: True
        pre_encode: False
        resume_from_steps: 0
        is_eval: True

    model:
        pretrain_model: '/cache/model/iter_149999.pth'
        fix_encoder: True
        full_tokens: True
        fix_decoder: False



    tokenizer:
        is_text_tokenized: False
        pretrained_dit_path: '/cache/data/sd3_medium.ckpt'
        params:
            image_size: 256
            k: 512
            stages: '1000'
            k_per_stage: '512'
            gradient_checkpointing: False
            in_channels: 16
            encoder_hidden_size: 16
            ema_enc: False
            enc_decay: 0.99
            L2_lr: 0.
            two_part_losses: False
            
            diffusion_type: 'flow'
            noise_schedule_config:
                schedule: 'log_norm'
                parameterization: 'velocity'
                force_recon: False
                m: 0.0
                s: 1.0
            
            enc: 'Enc-Qformer-Uni-XL/2'
            enable_enc_variable_size: True
            encoder_config:
                time_adaln: True
                qformer_mode: 'dual'
                pre_norm: False
                post_norm: True
                xavier_init: False
                qk_norm: False
                attn_mask: False
                
            quantizer_config:
                codebook_size: 32768
                code_dim: 16
                w_diversity: 1.0
                ema_entropy_ratio: 0.8
                w_commit: 1.0
                decay: 0.99
                dead_code_threshold: 0.2
                reset_cluster_size: 0.2
                smart_react: True
                continuous: False
                reg: [0.1, 0.3]
                K: 512

            model: 'MMDiT_XL_Renderer'
            decoder_config:
                repeat: True
                sd3_cond_pooling: None
                class_dropout_prob: 0.
                train_filter: 'all'
                freeze_filter: ''
                init_method: None
                time_adaln: 'pos_emb'
        
    '''