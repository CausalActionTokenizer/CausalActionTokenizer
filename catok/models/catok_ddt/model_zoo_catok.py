import torch
from catok.models.catok_ddt.encoders import QformerEncoder, Encoder
from catok.models.catok_ddt.sd3.mmdit import MMDiT, MMDiT_Action
from functools import partial

def MMDiT_XL_Action(**kwargs):
    # Allow explicit decoder_hidden_dim to override default hidden_size (1536 = 64 * 24)
    decoder_hidden_dim = kwargs.pop("decoder_hidden_dim", None)
    hidden_size = decoder_hidden_dim if decoder_hidden_dim is not None else 1536
    context_embedder_config = {
        "target": "torch.nn.Linear",
        "params": {"in_features": kwargs['encoder_hidden_size'], "out_features": hidden_size},
    }
    diffusion_model = MMDiT_Action(
        pos_embed_scaling_factor=None,
        pos_embed_offset=None,
        pos_embed_max_size=192,
        # patch_size=2,
        depth=24,
        # num_patches=36864,
        adm_in_channels=kwargs['encoder_hidden_size'],
        context_embedder_config=context_embedder_config,
        device='cpu',
        dtype=torch.float,
        decoder_hidden_dim=decoder_hidden_dim,
        **kwargs
    )
    return diffusion_model

def MMDiT_Tiny_Action(**kwargs):
    # Allow explicit decoder_hidden_dim to override default hidden_size (256 = 64 * 4)
    decoder_hidden_dim = kwargs.pop("decoder_hidden_dim", None)
    hidden_size = decoder_hidden_dim if decoder_hidden_dim is not None else 256
    context_embedder_config = {
        "target": "torch.nn.Linear",
        "params": {"in_features": kwargs['encoder_hidden_size'], "out_features": hidden_size},
    }
    diffusion_model = MMDiT_Action(
        pos_embed_scaling_factor=None,
        pos_embed_offset=None,
        pos_embed_max_size=192,
        # patch_size=2,
        depth=4,
        # num_patches=36864,
        adm_in_channels=kwargs['encoder_hidden_size'],
        context_embedder_config=context_embedder_config,
        device='cpu',
        dtype=torch.float,
        decoder_hidden_dim=decoder_hidden_dim,
        **kwargs
    )
    return diffusion_model

def MMDiT_Medium_Action(**kwargs):
    depth = 6
    # Allow explicit decoder_hidden_dim to override default hidden_size (384 = 64 * 6)
    decoder_hidden_dim = kwargs.pop("decoder_hidden_dim", None)
    hidden_size = decoder_hidden_dim if decoder_hidden_dim is not None else 64 * depth
    context_embedder_config = {
        "target": "torch.nn.Linear",
        "params": {"in_features": kwargs['encoder_hidden_size'], "out_features": hidden_size},
    }
    diffusion_model = MMDiT_Action(
        pos_embed_scaling_factor=None,
        pos_embed_offset=None,
        pos_embed_max_size=192,
        # patch_size=2,
        depth=depth,
        # num_patches=36864,
        adm_in_channels=kwargs['encoder_hidden_size'],
        context_embedder_config=context_embedder_config,
        device='cpu',
        dtype=torch.float,
        decoder_hidden_dim=decoder_hidden_dim,
        **kwargs
    )
    return diffusion_model

def MMDiT_Action_Auto(depth, **kwargs):
    dim_per_depth = kwargs.pop("dim_per_depth", 64)
    # Allow explicit decoder_hidden_dim to override computed hidden_size
    decoder_hidden_dim = kwargs.pop("decoder_hidden_dim", None)
    hidden_size = decoder_hidden_dim if decoder_hidden_dim is not None else dim_per_depth * depth
    context_embedder_config = {
        "target": "torch.nn.Linear",
        "params": {"in_features": kwargs['encoder_hidden_size'], "out_features": hidden_size},
    }
    diffusion_model = MMDiT_Action(
        pos_embed_scaling_factor=None,
        pos_embed_offset=None,
        pos_embed_max_size=192,
        # patch_size=2,
        depth=depth,
        # num_patches=36864,
        adm_in_channels=kwargs['encoder_hidden_size'],
        context_embedder_config=context_embedder_config,
        device='cpu',
        dtype=torch.float,
        dim_per_depth=dim_per_depth,
        decoder_hidden_dim=decoder_hidden_dim,  # Pass to MMDiT_Action
        **kwargs
    )
    return diffusion_model

def Enc_Qformer_Bi_Tiny(**kwargs):
    # Allow explicit encoder_hidden_dim to override default hidden_size (128)
    encoder_hidden_dim = kwargs.pop("encoder_hidden_dim", None)
    hidden_size = encoder_hidden_dim if encoder_hidden_dim is not None else 128
    return QformerEncoder(
        # patch_size=2, 
        hidden_size=hidden_size, num_heads=4, depth=4,
        query_dim=hidden_size, query_heads=4, 
        bidirectional=True, 
        **kwargs
    )
    
def Enc_Qformer_Bi_Medium(**kwargs):
    return Enc_Qformer_Bi_Auto(depth=6, **kwargs)
    
def Enc_Qformer_Bi_Auto(depth, **kwargs):
    num_heads = depth
    dim_per_depth = kwargs.pop("dim_per_depth", 64)
    # Allow explicit encoder_hidden_dim to override computed hidden_size
    encoder_hidden_dim = kwargs.pop("encoder_hidden_dim", None)
    hidden_size = encoder_hidden_dim if encoder_hidden_dim is not None else dim_per_depth * depth
    # Allow explicit num_heads override
    num_heads = kwargs.pop("num_heads", num_heads)
    return QformerEncoder(
        # patch_size=2, 
        hidden_size=hidden_size, num_heads=num_heads, depth=depth,
        query_dim=hidden_size, query_heads=num_heads,
        bidirectional=True, 
        **kwargs
    )
    
def Enc_Qformer_Bi_Medium_Heter(**kwargs):
    depth = 4
    num_heads = 4
    # Allow explicit encoder_hidden_dim to override default hidden_size (256 = 64 * 4)
    encoder_hidden_dim = kwargs.pop("encoder_hidden_dim", None)
    hidden_size = encoder_hidden_dim if encoder_hidden_dim is not None else 64 * depth
    return QformerEncoder(
        # patch_size=2, 
        hidden_size=hidden_size, num_heads=num_heads, depth=depth,
        query_dim=16, query_heads=num_heads,
        bidirectional=True, 
        **kwargs
    )

def Enc_Tiny(**kwargs):
    # Allow explicit encoder_hidden_dim to override default hidden_size (256)
    encoder_hidden_dim = kwargs.pop("encoder_hidden_dim", None)
    hidden_size = encoder_hidden_dim if encoder_hidden_dim is not None else 256
    return Encoder(
        # patch_size=8, 
        hidden_size=hidden_size, num_heads=4, **kwargs)

def Enc_Qformer_Bi_L(**kwargs):
    # Allow explicit encoder_hidden_dim to override default hidden_size (16)
    encoder_hidden_dim = kwargs.pop("encoder_hidden_dim", None)
    hidden_size = encoder_hidden_dim if encoder_hidden_dim is not None else 16
    return QformerEncoder(
        # patch_size=2, 
        hidden_size=hidden_size, num_heads=2, depth=24,
        query_dim=hidden_size, query_heads=2, bidirectional=True, **kwargs
    )
    
def Enc_Qformer_Uni_L(**kwargs):
    # Allow explicit encoder_hidden_dim to override default hidden_size (64)
    encoder_hidden_dim = kwargs.pop("encoder_hidden_dim", None)
    hidden_size = encoder_hidden_dim if encoder_hidden_dim is not None else 64
    return QformerEncoder(
        # patch_size=2, 
        hidden_size=hidden_size, num_heads=4, depth=20,
        query_dim=128, query_heads=8, bidirectional=False, **kwargs
    )

L_depths = list(range(2, 25))
L_names = [f"Enc-Qformer-Bi-Auto{d}" for d in L_depths]
L_kv = [partial(Enc_Qformer_Bi_Auto, depth=d) for d in L_depths]
dict_to_append = dict(zip(L_names, L_kv))

# No-VQ (continuous) encoder variants - bypass vector quantization
def Enc_Qformer_Bi_Auto_NoVQ(depth, **kwargs):
    """
    Continuous encoder variant that bypasses vector quantization.
    Uses the same architecture as Enc_Qformer_Bi_Auto but with no_vq=True.
    """
    kwargs['no_vq'] = True
    return Enc_Qformer_Bi_Auto(depth=depth, **kwargs)

# Generate no-VQ encoder variants for each depth
L_names_novq = [f"Enc-Qformer-Bi-Auto{d}-NoVQ" for d in L_depths]
L_kv_novq = [partial(Enc_Qformer_Bi_Auto_NoVQ, depth=d) for d in L_depths]
dict_to_append_novq = dict(zip(L_names_novq, L_kv_novq))

Enc_models = {
    'Enc_Tiny': Enc_Tiny,
    'Enc-Qformer-Bi-Tiny': Enc_Qformer_Bi_Tiny,
    'Enc-Qformer-Bi-Medium': Enc_Qformer_Bi_Medium,
    'Enc-Qformer-Bi-Medium-Heter': Enc_Qformer_Bi_Medium_Heter,
    'Enc-Qformer-Bi-L': Enc_Qformer_Bi_L,
    'Enc-Qformer-Uni-L': Enc_Qformer_Uni_L,
    **dict_to_append,
    **dict_to_append_novq,  # Add no-VQ variants
}

L_depths = list(range(2, 25))
L_names = [f"MMDiT-Action-Auto{d}" for d in L_depths]
L_kv = [partial(MMDiT_Action_Auto, depth=d) for d in L_depths]
dict_to_append = dict(zip(L_names, L_kv))

DiT_models = {
    "MMDiT_XL_Action": MMDiT_XL_Action,
    "MMDiT_Tiny_Action": MMDiT_Tiny_Action,
    "MMDiT_Medium_Action": MMDiT_Medium_Action,
    **dict_to_append
}