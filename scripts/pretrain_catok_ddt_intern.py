import torch
import numpy as np
import random
import hydra
from omegaconf import DictConfig, OmegaConf
from catok.models.catok_ddt.action_tokenizer import ActionTokenizer
from catok.training.dataset import make_dataset
import tqdm

from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import math
import wandb
import os


def is_distributed():
    """Check if running in distributed mode."""
    return (
        torch.distributed.is_available() 
        and torch.distributed.is_initialized() 
        and torch.distributed.get_world_size() > 1
    )


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def pretrain(cfg: DictConfig):
    
    # Set seed for reproducibility
    seed = cfg.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Check if distributed or single GPU mode
    use_ddp = cfg.get('use_ddp', True)
    
    if use_ddp:
        # +++ DDP Imports +++
        import torch.distributed as dist
        from accelerate import PartialState
        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.utils.data.distributed import DistributedSampler
        
        # Initialize distributed environment using accelerate.PartialState
        distributed_state = PartialState()
        torch.cuda.set_device(device_id := distributed_state.local_process_index)
        device = f'cuda:{device_id}'
        is_main_process = distributed_state.is_main_process
        world_size = distributed_state.num_processes
        rank = distributed_state.process_index
    else:
        # Single GPU mode
        device_id = 0
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        is_main_process = True
        world_size = 1
        rank = 0
        distributed_state = None
    
    cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    
    warmup = cfg['train'].get('warmup', 0)
    action_dim = cfg['tokenizer']['params']['action_dim']
    no_gripper = cfg['dataset'].get('drop_gripper', True)
    dct_loss = cfg['tokenizer']['params'].get('dct_loss', False)
    padding = cfg['dataset'].get('padding', True)
    encmode = cfg['tokenizer']['params'].get('encoding_mode', None)

    if no_gripper:
        def fetch_func(x):
            x = x[:, :, :-1]
            if padding:
                # padding to action_dim
                x = np.pad(x, ((0, 0), (0, 0), (0, action_dim - x.shape[-1])), mode='constant')
            return x
    else:   
        fetch_func = lambda x: x
    
    exp_name = cfg['wandb_expname'] + (
        f"-K{cfg['tokenizer']['params']['k']}"
        f"-cs{cfg['tokenizer']['params']['quantizer_config']['codebook_size']}" # codebook_size
        f"-cd{cfg['tokenizer']['params']['quantizer_config']['code_dim']}" # codebook_dim
        f"-enc{cfg['tokenizer']['params']['enc']}" # encoder type
        f"-dec{cfg['tokenizer']['params']['model']}" # decoder type
        f"-bs{cfg['train']['batch_size']}-epochs{cfg['train']['epochs']}" # batch size
        f"-normalizer{cfg['tokenizer']['params']['normalizer']}-scale{cfg['tokenizer']['params']['normalizer_config'].get('scale_factor', 1)}" # normalizer type
        f"-wp{warmup}" # warmup
        f"-{cfg['dataset']['dataset_name']}" # dataset name
        f"-{cfg['tokenizer']['params']['noise_schedule_config']['parameterization']}" # noise schedule parameterization
    )

    if no_gripper:
        exp_name += "-no_gripper"
    
    if dct_loss:
        dct_loss_weight = cfg['tokenizer']['params'].get('dct_loss_weight', 0.5)
        exp_name += f"-dct_w{dct_loss_weight}"
    
    if encmode:
        exp_name += f"-em{encmode}"

    # VAE latent settings: control Qformer and DiT inputs
    use_vae_latent = cfg['tokenizer']['params'].get('use_vae_latent', False)
    use_vae_for_qformer = cfg['tokenizer']['params'].get('use_vae_for_qformer', use_vae_latent)
    use_vae_for_dit = cfg['tokenizer']['params'].get('use_vae_for_dit', use_vae_latent)
    
    if use_vae_for_qformer or use_vae_for_dit:
        vae_cfg = cfg['tokenizer']['params'].get('vae_config') or {}
        vae_ckpt = cfg['tokenizer']['params'].get('vae_ckpt_path')
        ld = vae_cfg.get('latent_dim', '?')
        
        # Indicate which components use VAE latent: Q=Qformer, D=DiT
        vae_parts = ""
        if use_vae_for_qformer:
            vae_parts += "Q"
        if use_vae_for_dit:
            vae_parts += "D"
        exp_name += f"-vae_{vae_parts}_L{ld}"
        
        # Freeze status
        freeze_vae = cfg['tokenizer']['params'].get('freeze_vae', True)
        exp_name += "-fvae" if freeze_vae else "-tvae"
    
    # +++ 2. Restrict WandB initialization to the main process (rank 0) +++
    if is_main_process and cfg['use_wandb']:
        wandb.init(name=exp_name, project=cfg['wandb_project'], entity=cfg['wandb_entity'], config=cfg)

    tokenizer_config = cfg['tokenizer']['params'].copy()
    tokenizer = ActionTokenizer(
        **tokenizer_config
    ).to(device)
    
    # +++ 3. Wrap the model with DDP if in distributed mode +++
    if use_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        tokenizer = DDP(
            tokenizer, 
            device_ids=[device_id], 
            find_unused_parameters=True, 
            gradient_as_bucket_view=True
        )
    
    # Load ActionOnlyDataset
    train_dataset = make_dataset(cfg['dataset']['dataset_name'], cfg['dataset']['dataset_config'])

    if use_ddp:
        # Check dataset length consistency
        local_len = torch.tensor(len(train_dataset), device=device)
        all_lens = [torch.zeros_like(local_len) for _ in range(world_size)]
        dist.all_gather(all_lens, local_len)
        all_lens_vals = [l.item() for l in all_lens]
        if len(set(all_lens_vals)) > 1:
            if is_main_process:
                print(f"ERROR: Dataset lengths differ across ranks: {all_lens_vals}")
            raise RuntimeError(f"Dataset lengths mismatch: {all_lens_vals}")
        elif is_main_process:
            print(f"Dataset length consistency check passed: {all_lens_vals[0]}")
    
    # Use DistributedSampler for DDP, or regular shuffle for single GPU
    if use_ddp:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg['train']['batch_size'],
            sampler=train_sampler,
            num_workers=cfg.get('dataloader_workers', 2),
            pin_memory=True,
            drop_last=True,
        )
    else:
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg['train']['batch_size'],
            shuffle=True,
            num_workers=cfg.get('dataloader_workers', 2),
            pin_memory=True,
            drop_last=True,
        )
    
    optimizer = torch.optim.AdamW(
        tokenizer.parameters(),
        lr=cfg['train']['learning_rate'] * world_size,  # Scale LR by world size
        weight_decay=cfg['train']['weight_decay'],
    )
    
    num_training_steps = cfg['train']['epochs'] * len(train_loader) if hasattr(train_loader, '__len__') else cfg['train']['max_steps']
    warmup_steps = int(num_training_steps * warmup / 100.0) if warmup > 0 else 0
    
    if warmup == 0:
        # Use a warmup scheduler if specified in the config
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - warmup_steps,
            eta_min=0.0,
        )
    else:
        def combine_scheduler(current_step: int):
            if current_step < warmup_steps:
                # linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            
            # Cosine Annealing after warmup
            cosine_steps = num_training_steps - warmup_steps
            step_in_cosine = current_step - warmup_steps

            return 0.5 * (1 + math.cos(math.pi * step_in_cosine / cosine_steps))

        scheduler = LambdaLR(
            optimizer, 
            lr_lambda=combine_scheduler
        )

    save_dir = cfg['ckpt_dir']
    if is_main_process:
        os.makedirs(save_dir, exist_ok=True)

    try:
        time_step = 0
        for epoch in tqdm.trange(cfg['train']['epochs'], disable=(not is_main_process)):
            # Set epoch for DistributedSampler to ensure proper shuffling
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            
            for batch in tqdm.tqdm(train_loader, disable=(not is_main_process)):
                act = fetch_func(batch)
                # from catok.utils.visualize import visualize_action_chunks
                # normalizer = tokenizer.normalizer
                # visualize_action_chunks(normalizer.normalize(act[:5]), save_path="tmp", file_name="action_chunks.png")
                # import ipdb; ipdb.set_trace()
                act = torch.from_numpy(act).to(device) if isinstance(act, np.ndarray) else act.to(device)
                
                # Determine if we should generate visualization image (e.g., every N steps)
                log_image_every = cfg.get('log_image_every_n_steps', 100)
                should_log_image = (time_step % log_image_every == 0) if is_main_process else False
                
                # The DDP model automatically handles the forward pass on the correct data
                # and gradient synchronization during the backward pass.
                # Note: Access the original model via `tokenizer.module` if needed.
                loss, loss_dict = tokenizer(act, full_tokens=cfg['train'].get('full_tokens', False), return_image=should_log_image)
                
                # +++ 6. Restrict logging to the main process (rank 0) +++
                if is_main_process and cfg['use_wandb'] and time_step % cfg.get('log_every_n_steps', 10) == 0:
                    # You can optionally average metrics across all GPUs for more accuracy
                    # dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                    
                    # Base losses
                    log_data = {
                        "pretrain/loss": loss_dict['loss'],
                        "pretrain/commit_loss": loss_dict['commit_loss'],
                        "pretrain/diversity_entropy": loss_dict['diversity_entropy'],
                        "pretrain/deterministic_entropy": loss_dict['deterministic_entropy'],
                        "pretrain/code_book_usage": np.exp(loss_dict['deterministic_entropy']) / cfg['tokenizer']['params']['quantizer_config']['codebook_size'],
                        "pretrain/dm_loss": loss_dict.get('dm_loss', loss_dict.get('dm_mse', 0.0)),
                        "pretrain/loss_small": loss_dict['loss_small'],
                        "pretrain/loss_mid": loss_dict['loss_mid'],
                        "pretrain/loss_large": loss_dict['loss_large'],
                        "pretrain/loss_uncon": loss_dict.get('loss_uncon', 0.0),
                    }
                    
                    # L1, L2, DCT losses (always computed)
                    if 'loss_l1' in loss_dict:
                        log_data["pretrain/loss_l1"] = loss_dict['loss_l1']
                    if 'loss_l2' in loss_dict:
                        log_data["pretrain/loss_l2"] = loss_dict['loss_l2']
                    if 'loss_dct' in loss_dict:
                        log_data["pretrain/loss_dct"] = loss_dict['loss_dct']
                    
                    details = False
                    if details:
                        # Weighted losses (if applicable)
                        if 'loss_l1_weighted' in loss_dict:
                            log_data["pretrain/loss_l1_weighted"] = loss_dict['loss_l1_weighted']
                        if 'loss_l2_weighted' in loss_dict:
                            log_data["pretrain/loss_l2_weighted"] = loss_dict['loss_l2_weighted']
                        if 'loss_dct_weighted' in loss_dict:
                            log_data["pretrain/loss_dct_weighted"] = loss_dict['loss_dct_weighted']
                        
                        # Reconstruction losses (if force_recon is enabled)
                        if 'recon_l1' in loss_dict:
                            log_data["pretrain/recon_l1"] = loss_dict['recon_l1']
                        if 'recon_l2' in loss_dict:
                            log_data["pretrain/recon_l2"] = loss_dict['recon_l2']
                    
                    # Legacy dm_mse for backward compatibility
                    if 'dm_mse' in loss_dict:
                        log_data["pretrain/dm_mse"] = loss_dict['dm_mse']
                    
                    # VAE latent diffusion: VAE recon loss (when use_vae_latent and not freeze_vae)
                    if 'vae_recon_loss' in loss_dict:
                        log_data["pretrain/vae_recon_loss"] = loss_dict['vae_recon_loss']
                    
                    # Embedding statistics (encoder)
                    embedding_stats_keys = [
                        'action_emb_l2_mean', 'action_emb_l2_std',
                        'pos_emb_l2_mean', 'pos_emb_l2_std',
                        'combined_emb_l2_mean', 'combined_emb_l2_std',
                        'query_init_l2_mean', 'query_init_l2_std', 'query_diversity',
                    ]
                    for key in embedding_stats_keys:
                        if key in loss_dict:
                            log_data[f"embedding/{key}"] = loss_dict[key]
                    
                    # Per-layer embedding statistics
                    for i in range(32):  # Support up to 32 layers
                        layer_keys = [
                            f'layer_{i}_action_l2_mean', f'layer_{i}_action_l2_std',
                            f'layer_{i}_query_l2_mean', f'layer_{i}_query_l2_std',
                            f'layer_{i}_attn_pos_emb_l2_mean', f'layer_{i}_attn_pos_emb_l2_std',
                        ]
                        for key in layer_keys:
                            if key in loss_dict:
                                log_data[f"embedding/{key}"] = loss_dict[key]
                    
                    # Add action visualization image if available
                    if 'action_visualization' in loss_dict and loss_dict['action_visualization'] is not None:
                        log_data["pretrain/action_visualization"] = wandb.Image(loss_dict['action_visualization'])
                    
                    wandb.log(log_data, step=time_step)
                optimizer.zero_grad()
                loss.backward() # Gradients are automatically synced here
                
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    tokenizer.parameters(),
                    max_norm=1.0,
                )
                
                if is_main_process and cfg['use_wandb']:
                    wandb.log({"pretrain/grad_norm": grad_norm}, step=time_step)
                
                optimizer.step()
                next_lr = scheduler.get_last_lr()[0]
                
                if is_main_process and cfg['use_wandb']:
                    wandb.log({"pretrain/learning_rate": next_lr}, step=time_step)
                scheduler.step()
                time_step += 1
                
                # +++ 7. Only save the model checkpoint from the main process +++
                if is_main_process and time_step % cfg['save_every_n_steps'] == 0:
                    # Get model state dict (handle DDP wrapper)
                    model_state = tokenizer.module.state_dict() if use_ddp else tokenizer.state_dict()
                    save_path = os.path.join(save_dir, f"{time_step}.pth")
                    torch.save(
                        {
                            'model_state_dict': model_state,
                            'config': cfg,
                        },
                        save_path,
                    )
    finally:
        # Finish wandb logging before cleanup
        if is_main_process and cfg.get('use_wandb', False):
            wandb.finish()
        
        # Clean up distributed process group (always called, even on exception)
        if use_ddp:
            if dist.is_initialized():
                print("Destroying process group")
                dist.destroy_process_group()

if __name__ == "__main__":
    pretrain()