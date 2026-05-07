"""
Pretrain Action VAE model.

This script trains a VAE to encode action sequences (B, T, D) into continuous latents.
Based on pretrain_catok_ddt.py but adapted for VAE training.
"""

import torch
import numpy as np
import random
import hydra
from omegaconf import DictConfig, OmegaConf
from catok.models.action_vae import ActionVAE
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


@hydra.main(config_path="../conf", config_name="config_vae", version_base="1.3")
def pretrain(cfg: DictConfig):
    
    # Set seed for reproducibility
    seed = cfg.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Check if distributed or single GPU mode
    use_ddp = cfg.get('use_ddp', False)
    
    # Check if we're actually in a distributed environment
    if use_ddp:
        try:
            import torch.distributed as dist
            from accelerate import PartialState
            from torch.nn.parallel import DistributedDataParallel as DDP
            from torch.utils.data.distributed import DistributedSampler
            
            # Initialize distributed environment using accelerate.PartialState
            distributed_state = PartialState()
            
            # Check if actually running distributed
            if distributed_state.num_processes > 1:
                torch.cuda.set_device(device_id := distributed_state.local_process_index)
                device = f'cuda:{device_id}'
                is_main_process = distributed_state.is_main_process
                world_size = distributed_state.num_processes
                rank = distributed_state.process_index
            else:
                # Fallback to single GPU mode
                use_ddp = False
                raise RuntimeError("Single process detected, falling back to single GPU mode")
        except Exception as e:
            print(f"DDP initialization failed: {e}, falling back to single GPU mode")
            use_ddp = False
    
    if not use_ddp:
        # Single GPU mode
        device_id = 0
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        is_main_process = True
        world_size = 1
        rank = 0
        distributed_state = None
    
    cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    
    warmup = cfg['train'].get('warmup', 0)
    action_dim = cfg['vae']['params']['action_dim']
    no_gripper = cfg['dataset'].get('drop_gripper', True)
    padding = cfg['dataset'].get('padding', True)
    encmode = cfg['vae']['params'].get('encoding_mode', None)

    if no_gripper:
        def fetch_func(x):
            x = x[:, :, :-1]
            if padding:
                # padding to action_dim
                x = np.pad(x, ((0, 0), (0, 0), (0, action_dim - x.shape[-1])), mode='constant')
            return x
    else:   
        fetch_func = lambda x: x
    
    # Build experiment name
    exp_name = cfg['wandb_expname'] + (
        f"-latent{cfg['vae']['params']['latent_dim']}"
        f"-hidden{'-'.join(map(str, cfg['vae']['params']['hidden_dims']))}"
        f"-kl{cfg['vae']['params']['kl_weight']}"
        f"-{'transformer' if cfg['vae']['params']['use_transformer'] else 'conv'}"
        f"-layers{cfg['vae']['params']['num_transformer_layers']}"
        f"-bs{cfg['train']['batch_size']}-epochs{cfg['train']['epochs']}"
        f"-normalizer{cfg['vae']['params']['normalizer']}"
        f"-wp{warmup}"
        f"-{cfg['dataset']['dataset_name']}"
    )

    if no_gripper:
        exp_name += "-no_gripper"
    
    if encmode:
        exp_name += f"-em{encmode}"
    
    # Add magnitude embedding info to exp_name
    use_mag = cfg['vae']['params'].get('use_magnitude_embedding', False)
    if use_mag:
        mag_type = cfg['vae']['params'].get('magnitude_type', 'rms')
        mag_weight = cfg['vae']['params'].get('magnitude_loss_weight', 0.1)
        exp_name += f"-mag_{mag_type}_w{mag_weight}"
    
    # Add loss type info to exp_name
    use_huber = cfg['vae']['params'].get('use_huber_loss', False)
    if use_huber:
        huber_delta = cfg['vae']['params'].get('huber_delta', 1.0)
        exp_name += f"-huber_d{huber_delta}"
    else:
        recon_l1_weight = cfg['vae']['params'].get('recon_l1_weight', 0.0)
        if recon_l1_weight > 0:
            exp_name += f"-l1w{recon_l1_weight}"
    
    # +++ 2. Restrict WandB initialization to the main process (rank 0) +++
    if is_main_process and cfg['use_wandb']:
        wandb.init(name=exp_name, project=cfg['wandb_project'], entity=cfg['wandb_entity'], config=cfg)

    vae_config = cfg['vae']['params'].copy()
    vae = ActionVAE(**vae_config).to(device)
    
    if is_main_process:
        print(f"VAE model parameters: {sum(p.numel() for p in vae.parameters()):,}")
    
    # +++ 3. Wrap the model with DDP if in distributed mode +++
    if use_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        vae = DDP(
            vae, 
            device_ids=[device_id], 
            find_unused_parameters=False, 
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
        vae.parameters(),
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
                act = torch.from_numpy(act).to(device) if isinstance(act, np.ndarray) else act.to(device)
                
                # Forward pass
                loss, loss_dict = vae(act, return_loss=True)
                
                # +++ 6. Restrict logging to the main process (rank 0) +++
                if is_main_process and cfg['use_wandb'] and time_step % cfg.get('log_every_n_steps', 10) == 0:
                    log_data = {
                        "pretrain/loss": loss_dict['loss'],
                        "pretrain/recon_loss": loss_dict['recon_loss'],
                        "pretrain/recon_l1_loss": loss_dict['recon_l1_loss'],
                        "pretrain/recon_l1_loss_weighted": loss_dict['recon_l1_loss_weighted'],
                        "pretrain/huber_loss": loss_dict['huber_loss'],
                        "pretrain/kl_loss": loss_dict['kl_loss'],
                        "pretrain/kl_loss_weighted": loss_dict['kl_loss_weighted'],
                        "pretrain/recon_l1_orig": loss_dict['recon_l1_orig'],
                        "pretrain/recon_l2_orig": loss_dict['recon_l2_orig'],
                        "latent/mu_mean": loss_dict['mu_mean'],
                        "latent/mu_std": loss_dict['mu_std'],
                        "latent/logvar_mean": loss_dict['logvar_mean'],
                        "latent/z_mean": loss_dict['z_mean'],
                        "latent/z_std": loss_dict['z_std'],
                    }
                    
                    # Log magnitude losses if enabled
                    if 'mag_loss' in loss_dict:
                        log_data["pretrain/mag_loss"] = loss_dict['mag_loss']
                        log_data["pretrain/mag_loss_weighted"] = loss_dict['mag_loss_weighted']
                    if 'mag_mean' in loss_dict:
                        log_data["magnitude/mag_mean"] = loss_dict['mag_mean']
                        log_data["magnitude/mag_std"] = loss_dict['mag_std']
                    if 'mag_pred_mean' in loss_dict:
                        log_data["magnitude/mag_pred_mean"] = loss_dict['mag_pred_mean']
                        log_data["magnitude/mag_pred_std"] = loss_dict['mag_pred_std']
                    
                    wandb.log(log_data, step=time_step)
                
                optimizer.zero_grad()
                loss.backward()
                
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    vae.parameters(),
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
                    model_state = vae.module.state_dict() if use_ddp else vae.state_dict()
                    save_path = os.path.join(save_dir, f"{exp_name}_{time_step}.pth")
                    torch.save(
                        {
                            'model_state_dict': model_state,
                            'config': cfg,
                        },
                        save_path,
                    )
                    if is_main_process:
                        print(f"Saved checkpoint to {save_path}")
        
        # Save final checkpoint
        if is_main_process:
            model_state = vae.module.state_dict() if use_ddp else vae.state_dict()
            save_path = os.path.join(save_dir, f"{exp_name}_final.pth")
            torch.save(
                {
                    'model_state_dict': model_state,
                    'config': cfg,
                },
                save_path,
            )
            print(f"Saved final checkpoint to {save_path}")
            
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
