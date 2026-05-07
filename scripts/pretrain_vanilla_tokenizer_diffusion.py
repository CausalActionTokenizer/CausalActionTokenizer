import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_dct import dct
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

import tqdm
import hydra
import math
import wandb
import os
import sys
import random
from omegaconf import DictConfig, OmegaConf

from catok.models.catok_ddt.vanilla_utils import construct_model
from catok.models.catok_ddt.vanilla_tokenizer import VanillaVQDiffusion
from catok.training.dataset import make_dataset
from catok.training.statistics import build_action_normalizer


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
        device_id = distributed_state.local_process_index
        torch.cuda.set_device(device_id)
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
    
    # model
    vq_mode = cfg['tokenizer']['vq']['vq_mode']
    max_reset_steps = cfg['tokenizer']['vq']['max_reset_steps']
    model = construct_model(cfg['tokenizer'], device=device)
    model.train()
    if use_ddp:
        model = DDP(
            model, 
            device_ids=[device_id], 
            find_unused_parameters=True, 
            gradient_as_bucket_view=True
        )
    save_dir = cfg['ckpt_dir']
    if is_main_process:
        os.makedirs(save_dir, exist_ok=True)

    # dataset
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

    # dataloader
    if use_ddp:
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
            num_workers=cfg.get('dataloader_workers', 4),
            pin_memory=True,
            drop_last=True,
        )

    # data preprocess
    normalizer_cfg = cfg['normalizer']
    normalizer = build_action_normalizer(
        method=normalizer_cfg['type'],
        normalizer_config=normalizer_cfg['config'],
    )
    def fetch_func(x):
        # x = x[:, :, :-1]
        action_dim = cfg['tokenizer']['basic']['action_dim']
        x = np.pad(x, ((0, 0), (0, 0), (0, action_dim - x.shape[-1])), mode='constant')
        return x
    
    # loss function
    if cfg['train']['loss_fn'] == 'l1':
        loss_fn = nn.L1Loss()
    elif cfg['train']['loss_fn'] == 'l2':
        loss_fn = nn.MSELoss()
    else:
        raise ValueError('Unknown loss function.')
    
    # optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['train']['learning_rate'] * world_size,
        weight_decay=cfg['train']['weight_decay']
    )
    num_training_steps = cfg['train']['epochs'] * len(train_loader)
    warmup = cfg['train'].get('warmup', 0)
    warmup_steps = int(num_training_steps * warmup / 100.0) if warmup > 0 else 0
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

    # logger
    exp_name = cfg["wandb_expname"]
    if is_main_process and cfg['use_wandb']:
        wandb.init(name=exp_name, project=cfg['wandb_project'], config=cfg)

    epochs = cfg['train']['epochs']
    dct_loss = cfg['train']['dct_loss']
    consistency_weight = cfg['tokenizer'].get('decoder', {}).get('vanilla_consistency_weight', 0.0)
    consistency_enabled = consistency_weight > 0
    
    if is_main_process:
        print("DataLoader Len: ", len(train_loader))
    steps = 0
    for epoch in tqdm.trange(epochs, disable=(not is_main_process)):
        # Set epoch for DistributedSampler to ensure proper shuffling
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
                
        for batch in tqdm.tqdm(train_loader, disable=(not is_main_process)):
            # batch_x shape: (B, 10, 8)
            steps += 1
            optimizer.zero_grad()
            
            action_chunks = fetch_func(batch)
            action_chunks = normalizer.normalize(action_chunks)
            action_chunks = torch.from_numpy(action_chunks)
            action_chunks = action_chunks.to(device=device, dtype=torch.float32)
            
            recon_x, loss, loss_dict, indices, perplexity = model(action_chunks)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            if vq_mode in ['RVQ_EMA', 'VQ_EMA'] and steps == max_reset_steps:
                if use_ddp:
                    model.module.vq.set_threshold(0)
                else:
                    model.vq.set_threshold(0)

            if is_main_process and cfg['use_wandb'] and steps % cfg.get('log_every_n_steps', 10) == 0:
                log_dict = {"recon_loss": loss_dict['recon']}
                if vq_mode == 'RVQ_EMA':
                    log_dict["vq_loss"] = loss_dict['vq_loss']
                    for i in range(len(indices)):
                        log_dict[f"active_code_{i}"] = len(torch.unique(indices[i]))
                        log_dict[f"perplexity_{i}"] = perplexity[i]
                elif vq_mode is not None:
                    log_dict["vq_loss"] = loss_dict['vq_loss']
                    log_dict["active_code"] = len(torch.unique(indices))
                    log_dict["perplexity"] = perplexity
                if dct_loss:
                    log_dict["dct_loss"] = loss_dict["dct"]
                if consistency_enabled and "consistency" in loss_dict:
                    log_dict["consistency_loss"] = loss_dict["consistency"]
                if "overlap_infonce" in loss_dict:
                    log_dict["overlap_infonce"] = loss_dict["overlap_infonce"]
                wandb.log(log_dict, step=steps)

            if is_main_process and steps % cfg['save_every_n_steps'] == 0:
                # Get model state dict (handle DDP wrapper)
                model_state = model.module.state_dict() if use_ddp else model.state_dict()
                save_path = os.path.join(save_dir, f"{steps:06d}.pth")
                torch.save({
                    'model_state_dict': model_state,
                    'config': cfg,
                },
                    save_path,
                )

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