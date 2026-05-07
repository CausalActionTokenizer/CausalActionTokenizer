# Copyright 2024 Stability AI, The HuggingFace Team and The InstantX Team. All rights reserved.

# Copyright (C) 2025. All rights reserved.

# Modified this file to extend the branch.

# Licensed under MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/license/mit
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================



import torch
from einops import rearrange
from catok.utils.transformation import dct_differentiable
from catok.utils.visualize import visualize_action_chunks

TRADITION = 1000

def append_to_shape(t, x_shape):
    return t.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    B = tensor.shape[0]
    loss = tensor.reshape(B, -1)
    loss = torch.sum(loss, dim=1) / loss.shape[1]
    return loss


def sum_flat(tensor):
    """
    Take the sum over all non-batch dimensions.
    """
    return tensor.sum(dim=list(range(1, len(tensor.shape))))

class RectifiedFlow(torch.nn.Module):
    def __init__(self, num_timesteps=100, start=1.0, cut_of_k=None, schedule="log_norm", val_schedule='shift', parameterization='x0', shift=1.0, m=0, s=1, force_recon=False, device='cuda',
                 is_eval=False, loss_config=None):
        super().__init__()
        self.schedule = schedule
        self.parameterization = parameterization
        self.m = m
        self.s = s
        self.shift = shift
        self.is_eval = is_eval  # eval
        self.num_timesteps = num_timesteps
        self.start = start
        self.make_schedule(schedule=val_schedule, args=shift)
        self.device = device
        self.force_recon = force_recon
        self.cut_of_k = cut_of_k
        self.t_trajectory = self.schedule_by_uniform
        
        # Default loss config
        default_loss_config = {
            'use_l1_loss': False,
            'use_l2_loss': True,
            'use_dct_l1_loss': False,
            'use_dct_l2_loss': True,
            'l1_loss_weight': 1.0,
            'l2_loss_weight': 1.0,
            'dct_l1_loss_weight': 0.5,
            'dct_l2_loss_weight': 0.5,
        }
        if loss_config is None:
            loss_config = default_loss_config
        else:
            # Merge with defaults
            loss_config = {**default_loss_config, **loss_config}
        self.loss_config = loss_config
     
            
    def make_schedule(self, schedule="uniform", args=None):
        base_t = torch.linspace(self.start, 0, self.num_timesteps+1).cuda()
        if schedule == "uniform":
            scheduled_t = base_t
        elif schedule == "shift":
            scheduled_t =self.shift * base_t / (1 + (self.shift - 1) * base_t)
        elif schedule == "align_resolution":
            e = torch.e
            res1, s1, res2, s2, target_res, c = args
            m = (s1 -s2) / (res1 - res2) * (target_res - res1) + s1
            scheduled_t = e ** m / (e ** m + (1/base_t - 1) ** c)
        self.register_buffer("timestep_map", scheduled_t[:-1] * TRADITION)
        self.register_buffer("scheduled_t", scheduled_t[:-1])
        self.register_buffer("scheduled_t_prev", scheduled_t[1:])
        self.register_buffer("one_minus_scheduled_t", 1-scheduled_t[:-1])
    
    def shift_t(self, t, shift):
        return shift * t / (1 + (shift - 1) * t)

    def q_sample(self, x, t, noise=None):
        t = append_to_shape(t, x.shape)
        if noise is None:
            noise = torch.randn_like(x)
        return t * noise + (1 - t) * x

    def get_target(self, x, noise):
        target = noise - x
        return target
    
    def schedule_by_uniform(self, t):
        return t
    
    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, recon_ratio=None, original_t=None, 
                       loss_config=None, return_image=False):
        """
        Compute training losses for a single timestep.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :param loss_config: dict with loss configuration. If None, uses instance default.
            Keys: use_l1_loss, use_l2_loss, use_dct_loss, l1_loss_weight, l2_loss_weight, dct_loss_weight
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)
        
        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "velocity":
            target = noise - x_start
        else:
            raise NotImplementedError()
            
        terms = {}
        # print(x_t.shape) N T A
        v, drop_ids = model(x_t, t, **model_kwargs)
        v_gt = noise - x_start

        # Check and fix shape mismatch for v (model output)
        # This can happen when using patch embedder, where v length might be T*A instead of T
        B_x, T_x, A_x = x_start.shape
        B_v, Len_v, A_v = v.shape
        
        if v.shape != x_start.shape:
            # Verify batch size matches
            if B_v != B_x:
                raise RuntimeError(
                    f"Batch size mismatch: x_start.shape={x_start.shape}, v.shape={v.shape}"
                )
            
            # Check if total elements match for direct reshape
            total_elements_v = v.numel()
            total_elements_x = x_start.numel()
            
            if total_elements_v == total_elements_x:
                # Total elements match, can reshape directly
                v = v.reshape(B_x, T_x, A_x)
            elif Len_v == T_x * A_x and A_v == A_x:
                # Special case: v is (B, T*A, A) and needs to be (B, T, A)
                # Reshape to (B, T, A, A) then take mean or first along last dim
                # Actually, if v is (B, T*A, A), we can reshape to (B, T, A, A) and take mean
                v = v.reshape(B_x, T_x, A_x, A_v)
                # Take mean along the last dimension (the extra A dimension)
                v = v.mean(dim=-1)  # Result: (B, T, A)
            elif Len_v % T_x == 0 and A_v == A_x:
                # Len_v is a multiple of T_x, e.g., v is (B, k*T, A) where k = Len_v // T_x
                k = Len_v // T_x
                # Reshape to (B, T, k, A) and take mean along k dimension
                v = v.reshape(B_x, T_x, k, A_x)
                v = v.mean(dim=2)  # Result: (B, T, A)
            elif A_v != A_x:
                raise RuntimeError(
                    f"Action dimension mismatch: v.shape={v.shape}, x_start.shape={x_start.shape}, "
                    f"A_v={A_v} != A_x={A_x}"
                )
            else:
                # Use interpolation to match the length
                # Reshape v to (B, Len_v, A_v) and interpolate to (B, T_x, A_x)
                v = v.permute(0, 2, 1)  # (B, A_v, Len_v)
                v = torch.nn.functional.interpolate(
                    v.unsqueeze(-1),  # (B, A_v, Len_v, 1)
                    size=(T_x, 1),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(-1)  # (B, A_v, T_x)
                v = v.permute(0, 2, 1)  # (B, T_x, A_v)
                if A_v != A_x:
                    # If action dims don't match, use linear layer or raise error
                    raise RuntimeError(
                        f"Action dimension mismatch after interpolation: "
                        f"v.shape={v.shape}, expected A_x={A_x}"
                    )
        
        if self.force_recon:
            assert self.parameterization == 'velocity'
            # N T A version
            model_output = x_t - rearrange(t, 'b -> b 1 1') * v
            target = x_start
        else:
            model_output = v 

        # Use provided loss_config or fall back to instance default
        if loss_config is None:
            loss_config = self.loss_config
        else:
            # Merge with instance default
            loss_config = {**self.loss_config, **loss_config}
        
        use_l1 = loss_config['use_l1_loss']
        use_l2 = loss_config['use_l2_loss']
        use_dct_l1 = loss_config['use_dct_l1_loss']
        use_dct_l2 = loss_config['use_dct_l2_loss']
        w_l1 = loss_config['l1_loss_weight']
        w_l2 = loss_config['l2_loss_weight']
        w_dct_l1 = loss_config['dct_l1_loss_weight']
        w_dct_l2 = loss_config['dct_l2_loss_weight']

        # Compute all losses (always compute for logging, but only add to total loss if enabled)
        # Check and fix shape mismatch between target and model_output
        # This can happen when using patch embedder, where model_output length might be T*A instead of T
        if model_output.shape != target.shape:
            B_target, T_target, A_target = target.shape
            B_model, Len_model, A_model = model_output.shape
            
            # Check if model_output needs to be reshaped
            if Len_model == T_target * A_target and A_model == A_target:
                # Reshape from (B, T*A, A) to (B, T, A)
                model_output = model_output.reshape(B_model, T_target, A_target)
            elif Len_model == T_target and A_model == A_target:
                # Already correct shape, but dimensions might be swapped
                pass
            else:
                # Try to infer the correct shape
                # If model_output has more tokens, try to reshape assuming it's T*A
                if Len_model > T_target and Len_model % T_target == 0:
                    # Assume model_output is (B, T*A, A) and reshape to (B, T, A)
                    model_output = model_output.reshape(B_model, T_target, A_target)
                else:
                    # If shapes still don't match, raise an error with informative message
                    raise RuntimeError(
                        f"Shape mismatch between target and model_output: "
                        f"target.shape={target.shape}, model_output.shape={model_output.shape}. "
                        f"Expected model_output to have shape ({B_target}, {T_target}, {A_target}) "
                        f"or ({B_target}, {T_target * A_target}, {A_target})"
                    )
        
        diff = target - model_output
        # for debugging and visualization
        if return_image:
            # Return PIL Image for wandb logging
            action_image = visualize_action_chunks(
                target[:5], "tmp", file_name=f"action_chunks.png", 
                recon_chunks=model_output[:5].detach(), return_image=True
            )
            terms["action_visualization"] = action_image
        
        # L1 loss
        if "loss_mask" in model_kwargs:
            loss_mask = model_kwargs["loss_mask"].unsqueeze(1).repeat(1, target.shape[1], 1, 1)
            terms["loss_l1"] = sum_flat(torch.abs(diff) * loss_mask.float()) / sum_flat(loss_mask)
            terms["loss_l2"] = sum_flat((diff ** 2) * loss_mask.float()) / sum_flat(loss_mask)
        else:
            terms["loss_l1"] = mean_flat(torch.abs(diff))
            terms["loss_l2"] = mean_flat((diff ** 2))
        
        # DCT loss (always compute for logging)
        dct_diff = dct_differentiable(x_start) - dct_differentiable(v)

        terms["loss_dct_l1"] = mean_flat(torch.abs(dct_diff))
        terms["loss_dct_l2"] = mean_flat((dct_diff ** 2))
        if use_dct_l1:
            terms["loss_dct"] = terms["loss_dct_l1"]
        else:
            terms["loss_dct"] = terms["loss_dct_l2"]
        # Compute total loss based on enabled losses and weights
        total_loss = torch.zeros_like(terms["loss_l2"])
        if use_l1:
            total_loss = total_loss + w_l1 * terms["loss_l1"]
        if use_l2:
            total_loss = total_loss + w_l2 * terms["loss_l2"]
        if use_dct_l1:
            total_loss = total_loss + w_dct_l1 * terms["loss_dct_l1"]
        if use_dct_l2:
            total_loss = total_loss + w_dct_l2 * terms["loss_dct_l2"]
        
        terms["loss"] = total_loss
        
        # Handle recon_ratio for force_recon mode
        if recon_ratio != 1.0 and self.force_recon:
            recon_diff = v_gt - v
            recon_l1 = mean_flat(torch.abs(recon_diff))
            recon_l2 = mean_flat((recon_diff ** 2))
            
            recon_loss = torch.zeros_like(recon_l2)
            if use_l1:
                recon_loss = recon_loss + w_l1 * recon_l1
            if use_l2:
                recon_loss = recon_loss + w_l2 * recon_l2
            
            terms["loss"] = recon_ratio * terms["loss"] + (1 - recon_ratio) * recon_loss
        
        # Always record recon losses for logging (even if not used in total loss)
        if self.force_recon:
            recon_diff = v_gt - v
            terms["recon_l1"] = mean_flat(torch.abs(recon_diff))
            terms["recon_l2"] = mean_flat((recon_diff ** 2))
        else:
            # Set to zero if not in force_recon mode
            terms["recon_l1"] = torch.zeros_like(terms["loss_l1"])
            terms["recon_l2"] = torch.zeros_like(terms["loss_l2"])
        
        # Additional metrics (always use L2 for these)
        terms["mse"] = terms["loss_l2"]
        if (t <= 0.35).sum() > 0:
            terms["small"] = terms["mse"][t <= 0.35].mean()
        else:
            terms["small"] = torch.tensor(0.0, device=diff.device, dtype=diff.dtype)
        if ((0.35 < t) & (t <= 0.7)).sum() > 0:
            terms["mid"] = terms["mse"][(0.35 < t) & (t <= 0.7)].mean()
        else:
            terms["mid"] = torch.tensor(0.0, device=diff.device, dtype=diff.dtype)
        if (t > 0.7).sum() > 0:
            terms["large"] = terms["mse"][t > 0.7].mean()
        else:
            terms["large"] = torch.tensor(0.0, device=diff.device, dtype=diff.dtype)
        if drop_ids == None or drop_ids.sum() <= 0:
            terms["uncon"] = torch.tensor(0.0, device=diff.device, dtype=diff.dtype)
        else:
            terms["uncon"] = terms["mse"][drop_ids].mean()
        
        return terms

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        K = 512,
        start_t=None,
        model_kwargs=None,
        uncond_scale=1.0,
        uncond_y=None,
        uncond_c=None,
        x_0=None,
        encoder=None,
        diti=None,
        dit=None,
        ori_hidden_states=None,
        cond_vary=False,
        super_mask=None,
        device=None,
        t2k = 1.,
        **kwargs,
    ):
        batch_size = shape[0]
        if device is None:
            device = next(model.parameters()).device
 
        if noise is None:
            img = torch.randn(*shape, device=device)
        else:
            img = noise

        encoder_hidden_states = model_kwargs['encoder_hidden_states']

        # doc_to_save = {
        #     "t": [],
        # #    "mask": [],
        #     "Tim": [],
        #     "Fim": [],
        # }
        for i, step in enumerate(self.scheduled_t):
            t = torch.tensor([step] * batch_size, device=device)  # step：1~0
            with torch.no_grad():
                if cond_vary:
                    if diti.stages != None:
                        t_mapped = torch.tensor([self.timestep_map[i]]*batch_size, device=device).long()
                        t_tmp = t_mapped
                    else:
                        t_mapped = torch.tensor([(self.timestep_map[i])/1000.0]*batch_size, device=device)
                        t_tmp = (t2k * t_mapped).clamp(0, 1.0)
                    
                    k = diti.to_indices(t_tmp)
                    t = self.shift_t(t, self.shift)  # => 512 noise t
                    
                    if self.is_eval == False:
                        _, _, _, mask, _, _, _ = encoder(x=x_0, hidden_states=ori_hidden_states, d=k)
                    else:
                        _, _, _, mask, _, _, _ = encoder(x=x_0, d=k, kwargs=kwargs)

                    # doc_to_save["t"].append(t[0].item())
                    # # doc_to_save["mask"].append(mask[0].item())
                    # doc_to_save["Tim"].append(torch.sum(mask[0]).item())
                    # doc_to_save["Fim"].append(32 - torch.sum(mask[0]).item())
                    
                    
                    if self.cut_of_k is not None and self.cut_of_k < 1:
                        padding_size = K - encoder_hidden_states.shape[1]
                        padding_tensor = torch.zeros(encoder_hidden_states.shape[0], padding_size, encoder_hidden_states.shape[2]).cuda()
                        encoder_hidden_states = torch.cat((encoder_hidden_states, padding_tensor), dim=1)
                        padding_mask = torch.zeros(mask.shape[0], padding_size).cuda().bool()
                        mask = torch.cat((mask, padding_mask), dim=1)
                        super_mask_1 = torch.cat((super_mask, padding_mask), dim=1)
                    else:
                        super_mask_1 = super_mask
                        
                    if super_mask is not None:
                        mask = mask * super_mask_1

                    model_kwargs['encoder_hidden_states'] = encoder_hidden_states
                    model_kwargs['mask'] = mask

                    if encoder_hidden_states.sum() == 0 and dit is not None:
                        print("No condition is given...")
                        model_kwargs = {
                            'y': torch.tensor([1000] * len(x_0)).to(x_0.device)
                        }
                        model_to_use = dit
                    else:
                        model_to_use = model
                else:
                    model_to_use = model
 
                # Increment step counter for attention mask visualization hook
                from catok.utils.attn_mask_hook import get_attn_mask_hook
                hook = get_attn_mask_hook()
                if hook is not None:
                    hook.increment_step()
                
                img, pred_x0 = self.sample_one_step(
                    model_to_use,
                    img,
                    t,
                    index=i,
                    model_kwargs=model_kwargs,
                    cfg_scale=uncond_scale,
                    uncond_y=uncond_y,
                    uc=uncond_c,
                    **kwargs,
                )
        # # save doc
        # import pandas as pd
        # df = pd.DataFrame(doc_to_save)
        # df.to_csv("doc_to_save.csv", index=False)
        # import matplotlib.pyplot as plt
        # # set fig size
        # plt.figure(figsize=(30, 30))
        # # draw with t as x axis
        # # t should be exihibited explicitly on the x axis by h line
        # for t in df["t"]:
        #     plt.axvline(x=t, color="red", linestyle="--", linewidth=1)
        # for k in df["Tim"]:
        #     plt.axhline(y=k, color="blue", linestyle="--", linewidth=1)
        # plt.xlabel("t")
        # plt.ylabel("Number of tokens")
        # plt.title("Number of tokens in masks over time")
        # plt.plot(df["t"], df["Tim"], label="T in masks")
        # # x ticks should align with the h lines
        # plt.xticks(df["t"])
        # # y ticks should align with the v lines
        # plt.yticks(df["Fim"])
        # plt.plot(df["t"], df["Fim"], label="F in masks")
        # plt.legend()
        # plt.savefig("doc_to_save.png", dpi=300)
        # plt.close()
        # exit()
        return img
 
    def sample_one_step(
        self,
        model,
        x,
        t,
        index,
        model_kwargs=None,
        cfg_scale=1.0,
        uncond_y=None,
        uc=None,
        **kwargs,
    ):
        if model_kwargs is None:
            model_kwargs = {}
        b, *_, device = *x.shape, x.device
        # for action, shape is [N, T, D]
        # a_t = torch.full((b, 1, 1, 1), self.scheduled_t[index], device=device)
        # a_prev = torch.full((b, 1, 1, 1), self.scheduled_t_prev[index], device=device)
        a_t = torch.full((b, 1, 1), self.scheduled_t[index], device=device)
        a_prev = torch.full((b, 1, 1), self.scheduled_t_prev[index], device=device)
        
        if cfg_scale == 1.0:
            if self.is_eval == True:
                x = x.float()
            out, _ = model(x, t, **model_kwargs)
        else:
            context = model_kwargs['encoder_hidden_states']
            ori_mask = model_kwargs['mask']
            uncond_mask = torch.zeros(ori_mask.size(), dtype=torch.int, device=ori_mask.device)
            
            if self.is_eval == True:
                x = x.float()
            out_uncond = model.cfg_inference(x, t, None, None, mask = uncond_mask, shape=context.shape[1])
            out, _ = model(x, t, None, context, mask = ori_mask, shape=context.shape[1])
            out = out_uncond + cfg_scale * (out - out_uncond)
            
        img, pred_x0 = self.base_step(
            x, out, a_t=a_t, a_prev=a_prev, **kwargs
        )
        return img, pred_x0
    
    def base_step(self, x, v, a_t, a_prev):
        # Base sampler uses Euler numerical integrator.
        x_prev, pred_x0 = self.euler_step(x, v, a_t, a_prev)
        return x_prev, pred_x0
 
    def euler_step(self, x, v, a_t, a_prev):
        if self.parameterization == "velocity":
            x_prev = x - (a_t - a_prev) * v
            pred_x0 = x - a_t * v
        elif self.parameterization == "x0":
            x_prev = v + a_prev * (x - v) / a_t
            pred_x0 = v
            
        return x_prev, pred_x0
        
if __name__ == "__main__":
    # python CausalActionTokenizer/catok/models/catok_ddt/sd3/rectified_flow.py
    import torch.distributed as dist
    dist.init_process_group(backend='nccl', init_method='env://')
    
    rf = RectifiedFlow(num_timesteps=1000, start=1.0, cut_of_k=None, schedule="log_norm", val_schedule='shift', parameterization='x0', shift=1.0, m=0, s=1, force_recon=False, device='cuda',
                 is_eval=False).to('cuda')
    
    # example model
    class ExampleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(7, 7)

        def forward(self, x, t, **kwargs):
            drop_ids = torch.rand(x.shape[0])
            return self.linear(x)
        
        
    model = ExampleModel()
    x_start = torch.randn(20, 8, 7) # N T A
    t = torch.rand(x_start.shape[0])
    model_output = model(x_start)
    print(model_output.shape)  # should be same as x_start shape

    
    print(t.shape)
    