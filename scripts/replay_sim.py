#!/usr/bin/env python3
"""
CATok LIBERO Simulator Replay

这个脚本在LIBERO模拟器中渲染：
1. GT轨迹视频：使用demo数据中保存的states
2. Reconstructed轨迹视频：将重建的delta joint累加后修改state中的joint部分

支持带gripper的重建轨迹可视化（与visualization_traj_clean.py一致的重建流程）：
- no_gripper=False: 模型训练时包含gripper维度，重建结果包含gripper
- no_gripper=True:  模型训练时不包含gripper维度，渲染时使用原始gripper

请注意：
- LIBERO环境的action space是EEF (7D: 6D pose + 1D gripper)
- CATok训练的是delta joint actions (8D: 7D delta joint + 1D gripper) 或 7D (no_gripper)
- 两者不能直接转换，所以通过修改mujoco state中的joint positions和gripper来模拟重建效果

Usage:
    # 基础用法：只生成GT视频和关节对比图
    python scripts/replay_sim.py \
        --ckpt_path /path/to/catok_model.pth \
        --benchmark libero_10 \
        --task_id 0 \
        --demo_id 0 \
        --output_dir outputs/sim_replay

    # 渲染重建轨迹视频（包含gripper）
    python scripts/replay_sim.py \
        --ckpt_path /path/to/catok_model.pth \
        --benchmark libero_10 \
        --task_id 0 \
        --demo_id 0 \
        --output_dir outputs/sim_replay \
        --render_recon
"""

import sys
sys.path.insert(0, ".")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import numpy as np
import argparse
from pathlib import Path
import imageio
import h5py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# LIBERO imports
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# CATok imports
from catok.infer.CatokPipeline import CatokDdtPipeline


# ===========================================================================
# Utility functions (matching visualization_traj_clean.py)
# ===========================================================================

def get_step_batch(x, horizon):
    """Split a sequence into non-overlapping chunks of size horizon."""
    batches = []
    clips = len(x) // horizon
    for i in range(clips):
        batch = x[i * horizon : (i + 1) * horizon]
        batches.append(batch)
    if isinstance(x, torch.Tensor):
        batches = torch.stack(batches)
    else:
        batches = np.stack(batches)
    return batches


def concat_step_batches(batches):
    """Concatenate step batches back into a flat sequence."""
    if isinstance(batches, torch.Tensor):
        return torch.cat(tuple(batches), dim=0)
    else:
        return np.concatenate(tuple(batches), axis=0)


# ===========================================================================
# Data loading and environment helpers
# ===========================================================================

def load_demo_data(demo_file: str, demo_id: int = 0):
    with h5py.File(demo_file, "r") as f:
        demo_key = f"data/demo_{demo_id}"

        if demo_key not in f:
            available = [k for k in f["data"].keys()]
            raise ValueError(f"Demo {demo_id} not found. Available: {available}")

        demo = f[demo_key]

        states = demo["states"][()]  # Full mujoco states
        actions = demo["actions"][()]  # EEF actions (7D)

        return {
            'states': states,
            'actions': actions,
            'demo_id': demo_id,
        }


def extract_joint_from_state(env, state):
    """
    从mujoco state中提取joint positions和gripper qpos

    LIBERO使用Panda机械臂，有7个关节 + 2个gripper关节
    """
    env.set_init_state(state)
    joint_pos = env.sim.data.qpos[:7].copy()
    gripper_pos = env.sim.data.qpos[7:9].copy()  # Gripper有2个对称关节
    return joint_pos, gripper_pos


def render_trajectory_video(env, states, output_path: str, fps: int = 30,
                            camera_name: str = "agentview_image"):
    """渲染轨迹视频：遍历每一步的state并渲染"""
    frames = []

    for i, state in enumerate(states):
        env.set_init_state(state)
        obs, _, _, _ = env.step([0.] * 7)
        frame = obs[camera_name][::-1]  # Flip vertically
        frames.append(frame)

        if (i + 1) % 50 == 0:
            print(f"    Rendered {i+1}/{len(states)} frames")

    print(f"  Saving video to {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"  Saved: {output_path}")

    return frames


# ===========================================================================
# Gripper mapping: 1D action value <-> 2D mujoco qpos
# ===========================================================================

def build_gripper_mapping(gripper_actions_1d, gripper_qpos_2d):
    """
    构建从1D gripper action值到2D gripper qpos的线性映射

    通过demo数据中action gripper值和对应的mujoco gripper qpos，
    建立线性映射关系。

    Args:
        gripper_actions_1d: (T, 1) 1D gripper action values
        gripper_qpos_2d: (T, 2) 2D gripper qpos values from mujoco

    Returns:
        mapping function: 1D gripper value -> 2D qpos array
    """
    ga = gripper_actions_1d.flatten()
    gq = gripper_qpos_2d[:, 0]  # 两个finger通常对称，取第一个

    ga_min, ga_max = ga.min(), ga.max()
    gq_min, gq_max = gq.min(), gq.max()

    print(f"  Gripper action range: [{ga_min:.4f}, {ga_max:.4f}]")
    print(f"  Gripper qpos range:   [{gq_min:.4f}, {gq_max:.4f}]")

    def mapping(gripper_val):
        """Map 1D gripper action value to 2D symmetric gripper qpos."""
        if ga_max - ga_min < 1e-8:
            qpos_val = gq_min
        else:
            ratio = np.clip((gripper_val - ga_min) / (ga_max - ga_min), 0.0, 1.0)
            qpos_val = gq_min + ratio * (gq_max - gq_min)
        return np.array([qpos_val, qpos_val])

    return mapping


# ===========================================================================
# State modification for simulator rendering
# ===========================================================================

def modify_state_joints(env, original_state, new_joint_pos, new_gripper_qpos=None):
    """
    修改mujoco state中的joint positions，可选修改gripper qpos

    Args:
        env: LIBERO environment
        original_state: 原始mujoco state
        new_joint_pos: (7,) 新的joint positions
        new_gripper_qpos: (2,) 新的gripper qpos，None则保持原始值
    """
    env.set_init_state(original_state)

    # 修改arm joint positions
    env.sim.data.qpos[:7] = new_joint_pos

    # 修改gripper joint positions（如果提供）
    if new_gripper_qpos is not None:
        env.sim.data.qpos[7:9] = new_gripper_qpos

    # Forward dynamics to update other state variables
    env.sim.forward()

    return env


# ===========================================================================
# CATok Reconstruction (matching visualization_traj_clean.py flow)
# ===========================================================================

def reconstruct_actions_with_catok(pipeline, joint_states, gripper_actions,
                                   horizon: int = 8, no_gripper: bool = False,
                                   action_dim: int = 7, device: str = 'cuda'):
    """
    使用CATok重建动作（与visualization_traj_clean.py流程一致）

    数据格式说明（对应Libero RLDSStateActionDataset with extra_delta=True）：
        - states:  (T, 7)  = joint positions
        - actions: (T, 8)  = [joint_positions(7D), gripper(1D)]
        - delta计算方式: actions[..., :7] -= states[:, :1, :] （gripper维度保持不变）
        - 重建时: absolute[..., :7] = pred_delta[..., :7] + states[:, :1, :]
                  absolute[..., 7]  = pred_delta[..., 7]  (gripper直接输出)

    Args:
        pipeline: CatokDdtPipeline
        joint_states: (T, 7) joint positions from mujoco
        gripper_actions: (T, 1) gripper values from EEF actions
        horizon: chunk size (must match model training config)
        no_gripper: if True, model was trained without gripper dimension
        action_dim: model's action_dim from config
        device: 'cuda' or 'cpu'

    Returns:
        recon_joints:      (valid_T, 7) reconstructed joint trajectory
        recon_gripper:     (valid_T, 1) reconstructed gripper (or original if no_gripper)
        gt_joints_valid:   (valid_T, 7) GT joint trajectory (truncated to valid length)
        gt_gripper_valid:  (valid_T, 1) GT gripper trajectory (truncated to valid length)
    """
    T = len(joint_states)

    if T < horizon:
        print(f"  Warning: trajectory too short ({T} < {horizon}), skipping reconstruction")
        return joint_states, gripper_actions, joint_states, gripper_actions

    # ---- Step 1: Build actions and states ----
    # actions: (T, 8) = [joint_positions, gripper]
    # states:  (T, 7) = joint_positions
    #
    # LIBERO HDF5 gripper format: [-1, 1] (-1=open, 1=close)
    # CATok training gripper format: [0, 1] (0=open, 1=close)
    # Conversion: gripper_catok = (1 + gripper_libero) / 2
    gripper_catok = (1.0 + gripper_actions) / 2.0
    print(f"  Gripper conversion: LIBERO [{gripper_actions.min():.2f}, {gripper_actions.max():.2f}] "
          f"-> CATok [{gripper_catok.min():.2f}, {gripper_catok.max():.2f}]")

    full_actions = np.concatenate([joint_states, gripper_catok], axis=-1)  # (T, 8)
    states_np = joint_states.copy()  # (T, 7)

    # ---- Step 2: Apply no_gripper preprocessing (matching fetch_func) ----
    if no_gripper:
        # Strip gripper dim, pad to action_dim
        processed_actions = full_actions[:, :-1].copy()  # (T, 7)
        pad_size = action_dim - processed_actions.shape[-1]
        if pad_size > 0:
            processed_actions = np.pad(processed_actions, ((0, 0), (0, pad_size)), mode='constant')
        actual_joint_dims = 7
        actual_action_dims = processed_actions.shape[-1]
        print(f"  no_gripper=True: stripped gripper, padded to action_dim={action_dim}")
    else:
        # Keep gripper, pad to action_dim if needed
        processed_actions = full_actions.copy()  # (T, 8)
        pad_size = action_dim - processed_actions.shape[-1]
        if pad_size > 0:
            processed_actions = np.pad(processed_actions, ((0, 0), (0, pad_size)), mode='constant')
        actual_joint_dims = 7
        actual_action_dims = processed_actions.shape[-1]
        print(f"  no_gripper=False: kept gripper, action_dim={action_dim}")

    # ---- Step 3: Chunk into non-overlapping batches ----
    actions_tensor = torch.from_numpy(processed_actions).float().to(device)
    states_tensor = torch.from_numpy(states_np).float().to(device)

    actions_batch = get_step_batch(actions_tensor, horizon)  # (B, H, action_dim)
    states_batch = get_step_batch(states_tensor, horizon)    # (B, H, 7)

    num_chunks = actions_batch.shape[0]
    valid_T = num_chunks * horizon

    print(f"  T={T}, horizon={horizon}, chunks={num_chunks}, valid_T={valid_T}")
    print(f"  actions_batch: {actions_batch.shape}, states_batch: {states_batch.shape}")

    # Save states for reconstruction
    states_batch_saved = states_batch.clone()

    # ---- Step 4: Compute delta (matching visualization_traj_clean.py) ----
    # delta[..., :7] = actions[..., :7] - states[:, :1, :]   (joints relative to window start)
    # delta[..., 7]  = actions[..., 7]                        (gripper stays as-is, if present)
    # delta[..., 8:] = actions[..., 8:]                       (padding stays as-is)
    delta_batch = actions_batch.clone()
    delta_batch[:, :, :actual_joint_dims] = (
        delta_batch[:, :, :actual_joint_dims] - states_batch[:, :1, :]
    )

    print(f"  delta_batch: {delta_batch.shape}")

    # ---- Step 5: Encode & Decode through CATok ----
    with torch.no_grad():
        tokens = pipeline.encoding(delta_batch, device=device)
        pred_delta = pipeline.decoding(tokens, device=device)

    if isinstance(pred_delta, torch.Tensor):
        pred_delta = pred_delta.cpu()
    delta_batch_cpu = delta_batch.cpu()
    states_batch_cpu = states_batch_saved.cpu()

    # ---- Step 6: Compute reconstruction errors in delta space ----
    if no_gripper:
        gt_for_error = delta_batch_cpu[:, :, :7].numpy()
        pred_for_error = pred_delta[:, :, :7].numpy() if isinstance(pred_delta, np.ndarray) else pred_delta[:, :, :7].numpy()
    else:
        # Error on joints + gripper (first 8 dims)
        n_dims = min(8, delta_batch_cpu.shape[-1])
        gt_for_error = delta_batch_cpu[:, :, :n_dims].numpy()
        pred_for_error = pred_delta[:, :, :n_dims].numpy() if isinstance(pred_delta, np.ndarray) else pred_delta[:, :, :n_dims].numpy()

    delta_mse = np.mean((gt_for_error - pred_for_error) ** 2)
    delta_mae = np.mean(np.abs(gt_for_error - pred_for_error))
    print(f"  Delta Reconstruction MSE: {delta_mse:.6f}, MAE: {delta_mae:.6f}")

    # ---- Step 7: Convert back to absolute coordinates ----
    if no_gripper:
        # Reconstruct joints only, use original gripper
        recon_joints_batch = pred_delta[:, :, :7] + states_batch_cpu[:, :1, :]
        recon_joints = concat_step_batches(recon_joints_batch).numpy()
        recon_gripper = gripper_actions[:valid_T].copy()
    else:
        # Reconstruct joints AND gripper
        # joints: pred_delta[:, :, :7] + window_start_state
        recon_joints_batch = pred_delta[:, :, :7] + states_batch_cpu[:, :1, :]
        # gripper: pred_delta[:, :, 7] is directly the gripper value (not delta)
        recon_gripper_batch = pred_delta[:, :, 7:8]

        recon_joints = concat_step_batches(recon_joints_batch).numpy()
        recon_gripper = concat_step_batches(recon_gripper_batch).numpy()

    # ---- Step 7b: Convert gripper back to LIBERO format [-1, 1] ----
    # 保存 CATok 格式的 gripper 用于可视化调试
    if not no_gripper:
        recon_gripper_catok = recon_gripper.copy()  # [0, 1] format
        gt_gripper_catok = gripper_catok[:valid_T].copy()  # [0, 1] format

        # CATok [0, 1] -> LIBERO [-1, 1]: gripper_libero = 2 * gripper_catok - 1
        print(f"  Gripper before conversion back: [{recon_gripper.min():.2f}, {recon_gripper.max():.2f}]")
        recon_gripper = 2.0 * recon_gripper - 1.0
        print(f"  Gripper after conversion back:  [{recon_gripper.min():.2f}, {recon_gripper.max():.2f}]")
    else:
        recon_gripper_catok = None
        gt_gripper_catok = None

    # GT values for comparison (original LIBERO format [-1, 1])
    gt_joints_valid = joint_states[:valid_T]
    gt_gripper_valid = gripper_actions[:valid_T]

    # ---- Step 8: Report absolute position errors ----
    abs_mse = np.mean((recon_joints - gt_joints_valid) ** 2)
    abs_mae = np.mean(np.abs(recon_joints - gt_joints_valid))
    print(f"  Absolute Joint Position MSE: {abs_mse:.6f}, MAE: {abs_mae:.6f}")

    if not no_gripper:
        gripper_mse = np.mean((recon_gripper - gt_gripper_valid) ** 2)
        gripper_mae = np.mean(np.abs(recon_gripper - gt_gripper_valid))
        print(f"  Gripper Reconstruction MSE: {gripper_mse:.6f}, MAE: {gripper_mae:.6f}")

    return (recon_joints, recon_gripper, gt_joints_valid, gt_gripper_valid,
            recon_gripper_catok, gt_gripper_catok)


# ===========================================================================
# Comparison figure (supports gripper)
# ===========================================================================

def create_comparison_figure(gt_joints, recon_joints, output_path: str,
                             gt_gripper=None, recon_gripper=None):
    """
    创建GT vs Reconstructed的对比图，支持joints (7D) 和可选的gripper (1D)
    """
    T = len(gt_joints)
    num_joints = gt_joints.shape[1]
    has_gripper = gt_gripper is not None and recon_gripper is not None
    total_plots = num_joints + (1 if has_gripper else 0)

    fig, axes = plt.subplots(total_plots, 1, figsize=(14, 2.5 * total_plots), sharex=True)
    if total_plots == 1:
        axes = [axes]

    steps = np.arange(T)

    # Plot joints
    for i in range(num_joints):
        axes[i].plot(steps, gt_joints[:, i], 'b-', linewidth=2, label='GT', alpha=0.8)
        axes[i].plot(steps, recon_joints[:, i], 'r--', linewidth=2, label='Reconstructed', alpha=0.8)
        axes[i].set_ylabel(f'Joint {i+1}', fontsize=11)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc='upper right')

    # Plot gripper
    if has_gripper:
        axes[num_joints].plot(steps, gt_gripper.flatten(), 'b-', linewidth=2, label='GT Gripper', alpha=0.8)
        axes[num_joints].plot(steps, recon_gripper.flatten(), 'r--', linewidth=2, label='Recon Gripper', alpha=0.8)
        axes[num_joints].set_ylabel('Gripper', fontsize=11)
        axes[num_joints].grid(True, alpha=0.3)
        axes[num_joints].legend(loc='upper right')

    axes[-1].set_xlabel('Time Step', fontsize=11)

    title = 'Joint Trajectory Comparison: GT vs Reconstructed'
    if has_gripper:
        title += ' (with Gripper)'
    fig.suptitle(title, fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison figure: {output_path}")


def create_full_action_figure(gt_joints, recon_joints, output_path: str,
                              gt_gripper=None, recon_gripper=None,
                              gt_gripper_catok=None, recon_gripper_catok=None):
    """
    创建完整的动作对比图：7个joint + gripper（同时显示LIBERO和CATok格式）
    所有维度画在同一张图上，便于整体对比。

    Args:
        gt_joints:          (T, 7) GT joint positions
        recon_joints:       (T, 7) Reconstructed joint positions
        gt_gripper:         (T,) or (T,1) GT gripper in LIBERO format [-1, 1]
        recon_gripper:      (T,) or (T,1) Reconstructed gripper in LIBERO format [-1, 1]
        gt_gripper_catok:   (T,) or (T,1) GT gripper in CATok format [0, 1] (optional)
        recon_gripper_catok:(T,) or (T,1) Reconstructed gripper in CATok format [0, 1] (optional)
    """
    T = len(gt_joints)
    has_gripper = gt_gripper is not None and recon_gripper is not None
    has_catok_gripper = gt_gripper_catok is not None and recon_gripper_catok is not None

    # Number of subplots: 7 joints + 1 gripper (LIBERO) + 1 gripper (CATok, optional)
    n_rows = 7
    if has_gripper:
        n_rows += 1
    if has_catok_gripper:
        n_rows += 1

    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 2.2 * n_rows), sharex=True)
    steps = np.arange(T)

    # ---- Plot 7 joints ----
    joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4',
                   'Joint 5', 'Joint 6', 'Joint 7']
    for i in range(7):
        ax = axes[i]
        ax.plot(steps, gt_joints[:, i], 'b-', linewidth=1.5, label='GT', alpha=0.8)
        ax.plot(steps, recon_joints[:, i], 'r--', linewidth=1.5, label='Recon', alpha=0.7)
        ax.set_ylabel(joint_names[i], fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

        # 计算每个joint的误差
        mae = np.mean(np.abs(gt_joints[:, i] - recon_joints[:, i]))
        ax.set_title(f'{joint_names[i]}  (MAE={mae:.5f})', fontsize=9, loc='left')

    # ---- Plot gripper (LIBERO format: -1=open, 1=close) ----
    row_idx = 7
    if has_gripper:
        ax = axes[row_idx]
        gt_g = np.asarray(gt_gripper).flatten()
        recon_g = np.asarray(recon_gripper).flatten()

        ax.plot(steps, gt_g, 'b-', linewidth=1.5, label='GT Gripper', alpha=0.8)
        ax.plot(steps, recon_g, 'r--', linewidth=1.5, label='Recon Gripper', alpha=0.7)
        ax.set_ylabel('Gripper\n(LIBERO)', fontsize=10)
        ax.set_ylim(-1.5, 1.5)
        ax.axhline(y=-1.0, color='gray', linestyle=':', alpha=0.4, label='-1 (open)')
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.4, label='1 (close)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=7, ncol=2)

        gripper_mae = np.mean(np.abs(gt_g - recon_g))
        ax.set_title(f'Gripper LIBERO [-1,1]  (MAE={gripper_mae:.5f})', fontsize=9, loc='left')
        row_idx += 1

    # ---- Plot gripper (CATok format: 0=open, 1=close) ----
    if has_catok_gripper:
        ax = axes[row_idx]
        gt_gc = np.asarray(gt_gripper_catok).flatten()
        recon_gc = np.asarray(recon_gripper_catok).flatten()

        ax.plot(steps, gt_gc, 'b-', linewidth=1.5, label='GT Gripper (CATok)', alpha=0.8)
        ax.plot(steps, recon_gc, 'r--', linewidth=1.5, label='Recon Gripper (CATok)', alpha=0.7)
        ax.set_ylabel('Gripper\n(CATok)', fontsize=10)
        ax.set_ylim(-0.3, 1.3)
        ax.axhline(y=0.0, color='gray', linestyle=':', alpha=0.4, label='0 (open)')
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.4, label='1 (close)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=7, ncol=2)

        gripper_catok_mae = np.mean(np.abs(gt_gc - recon_gc))
        ax.set_title(f'Gripper CATok [0,1]  (MAE={gripper_catok_mae:.5f})', fontsize=9, loc='left')
        row_idx += 1

    axes[-1].set_xlabel('Time Step', fontsize=11)

    fig.suptitle('Full Action Comparison: GT vs Reconstructed (Joints + Gripper)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved full action comparison figure: {output_path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='CATok LIBERO Simulator Replay')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to CATok checkpoint')
    parser.add_argument('--benchmark', type=str, default='libero_10',
                        choices=['libero_10', 'libero_spatial', 'libero_object', 'libero_goal'],
                        help='LIBERO benchmark suite')
    parser.add_argument('--task_id', type=int, default=0,
                        help='Task ID within the benchmark')
    parser.add_argument('--demo_id', type=int, default=0,
                        help='Demo ID to replay')
    parser.add_argument('--output_dir', type=str, default='outputs/sim_replay',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--camera', type=str, default='agentview_image',
                        choices=['agentview_image', 'robot0_eye_in_hand_image'])
    parser.add_argument('--render_recon', action='store_true',
                        help='Also render reconstructed trajectory video (with gripper if available)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # Load benchmark and task
    # ==========================================================================
    print(f"\n{'='*60}")
    print(f"Loading LIBERO benchmark: {args.benchmark}")
    print(f"{'='*60}")

    benchmark_dict = benchmark.get_benchmark_dict()
    bench = benchmark_dict[args.benchmark]()

    task = bench.get_task(args.task_id)
    print(f"Task: {task.name}")
    print(f"Language: {task.language}")

    # Get paths
    bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file
    )

    demo_file = os.path.join(
        get_libero_path("datasets"),
        bench.get_task_demonstration(args.task_id),
    )

    init_states_file = os.path.join(
        get_libero_path("init_states"),
        task.problem_folder,
        task.init_states_file
    )

    print(f"BDDL: {bddl_file}")
    print(f"Demo: {demo_file}")
    print(f"Init states: {init_states_file}")

    # Check if files exist
    if not os.path.exists(bddl_file):
        print(f"ERROR: BDDL file not found: {bddl_file}")
        print("Please check ~/.libero/config.yaml and ensure bddl_files path is correct.")
        return

    if not os.path.exists(demo_file):
        print(f"ERROR: Demo file not found: {demo_file}")
        print("\nTo fix this, edit ~/.libero/config.yaml and set 'datasets' to the correct path.")
        print("Example: datasets: /path/to/libero/datasets")
        print("\nOr download LIBERO datasets from: https://github.com/Lifelong-Robot-Learning/LIBERO")
        return

    # ==========================================================================
    # Load demo data
    # ==========================================================================
    print(f"\n{'='*60}")
    print(f"Loading demo {args.demo_id}...")
    print(f"{'='*60}")

    demo_data = load_demo_data(demo_file, args.demo_id)
    states = demo_data['states']
    actions = demo_data['actions']

    print(f"States shape: {states.shape}")
    print(f"Actions shape: {actions.shape}")

    # ==========================================================================
    # Create environment
    # ==========================================================================
    print(f"\n{'='*60}")
    print("Creating LIBERO environment...")
    print(f"{'='*60}")

    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": 256,
        "camera_widths": 256,
    }

    env = OffScreenRenderEnv(**env_args)
    env.reset()

    # ==========================================================================
    # Extract joint trajectories and gripper qpos from mujoco states
    # ==========================================================================
    print(f"\n{'='*60}")
    print("Extracting joint trajectories and gripper qpos...")
    print(f"{'='*60}")

    joint_positions = []
    gripper_qpos_list = []

    for i, state in enumerate(states):
        joint_pos, gripper_pos = extract_joint_from_state(env, state)
        joint_positions.append(joint_pos)
        gripper_qpos_list.append(gripper_pos)

        if (i + 1) % 50 == 0:
            print(f"  Extracted {i+1}/{len(states)} states")

    joint_positions = np.array(joint_positions)    # (T, 7)
    gripper_qpos_all = np.array(gripper_qpos_list)  # (T, 2) mujoco gripper qpos

    print(f"Joint positions shape: {joint_positions.shape}")
    print(f"Gripper qpos shape: {gripper_qpos_all.shape}")

    # ==========================================================================
    # Load CATok and read model config
    # ==========================================================================
    print(f"\n{'='*60}")
    print("Loading CATok model and reconstructing...")
    print(f"{'='*60}")

    pipeline = CatokDdtPipeline(args.ckpt_path, device=args.device)

    ckpt = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    horizon = cfg['dataset']['dataset_config'].get('horizon', 8)
    no_gripper = cfg['dataset']['dataset_config'].get('no_gripper', False)
    action_dim = cfg['tokenizer']['params']['action_dim']

    print(f"\nModel config:")
    print(f"  horizon:    {horizon}")
    print(f"  no_gripper: {no_gripper}")
    print(f"  action_dim: {action_dim}")

    # 使用原始EEF action中的gripper (1D)
    gripper_from_actions = actions[:, -1:]  # (T, 1)

    # ==========================================================================
    # Reconstruct with CATok
    # ==========================================================================
    print(f"\n{'='*60}")
    print("Reconstructing trajectory with CATok...")
    print(f"{'='*60}")

    (recon_joints, recon_gripper, gt_joints_valid, gt_gripper_valid,
     recon_gripper_catok, gt_gripper_catok) = \
        reconstruct_actions_with_catok(
            pipeline, joint_positions, gripper_from_actions,
            horizon=horizon, no_gripper=no_gripper,
            action_dim=action_dim, device=args.device
        )

    valid_T = len(recon_joints)

    # ==========================================================================
    # Render GT video
    # ==========================================================================
    print(f"\n{'='*60}")
    print("Rendering GT trajectory video...")
    print(f"{'='*60}")

    gt_video_path = str(output_dir / f"gt_{args.benchmark}_task{args.task_id}_demo{args.demo_id}.mp4")
    gt_frames = render_trajectory_video(env, states, gt_video_path, fps=args.fps, camera_name=args.camera)

    # ==========================================================================
    # Create comparison figures
    # ==========================================================================
    print(f"\n{'='*60}")
    print("Creating trajectory comparison figures...")
    print(f"{'='*60}")

    # Figure 1: Joint-level comparison
    fig_path = str(output_dir / f"joints_{args.benchmark}_task{args.task_id}_demo{args.demo_id}.png")
    if no_gripper:
        create_comparison_figure(gt_joints_valid, recon_joints, fig_path)
    else:
        create_comparison_figure(
            gt_joints_valid, recon_joints, fig_path,
            gt_gripper=gt_gripper_valid,
            recon_gripper=recon_gripper
        )

    # Figure 2: Full action comparison (joints + gripper with both LIBERO and CATok format)
    full_fig_path = str(output_dir / f"full_actions_{args.benchmark}_task{args.task_id}_demo{args.demo_id}.png")
    if no_gripper:
        create_full_action_figure(gt_joints_valid, recon_joints, full_fig_path)
    else:
        create_full_action_figure(
            gt_joints_valid, recon_joints, full_fig_path,
            gt_gripper=gt_gripper_valid,
            recon_gripper=recon_gripper,
            gt_gripper_catok=gt_gripper_catok,
            recon_gripper_catok=recon_gripper_catok
        )

    # ==========================================================================
    # Optionally render reconstructed video (with gripper)
    # ==========================================================================
    if args.render_recon and recon_joints is not None:
        print(f"\n{'='*60}")
        gripper_status = "with gripper" if not no_gripper else "without gripper (using original)"
        print(f"Rendering reconstructed trajectory video ({gripper_status})...")
        print(f"{'='*60}")

        recon_frames = []

        for i in range(valid_T):
            # 1) 用原始 state 恢复完整的 mujoco 状态
            env.set_init_state(states[i])

            # 2) 覆盖 arm joint positions 为重建值
            env.sim.data.qpos[:7] = recon_joints[i]
            env.sim.forward()

            # 3) 构造 action: EEF delta=0 (arm 不动), gripper=重建的开合值 [-1, 1]
            #    LIBERO action: [6D EEF delta, 1D gripper]
            #    gripper 是绝对开合指令，由 robosuite gripper controller 处理
            if not no_gripper and recon_gripper is not None:
                gripper_val = float(recon_gripper[i, 0] if recon_gripper.ndim > 1 else recon_gripper[i])
            else:
                # no_gripper: 使用原始 demo 中的 gripper action
                gripper_val = float(actions[i, -1])

            step_action = [0.] * 6 + [gripper_val]  # 7D: [6D EEF zeros, 1D gripper]
            obs, _, _, _ = env.step(step_action)
            frame = obs[args.camera][::-1]
            recon_frames.append(frame)

            if (i + 1) % 50 == 0:
                print(f"    Rendered {i+1}/{valid_T} frames")

        # GT frames truncated to valid_T for comparison
        gt_frames_valid = gt_frames[:valid_T]

        recon_video_path = str(output_dir / f"recon_{args.benchmark}_task{args.task_id}_demo{args.demo_id}.mp4")
        imageio.mimsave(recon_video_path, recon_frames, fps=args.fps)
        print(f"  Saved: {recon_video_path}")

        # 创建并排对比视频
        print("  Creating side-by-side comparison video...")
        comparison_frames = []
        for gt_f, recon_f in zip(gt_frames_valid, recon_frames):
            combined = np.concatenate([gt_f, recon_f], axis=1)
            comparison_frames.append(combined)

        comparison_video_path = str(output_dir / f"comparison_{args.benchmark}_task{args.task_id}_demo{args.demo_id}.mp4")
        imageio.mimsave(comparison_video_path, comparison_frames, fps=args.fps)
        print(f"  Saved: {comparison_video_path}")

    env.close()

    # ==========================================================================
    # Summary
    # ==========================================================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Task: {task.name}")
    print(f"Demo length: {len(states)} steps")
    print(f"Valid length: {valid_T} steps (truncated to horizon={horizon} chunks)")
    print(f"Model config: no_gripper={no_gripper}, horizon={horizon}, action_dim={action_dim}")
    print(f"GT video: {gt_video_path}")
    print(f"Joint comparison: {fig_path}")
    if args.render_recon:
        print(f"Reconstructed video: {recon_video_path}")
        print(f"Side-by-side comparison: {comparison_video_path}")
        if not no_gripper:
            print(f"  (gripper was reconstructed by CATok)")
        else:
            print(f"  (gripper used original values from demo)")
    print(f"\nOutput directory: {output_dir}")

if __name__ == "__main__":
    main()
