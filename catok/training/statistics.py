import numpy as np
import tqdm
import random
import json
import os
from pathlib import Path
from functools import partial
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed


class ActionNormalizerBase:
    def __init__(self, stats_path):
        with open(stats_path, "r") as f:
            self.stats = json.load(f)

    def normalize(self, action):
        raise NotImplementedError

    def denormalize(self, action):
        raise NotImplementedError

class ZScoreNormalizer(ActionNormalizerBase):
    def __init__(self, stats_path, eps=1e-8, action_dim=7, clip=True):
        super().__init__(stats_path)
        self.stats_path = stats_path
        self.action_dim = action_dim
        self.mean = np.array(self.stats['action']["mean"])
        self.std = np.array(self.stats['action']["std"])
        self.eps = eps

    def normalize(self, action):
        original_dim = action.shape[-1]
        normalized = (action[..., :self.action_dim] - self.mean[:self.action_dim]) / (self.std[:self.action_dim] + self.eps)
        if original_dim > self.action_dim:
            normalized = np.concatenate([normalized, action[..., self.action_dim:]], axis=-1)
        return normalized

    def denormalize(self, action):
        original_dim = action.shape[-1]
        denormalized = action[..., :self.action_dim] * (self.std[:self.action_dim] + self.eps) + self.mean[:self.action_dim]
        if original_dim > self.action_dim:
            denormalized = np.concatenate([denormalized, action[..., self.action_dim:]], axis=-1)
        return denormalized

class PercentileSymmetricNormalizer(ActionNormalizerBase):
    def __init__(self, stats_path, clip=True):
        super().__init__(stats_path)
        q01 = np.array(self.stats['action']["q01"])
        q99 = np.array(self.stats['action']["q99"])
        self.scale = np.maximum(np.abs(q01), np.abs(q99))
        self.clip = clip

    def normalize(self, action):
        out = action / self.scale
        if self.clip:
            out = np.clip(out, -1.0, 1.0)
        return out

    def denormalize(self, action):
        return action * self.scale

class AbsMaxNormalizer(ActionNormalizerBase):
    def __init__(self, stats_path, clip=True):
        super().__init__(stats_path)
        self.scale = np.array(self.stats['action']["abs_max"])
        self.clip = clip

    def normalize(self, action):
        out = action / self.scale
        if self.clip:
            out = np.clip(out, -1.0, 1.0)
        return out

    def denormalize(self, action):
        return action * self.scale

# Min–Max Normalizer（一般不推荐，但有时有用）
class MinMaxNormalizer(ActionNormalizerBase):
    def __init__(self, stats_path):
        super().__init__(stats_path)
        self.min = np.array(self.stats['action']["min"])
        self.max = np.array(self.stats['action']["max"])

    def normalize(self, action):
        return 2 * (action - self.min) / (self.max - self.min) - 1

    def denormalize(self, action):
        return (action + 1) * (self.max - self.min) / 2 + self.min

class QQNormalizer(ActionNormalizerBase):
    def __init__(self, stats_path, clip=True, transform='identity', stats_key='action', action_dim=None): # action dim for backward compatibility
        super().__init__(stats_path)
        self.q01 = np.array(self.stats[stats_key]["q01"])
        self.q99 = np.array(self.stats[stats_key]["q99"])
        self.clip = clip
        self.transform = transform
        self.action_dim = action_dim

    def normalize(self, action):
        action[...,:self.action_dim] = (action[...,:self.action_dim] - self.q01[None,:]) / (self.q99 - self.q01 + 1e-7)[None,:] * 2.0 - 1.0
        
        if self.clip:
            action = np.clip(action, -1.0, 1.0)

        return action
    
    def denormalize(self, action):
        action = (action + 1.0) / 2.0 * (self.q99 - self.q01 + 1e-7)[None,:] + self.q01[None,:]

        return action

class ScaleOnlyNormalizer(ActionNormalizerBase):
    def __init__(self, stats_path, method="q99", clip=True, scale_factor=1.0, no_gripper=False):
        super().__init__(stats_path)
        self.stats_path = stats_path
        self.scale_factor = scale_factor
        self.method = method
        if method == "q99":
            q01 = np.array(self.stats['action']["q01"])
            q99 = np.array(self.stats['action']["q99"])
            self.scale = np.maximum(np.abs(q01), np.abs(q99))
        elif method == "abs_max":
            self.scale = np.array(self.stats['action']["abs_max"])
        else:
            raise ValueError(f"Unknown method: {method}")

        if no_gripper:
            self.scale = self.scale[:-1]
        
        self.clip = clip

    def __str__(self):
        return f"ScaleOnlyNormalizer(scale_factor={self.scale_factor}, method={self.method}, clip={self.clip}, stats_path={self.stats_path})"

    def normalize(self, action):
        out = action / self.scale
        if self.clip:
            out = np.clip(out, -1.0, 1.0)
        return out * self.scale_factor

    def denormalize(self, action):
        return action / self.scale_factor * self.scale

def build_action_normalizer(method, normalizer_config):
    """Build an action normalizer based on method and config.
    
    Args:
        method (str): Normalizer type: "zscore", "percentile", "absmax", "minmax", "scale_only", or None
        normalizer_config (dict): Configuration dict with parameters for the normalizer
        
    Returns:
        ActionNormalizerBase or None: The normalizer instance, or None if method is None
    """
    if method is None or normalizer_config is None:
        return None
    if method == "zscore":
        return ZScoreNormalizer(**normalizer_config)
    elif method == "percentile":
        return PercentileSymmetricNormalizer(**normalizer_config)
    elif method == "absmax":
        return AbsMaxNormalizer(**normalizer_config)
    elif method == "minmax":
        return MinMaxNormalizer(**normalizer_config)
    elif method == "scale_only":
        return ScaleOnlyNormalizer(**normalizer_config)
    elif method == "qq":
        return QQNormalizer(**normalizer_config)
    else:
        class DummyNormalizer(ActionNormalizerBase):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
            def normalize(self, action):
                return action
            def denormalize(self, action):
                return action
        # raise warning
        print(f"Warning: Unknown normalizer method: {method}, using dummy normalizer")
        return DummyNormalizer(**normalizer_config)
#--------------Compute statistics--------------


def _process_parquet_file(parquet_path, extra_delta=True):
    """Process a single parquet file and return delta actions."""
    try:
        df = pd.read_parquet(parquet_path)
        
        # Extract actions
        joint_pos = np.stack(df['actions.joint.position'].values)
        gripper_pos = np.stack(df['actions.gripper.position'].values)
        
        state_joint_pos = np.stack(df['states.joint.position'].values)
        state_gripper_pos = np.stack(df['states.gripper.position'].values)
        
        if state_gripper_pos.ndim == 1:
            state_gripper_pos = state_gripper_pos[:, None]
        if gripper_pos.ndim == 1:
            gripper_pos = gripper_pos[:, None]

        states = np.concatenate([state_joint_pos, state_gripper_pos], axis=-1).astype(np.float32)
        actions = np.concatenate([joint_pos, gripper_pos], axis=-1).astype(np.float32)
        
        if extra_delta:
            actions = actions - states
        
        return states, actions
    except Exception as e:
        print(f"Error processing {parquet_path}: {e}")
        return None


def _compute_stats_for_array(arr, name=""):
    """Compute statistics for a numpy array."""
    stats = {
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "abs_max": np.abs(arr).max(axis=0).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
    }
    return stats


def compute_action_statistics_fast(data_root_dir="data/robot_dataset",
                                    max_episodes=5000, 
                                    num_workers=8, 
                                    extra_delta=True):
    """Compute state and action statistics using parallel processing on parquet files directly.
    
    Returns and saves statistics for both states and actions.
    """
    
    root_dir = Path(data_root_dir)
    
    # Find all parquet files
    print("Finding parquet files...")
    parquet_files = list(root_dir.rglob("data/chunk-*/episode_*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")
    
    # Sample if too many
    if len(parquet_files) > max_episodes:
        parquet_files = random.sample(parquet_files, max_episodes)
        print(f"Sampled {max_episodes} files")
    
    # Process in parallel
    all_states = []
    all_actions = []
    print(f"Processing with {num_workers} workers...")
    
    # Use partial to fix extra_delta parameter
    process_fn = partial(_process_parquet_file, extra_delta=extra_delta)
    
    with ProcessPoolExecutor(max_workers=None) as executor:
        futures = {executor.submit(process_fn, str(p)): p for p in parquet_files}
        
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result is not None:
                states, actions = result
                all_states.append(states)
                all_actions.append(actions)
    
    # Concatenate all
    print("Computing statistics...")
    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    print(f"Total samples: states={len(all_states)}, actions={len(all_actions)}")
    
    # Compute statistics for both
    action_type = "delta" if extra_delta else "absolute"
    
    action_stats = _compute_stats_for_array(all_actions)
    state_stats = _compute_stats_for_array(all_states)
    
    # Combined stats (for backward compatibility, use action stats as default)
    stats = {
        "action_type": action_type,
        # Separate action and state stats
        "action": action_stats,
        "state": state_stats,
    }
    
    print("=== Action Statistics ===")
    for k, v in action_stats.items():
        print(f"  {k}: {v}")
    
    print("=== State Statistics ===")
    for k, v in state_stats.items():
        print(f"  {k}: {v}")
    
    stats_path = os.path.join(data_root_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Saved to {stats_path}")

    return stats


if __name__ == "__main__":
    # Use fast version with parallel processing
    compute_action_statistics_fast(data_root_dir="data/openpi/libero", num_workers=8, max_episodes=40000, extra_delta=True)