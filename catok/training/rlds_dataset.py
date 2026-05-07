"""
RLDS State-Action Dataset

Wraps load_rlds_state_action from catok.utils_catok.data_utils as a
PyTorch Dataset interface for loading state and action data (no images)
in RLDS format.

Each sample is a (state_window, action_window) pair, both shaped (horizon, dim).
Supports action_only / state_only modes.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Sampler, WeightedRandomSampler

from catok.utils_catok.data_utils import load_rlds_state_action
from catok.training.batch_transform import wrap_dataset_with_transform


# =============================================================================
# DDP-compatible weighted sampler
# =============================================================================

class DistributedWeightedSampler(Sampler[int]):
    """
    Weighted sampling with DDP partitioning: each rank samples only its own
    partition of indices according to the given weights.
    Call set_epoch(epoch) at the start of each epoch.
    """

    def __init__(
        self,
        weights,
        num_samples: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        drop_last: bool = False,
    ):
        self.weights = weights
        self.num_samples = num_samples
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self.drop_last = drop_last

        if drop_last and num_samples % world_size != 0:
            self.num_samples = num_samples - num_samples % world_size
        self.num_samples_per_rank = self.num_samples // world_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Shuffle then partition (consistent with DistributedSampler)
        shuffled = torch.randperm(len(self.weights), generator=g).tolist()
        rank_indices = shuffled[self.rank : self.num_samples : self.world_size]

        if len(rank_indices) == 0:
            return iter([])

        rank_weights = self.weights[rank_indices]
        rank_weights = rank_weights / rank_weights.sum()

        sampled_local = torch.multinomial(
            rank_weights,
            num_samples=self.num_samples_per_rank,
            replacement=True,
            generator=g,
        )
        return iter([rank_indices[i] for i in sampled_local.tolist()])

    def __len__(self) -> int:
        return self.num_samples_per_rank


class RLDSStateActionDataset:
    """
    PyTorch Dataset: loads state and action from RLDS.

    Each sample is a sliding window of length horizon.
    - horizon=1: single step (state, action)
    - horizon>1: multi-step window (state_window, action_window)

    Output modes (mutually exclusive):
      - action_only=True: returns only action_window, shape (horizon, action_dim)
      - state_only=True:  returns only state_window, shape (horizon, state_dim)
      - both False:       returns (state_window, action_window)
    """

    def __init__(
        self,
        root_dir: str,
        max_episodes: Optional[int] = None,
        horizon: int = 1,
        preload_cache_path: Optional[str] = None,
        num_parallel_reads: int = 4,
        action_only: bool = False,
        state_only: bool = False,
        debug: bool = False,
        dataset_name: str = None,
    ):
        """
        Args:
            root_dir: RLDS root directory
            max_episodes: maximum number of episodes to load; None means all
            horizon: number of steps per sample (sliding window length)
            preload_cache_path: pickle cache path for faster repeated loading
            num_parallel_reads: number of parallel TFRecord readers
            action_only: return only action
            state_only: return only state
            debug: debug mode, load only one episode
        """
        if action_only and state_only:
            raise ValueError("action_only and state_only are mutually exclusive")

        self.horizon = horizon
        self.action_only = action_only
        self.state_only = state_only

        self.episode_states, self.episode_actions = load_rlds_state_action(
            root_dir=root_dir,
            max_episodes=max_episodes,
            preload_cache_path=preload_cache_path,
            num_parallel_reads=num_parallel_reads,
            debug=debug,
            dataset_name=dataset_name,
        )

        # Build sample index: (episode_idx, start_idx)
        self.samples: List[Tuple[int, int]] = []
        for ep_idx, states in enumerate(self.episode_states):
            num_samples = len(states) - self.horizon + 1
            for start_idx in range(num_samples):
                self.samples.append((ep_idx, start_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        ep_idx, start_idx = self.samples[idx]
        end_idx = start_idx + self.horizon

        if self.action_only:
            return np.asarray(self.episode_actions[ep_idx][start_idx:end_idx], dtype=np.float32)
        if self.state_only:
            return np.asarray(self.episode_states[ep_idx][start_idx:end_idx], dtype=np.float32)

        state_window = np.asarray(self.episode_states[ep_idx][start_idx:end_idx], dtype=np.float32)
        action_window = np.asarray(self.episode_actions[ep_idx][start_idx:end_idx], dtype=np.float32)
        return state_window, action_window


# =============================================================================
# Unified DataLoader factory
# =============================================================================

def make_rlds_dataloader(
    dataset_specs: List[Tuple[str, float]],
    data_root: str = "data",
    horizon: int = 10,
    batch_size: int = 64,
    action_only: bool = True,
    num_parallel_reads: int = 8,
    max_episodes: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    debug: bool = False,
    # normalize + pad (done at dataset load time to avoid mixing stats and unify dims)
    target_action_dim: Optional[int] = None,
    target_horizon: Optional[int] = None,
    stats_path_map: Optional[Dict[str, str]] = None,
    normalizer_method: str = "qq",
    normalizer_method_map: Optional[Dict[str, str]] = None,
    normalizer_config: Optional[dict] = None,
    action_dim_map: Optional[Dict[str, int]] = None,
    horizon_map: Optional[Dict[str, int]] = None,
    action_start_map: Optional[Dict[str, int]] = None,
    # eval mode: sequential traversal, no weighted sampling
    eval_mode: bool = False,
    # DDP support
    distributed: bool = False,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    drop_last: bool = False,
) -> DataLoader:
    """
    Build a weighted DataLoader from a list of dataset specs.

    Args:
        dataset_specs: [(dataset_path, weight), ...]
            e.g. [("bridge", 1.0), ("Libero_RLDS/libero_spatial_no_noops", 0.5)]
            - dataset_path: path relative to data_root
            - weight: sampling weight; higher means sampled more often
        data_root: data root directory, default "data"
        horizon: sliding window length
        batch_size: batch size
        action_only: return only action
        num_parallel_reads: number of parallel TFRecord readers
        max_episodes: max episodes per dataset; None means all
        num_workers: DataLoader worker count; 0 avoids ConnectionResetError on CephFS
        pin_memory: whether to pin memory
        debug: debug mode, load only one episode per dataset
        target_action_dim: pad output action to this dim; None means no padding
        stats_path_map: dataset_path -> stats_path for per-dataset normalization
        normalizer_method: default normalizer type
        normalizer_method_map: dataset_path -> normalizer type (e.g. bridge uses "bridge")
        normalizer_config: extra normalizer config (e.g. clip, action_dim)
        distributed: DDP mode; uses DistributedWeightedSampler when True
        rank: rank for DDP; None auto-detects via accelerate.PartialState
        world_size: world_size for DDP; None auto-detects via PartialState
        drop_last: drop last incomplete batch to align batch counts across ranks (recommended for DDP)

    Note:
        In DDP mode, call loader.sampler.set_epoch(epoch) each epoch to vary shuffling.

    Returns:
        DataLoader
    """
    if distributed:
        try:
            from accelerate import PartialState
            state = PartialState()
            _rank = state.process_index
            _world_size = state.num_processes
        except ImportError:
            _rank = rank if rank is not None else 0
            _world_size = world_size if world_size is not None else 1

    datasets: List[Union[RLDSStateActionDataset, object]] = []
    dataset_weights: List[Tuple[float, int]] = []  # (weight, len) per dataset

    for dataset_path, weight in dataset_specs:
        root_dir = f"{data_root}/{dataset_path}".rstrip("/")
        cache_path = f"{root_dir}/rlds_state_action_cache.pkl"

        # Each dataset can have its own horizon (sliding window length)
        ds_horizon = horizon_map.get(dataset_path, horizon) if horizon_map else horizon

        ds = RLDSStateActionDataset(
            root_dir=root_dir,
            max_episodes=max_episodes,
            horizon=ds_horizon,
            preload_cache_path=cache_path,
            num_parallel_reads=num_parallel_reads,
            action_only=action_only,
            state_only=False,
            debug=debug,
        )

        # Wrap with transform if normalize / pad params are specified
        need_transform = (
            target_action_dim is not None
            or stats_path_map is not None
            or (horizon_map and ds_horizon != (target_horizon or horizon))
        )
        if need_transform:
            stats_path = stats_path_map.get(dataset_path) if stats_path_map else None
            method = (
                normalizer_method_map.get(dataset_path, normalizer_method)
                if normalizer_method_map
                else normalizer_method
            )
            # Build per-dataset normalizer_config with optional per-dataset action_dim
            ds_normalizer_config = dict(normalizer_config) if normalizer_config else {}
            if action_dim_map and dataset_path in action_dim_map:
                ds_normalizer_config['action_dim'] = action_dim_map[dataset_path]
            # Only pad horizon when this dataset's horizon differs from target
            ds_target_horizon = target_horizon if (ds_horizon != (target_horizon or horizon)) else None
            ds_action_start = action_start_map.get(dataset_path, 0) if action_start_map else 0
            ds = wrap_dataset_with_transform(
                dataset=ds,
                stats_path=stats_path,
                target_action_dim=target_action_dim if target_action_dim is not None else 8,
                target_horizon=ds_target_horizon,
                action_only=action_only,
                normalizer_method=method,
                normalizer_config=ds_normalizer_config,
                action_start=ds_action_start,
            )

        datasets.append(ds)
        dataset_weights.append((weight, len(ds)))

    concat_ds = ConcatDataset(datasets)
    num_total = len(concat_ds)

    if eval_mode:
        # Eval mode: sequential traversal, no weighted sampling
        sampler = None
        loader = DataLoader(
            concat_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=num_workers > 0,
        )
    else:
        # Vectorized weight construction to avoid per-sample extend
        weight_arrays = [np.full(n, w / n, dtype=np.float64) for w, n in dataset_weights]
        weights = torch.from_numpy(np.concatenate(weight_arrays))

        if distributed:
            sampler = DistributedWeightedSampler(
                weights=weights,
                num_samples=num_total,
                rank=_rank,
                world_size=_world_size,
                drop_last=drop_last,
            )
        else:
            sampler = WeightedRandomSampler(
                weights,
                num_samples=num_total,
                replacement=True,
            )

        loader = DataLoader(
            concat_ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=num_workers > 0,
        )

    return sampler, loader, concat_ds

# =============================================================================
# CLI test
# =============================================================================

if __name__ == "__main__":
    # Build DataLoader via unified interface (with normalize + pad)
    _, loader, _ = make_rlds_dataloader(
        dataset_specs=[
            ("bridge", 1.0),
            ("Libero_RLDS/libero_spatial_no_noops", 5.0),
            ("Libero_RLDS/libero_goal_no_noops", 5.0),
            ("Libero_RLDS/libero_object_no_noops", 5.0),
            ("Libero_RLDS/libero_10_no_noops", 5.0),
        ],
        data_root="data",
        horizon=10,
        batch_size=64,
        action_only=True,
        target_action_dim=8,
        stats_path_map={
            "bridge": "data/bridge/stats.json",
            "Libero_RLDS/libero_spatial_no_noops": "data/Libero_RLDS/libero_spatial_no_noops/stats.json",
            "Libero_RLDS/libero_goal_no_noops": "data/Libero_RLDS/libero_goal_no_noops/stats.json",
            "Libero_RLDS/libero_object_no_noops": "data/Libero_RLDS/libero_object_no_noops/stats.json",
            "Libero_RLDS/libero_10_no_noops": "data/Libero_RLDS/libero_10_no_noops/stats.json",
        },
        normalizer_method_map={"bridge": "bridge", "Libero_RLDS/libero_spatial_no_noops": "qq", "Libero_RLDS/libero_goal_no_noops": "qq", "Libero_RLDS/libero_object_no_noops": "qq", "Libero_RLDS/libero_10_no_noops": "qq"},
        normalizer_config={"clip": True, "action_dim": 7},
    )
    batch = next(iter(loader))
    print("action:", batch.shape, batch.dtype)  # (64, 10, 8)
