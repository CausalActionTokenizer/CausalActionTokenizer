import os
import json
import random
import re
import numpy as np
import pandas as pd
import tqdm
import pickle
from pathlib import Path
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from collections import OrderedDict
from functools import partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

try:
    from decord import VideoReader, cpu
except ImportError:
    print("Warning: decord not found. Video loading will fail.")
    VideoReader = None
    cpu = None

try:
    import tensorflow_datasets as tfds
    import tensorflow as tf
    # Disable GPU for TensorFlow to avoid conflicts with PyTorch
    tf.config.set_visible_devices([], 'GPU')
except ImportError:
    print("Warning: tensorflow_datasets not found. RLDS loading will fail.")
    tfds = None
    tf = None


def make_dataset(dataset_name, dataset_config):
    if dataset_name == 'libero_rlds':
        return RLDSStateActionDataset(**dataset_config)
    elif dataset_name.startswith('lerobot'):
        return StateActionDataset(**dataset_config)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


def _load_state_action_from_parquet(parquet_path, down_sampling_frequecy=1):
    """Load actions from a single parquet file."""
    try:
        df = pd.read_parquet(parquet_path)
        
        # Extract actions
        joint_pos = np.stack(df['actions.joint.position'].values)
        gripper_pos = np.stack(df['actions.gripper.position'].values)

        state_joint_pos = np.stack(df['states.joint.position'].values)
        state_gripper_pos = np.stack(df['states.gripper.position'].values)
        
        # Handle gripper shape
        if gripper_pos.ndim == 1:
            gripper_pos = gripper_pos[:, None]
        if state_gripper_pos.ndim == 1:
            state_gripper_pos = state_gripper_pos[:, None]

        states = np.concatenate([state_joint_pos, state_gripper_pos], axis=-1).astype(np.float32)
        actions = np.concatenate([joint_pos, gripper_pos], axis=-1).astype(np.float32)

        states = states[::down_sampling_frequecy]
        actions = actions[::down_sampling_frequecy]

        return states, actions

    except Exception as e:
        print(f"Error processing {parquet_path}: {e}")
        return None

def _load_state_action_libero(parquet_path, extra_delta=False, absolute_gripper=False):
    """Load actions from a single parquet file."""
    try:
        df = pd.read_parquet(parquet_path)
        print(df.columns)
        # Extract actions
        # state: "Robot EEF state (6D pose, 2D gripper)."
        states = np.stack(df['state'].values)
        joint_states = np.stack(df['joint_state'].values)

        eef_actions = np.stack(df['action'].values)
        
        # always true
        if extra_delta:
            actions = joint_states[1:] - joint_states[:-1]
            gripper_actions = eef_actions[1:, -1]
            actions = np.concatenate([actions, gripper_actions], axis=-1)
        else:
            actions = np.concatenate([joint_states, eef_actions[:, -1]], axis=-1)

        return joint_states[:-1], actions

        
        # action: "Robot EEF action (6D pose, 1D gripper)."
        # actions = np.stack(df['action'].values)
        
        if extra_delta:
            actions = actions - states
        
    except Exception as e:
        print(f"Error processing {parquet_path}: {e}")
        return None
    return states, actions


def _get_parquet_length(parquet_path, horizon, down_sampling_frequecy=1):
    """Get the number of valid samples from a parquet file without loading all data.
    
    Args:
        parquet_path: Path to parquet file
        horizon: Required horizon length
        down_sampling_frequecy: Downsampling frequency (e.g., 3 means take every 3rd frame)
    
    Returns:
        (path_str, num_samples): Tuple of path string and number of valid samples after downsampling
    """
    try:
        df = pd.read_parquet(parquet_path)
        original_length = len(df)
        # Apply downsampling: length after downsampling = ceil(original_length / down_sampling_frequecy)
        downsampled_length = (original_length + down_sampling_frequecy - 1) // down_sampling_frequecy
        num_samples = max(0, downsampled_length - horizon + 1)
        return str(parquet_path), num_samples
    except Exception as e:
        print(f"Error getting length from {parquet_path}: {e}")
        return str(parquet_path), 0


class StateActionDataset(Dataset):
    """Unified State/Action dataset with optional lazy loading.
    
    Each item returns state and/or action windows based on configuration.
    
    Output modes (mutually exclusive):
        - action_only=True: Returns action window of shape (horizon, action_dim)
        - state_only=True: Returns state window of shape (horizon, state_dim)
        - Both False: Returns tuple (state_window, action_window)
    
    Loading modes:
        - lazy_loading=False (default): Pre-loads all data into memory. Fast access, high memory usage.
        - lazy_loading=True: Loads data on-demand with LRU cache. Low memory usage, slightly slower.
    """
    
    def __init__(
        self,
        root_dir,
        horizon=8,
        extra_delta=False,
        delta_method="relative", # "delta" or "relative"
        max_episodes=None,
        num_workers=None,
        lazy_loading=False,
        cache_size=1000,
        action_only=False,
        state_only=False,
        absolute_gripper=True,
        down_sampling_frequecy=1,
        _load_func=_load_state_action_from_parquet,
        pattern="data/chunk-*/episode_*.parquet",
    ):
        """
        Args:
            root_dir (str): Path to the root directory (e.g. data/robot_dataset)
            horizon (int): Number of consecutive steps per sample
            extra_delta (bool): If True, compute delta actions (action - state)
            max_episodes (int): Maximum number of episodes to load (None = all)
            num_workers (int): Number of parallel workers for loading
            lazy_loading (bool): If True, load data on-demand instead of pre-loading all into memory
            cache_size (int): Number of episodes to cache in memory when lazy_loading=True
            action_only (bool): If True, only return actions (mutually exclusive with state_only)
            state_only (bool): If True, only return states (mutually exclusive with action_only)
            absolute_gripper (bool): If True, use absolute gripper position for delta computation
            _load_func (callable): Function to load state/action data from parquet files
            pattern (str): Glob pattern to match parquet files (e.g. "data/chunk-*/episode_*.parquet")
        """
        if action_only and state_only:
            raise ValueError("action_only and state_only are mutually exclusive")
        
        self.root_dir = Path(root_dir)
        self.horizon = horizon
        self.extra_delta = extra_delta
        self.delta_method = delta_method
        self.lazy_loading = lazy_loading
        self.cache_size = cache_size
        self.action_only = action_only
        self.state_only = state_only
        self.absolute_gripper = absolute_gripper

        # load function for different loading setting.
        self.down_sampling_frequecy = down_sampling_frequecy
        self._load_func = partial(_load_func, down_sampling_frequecy=down_sampling_frequecy)
        self.pattern = pattern
        
        # Find all parquet files (with caching)
        parquet_files = self._get_parquet_files_cached()
        
        # Sample if max_episodes is specified
        if max_episodes is not None and len(parquet_files) > max_episodes:
            parquet_files = random.sample(parquet_files, max_episodes)
            print(f"Sampled {max_episodes} episodes")
        
        if lazy_loading:
            self._init_lazy_loading(parquet_files, num_workers)
        else:
            self._init_preload(parquet_files, num_workers)
    
    def _get_parquet_files(self):
        """Get parquet files list."""
        print(f"Finding parquet files in {self.root_dir} with pattern: {self.pattern}")
        parquet_files = list(self.root_dir.rglob(self.pattern))
        print(f"Found {len(parquet_files)} parquet files")
        return parquet_files

    def _get_cache_filename(self):
        """Generate cache filename based on pattern."""
        # Sanitize pattern to create a valid filename
        # Replace special characters with underscores
        safe_pattern = re.sub(r'[^\w\-_./]', '_', self.pattern)
        safe_pattern = safe_pattern.replace('/', '_').replace('*', 'star')
        cache_name = f"parquet_files_cache_{safe_pattern}.json"
        return self.root_dir / cache_name

    def _get_parquet_files_cached(self):
        """Get parquet files list, using cache if available."""
        cache_file = self._get_cache_filename()
        
        if cache_file.exists():
            print(f"Loading parquet file list from cache: {cache_file}")
            with open(cache_file, 'r') as f:
                parquet_paths = json.load(f)
            # Convert to Path objects
            parquet_files = [Path(p) for p in parquet_paths]
            print(f"Loaded {len(parquet_files)} parquet files from cache")
        else:
            print(f"Finding parquet files in {self.root_dir} with pattern: {self.pattern}")
            parquet_files = list(self.root_dir.rglob(self.pattern))
            print(f"Found {len(parquet_files)} parquet files")
            
            # Save to cache
            parquet_paths = [str(p) for p in parquet_files]
            with open(cache_file, 'w') as f:
                json.dump(parquet_paths, f)
            print(f"Saved parquet file list to cache: {cache_file}")
        
        return parquet_files
    
    def _init_preload(self, parquet_files, num_workers):
        """Pre-load all data into memory."""
        # Limit num_workers to prevent resource exhaustion
        if num_workers is None:
            num_workers = min(8, os.cpu_count() or 1)
        num_workers = max(1, min(num_workers, len(parquet_files)))  # Don't exceed number of files
        
        print(f"Loading data with {num_workers} workers...")
        all_episode_states = []
        all_episode_actions = []
        
        process_fn = partial(self._load_func)
        
        # Use ThreadPoolExecutor for I/O-bound tasks (reading parquet files)
        # This avoids process overhead and file descriptor issues
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_fn, str(p)): p for p in parquet_files}
            
            for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                if result is None:
                    continue
                states, actions = result
                if states is not None and len(states) >= self.horizon:
                    all_episode_states.append(states)
                    all_episode_actions.append(actions)
        
        # Build index: (episode_idx, start_frame)
        self.episode_states = all_episode_states
        self.episode_actions = all_episode_actions
        self.samples = []  # List of (episode_idx, start_idx)
        
        for ep_idx, states in enumerate(self.episode_states):
            num_samples = len(states) - self.horizon + 1
            for start_idx in range(num_samples):
                self.samples.append((ep_idx, start_idx))
        
        print(f"Loaded {len(self.episode_states)} episodes, {len(self.samples)} total samples")
    
    def _init_lazy_loading(self, parquet_files, num_workers):
        """Initialize lazy loading: only build index, don't load data."""
        # Limit num_workers to prevent resource exhaustion
        if num_workers is None:
            num_workers = min(8, os.cpu_count() or 1)
        num_workers = max(1, min(num_workers, len(parquet_files)))  # Don't exceed number of files
        
        print(f"Building index with {num_workers} workers (lazy loading mode)...")
        
        # Get lengths of all parquet files in parallel
        # Pass down_sampling_frequecy to account for downsampling in length calculation
        process_fn = partial(_get_parquet_length, horizon=self.horizon, down_sampling_frequecy=self.down_sampling_frequecy)
        
        episode_info = []  # List of (path, num_samples)
        # Use ThreadPoolExecutor for I/O-bound tasks (reading parquet files)
        # This avoids process overhead and file descriptor issues
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_fn, str(p)) for p in parquet_files]
            
            for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                path, num_samples = future.result()
                if num_samples > 0:
                    episode_info.append((path, num_samples))
        
        # Build sample index: (episode_idx, start_idx)
        self.episode_paths = [info[0] for info in episode_info]  # List of paths
        self.samples = []
        
        for ep_idx, (path, num_samples) in enumerate(episode_info):
            for start_idx in range(num_samples):
                self.samples.append((ep_idx, start_idx))
        
        # LRU cache for loaded episodes
        self._cache = OrderedDict()
        self.episode_states = None  # Not used in lazy mode
        self.episode_actions = None  # Not used in lazy mode
        
        print(f"Indexed {len(self.episode_paths)} episodes, {len(self.samples)} total samples (lazy loading)")
    
    def _load_episode(self, ep_idx):
        """Load an episode with LRU caching."""
        if ep_idx in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(ep_idx)
            return self._cache[ep_idx]
        
        # Load from disk
        path = self.episode_paths[ep_idx]
        states, actions = self._load_func(path)
        
        # Add to cache
        self._cache[ep_idx] = (states, actions)
        
        # Evict oldest if cache is full
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        
        return states, actions
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        ep_idx, start_idx = self.samples[idx]
        
        if self.lazy_loading:
            states, actions = self._load_episode(ep_idx)
        else:
            states = self.episode_states[ep_idx]
            actions = self.episode_actions[ep_idx]
        
        # Safety check: ensure we have enough data after downsampling
        # This can happen if the actual parquet file length differs from what was calculated
        available_length = len(states)
        if start_idx + self.horizon > available_length:
            # Clamp start_idx to ensure we can extract a full window
            start_idx = max(0, available_length - self.horizon)
            # If still not enough, pad with last frame
            if start_idx < 0:
                # Not enough data even after clamping, pad with last frame
                state_window = np.tile(states[-1:], (self.horizon, 1)) if len(states) > 0 else np.zeros((self.horizon, states.shape[1] if len(states) > 0 else 8), dtype=np.float32)
                action_window = np.tile(actions[-1:], (self.horizon, 1)) if len(actions) > 0 else np.zeros((self.horizon, actions.shape[1] if len(actions) > 0 else 8), dtype=np.float32)
            else:
                # Extract windows: (horizon, dim)
                # Use .copy() to ensure contiguous arrays that can be resized during collation
                state_window = states[start_idx:start_idx + self.horizon].copy()
                action_window = actions[start_idx:start_idx + self.horizon].copy()
                # Pad if needed
                if len(state_window) < self.horizon:
                    padding_needed = self.horizon - len(state_window)
                    state_window = np.concatenate([state_window, np.tile(state_window[-1:], (padding_needed, 1))], axis=0)
                if len(action_window) < self.horizon:
                    padding_needed = self.horizon - len(action_window)
                    action_window = np.concatenate([action_window, np.tile(action_window[-1:], (padding_needed, 1))], axis=0)
        else:
            # Extract windows: (horizon, dim)
            # Use .copy() to ensure contiguous arrays that can be resized during collation
            state_window = states[start_idx:start_idx + self.horizon].copy()
            action_window = actions[start_idx:start_idx + self.horizon].copy()
        
        if self.extra_delta: # always absolute gripper
            if self.delta_method == "delta":
                action_window[:, :-1] = action_window[:, :-1] - state_window[:, :-1]
            elif self.delta_method == "relative":
                action_window[:, :-1] = action_window[:, :-1] - state_window[:1, :-1]
            else:
                raise ValueError(f"Invalid delta method: {self.delta_method}")
        if self.action_only:
            return action_window
        elif self.state_only:
            return state_window
        else:
            return state_window, action_window


# Backward compatibility aliases
ActionOnlyDataset = lambda **kwargs: StateActionDataset(action_only=True, **kwargs)
StateOnlyDataset = lambda **kwargs: StateActionDataset(state_only=True, **kwargs)


# actions[-1] [-1, 1] Gripper

class RLDSStateActionDataset(Dataset):
    """Dataset for loading LIBERO data in RLDS TFRecord format.
    
    This dataset loads data from RLDS TFRecord files (e.g., from data/Libero_RLDS).
    
    Data structure from RLDS:
        - action: (7,) float32 - Robot EEF action
        - observation.state: (8,) float32 - Robot EEF state (6D pose, 2D gripper)
        - observation.joint_state: (7,) float32 - Robot joint angles
    
    Output modes (mutually exclusive):
        - action_only=True: Returns action window of shape (horizon, action_dim)
        - state_only=True: Returns state window of shape (horizon, state_dim)
        - Both False: Returns tuple (state_window, action_window)
    """
    
    def __init__(
        self,
        root_dir,
        horizon=8,
        extra_delta=True,
        max_episodes=None,
        action_only=False,
        state_only=False,
        absolute_gripper=True,
        action_space="eef",
        use_joint_state=False,
        dataset_names=None,
        debug=False,
        preload=False
    ):
        """
        Args:
            root_dir (str): Path to the Libero_RLDS directory
            horizon (int): Number of consecutive steps per sample
            extra_delta (bool): If True, compute delta actions (relative to first state in window)
            max_episodes (int): Maximum number of episodes to load (None = all)
            action_only (bool): If True, only return actions
            state_only (bool): If True, only return states
            absolute_gripper (bool): If True, use absolute gripper position for delta
            use_joint_state (bool): If True, use joint_state (7D) and delta joint; else use EEF state
            use_delta_eef (bool): If True, use EEF state (7D) and delta EEF (next_eef - current_eef).
                                  When True, use_joint_state is effectively False.
            dataset_names (list): List of dataset names to load (e.g., ['libero_10_no_noops'])
                                  If None, loads all available datasets
            debug (bool): If True, only load one episode for debugging
        """
        if tfds is None:
            raise ImportError("tensorflow_datasets is required for RLDS loading. Install with: pip install tensorflow_datasets")
        
        if action_only and state_only:
            raise ValueError("action_only and state_only are mutually exclusive")
        
        self.root_dir = Path(root_dir)
        self.horizon = horizon
        self.extra_delta = extra_delta
        self.action_only = action_only
        self.state_only = state_only
        self.absolute_gripper = absolute_gripper
        self.action_space = action_space

        if self.action_space == "eef":
            self.use_joint_state = False
        elif self.action_space == "joint":
            self.use_joint_state = True
        else:
            raise ValueError(f"Invalid action space: {self.action_space}")
        self.debug = debug

        if preload:
            if use_joint_state and extra_delta:
                with open('./data/Libero_RLDS/libero_states.pkl', 'rb') as f:
                    self.episode_states = pickle.load(f)
                with open('./data/Libero_RLDS/libero_actions.pkl', 'rb') as f:
                    self.episode_actions = pickle.load(f)
                with open('./data/Libero_RLDS/libero_samples.pkl', 'rb') as f:
                    self.samples = pickle.load(f)
            elif not use_joint_state and not extra_delta:
                with open('./data/Libero_RLDS/libero_states_EEF_H20.pkl', 'rb') as f:
                    self.episode_states = pickle.load(f)
                with open('./data/Libero_RLDS/libero_actions_EEF_H20.pkl', 'rb') as f:
                    self.episode_actions = pickle.load(f)
                with open('./data/Libero_RLDS/libero_samples_EEF_H20.pkl', 'rb') as f:
                    self.samples = pickle.load(f)
        
        else:
            # Find all dataset directories
            if dataset_names is None:
                dataset_dirs = [d for d in self.root_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
            else:
                dataset_dirs = [self.root_dir / name for name in dataset_names]
            
            print(f"Loading LIBERO RLDS datasets from {self.root_dir}")
            print(f"Found datasets: {[d.name for d in dataset_dirs]}")
            
            # Load all episodes
            self.episode_states = []
            self.episode_actions = []
            
            for dataset_dir in dataset_dirs:
                self._load_dataset(dataset_dir)
                if self.debug and len(self.episode_states) > 0:
                    break
            
            # Sample if max_episodes is specified
            if max_episodes is not None and len(self.episode_states) > max_episodes:
                indices = random.sample(range(len(self.episode_states)), max_episodes)
                self.episode_states = [self.episode_states[i] for i in indices]
                self.episode_actions = [self.episode_actions[i] for i in indices]
                print(f"Sampled {max_episodes} episodes")
            
            # Build sample index
            self.samples = []
            for ep_idx, states in enumerate(self.episode_states):
                num_samples = len(states) - self.horizon + 1
                for start_idx in range(num_samples):
                    self.samples.append((ep_idx, start_idx))
            
        print(f"Loaded {len(self.episode_states)} episodes, {len(self.samples)} total samples")
    
    def _load_dataset(self, dataset_dir):
        """Load a single RLDS dataset."""
        # Find the version directory (e.g., 1.0.0)
        version_dirs = [d for d in dataset_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        if not version_dirs:
            print(f"Warning: No version directory found in {dataset_dir}")
            return
        
        version_dir = version_dirs[0]  # Use first version found
        
        # Load dataset using tensorflow_datasets
        builder = tfds.builder_from_directory(str(version_dir))
        ds = builder.as_dataset(split='train')
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        print(f"Loading episodes from {dataset_dir.name}...")
        
        for episode in tqdm.tqdm(ds, desc=f"Loading {dataset_dir.name}"):
            states_list = []
            actions_list = []
            
            for step in episode['steps']:
                if self.action_space == "eef":
                    # EEF state: 8D (6D pose + 2D gripper), use 7D (6D pose + 1D gripper)
                    state = step['observation']['state'].numpy()[:7]
                elif self.use_joint_state:
                    state = step['observation']['joint_state'].numpy()
                else:
                    state = step['observation']['state'].numpy()
                action = step['action'].numpy()
                
                states_list.append(state)
                actions_list.append(action)
            
            if len(states_list) < self.horizon:
                continue
            
            states = np.array(states_list, dtype=np.float32)
            actions = np.array(actions_list, dtype=np.float32)
            
            if self.action_space == "eef":
                pass
            elif self.use_joint_state:
                # Delta = next_joint_state - current_joint_state (7D)
                delta_actions = states[1:]
                # Gripper position from action (last dim of EEF action, 1D)
                gripper_actions = actions[:-1, -1:]
                # Final action: (delta_joint_state, gripper_position) = 8D
                actions = np.concatenate([delta_actions, gripper_actions], axis=-1)
                states = states[:-1]
            
            if len(states) >= self.horizon:
                self.episode_states.append(states)
                self.episode_actions.append(actions)
                if self.debug:
                    break
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        ep_idx, start_idx = self.samples[idx]
        
        states = self.episode_states[ep_idx]
        actions = self.episode_actions[ep_idx]
        
        # Extract windows: (horizon, dim)
        # Use .copy() to ensure contiguous arrays that can be resized during collation
        state_window = states[start_idx:start_idx + self.horizon].copy()
        action_window = actions[start_idx:start_idx + self.horizon].copy()

        if self.extra_delta and self.use_joint_state:
            # For use_joint_state: action 8D, state 7D -> subtract full state from action[:, :-1]
            ndim = min(action_window.shape[1] - 1, state_window.shape[1])
            action_window[:, :-1] = action_window[:, :-1] - state_window[:1, :ndim]

        if self.action_only:
            return action_window
        elif self.state_only:
            return state_window
        else:
            return state_window, action_window

class StateActionIterableDataset(IterableDataset):
    """Iterable State/Action dataset for streaming large-scale data.
    
    This dataset streams data from parquet files without loading everything into memory.
    Suitable for very large datasets that don't fit in memory.
    
    Output modes (mutually exclusive):
        - action_only=True: Yields action window of shape (horizon, action_dim)
        - state_only=True: Yields state window of shape (horizon, state_dim)
        - Both False: Yields tuple (state_window, action_window)
    
    Multi-worker support:
        - Automatically shards data across workers when num_workers > 0 in DataLoader
        - Each worker processes a different subset of parquet files
    """
    
    def __init__(
        self,
        root_dir,
        horizon=8,
        extra_delta=False,
        max_episodes=None,
        shuffle=True,
        action_only=False,
        state_only=False,
    ):
        """
        Args:
            root_dir (str): Path to the root directory (e.g. data/robot_dataset)
            horizon (int): Number of consecutive steps per sample
            extra_delta (bool): If True, compute delta actions (action - state)
            max_episodes (int): Maximum number of episodes to use (None = all)
            shuffle (bool): If True, shuffle episodes and samples within episodes
            action_only (bool): If True, only return actions (mutually exclusive with state_only)
            state_only (bool): If True, only return states (mutually exclusive with action_only)
        """
        if action_only and state_only:
            raise ValueError("action_only and state_only are mutually exclusive")
        
        self.root_dir = Path(root_dir)
        self.horizon = horizon
        self.extra_delta = extra_delta
        self.shuffle = shuffle
        self.action_only = action_only
        self.state_only = state_only
        
        # Get parquet files (with caching)
        self.parquet_files = self._get_parquet_files_cached()
        
        # Sample if max_episodes is specified
        if max_episodes is not None and len(self.parquet_files) > max_episodes:
            self.parquet_files = random.sample(self.parquet_files, max_episodes)
            print(f"Sampled {max_episodes} episodes")
        
        print(f"StateActionIterableDataset: {len(self.parquet_files)} episodes")
    
    def _get_parquet_files_cached(self):
        """Get parquet files list, using cache if available."""
        cache_file = self.root_dir / "parquet_files_cache.json"
        
        if cache_file.exists():
            print(f"Loading parquet file list from cache: {cache_file}")
            with open(cache_file, 'r') as f:
                parquet_paths = json.load(f)
            parquet_files = [Path(p) for p in parquet_paths]
            print(f"Loaded {len(parquet_files)} parquet files from cache")
        else:
            print(f"Finding parquet files in {self.root_dir}...")
            # Use glob pattern that's more efficient - find data dirs first, then parquet files
            parquet_files = []
            data_dirs = list(self.root_dir.glob("**/data"))
            print(f"Found {len(data_dirs)} data directories, scanning for parquet files...")
            for data_dir in tqdm.tqdm(data_dirs, desc="Scanning directories"):
                parquet_files.extend(data_dir.glob("chunk-*/episode_*.parquet"))
            print(f"Found {len(parquet_files)} parquet files")
            
            # Save to cache
            parquet_paths = [str(p) for p in parquet_files]
            with open(cache_file, 'w') as f:
                json.dump(parquet_paths, f)
            print(f"Saved parquet file list to cache: {cache_file}")
        
        return parquet_files
    
    def _get_worker_files(self):
        """Get the subset of files for the current worker."""
        worker_info = get_worker_info()
        
        if worker_info is None:
            # Single-process loading
            return self.parquet_files
        else:
            # Multi-process loading: shard files across workers
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            
            # Distribute files evenly across workers
            files_per_worker = len(self.parquet_files) // num_workers
            remainder = len(self.parquet_files) % num_workers
            
            start_idx = worker_id * files_per_worker + min(worker_id, remainder)
            end_idx = start_idx + files_per_worker + (1 if worker_id < remainder else 0)
            
            return self.parquet_files[start_idx:end_idx]
    
    def _iterate_episode(self, parquet_path):
        """Iterate over samples from a single episode."""
        result = _load_state_action_from_parquet(str(parquet_path))
        if result is None:
            return
        
        states, actions = result
        num_samples = len(states) - self.horizon + 1
        
        if num_samples <= 0:
            return
        
        # Generate sample indices
        indices = list(range(num_samples))
        if self.shuffle:
            random.shuffle(indices)
        
        for start_idx in indices:
            # Use .copy() to ensure contiguous arrays that can be resized during collation
            state_window = states[start_idx:start_idx + self.horizon].copy()
            action_window = actions[start_idx:start_idx + self.horizon].copy()
            
            if self.action_only:
                yield action_window
            elif self.state_only:
                yield state_window
            else:
                yield state_window, action_window
    
    def __iter__(self):
        """Iterate over all samples."""
        worker_files = self._get_worker_files()
        
        # Optionally shuffle episode order
        if self.shuffle:
            worker_files = worker_files.copy()
            random.shuffle(worker_files)
        
        for parquet_path in worker_files:
            yield from self._iterate_episode(parquet_path)


class RobotDataDataset(Dataset):
    def __init__(
        self, 
        root_dir, 
        train=True,
        cache_size=64,
        read_video=True,
        action_horizon=1,
        state_horizon=1,
        extra_delta=False,
        extra_delta_state=False,
    ):
        """
        Args:
            root_dir (str): Path to the root directory (e.g. data/robot_dataset)
            train (bool): Whether to load train or validation data (currently loads all found)
            cache_size (int): Number of open video readers/dataframes to cache
            read_video (bool): Whether to read video frames. If False, image fields will be None.
            action_horizon (int): Number of consecutive action steps to load. Output shape: (action_horizon, action_dim)
            state_horizon (int): Number of consecutive state steps to load. Output shape: (state_horizon, state_dim)
            extra_delta (bool): If True, compute delta actions (a[t+1] - a[t])
            extra_delta_state (bool): If True, compute delta states (s[t+1] - s[t])
        """
        self.root_dir = Path(root_dir)
        self.read_video = read_video
        self.action_horizon = action_horizon + 1 if extra_delta else action_horizon
        # state_horizon is not affected by extra_delta_state (state does not do delta)
        self.state_horizon = state_horizon
        self.extra_delta = extra_delta
        self.extra_delta_state = extra_delta_state  # kept for compatibility, but not used for state
        self.episodes = []
        self.cumulative_lengths = [0]
        
        # The effective horizon is the max of action_horizon and state_horizon
        self.effective_horizon = max(self.action_horizon, self.state_horizon)
        
        # Walk through the directory to find all episodes.jsonl files
        # Structure: sim/pick_and_place_tasks/task_name/meta/episodes.jsonl
        # We start search from self.root_dir
        
        print(f"Scanning {self.root_dir} for episodes...")
        
        # Finding all meta/episodes.jsonl files
        # Using glob with recursive search might be slow if there are too many files, 
        # but we are targeting specific pattern.
        meta_files = sorted(list(self.root_dir.rglob("meta/episodes.jsonl")))
        
        print(f"Found {len(meta_files)} meta files. Indexing episodes...")
        
        for meta_file in meta_files:
            task_dir = meta_file.parent.parent
            # Check if data directory exists
            data_dir = task_dir / "data"
            video_dir = task_dir / "videos"
            
            if not data_dir.exists() or not video_dir.exists():
                continue
                
            try:
                with open(meta_file, 'r') as f:
                    for line in f:
                        ep_meta = json.loads(line)
                        ep_idx = ep_meta["episode_index"]
                        length = ep_meta.get("length", 0)
                        
                        # Assuming at least one instruction exists
                        tasks = ep_meta.get("tasks", [])
                        instruction = tasks[0] if len(tasks) > 0 else ""
                        
                        # Calculate chunk path
                        chunk_id = ep_idx // 1000
                        chunk_str = f"chunk-{chunk_id:03d}"
                        
                        ep_filename_parquet = f"episode_{ep_idx:06d}.parquet"
                        ep_filename_mp4 = f"episode_{ep_idx:06d}.mp4"
                        
                        parquet_path = data_dir / chunk_str / ep_filename_parquet
                        video_head_path = video_dir / chunk_str / "images.rgb.head" / ep_filename_mp4
                        video_hand_path = video_dir / chunk_str / "images.rgb.hand" / ep_filename_mp4
                        
                        # Verification (optional, can be slow)
                        # if not parquet_path.exists(): continue
                        
                        self.episodes.append({
                            "parquet_path": str(parquet_path),
                            "video_head_path": str(video_head_path),
                            "video_hand_path": str(video_hand_path),
                            "instruction": instruction,
                            "length": length
                        })
                        # Adjust length to account for effective_horizon (discard last effective_horizon-1 frames)
                        valid_length = max(0, length - self.effective_horizon + 1)
                        self.cumulative_lengths.append(self.cumulative_lengths[-1] + valid_length)
                        
            except Exception as e:
                print(f"Error reading {meta_file}: {e}")
                
        print(f"Indexed {len(self.episodes)} episodes with {self.cumulative_lengths[-1]} total frames.")
        
        # LRU Cache for readers
        self.cache_size = cache_size
        self._parquet_cache = OrderedDict()
        self._video_cache = OrderedDict()

    def __len__(self):
        return self.cumulative_lengths[-1]

    def _get_parquet(self, path):
        if path in self._parquet_cache:
            self._parquet_cache.move_to_end(path)
            return self._parquet_cache[path]
        
        df = pd.read_parquet(path)
        
        if len(self._parquet_cache) >= self.cache_size:
            self._parquet_cache.popitem(last=False)
        self._parquet_cache[path] = df
        return df

    def _get_video_reader(self, path):
        if VideoReader is None:
            raise ImportError("decord is required for video loading. Install with: pip install decord")
        
        if path in self._video_cache:
            self._video_cache.move_to_end(path)
            return self._video_cache[path]
        
        if not os.path.exists(path):
            # Fallback or error
             raise FileNotFoundError(f"Video file not found: {path}")

        vr = VideoReader(path, ctx=cpu(0))
        
        if len(self._video_cache) >= self.cache_size:
            self._video_cache.popitem(last=False)
        self._video_cache[path] = vr
        return vr

    def __getitem__(self, idx):
        # Binary search to find the episode
        # bisect_right returns the insertion point to maintain order.
        # self.cumulative_lengths is [0, len1, len1+len2, ...]
        # if idx is between cum_len[i] and cum_len[i+1], it belongs to episode i.
        
        import bisect
        ep_idx = bisect.bisect_right(self.cumulative_lengths, idx) - 1
        
        episode = self.episodes[ep_idx]
        frame_idx = idx - self.cumulative_lengths[ep_idx]
        
        # Load Parquet Data
        df = self._get_parquet(episode["parquet_path"])
        
        # Extract row
        # Assuming frame_idx corresponds to row index in parquet
        # We need to verify if parquet has 'frame_index' column and if it aligns with row index
        # Usually it does.
        
        # Safety check for bounds - ensure we have room for effective_horizon frames
        # This handles cases where actual parquet length differs from metadata length
        max_valid_idx = len(df) - self.effective_horizon
        if max_valid_idx < 0 or len(df) < self.action_horizon:
            # Episode is too short, this shouldn't happen if cumulative_lengths is correct
            # But handle it gracefully by clamping frame_idx and ensuring we load what we can
            frame_idx = max(0, min(frame_idx, len(df) - 1))
            # Ensure we don't try to load more frames than available
            actual_action_horizon = min(self.action_horizon, len(df) - frame_idx)
            actual_state_horizon = min(self.state_horizon, len(df) - frame_idx)
            if actual_action_horizon < (2 if self.extra_delta else 1) or actual_state_horizon < 1:
                # Not enough frames, this is a data issue - raise an error or return None
                # For now, we'll try to load what we can and let _extra_delta handle it
                actual_action_horizon = max(actual_action_horizon, 1)
                actual_state_horizon = max(actual_state_horizon, 1)
        else:
            actual_action_horizon = self.action_horizon
            actual_state_horizon = self.state_horizon
            
        if frame_idx > max_valid_idx:
            frame_idx = max(0, max_valid_idx)
        
        def extract_master_action(row):
            joint_pos = row['master_actions.joint.position']
            gripper_pos = row['master_actions.gripper.position']
            
            if isinstance(joint_pos, np.ndarray):
                action_joint = joint_pos
            else:
                action_joint = np.array(joint_pos)
                
            if isinstance(gripper_pos, np.ndarray):
                action_gripper = gripper_pos
            else:
                action_gripper = np.array([gripper_pos]) if np.isscalar(gripper_pos) else np.array(gripper_pos)

            return np.concatenate([action_joint, action_gripper], axis=-1).astype(np.float32)

        # Helper function to extract action/state from a row
        def extract_action(row):
            joint_pos = row['actions.joint.position']
            gripper_pos = row['actions.gripper.position']
            
            if isinstance(joint_pos, np.ndarray):
                action_joint = joint_pos
            else:
                action_joint = np.array(joint_pos)
                
            if isinstance(gripper_pos, np.ndarray):
                action_gripper = gripper_pos
            else:
                action_gripper = np.array([gripper_pos]) if np.isscalar(gripper_pos) else np.array(gripper_pos)

            return np.concatenate([action_joint, action_gripper], axis=-1).astype(np.float32)
        
        def extract_state(row):
            # State: try slave_actions first, fallback to observations or master_actions
            if 'states.joint.position' in row.index:
                joint_pos = row['states.joint.position']
                gripper_pos = row['states.gripper.position']
            elif 'observations.joint.position' in row.index:
                joint_pos = row['observations.joint.position']
                gripper_pos = row['observations.gripper.position']
            else:
                # Fallback: use master_actions as state
                joint_pos = row['master_actions.joint.position']
                gripper_pos = row['master_actions.gripper.position']
            
            if isinstance(joint_pos, np.ndarray):
                state_joint = joint_pos
            else:
                state_joint = np.array(joint_pos)
                
            if isinstance(gripper_pos, np.ndarray):
                state_gripper = gripper_pos
            else:
                state_gripper = np.array([gripper_pos]) if np.isscalar(gripper_pos) else np.array(gripper_pos)

            return np.concatenate([state_joint, state_gripper], axis=-1).astype(np.float32)
        
        # Extract consecutive states for state_horizon steps (no delta for state)
        states = []
        for h in range(actual_state_horizon):
            cur_idx = frame_idx + h
            if cur_idx >= len(df):
                break
            row = df.iloc[cur_idx]
            s = extract_state(row)
            states.append(s)
        
        # Ensure we have at least one state, pad if necessary
        if len(states) == 0:
            # Fallback: use first available row
            if len(df) > 0:
                states.append(extract_state(df.iloc[min(frame_idx, len(df)-1)]))
            else:
                raise ValueError(f"Episode {ep_idx} has no data in parquet file")
        
        # Pad to expected state_horizon if needed
        while len(states) < self.state_horizon:
            states.append(states[-1])  # Repeat last state
        
        # Stack to (state_horizon, state_dim)
        state = np.stack(states[:self.state_horizon], axis=0)
        
        # Extract consecutive actions for action_horizon steps
        # Note: bounds are guaranteed by adjusted cumulative_lengths
        actions = []
        master_actions = []
        for h in range(actual_action_horizon):
            cur_idx = frame_idx + h
            if cur_idx >= len(df):
                break
            row = df.iloc[cur_idx]
            action = extract_action(row)
            master_action = extract_master_action(row)
            actions.append(action)
            master_actions.append(master_action)
        
        # Ensure we have at least the minimum required frames
        if len(actions) == 0:
            # Fallback: use first available row
            if len(df) > 0:
                fallback_row = df.iloc[min(frame_idx, len(df)-1)]
                actions.append(extract_action(fallback_row))
                master_actions.append(extract_master_action(fallback_row))
            else:
                raise ValueError(f"Episode {ep_idx} has no data in parquet file")
        
        # Pad to expected action_horizon if needed (before extra_delta)
        while len(actions) < self.action_horizon:
            actions.append(actions[-1])  # Repeat last action
            master_actions.append(master_actions[-1])
        
        # Stack to (action_horizon, action_dim)
        action = np.stack(actions[:self.action_horizon], axis=0)
        master_action = np.stack(master_actions[:self.action_horizon], axis=0)
        
        # Load Images (if read_video is True)
        img_head = None
        img_hand = None
        
        if self.read_video:
            vr_head = self._get_video_reader(episode["video_head_path"])
            vr_hand = self._get_video_reader(episode["video_hand_path"])
            
            # Ensure frame_idx is within video bounds
            # Note: VideoReader length might differ slightly from parquet length due to encoding/decoding issues
            # but usually they should match.
            
            vid_len_head = len(vr_head)
            vid_len_hand = len(vr_hand)
            
            read_idx_head = min(frame_idx, vid_len_head - 1)
            read_idx_hand = min(frame_idx, vid_len_hand - 1)
            
            img_head = vr_head[read_idx_head].asnumpy() # RGB, HWC
            img_hand = vr_hand[read_idx_hand].asnumpy() # RGB, HWC
        
        # Return format compatible with RLDS dict
        # "observation": {"image_primary": ..., "image_wrist": ...}
        
        if self.extra_delta:
            # Safety check: ensure we have at least 2 frames for delta computation
            if action.shape[0] < 2:
                # Not enough frames for delta, return zeros with expected shape
                expected_shape = (self.action_horizon - 1, action.shape[1])
                action = np.zeros(expected_shape, dtype=action.dtype)
            else:
                action = self._extra_delta(action)

        sample = {
            "observation": {
                "image_primary": img_head,
                "image_wrist": img_hand,
            },
            "state": state,
            "action": action,
            "master_action": master_action,
            "language_instruction": episode["instruction"],
            "dataset_name": "robot_dataset"
        }
        
        return sample

    @staticmethod
    def _extra_delta(action):
        return action[1:] - action[:-1]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="action_only", choices=["action_only", "state_only", "state_action", "robot_data"])
    parser.add_argument("--root_dir", type=str, default="data/robot_dataset")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--action_horizon", type=int, default=8)
    parser.add_argument("--state_horizon", type=int, default=8)
    parser.add_argument("--extra_delta", action="store_true")
    parser.add_argument("--lazy_loading", action="store_true")
    args = parser.parse_args()
    
    if args.dataset == "action_only":
        # Test StateActionDataset with action_only=True
        print("=== Testing StateActionDataset (action_only=True) ===")
        from catok.training.statistics import build_action_normalizer
        print(f"Building normalizer with stats path: {os.path.join(args.root_dir, 'stats.json')}")
        normalizer = build_action_normalizer(
            method="scale_only",
            normalizer_config={
                "stats_path": os.path.join(args.root_dir, "stats.json"),
                "method": "q99",
                "clip": True,
                "scale_factor": 1.0,
            }
        )
        ds = StateActionDataset(
            root_dir=args.root_dir,
            horizon=8,
            extra_delta=True,
            max_episodes=100,  # Limit for testing
            num_workers=4,
            lazy_loading=args.lazy_loading,
            action_only=True,
            absolute_gripper=True
        )
        print(f"Dataset length: {len(ds)}")
        
        rnd_idx = random.randint(0, len(ds) - 1)
        sample = ds[rnd_idx]
        print(f"Sample shape: {sample.shape}")  # (horizon, action_dim)
        print(f"Sample dtype: {sample.dtype}")
        print(f"Sample:\n{sample}")

        sample_normalized = normalizer.normalize(sample)
        print(f"Sample normalized: {sample_normalized}")

        # Statistics
        print("mean abs:", np.mean(np.abs(sample)))
        print("p95:", np.percentile(np.abs(sample), 95))
        print("p99:", np.percentile(np.abs(sample), 99))

        a = sample_normalized
        print("mean abs:", np.mean(np.abs(a)))
        print("p95:", np.percentile(np.abs(a), 95))
        print("p99:", np.percentile(np.abs(a), 99))
    
    elif args.dataset == "state_only":
        # Test StateActionDataset with state_only=True
        print("=== Testing StateActionDataset (state_only=True) ===")
        ds = StateActionDataset(
            root_dir=args.root_dir,
            horizon=args.horizon,
            extra_delta=args.extra_delta,
            max_episodes=100,  # Limit for testing
            num_workers=4,
            lazy_loading=args.lazy_loading,
            state_only=True,
        )
        print(f"Dataset length: {len(ds)}")
        
        rnd_idx = random.randint(0, len(ds) - 1)
        sample = ds[rnd_idx]
        print(f"Sample shape: {sample.shape}")  # (horizon, state_dim)
        print(f"Sample dtype: {sample.dtype}")
        print(f"Sample:\n{sample}")

        # Statistics
        print("mean abs:", np.mean(np.abs(sample)))
        print("p95:", np.percentile(np.abs(sample), 95))
        print("p99:", np.percentile(np.abs(sample), 99))
    
    elif args.dataset == "state_action":
        # Test StateActionDataset returning both state and action
        print("=== Testing StateActionDataset (both state and action) ===")
        ds = StateActionDataset(
            root_dir=args.root_dir,
            horizon=args.horizon,
            extra_delta=True,
            max_episodes=100,  # Limit for testing
            num_workers=4,
            lazy_loading=args.lazy_loading,
            absolute_gripper=True
        )
        print(f"Dataset length: {len(ds)}")
        
        rnd_idx = random.randint(0, len(ds) - 1)
        state, action = ds[rnd_idx]
        print(f"State shape: {state.shape}")   # (horizon, state_dim)
        print(f"Action shape: {action.shape}") # (horizon, action_dim)
        print(f"State:\n{state}")
        print(f"Action:\n{action}")

        # Statistics
        print("State - mean abs:", np.mean(np.abs(state)))
        print("Action - mean abs:", np.mean(np.abs(action)))

        for idx in range(len(ds)):
            state, action = ds[idx]
            print(state[..., -1])
            print(action[..., -1])
    
    else:
        # Test RobotDataDataset (read_video=False to skip video loading)
        print("=== Testing RobotDataDataset ===")
        ds = RobotDataDataset(
            root_dir=args.root_dir, 
            cache_size=2, 
            read_video=False, 
            action_horizon=args.action_horizon,
            state_horizon=args.state_horizon,
            extra_delta=args.extra_delta
        )
        print(f"Dataset length: {len(ds)}")
        
        # Print parquet columns for debugging
        if len(ds.episodes) > 0:
            sample_parquet = ds._get_parquet(ds.episodes[0]["parquet_path"])
            print(f"Parquet columns: {list(sample_parquet.columns)}")
        
        def print_sample(idx):
            sample = ds[idx]
            print("Sample keys:", sample.keys())
            print("State shape:", sample["state"].shape)
            print("Action shape:", sample["action"].shape)
            print("Image Primary:", sample["observation"]["image_primary"])
            print("Instruction:", sample["language_instruction"])
            
            print(f"state {sample['state']}")
            print(f"action {sample['action']}")
            print(f"master_action {sample['master_action']}")

        if len(ds) > 0:
            print_sample(120)

        sample = ds[120]
        state = sample['state']
        action = sample['action']
        master_action = sample['master_action']
        print(f"state {state}")
        print(f"action {action}")
        print(f"master_action {master_action}")