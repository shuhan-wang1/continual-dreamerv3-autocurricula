"""
Train DreamerV3 on Craftax environments.
Craftax is a JAX-based reimplementation of Crafter with better performance.
"""
import collections
import os
import pathlib
import shutil
import sys
from functools import partial as bind

# CRITICAL: Must set BEFORE importing jax (or any lib that imports jax).
# JAX reads this at import time and will pre-allocate 90% of GPU memory otherwise.
# Use JAX's default BFC allocator with preallocation to avoid memory fragmentation.
# The 'platform' allocator causes CUDA_ERROR_ILLEGAL_ADDRESS after ~20k steps
# due to severe fragmentation from repeated cudaMalloc/cudaFree calls.
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.90'

# NOTE: Do NOT set XLA_FLAGS here — internal.setup() in DreamerV3 will
# overwrite it completely.  Instead, we patch internal.setup() to preserve
# our flags (see below).

# Maximize GPU/CPU utilization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging noise
# Do NOT set XLA_PYTHON_CLIENT_ALLOCATOR='platform' — it causes memory fragmentation

# Parse arguments early to determine which DreamerV3 version to use
root = pathlib.Path(__file__).parent
# Import input_args before setting dreamerv3 path (it doesn't depend on dreamerv3)
sys.path.insert(0, str(root))
from input_args import parse_craftax_args
_args = parse_craftax_args()

# Add dreamerv3 to path based on --use_original_dreamer flag
if _args.use_original_dreamer:
    print("Using ORIGINAL DreamerV3 from dreamerv3-main/")
    sys.path.insert(0, str(root / 'dreamerv3-main'))
    sys.path.insert(0, str(root / 'dreamerv3-main' / 'dreamerv3'))
else:
    print("Using CONTINUOUS ENHANCED DreamerV3 from dreamerv3/")
    sys.path.insert(0, str(root / 'dreamerv3'))
    sys.path.insert(0, str(root / 'dreamerv3' / 'dreamerv3'))

import ast
import elements
import embodied
import numpy as np
import ruamel.yaml as yaml
import wandb

import jax
import jax.numpy as jnp

# NOTE: XLA_FLAGS are now handled automatically by internal.setup() in DreamerV3.
# It detects single vs multi-GPU and applies appropriate flags.
# We only set flags here that need to be set BEFORE JAX initializes.

# Allow implicit host-to-device transfers (needed for passing Python scalars to JIT functions)
# This will be overridden by internal.setup() if transfer_guard=True, but we set it here
# to allow early JAX operations before Agent is created.
jax.config.update("jax_transfer_guard", "allow")

# Performance: enable persistent compilation cache to avoid re-compiling JIT on restart
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

# Disable debug checks for speed
jax.config.update("jax_debug_nans", False)
jax.config.update("jax_disable_jit", False)

from dreamerv3.agent import Agent

import json
from collections import deque


# ============== Craftax Achievement Constants ==============
# These must be defined before CraftaxMetrics class which uses them

# Craftax achievement names (22 achievements across 5 tiers)
# These match the Craftax Achievement enum
CRAFTAX_ACHIEVEMENT_NAMES = [
    'collect_wood', 'place_table', 'eat_cow', 'collect_sapling',
    'collect_drink', 'make_wood_pickaxe', 'make_wood_sword',
    'place_stone', 'collect_stone', 'place_furnace', 'collect_coal',
    'collect_iron', 'make_stone_pickaxe', 'make_stone_sword',
    'make_iron_pickaxe', 'make_iron_sword', 'collect_diamond',
    'make_diamond_pickaxe', 'make_diamond_sword',
    'defeat_zombie', 'defeat_skeleton', 'wake_up_boss',
]

# Achievement tiers for depth calculation
ACHIEVEMENT_TIERS = {
    'collect_wood': 0, 'place_table': 0, 'eat_cow': 0, 'collect_sapling': 0,
    'collect_drink': 0, 'make_wood_pickaxe': 0, 'make_wood_sword': 0,
    'place_stone': 1, 'collect_stone': 1, 'place_furnace': 1, 'collect_coal': 1,
    'collect_iron': 1, 'make_stone_pickaxe': 1, 'make_stone_sword': 1,
    'make_iron_pickaxe': 2, 'make_iron_sword': 2, 'collect_diamond': 2,
    'make_diamond_pickaxe': 3, 'make_diamond_sword': 3,
    'defeat_zombie': 4, 'defeat_skeleton': 4, 'wake_up_boss': 4,
}

NUM_CRAFTAX_ACHIEVEMENTS = len(CRAFTAX_ACHIEVEMENT_NAMES)
NUM_ACHIEVEMENT_TIERS = 5  # Tiers 0-4


class CraftaxMetrics:
    """Comprehensive metrics tracker for Craftax training.

    Tracks all metrics from the specification:
    1. Core training metrics (logged every N steps)
    2. Per-episode online metrics
    3. Replay buffer diagnostics
    4. Continual learning metrics
    5. Exploration diagnostics

    All metrics are tracked per-task and aggregated. Every episode-end writes
    a JSONL record; a summary JSON is saved at the end of training.
    """

    def __init__(
        self,
        logdir: str,
        num_tasks: int = 1,
        window_size: int = 100,
        success_threshold: float = 1.0,
        jsonl_name: str = 'online_metrics.jsonl',
        summary_name: str = 'metrics_summary.json',
        num_achievements: int = NUM_CRAFTAX_ACHIEVEMENTS,
        num_tiers: int = NUM_ACHIEVEMENT_TIERS,
    ) -> None:
        self.logdir = logdir
        os.makedirs(logdir, exist_ok=True)
        self.num_tasks = int(num_tasks)
        self.window_size = int(window_size)
        self.success_threshold = float(success_threshold)
        self.jsonl_path = os.path.join(logdir, jsonl_name)
        self.summary_path = os.path.join(logdir, summary_name)
        self.num_achievements = num_achievements
        self.num_tiers = num_tiers

        # Per-task sliding windows for episode metrics
        self._returns = {i: deque(maxlen=window_size) for i in range(num_tasks)}
        self._lengths = {i: deque(maxlen=window_size) for i in range(num_tasks)}
        self._depths = {i: deque(maxlen=window_size) for i in range(num_tasks)}

        # Per-achievement tracking (sliding window per task)
        # Each entry is a binary vector of achievements for that episode
        self._achievement_history = {i: deque(maxlen=window_size) for i in range(num_tasks)}

        # Peak achievement rates for forgetting calculation
        self._peak_achievement_rates = {i: np.zeros(num_achievements) for i in range(num_tasks)}

        # Lifetime accumulators
        self._total_episodes = {i: 0 for i in range(num_tasks)}
        self._total_successes = {i: 0 for i in range(num_tasks)}
        self._max_return = {i: 0.0 for i in range(num_tasks)}
        self._max_depth = {i: -1 for i in range(num_tasks)}
        self._sum_return = {i: 0.0 for i in range(num_tasks)}
        self._personal_best_depth = -1  # Global personal best

        # Training metrics (updated via log_training_metrics)
        self._latest_training_metrics = {}

        # Replay buffer diagnostics (updated via log_replay_diagnostics)
        self._latest_replay_diagnostics = {}

        # Exploration diagnostics (updated via log_exploration_diagnostics)
        self._latest_exploration_diagnostics = {}

        # TD-error tracking per episode
        self._episode_td_errors = {i: deque(maxlen=window_size) for i in range(num_tasks)}

    # ---- Per-achievement metrics ----
    def _get_per_achievement_rates(self, task_id: int) -> np.ndarray:
        """Get rolling success rate for each achievement."""
        history = self._achievement_history[task_id]
        if not history:
            return np.zeros(self.num_achievements)
        # Stack all achievement vectors and compute mean
        arr = np.array(list(history), dtype=np.float32)
        return arr.mean(axis=0)

    def _update_peak_rates(self, task_id: int):
        """Update peak achievement rates for forgetting calculation."""
        current_rates = self._get_per_achievement_rates(task_id)
        self._peak_achievement_rates[task_id] = np.maximum(
            self._peak_achievement_rates[task_id], current_rates)

    def get_per_achievement_forgetting(self, task_id: int) -> np.ndarray:
        """Compute F_a = max_{t'<t} p_a(t') - p_a(t) for each achievement."""
        current_rates = self._get_per_achievement_rates(task_id)
        peak_rates = self._peak_achievement_rates[task_id]
        return np.maximum(0.0, peak_rates - current_rates)

    def get_aggregate_forgetting(self, task_id: int) -> float:
        """Mean forgetting across all achievements."""
        forgetting = self.get_per_achievement_forgetting(task_id)
        return float(forgetting.mean())

    def get_frontier_rate(self, task_id: int) -> float:
        """Fraction of recent episodes reaching personal-best depth."""
        depths = list(self._depths[task_id])
        if not depths or self._personal_best_depth < 0:
            return 0.0
        frontier_count = sum(1 for d in depths if d >= self._personal_best_depth)
        return frontier_count / len(depths)

    def _get_score_distribution(self, task_id: int) -> list:
        """Get fraction of episodes at each achievement tier."""
        depths = list(self._depths[task_id])
        if not depths:
            return [0.0] * (self.num_tiers + 1)  # +1 for tier -1

        counts = [0] * (self.num_tiers + 1)
        for d in depths:
            idx = d + 1  # Shift so -1 maps to index 0
            if 0 <= idx < len(counts):
                counts[idx] += 1

        total = len(depths)
        return [c / total for c in counts]

    # ---- per-task windowed stats ----
    def _windowed_mean_return(self, task_id: int) -> float:
        buf = self._returns[task_id]
        return float(np.mean(buf)) if buf else 0.0

    def _windowed_success_rate(self, task_id: int) -> float:
        buf = self._returns[task_id]
        if not buf:
            return 0.0
        return float(np.mean([1.0 if r >= self.success_threshold else 0.0 for r in buf]))

    def _windowed_mean_depth(self, task_id: int) -> float:
        buf = self._depths[task_id]
        return float(np.mean(buf)) if buf else -1.0

    def _windowed_mean_td_error(self, task_id: int) -> float:
        buf = self._episode_td_errors[task_id]
        return float(np.mean(buf)) if buf else 0.0

    def _windowed_max_td_error(self, task_id: int) -> float:
        buf = self._episode_td_errors[task_id]
        return float(np.max(buf)) if buf else 0.0

    # ---- aggregated stats ----
    def mean_return(self) -> float:
        """Mean of per-task windowed mean returns (only tasks with data)."""
        vals = [self._windowed_mean_return(i)
                for i in range(self.num_tasks) if self._returns[i]]
        return float(np.mean(vals)) if vals else 0.0

    def mean_success_rate(self) -> float:
        vals = [self._windowed_success_rate(i)
                for i in range(self.num_tasks) if self._returns[i]]
        return float(np.mean(vals)) if vals else 0.0

    def max_achievement_depth(self) -> float:
        """Best single-episode depth across all tasks."""
        return float(max(self._max_depth.values()))

    def per_task_mean_return(self):
        return [self._windowed_mean_return(i) for i in range(self.num_tasks)]

    def per_task_success_rate(self):
        return [self._windowed_success_rate(i) for i in range(self.num_tasks)]

    def per_task_max_return(self):
        return [float(self._max_return[i]) for i in range(self.num_tasks)]

    def per_task_max_depth(self):
        return [int(self._max_depth[i]) for i in range(self.num_tasks)]

    def per_task_episodes(self):
        return [self._total_episodes[i] for i in range(self.num_tasks)]

    def per_task_achievement_rates(self):
        """Get per-achievement success rates for all tasks."""
        return [self._get_per_achievement_rates(i).tolist() for i in range(self.num_tasks)]

    # ---- Training metrics logging ----
    def log_training_metrics(
        self,
        step: int,
        loss_obs: float = 0.0,
        loss_rew: float = 0.0,
        loss_con: float = 0.0,
        loss_dyn: float = 0.0,
        loss_rep: float = 0.0,
        loss_policy: float = 0.0,
        loss_value: float = 0.0,
        td_error_mean: float = 0.0,
        td_error_max: float = 0.0,
        ensemble_disagreement: float = 0.0,
        intrinsic_reward: float = 0.0,
        extrinsic_reward: float = 0.0,
        **extra_metrics
    ):
        """Log core training metrics."""
        self._latest_training_metrics = {
            'step': int(step),
            'loss/obs': float(loss_obs),
            'loss/rew': float(loss_rew),
            'loss/con': float(loss_con),
            'loss/dyn': float(loss_dyn),
            'loss/rep': float(loss_rep),
            'loss/policy': float(loss_policy),
            'loss/value': float(loss_value),
            'td_error/mean': float(td_error_mean),
            'td_error/max': float(td_error_max),
            'p2e/ensemble_disagreement': float(ensemble_disagreement),
            'p2e/intrinsic_reward': float(intrinsic_reward),
            'p2e/extrinsic_reward': float(extrinsic_reward),
            **{k: float(v) for k, v in extra_metrics.items()},
        }

    # ---- Replay buffer diagnostics ----
    def log_replay_diagnostics(
        self,
        step: int,
        buffer_size: int = 0,
        depth_distribution: list = None,
        mean_td_error: float = 0.0,
        mean_episode_age: float = 0.0,
        **extra
    ):
        """Log replay buffer diagnostic metrics."""
        self._latest_replay_diagnostics = {
            'step': int(step),
            'replay/buffer_size': int(buffer_size),
            'replay/depth_distribution': depth_distribution or [],
            'replay/mean_td_error': float(mean_td_error),
            'replay/mean_episode_age': float(mean_episode_age),
            **{f'replay/{k}': v for k, v in extra.items()},
        }

    # ---- Exploration diagnostics ----
    def log_exploration_diagnostics(
        self,
        step: int,
        imagined_value: float = 0.0,
        actual_value: float = 0.0,
        dream_accuracy: float = 0.0,
        intrinsic_extrinsic_ratio: float = 0.0,
        **extra
    ):
        """Log exploration diagnostic metrics."""
        self._latest_exploration_diagnostics = {
            'step': int(step),
            'explore/imagined_value': float(imagined_value),
            'explore/actual_value': float(actual_value),
            'explore/dream_accuracy': float(dream_accuracy),
            'explore/intr_extr_ratio': float(intrinsic_extrinsic_ratio),
            **{f'explore/{k}': v for k, v in extra.items()},
        }

    # ---- update & IO ----
    def update(
        self,
        task_id: int,
        step: int,
        score: float,
        length: int = 0,
        achievements: np.ndarray = None,
        achievement_depth: int = -1,
        td_error_mean: float = 0.0,
    ) -> dict:
        """Update metrics with episode results.

        Args:
            task_id: Current task ID.
            step: Global training step.
            score: Episode return/score.
            length: Episode length.
            achievements: Binary vector of achievements (NUM_CRAFTAX_ACHIEVEMENTS,).
            achievement_depth: Max achievement tier reached (-1 to 4).
            td_error_mean: Mean TD-error for this episode.
        """
        task_id = int(task_id)
        score = float(score)
        self._returns[task_id].append(score)
        self._lengths[task_id].append(int(length))
        self._depths[task_id].append(int(achievement_depth))
        self._episode_td_errors[task_id].append(float(td_error_mean))
        self._total_episodes[task_id] += 1

        if score >= self.success_threshold:
            self._total_successes[task_id] += 1
        if score > self._max_return[task_id]:
            self._max_return[task_id] = score
        if achievement_depth > self._max_depth[task_id]:
            self._max_depth[task_id] = achievement_depth
        if achievement_depth > self._personal_best_depth:
            self._personal_best_depth = achievement_depth
        self._sum_return[task_id] += score

        # Track per-achievement metrics
        if achievements is not None:
            ach_vec = np.asarray(achievements, dtype=np.bool_)
            if len(ach_vec) < self.num_achievements:
                padded = np.zeros(self.num_achievements, dtype=np.bool_)
                padded[:len(ach_vec)] = ach_vec
                ach_vec = padded
            self._achievement_history[task_id].append(ach_vec[:self.num_achievements])
            self._update_peak_rates(task_id)

        # Build comprehensive record
        per_ach_rates = self._get_per_achievement_rates(task_id).tolist()
        per_ach_forgetting = self.get_per_achievement_forgetting(task_id).tolist()

        record = {
            'step': int(step),
            'task': task_id,
            'score': score,
            'length': int(length),

            # Per-episode achievement info
            'achievements': achievements.tolist() if achievements is not None else [],
            'achievement_depth': int(achievement_depth),

            # Windowed per-task metrics
            'return_mean': self._windowed_mean_return(task_id),
            'success_rate': self._windowed_success_rate(task_id),
            'depth_mean': self._windowed_mean_depth(task_id),
            'td_error_mean': self._windowed_mean_td_error(task_id),
            'td_error_max': self._windowed_max_td_error(task_id),

            # Per-achievement rolling success rates
            'per_achievement_rates': per_ach_rates,

            # Score distribution histogram
            'score_distribution': self._get_score_distribution(task_id),

            # Continual learning metrics
            'per_achievement_forgetting': per_ach_forgetting,
            'aggregate_forgetting': self.get_aggregate_forgetting(task_id),
            'frontier_rate': self.get_frontier_rate(task_id),
            'personal_best_depth': self._personal_best_depth,

            # Aggregated across tasks
            'agg_return_mean': self.mean_return(),
            'agg_success_rate': self.mean_success_rate(),
            'agg_max_depth': self.max_achievement_depth(),

            # Per-task snapshots
            'per_task_return_mean': self.per_task_mean_return(),
            'per_task_success_rate': self.per_task_success_rate(),
            'per_task_max_return': self.per_task_max_return(),
            'per_task_max_depth': self.per_task_max_depth(),

            # Include latest training metrics if available
            **self._latest_training_metrics,

            # Include latest replay diagnostics if available
            **self._latest_replay_diagnostics,

            # Include latest exploration diagnostics if available
            **self._latest_exploration_diagnostics,
        }

        with open(self.jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        return record

    def save_summary(self) -> None:
        """Save final summary metrics."""
        payload = {
            'mean_return': self.mean_return(),
            'success_rate': self.mean_success_rate(),
            'max_achievement_depth': self.max_achievement_depth(),
            'personal_best_depth': self._personal_best_depth,

            # Per-task metrics
            'per_task_return_mean': self.per_task_mean_return(),
            'per_task_success_rate': self.per_task_success_rate(),
            'per_task_max_return': self.per_task_max_return(),
            'per_task_max_depth': self.per_task_max_depth(),
            'per_task_episodes': self.per_task_episodes(),
            'per_task_achievement_rates': self.per_task_achievement_rates(),
            'per_task_lifetime_mean': [
                float(self._sum_return[i] / max(1, self._total_episodes[i]))
                for i in range(self.num_tasks)
            ],

            # Continual learning summary
            'per_task_aggregate_forgetting': [
                self.get_aggregate_forgetting(i) for i in range(self.num_tasks)
            ],
            'per_task_frontier_rate': [
                self.get_frontier_rate(i) for i in range(self.num_tasks)
            ],

            # Config
            'num_tasks': self.num_tasks,
            'window_size': self.window_size,
            'success_threshold': self.success_threshold,
            'num_achievements': self.num_achievements,
            'achievement_names': CRAFTAX_ACHIEVEMENT_NAMES,
        }
        with open(self.summary_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

# Import Craftax
try:
    import craftax
    from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv
    from craftax.craftax.envs.craftax_pixels_env import CraftaxPixelsEnv
    from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv
    from craftax.craftax_classic.envs.craftax_pixels_env import CraftaxClassicPixelsEnv
    # Try to import achievement constants
    try:
        from craftax.craftax.constants import Achievement as CraftaxAchievement
        CRAFTAX_ACHIEVEMENTS_AVAILABLE = True
    except ImportError:
        CRAFTAX_ACHIEVEMENTS_AVAILABLE = False
    CRAFTAX_AVAILABLE = True
except ImportError:
    CRAFTAX_AVAILABLE = False
    CRAFTAX_ACHIEVEMENTS_AVAILABLE = False
    print("Warning: Craftax not installed. Install with: pip install craftax")


# ============== Craftax Environment Wrapper ==============
class CraftaxWrapper(embodied.Env):
    """Wrapper to convert Craftax JAX environments to embodied interface.

    Optimized to minimize GPU<->CPU transfers by keeping data on GPU as long
    as possible and using JIT-compiled processing functions.
    """

    def __init__(self, env_name='CraftaxSymbolic-v1', embedding_dim=256, use_embedding=True, seed=42, track_achievements=True):
        self._env_name = env_name
        self._embedding_dim = embedding_dim
        self._use_embedding = use_embedding
        self._seed = seed
        self._done = True
        self._track_achievements = track_achievements

        # Achievement tracking for current episode
        self._episode_achievements = np.zeros(NUM_CRAFTAX_ACHIEVEMENTS, dtype=np.bool_)
        self._cumulative_reward = 0.0

        # Use a base key and step counter for deterministic key generation
        # Pre-generate a batch of keys to avoid per-step host-to-device transfers
        self._base_key = jax.random.PRNGKey(seed)
        self._step_count = 0
        self._key_batch_size = 1024
        self._key_cache = jax.random.split(self._base_key, self._key_batch_size)
        self._key_idx = 0

        # Create the appropriate Craftax environment
        self._is_classic = 'Classic' in env_name
        if self._is_classic:
            if 'Symbolic' in env_name:
                self._env = CraftaxClassicSymbolicEnv()
            else:
                self._env = CraftaxClassicPixelsEnv()
        else:
            if 'Symbolic' in env_name:
                self._env = CraftaxSymbolicEnv()
            else:
                self._env = CraftaxPixelsEnv()

        # Get environment parameters
        self._env_params = self._env.default_params
        
        # Initialize state for space inference - use jit compiled reset
        self._reset_fn = jax.jit(self._env.reset)
        self._step_fn = jax.jit(self._env.step)
        
        # Get initial observation to determine shape
        init_key = self._key_cache[0]
        self._key_idx = 1
        obs, self._state = self._reset_fn(init_key, self._env_params)
        self._obs = obs
        
        # Determine observation shape from the observation (one-time cost)
        if hasattr(obs, 'shape'):
            self._obs_shape = tuple(int(x) for x in obs.shape)
        elif isinstance(obs, dict):
            if 'obs' in obs:
                self._obs_shape = tuple(int(x) for x in obs['obs'].shape)
            elif 'pixels' in obs:
                self._obs_shape = tuple(int(x) for x in obs['pixels'].shape)
            else:
                self._obs_shape = tuple(int(x) for x in jax.tree_util.tree_leaves(obs)[0].shape)
        else:
            obs_array = np.asarray(obs)
            self._obs_shape = obs_array.shape
        
        # Setup embedding projection on GPU and JIT the processing function
        if use_embedding:
            np.random.seed(42)
            flat_dim = int(np.prod(self._obs_shape))
            projection_np = np.random.randn(flat_dim, embedding_dim).astype(np.float32)
            projection_np /= np.sqrt(flat_dim)
            # Keep projection matrix on GPU
            self._projection = jax.device_put(jnp.array(projection_np))
            
            # JIT compile the full obs->embedding pipeline on GPU
            @jax.jit
            def process_obs_jit(obs):
                obs_flat = obs.reshape(-1).astype(jnp.float32)
                return jnp.dot(obs_flat, self._projection)
            self._process_obs_jit = process_obs_jit
        else:
            @jax.jit
            def process_obs_jit(obs):
                return obs.astype(jnp.float32)
            self._process_obs_jit = process_obs_jit
        
        # Number of actions
        self._num_actions = int(self._env.action_space(self._env_params).n)

    @property
    def obs_space(self):
        spaces = {}
        if self._use_embedding:
            spaces['embedding'] = elements.Space(np.float32, (self._embedding_dim,), -np.inf, np.inf)
        else:
            spaces['image'] = elements.Space(np.float32, self._obs_shape, 0.0, 1.0)
        spaces.update({
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool),
        })
        # Achievement tracking spaces (prefixed with log/ to exclude from agent training)
        # Use int32 instead of bool to avoid JAX iota dtype issues in replay buffer
        if self._track_achievements:
            spaces['log/achievements'] = elements.Space(np.int32, (NUM_CRAFTAX_ACHIEVEMENTS,), 0, 2)
            spaces['log/achievement_depth'] = elements.Space(np.int32)
        return spaces

    @property
    def act_space(self):
        return {
            'action': elements.Space(np.int32, (), 0, self._num_actions),
            'reset': elements.Space(bool),
        }

    def _get_rng_key(self):
        """Get next random key from pre-generated cache (no device transfer per step)."""
        if self._key_idx >= self._key_batch_size:
            # Regenerate key batch on GPU (amortized cost)
            self._step_count += self._key_batch_size
            self._key_cache = jax.random.split(
                jax.random.fold_in(self._base_key, self._step_count),
                self._key_batch_size
            )
            self._key_idx = 0
        key = self._key_cache[self._key_idx]
        self._key_idx += 1
        return key

    def _extract_obs(self, obs):
        """Extract raw observation array, staying on GPU."""
        if hasattr(obs, 'shape'):
            return obs
        elif isinstance(obs, dict):
            if 'obs' in obs:
                return obs['obs']
            elif 'pixels' in obs:
                return obs['pixels']
            else:
                return jax.tree_util.tree_leaves(obs)[0]
        return obs

    def _extract_achievements_from_state(self, state):
        """Extract achievement vector from Craftax state.

        The state contains achievements as a boolean array.
        Returns int32 to avoid JAX iota dtype issues with bool.
        """
        try:
            # Craftax stores achievements in state.achievements as a JAX array
            if hasattr(state, 'achievements'):
                # Convert to int32 to avoid JAX iota dtype issues
                achievements = np.asarray(state.achievements, dtype=np.int32)
                # Pad or truncate to expected size
                if len(achievements) < NUM_CRAFTAX_ACHIEVEMENTS:
                    padded = np.zeros(NUM_CRAFTAX_ACHIEVEMENTS, dtype=np.int32)
                    padded[:len(achievements)] = achievements
                    return padded
                return achievements[:NUM_CRAFTAX_ACHIEVEMENTS]
        except Exception:
            pass
        return np.zeros(NUM_CRAFTAX_ACHIEVEMENTS, dtype=np.int32)

    def _compute_achievement_depth(self, achievements):
        """Compute max achievement tier from achievement vector."""
        max_tier = -1
        for i, achieved in enumerate(achievements):
            if achieved and i < len(CRAFTAX_ACHIEVEMENT_NAMES):
                name = CRAFTAX_ACHIEVEMENT_NAMES[i]
                if name in ACHIEVEMENT_TIERS:
                    max_tier = max(max_tier, ACHIEVEMENT_TIERS[name])
        return max_tier

    def _to_numpy_result(self, obs_jax, reward, is_first, is_last, is_terminal, achievements=None):
        """Single batched transfer from GPU to CPU at the end of step."""
        # Process obs on GPU first
        processed = self._process_obs_jit(self._extract_obs(obs_jax))
        # Single device_get for the processed obs only
        obs_np = np.asarray(processed)
        key = 'embedding' if self._use_embedding else 'image'
        result = {
            key: obs_np,
            'reward': np.float32(reward),
            'is_first': is_first,
            'is_last': is_last,
            'is_terminal': is_terminal,
        }
        if self._track_achievements:
            if achievements is None:
                achievements = np.zeros(NUM_CRAFTAX_ACHIEVEMENTS, dtype=np.int32)
            # Ensure achievements are int32 to avoid JAX iota dtype issues
            result['log/achievements'] = np.asarray(achievements, dtype=np.int32)
            result['log/achievement_depth'] = np.int32(self._compute_achievement_depth(achievements))
        return result

    def step(self, action):
        if action['reset'] or self._done:
            reset_key = self._get_rng_key()
            self._obs, self._state = self._reset_fn(reset_key, self._env_params)
            self._done = False
            # Reset episode achievement tracking
            self._episode_achievements = np.zeros(NUM_CRAFTAX_ACHIEVEMENTS, dtype=np.bool_)
            self._cumulative_reward = 0.0
            achievements = self._extract_achievements_from_state(self._state) if self._track_achievements else None
            return self._to_numpy_result(self._obs, 0.0, True, False, False, achievements)

        # Get action - avoid Python int() conversion, keep as jax-compatible
        act = action['action']
        if isinstance(act, np.ndarray):
            act = int(act.item() if act.ndim == 0 else act[0])

        # Step environment using jitted step function (stays on GPU)
        step_key = self._get_rng_key()
        self._obs, self._state, reward, done, info = self._step_fn(
            step_key, self._state, act, self._env_params
        )

        # Only transfer done flag to CPU (single scalar)
        self._done = bool(done)
        # Transfer reward scalar to CPU
        reward_val = float(reward)
        self._cumulative_reward += reward_val

        # Extract achievements from state
        achievements = None
        if self._track_achievements:
            achievements = self._extract_achievements_from_state(self._state)
            # Update episode achievements (OR with current)
            self._episode_achievements = np.logical_or(self._episode_achievements, achievements)

        return self._to_numpy_result(
            self._obs, reward_val, False, self._done, self._done, achievements
        )

    def close(self):
        pass


class VectorCraftaxEnv:
    """Vectorized Craftax environment using jax.vmap.

    Runs N environments in a single process with one CUDA context,
    eliminating the multi-process overhead of the parallel Driver.
    All env step/reset logic stays on GPU; only one batched transfer per step.
    """

    def __init__(self, env_name='CraftaxSymbolic-v1', num_envs=16,
                 embedding_dim=256, use_embedding=True, seed=42, track_achievements=True):
        self._env_name = env_name
        self._num_envs = num_envs
        self._embedding_dim = embedding_dim
        self._use_embedding = use_embedding
        self._track_achievements = track_achievements

        # Create one Craftax env to get params / spaces
        if 'Classic' in env_name:
            if 'Symbolic' in env_name:
                self._env = CraftaxClassicSymbolicEnv()
            else:
                self._env = CraftaxClassicPixelsEnv()
        else:
            if 'Symbolic' in env_name:
                self._env = CraftaxSymbolicEnv()
            else:
                self._env = CraftaxPixelsEnv()

        self._env_params = self._env.default_params
        self._num_actions = int(self._env.action_space(self._env_params).n)

        # Vectorised reset / step via vmap (single JIT, single CUDA context)
        self._v_reset = jax.jit(jax.vmap(self._env.reset, in_axes=(0, None)))
        self._v_step = jax.jit(jax.vmap(self._env.step, in_axes=(0, 0, 0, None)))

        # Initial reset to get obs shape
        keys = jax.random.split(jax.random.PRNGKey(seed), num_envs)
        obs_batch, self._states = self._v_reset(keys, self._env_params)

        sample = jax.tree_util.tree_leaves(obs_batch)[0]
        self._obs_shape = tuple(int(x) for x in sample.shape[1:])  # drop batch dim

        # Projection for embedding (on GPU)
        if use_embedding:
            np.random.seed(42)
            flat_dim = int(np.prod(self._obs_shape))
            proj = np.random.randn(flat_dim, embedding_dim).astype(np.float32)
            proj /= np.sqrt(flat_dim)
            self._projection = jax.device_put(jnp.array(proj))

            @jax.jit
            def process_obs(obs):
                # obs: (N, *obs_shape)
                flat = obs.reshape(obs.shape[0], -1).astype(jnp.float32)
                return jnp.dot(flat, self._projection)
            self._process_obs = process_obs
        else:
            @jax.jit
            def process_obs(obs):
                return obs.astype(jnp.float32)
            self._process_obs = process_obs

        # JIT the full step logic (reset + step + process) to avoid per-step syncs
        # We'll use a wrapper that handles auto-reset inside JAX.
        @jax.jit
        def _batched_step(states, actions, dones, rng_keys, env_params, projection):
            """Step all envs; auto-reset any that are done."""
            # First handle resets for envs that are done
            new_obs_reset, new_states_reset = self._env.reset(rng_keys[0], env_params)
            # Then step all envs
            obs_step, new_states_step, rewards, new_dones, infos = self._env.step(
                rng_keys[0], states, actions, env_params)
            return obs_step, new_states_step, rewards, new_dones
        # ^ The above approach doesn't vectorize well; we'll use a simpler design.

        # RNG key management
        self._base_key = jax.random.PRNGKey(seed)
        self._step_count = 0
        self._dones = jnp.ones(num_envs, dtype=jnp.bool_)  # all need reset initially

        # Do the initial reset properly
        self._obs_jax = self._extract_obs_batch(obs_batch)
        self._dones = jnp.zeros(num_envs, dtype=jnp.bool_)

    def _extract_obs_batch(self, obs):
        """Extract raw observation array from vmapped output."""
        if hasattr(obs, 'shape'):
            return obs
        elif isinstance(obs, dict):
            for k in ('obs', 'pixels'):
                if k in obs:
                    return obs[k]
            return jax.tree_util.tree_leaves(obs)[0]
        return obs

    def _get_keys(self, n):
        """Get n random keys (amortised)."""
        self._step_count += 1
        return jax.random.split(
            jax.random.fold_in(self._base_key, self._step_count), n)

    def _extract_achievements_batch(self, states):
        """Extract achievement vectors from vectorized states.

        Args:
            states: Vectorized Craftax states.

        Returns:
            numpy array of shape (num_envs, NUM_CRAFTAX_ACHIEVEMENTS) as int32.
        """
        try:
            if hasattr(states, 'achievements'):
                # states.achievements should be (num_envs, num_achievements)
                # Use int32 to avoid JAX iota dtype issues with bool
                achievements = np.asarray(states.achievements, dtype=np.int32)
                if achievements.ndim == 1:
                    # Single env case - expand
                    achievements = achievements[np.newaxis, :]
                # Ensure correct shape
                if achievements.shape[1] < NUM_CRAFTAX_ACHIEVEMENTS:
                    padded = np.zeros((achievements.shape[0], NUM_CRAFTAX_ACHIEVEMENTS), dtype=np.int32)
                    padded[:, :achievements.shape[1]] = achievements
                    return padded
                return achievements[:, :NUM_CRAFTAX_ACHIEVEMENTS]
        except Exception:
            pass
        return np.zeros((self._num_envs, NUM_CRAFTAX_ACHIEVEMENTS), dtype=np.int32)

    def _compute_achievement_depths_batch(self, achievements):
        """Compute achievement depths for a batch of achievement vectors.

        Args:
            achievements: numpy array of shape (num_envs, NUM_CRAFTAX_ACHIEVEMENTS).

        Returns:
            numpy array of shape (num_envs,) with achievement depths.
        """
        depths = np.full(achievements.shape[0], -1, dtype=np.int32)
        for i, ach in enumerate(achievements):
            max_tier = -1
            for j, achieved in enumerate(ach):
                if achieved and j < len(CRAFTAX_ACHIEVEMENT_NAMES):
                    name = CRAFTAX_ACHIEVEMENT_NAMES[j]
                    if name in ACHIEVEMENT_TIERS:
                        max_tier = max(max_tier, ACHIEVEMENT_TIERS[name])
            depths[i] = max_tier
        return depths

    @property
    def obs_space(self):
        spaces = {}
        if self._use_embedding:
            spaces['embedding'] = elements.Space(
                np.float32, (self._embedding_dim,), -np.inf, np.inf)
        else:
            spaces['image'] = elements.Space(
                np.float32, self._obs_shape, 0.0, 1.0)
        spaces.update({
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool),
        })
        # Achievement tracking spaces (prefixed with log/ to exclude from agent training)
        # Use int32 instead of bool to avoid JAX iota dtype issues in replay buffer
        if self._track_achievements:
            spaces['log/achievements'] = elements.Space(np.int32, (NUM_CRAFTAX_ACHIEVEMENTS,), 0, 2)
            spaces['log/achievement_depth'] = elements.Space(np.int32)
        return spaces

    @property
    def act_space(self):
        return {
            'action': elements.Space(np.int32, (), 0, self._num_actions),
            'reset': elements.Space(bool),
        }

    def step(self, actions_batch):
        """Step all N envs at once.

        Args:
            actions_batch: dict with 'action' (N,) int, 'reset' (N,) bool.
        Returns:
            dict with batched obs, reward, flags  –  all numpy (N, ...).
        """
        reset_mask = actions_batch['reset'] | np.asarray(self._dones)
        acts = jnp.asarray(actions_batch['action'], dtype=jnp.int32)

        keys = self._get_keys(self._num_envs)

        # Auto-reset envs that are done  (all on GPU, vmapped)
        if np.any(reset_mask):
            reset_idx = np.where(reset_mask)[0]
            reset_keys = keys[reset_idx]
            # vmap over only the envs that need reset
            r_obs, r_states = jax.vmap(
                self._env.reset, in_axes=(0, None))(reset_keys, self._env_params)
            # Scatter back into full state tree
            r_obs_raw = self._extract_obs_batch(r_obs)
            self._obs_jax = self._obs_jax.at[reset_idx].set(r_obs_raw)
            self._states = jax.tree_util.tree_map(
                lambda full, part: full.at[reset_idx].set(part),
                self._states, r_states)

        # Step all envs  (on GPU, single vmap call)
        step_keys = self._get_keys(self._num_envs)
        obs_new, new_states, rewards, dones, _info = self._v_step(
            step_keys, self._states, acts, self._env_params)
        obs_raw = self._extract_obs_batch(obs_new)

        # For reset envs, the observation is from the reset, reward=0
        reset_jnp = jnp.asarray(reset_mask)
        # Where reset happened, keep the already-written reset obs
        obs_final = jnp.where(
            reset_jnp[:, None] if obs_raw.ndim == 2 else reset_jnp.reshape(-1, *([1]*(obs_raw.ndim-1))),
            self._obs_jax, obs_raw)
        rewards_final = jnp.where(reset_jnp, 0.0, rewards)
        dones_final = jnp.where(reset_jnp, False, dones)

        # For non-reset envs, update states
        self._states = jax.tree_util.tree_map(
            lambda ns, old: jnp.where(
                reset_jnp.reshape(-1, *([1]*(ns.ndim-1))), old, ns),
            new_states, self._states)
        self._obs_jax = jnp.where(
            reset_jnp[:, None] if obs_raw.ndim == 2 else reset_jnp.reshape(-1, *([1]*(obs_raw.ndim-1))),
            self._obs_jax, obs_raw)

        self._dones = dones_final

        # Process observations (embedding projection) — all on GPU, one call
        processed = self._process_obs(self._obs_jax)

        # === Single batched GPU→CPU transfer ===
        result_jax = (processed, rewards_final, dones_final)
        processed_np, rewards_np, dones_np = jax.device_get(result_jax)

        is_first = np.asarray(reset_mask, dtype=bool)
        is_last = np.asarray(dones_np, dtype=bool)

        key = 'embedding' if self._use_embedding else 'image'
        result = {
            key: np.asarray(processed_np, dtype=np.float32),
            'reward': np.asarray(rewards_np, dtype=np.float32),
            'is_first': is_first,
            'is_last': is_last,
            'is_terminal': is_last.copy(),
        }

        # Add achievement tracking (prefixed with log/ to exclude from agent training)
        if self._track_achievements:
            achievements = self._extract_achievements_batch(self._states)
            depths = self._compute_achievement_depths_batch(achievements)
            result['log/achievements'] = achievements
            result['log/achievement_depth'] = depths

        return result

    def close(self):
        pass


class VectorDriver:
    """Lightweight driver for VectorCraftaxEnv.

    Replaces embodied.Driver for vectorised JAX envs:
    - single process, single CUDA context
    - one vmap call per step for all envs
    - batched GPU→CPU transfer
    """

    def __init__(self, vec_env):
        self.vec_env = vec_env
        self.length = vec_env._num_envs
        self.act_space = vec_env.act_space
        self.obs_space = vec_env.obs_space
        self.callbacks = []
        self.carry = None
        self.acts = None
        self.reset()

    def reset(self, init_policy=None):
        self.acts = {
            k: np.zeros((self.length,) + v.shape, v.dtype)
            for k, v in self.act_space.items()}
        self.acts['reset'] = np.ones(self.length, bool)
        self.carry = init_policy(self.length) if init_policy else None

    def close(self):
        self.vec_env.close()

    def on_step(self, callback):
        self.callbacks.append(callback)

    def __call__(self, policy, steps=0, episodes=0):
        step, episode = 0, 0
        while step < steps or episode < episodes:
            step, episode = self._step(policy, step, episode)

    def _step(self, policy, step, episode):
        # Step all envs at once (single GPU call)
        obs = self.vec_env.step(self.acts)
        obs_no_log = {k: v for k, v in obs.items() if not k.startswith('log/')}

        self.carry, acts, outs = policy(self.carry, obs_no_log)
        if obs['is_last'].any():
            mask = ~obs['is_last']
            acts = {k: self._mask(v, mask) for k, v in acts.items()}
        self.acts = {**acts, 'reset': obs['is_last'].copy()}

        # Run callbacks per env
        trans = {**obs, **acts, **outs}
        for i in range(self.length):
            trn = {k: v[i] for k, v in trans.items()}
            for fn in self.callbacks:
                fn(trn, i)

        step += len(obs['is_first'])
        episode += int(obs['is_last'].sum())
        return step, episode

    def _mask(self, value, mask):
        while mask.ndim < value.ndim:
            mask = mask[..., None]
        return value * mask.astype(value.dtype)


# ============== Helper functions ==============
def wrap_env(craftax_env):
    """Wrap a Craftax environment for DreamerV3."""
    env = craftax_env
    for name, space in env.act_space.items():
        if name != 'reset' and not space.discrete:
            env = embodied.wrappers.NormalizeAction(env, name)
    env = embodied.wrappers.UnifyDtypes(env)
    env = embodied.wrappers.CheckSpaces(env)
    return env


def make_agent(config, env, args=None):
    """Create a DreamerV3 agent."""
    notlog = lambda k: not k.startswith('log/')
    obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
    act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
    # Build P2E config from args (only for enhanced version; original Agent has no P2E support)
    p2e_config = {}
    use_original = getattr(args, 'use_original_dreamer', False) if args is not None else False
    if args is not None and not use_original:
        p2e_config = {
            'plan2explore': getattr(args, 'plan2explore', False),
            'disag_models': getattr(args, 'disag_models', 10),
            'disag_target': getattr(args, 'disag_target', 'feat'),
            'expl_intr_scale': getattr(args, 'expl_intr_scale', 0.9),
            'expl_extr_scale': getattr(args, 'expl_extr_scale', 0.9),
        }
    agent_config = elements.Config(
        **config.agent,
        logdir=config.logdir,
        seed=config.seed,
        jax=config.jax,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        replay_context=config.replay_context,
        report_length=config.report_length,
        replica=config.replica,
        replicas=config.replicas,
        **p2e_config,
    )
    return Agent(obs_space, act_space, agent_config)


def make_selector(args, capacity, seed=0):
    """Create a replay selector based on sampling strategy args.

    Supports:
    - Uniform sampling (default)
    - Reservoir sampling (random eviction) - use with eviction='reservoir' in make_replay
    - Recency-biased sampling
    - 50:50 sampling (half uniform, half triangular-recency) - Continual-Dreamer strategy
    - Mixture of multiple strategies
    """
    from embodied.core import selectors

    # Check if using 50:50 sampling (Continual-Dreamer strategy)
    # This is the recommended setup for continual learning with 8+ tasks.
    # The "recent" half uses a triangular (linearly decaying) distribution
    # over the most recent `window_size` items, matching the paper exactly.
    recent_frac = getattr(args, 'recent_frac', 0.0)
    if recent_frac > 0:
        window_size = getattr(args, 'recent_window', 1000)
        # Triangular distribution: most recent items get highest probability,
        # linearly decaying to near-zero for the oldest item in the window.
        uprobs = np.linspace(1.0, 0.0, window_size, endpoint=False)
        selector_dict = {
            'uniform': selectors.Uniform(seed=seed),
            'recency': selectors.Recency(uprobs, seed=seed + 1),
        }
        fractions = {
            'uniform': 1.0 - recent_frac,
            'recency': recent_frac,
        }
        return selectors.Mixture(selector_dict, fractions, seed=seed + 2)

    # Check if using recency sampling (different from 50:50)
    if getattr(args, 'recency_sampling', False):
        # Create recency distribution - more recent items have higher probability
        uprobs = np.linspace(1.0, 0.1, min(capacity, 100000))
        return selectors.Recency(uprobs, seed=seed)

    # Check if using mixture of uniform and recency
    uniform_frac = getattr(args, 'uniform_frac', 1.0)
    recency_frac = getattr(args, 'recency_frac', 0.0)

    if recency_frac > 0 and uniform_frac < 1.0:
        # Mixture of uniform and recency
        uprobs = np.linspace(1.0, 0.1, min(capacity, 100000))
        selector_dict = {
            'uniform': selectors.Uniform(seed=seed),
            'recency': selectors.Recency(uprobs, seed=seed + 1),
        }
        fractions = {
            'uniform': uniform_frac,
            'recency': recency_frac,
        }
        return selectors.Mixture(selector_dict, fractions, seed=seed + 2)

    # Default to uniform sampling
    return selectors.Uniform(seed=seed)


def make_replay(config, directory, args=None):
    """Create a replay buffer with configurable sampling and eviction strategy.

    Supports:
    - FIFO eviction (default)
    - Reservoir eviction (random eviction for continual learning)

    Note: Custom selectors and 'eviction' parameter are only available in the continuous enhanced version.
    The original DreamerV3 only supports default uniform sampling.
    """
    length = config.batch_length + config.replay_context
    capacity = int(config.replay.size)

    # Check which version we're using
    use_original = getattr(args, 'use_original_dreamer', False) if args is not None else False

    # Build replay kwargs based on DreamerV3 version
    replay_kwargs = {
        'length': length,
        'capacity': capacity,
        'directory': directory,
        'online': config.replay.online,
        'chunksize': config.replay.chunksize,
    }

    # Only use custom selectors and eviction with continuous enhanced version
    if not use_original:
        # Create selector based on args
        selector = None
        eviction = 'fifo'

        if args is not None:
            selector = make_selector(args, capacity, seed=config.seed)
            # Use reservoir eviction if flag is set
            if getattr(args, 'reservoir_sampling', False):
                eviction = 'reservoir'

        replay_kwargs['selector'] = selector
        replay_kwargs['eviction'] = eviction
    else:
        # Original DreamerV3: use default selector (uniform) and no eviction parameter
        if args is not None and getattr(args, 'reservoir_sampling', False):
            print("Warning: --reservoir_sampling is not supported with --use_original_dreamer")
        if args is not None and getattr(args, 'recent_frac', 0.0) > 0:
            print("Warning: --recent_frac sampling is not supported with --use_original_dreamer")

    return embodied.replay.Replay(**replay_kwargs)


def make_logger(config, step):
    """Create a logger."""
    logdir = elements.Path(config.logdir)
    outputs = [
        elements.logger.TerminalOutput('.*', 'Agent'),
        elements.logger.JSONLOutput(logdir, 'metrics.jsonl'),
    ]
    return elements.Logger(step, outputs, multiplier=1)


def _get_size_overrides(size_preset):
    """Get model size overrides as properly nested dict.
    
    The size presets in configs.yaml use regex keys (e.g. '.*\\.rssm')
    which only work with elements.Flags, not config.update().
    We must manually expand them into the correct nested paths.
    """
    SIZES = {
        '1m':   dict(rssm=dict(deter=512,   hidden=64,   classes=4),  depth=4,  units=64),
        '12m':  dict(rssm=dict(deter=2048,  hidden=256,  classes=16), depth=16, units=256),
        '25m':  dict(rssm=dict(deter=3072,  hidden=384,  classes=24), depth=24, units=384),
        '50m':  dict(rssm=dict(deter=4096,  hidden=512,  classes=32), depth=32, units=512),
        '100m': dict(rssm=dict(deter=6144,  hidden=768,  classes=48), depth=48, units=768),
        '200m': dict(rssm=dict(deter=8192,  hidden=1024, classes=64), depth=64, units=1024),
        '400m': dict(rssm=dict(deter=12288, hidden=1536, classes=96), depth=96, units=1536),
    }
    s = SIZES[size_preset]
    rssm, depth, units = s['rssm'], s['depth'], s['units']
    return {
        'agent': {
            'dyn': {'rssm': rssm},
            'enc': {'simple': {'depth': depth, 'units': units}},
            'dec': {'simple': {'depth': depth, 'units': units}},
            'rewhead': {'units': units},
            'conhead': {'units': units},
            'policy': {'units': units},
            'value': {'units': units},
        },
    }


def load_config(args):
    """Load DreamerV3 config from YAML and merge with args.

    All non-architectural hyperparameters (replay size, train_ratio, envs,
    batch_size, etc.) are kept IDENTICAL between enhanced and original so
    that the only differences are the CL innovations:
      - Replay sampling strategy (50:50 vs uniform)
      - Plan2Explore exploration (on vs off)
      - Reservoir eviction (on vs off / FIFO)
    """
    use_original = getattr(args, 'use_original_dreamer', False)
    # Always load enhanced config so infrastructure params are identical.
    configs_path = root / 'dreamerv3' / 'dreamerv3' / 'configs.yaml'
    configs = yaml.YAML(typ='safe').load(configs_path.read_text())
    config = elements.Config(configs['defaults'])

    # Apply model size preset by manually overriding nested agent params.
    # The default config is size200m (~200M params) which will OOM on <=32GB GPUs.
    size_preset = getattr(args, 'model_size', '12m')
    size_overrides = _get_size_overrides(size_preset)
    config = config.update(size_overrides)
    print(f'Using model size preset: size{size_preset}')

    # Apply size preset if specified
    if args.cl_small and 'small' in configs:
        config = config.update(configs['small'])

    tag = args.tag + str(args.seed)

    # Build overrides from args
    # Default to 64 envs for JAX-based environments
    num_envs = int(args.envs) if getattr(args, 'envs', None) is not None else 64
    batch_length = getattr(args, 'batch_length', 32)
    # Scale train_ratio with envs * batch_length to keep GPU saturated.
    # With 128 envs and batch_length 256, we need high train_ratio
    # to generate enough gradient steps per env step.
    user_train_ratio = getattr(args, 'train_ratio', None)
    if user_train_ratio is not None:
        effective_train_ratio = user_train_ratio
    elif num_envs >= 64 and batch_length >= 128:
        effective_train_ratio = 128.0
    elif num_envs >= 32:
        effective_train_ratio = 96.0
    else:
        effective_train_ratio = 64.0
    run_overrides = {
        'logdir': f'{args.logdir}/craftax_{tag}',
        'seed': args.seed,
        'batch_size': args.batch_size,
        'batch_length': batch_length,
        'replay_context': 0,  # Disable replay context to avoid needing dyn/deter and dyn/stoch in replay
        'run': {
            'steps': int(args.steps),
            'log_every': 5000,      # Less frequent logging to reduce CPU overhead
            'save_every': 50000,    # Less frequent saving to reduce I/O
            'report_every': 100000, # Less frequent reporting
            'train_ratio': effective_train_ratio,
            'envs': num_envs,
            'debug': False,  # Disable debug mode for performance
        },
        'jax': {
            'prealloc': True,   # Use BFC allocator with preallocation to prevent fragmentation
            'platform': 'gpu',  # Must be 'gpu' (not 'cuda') for internal.setup() GPU flags
        },
    }
    run_overrides['replay'] = {'size': int(args.replay_capacity)}
    if getattr(args, 'eval_envs', None) is not None:
        run_overrides['run']['eval_envs'] = int(args.eval_envs)
    config = config.update(run_overrides)

    # --- Architectural-only overrides for original DreamerV3 baseline ---
    # Only change what the CL innovations introduce; keep everything else identical.
    if use_original:
        config = config.update({
            'replay': {
                'fracs': {'uniform': 1.0, 'priority': 0.0, 'recency': 0.0},
            },
        })
        print('=== Original DreamerV3 Baseline Config Verification ===')
        print(f'  replay.fracs:    {dict(config.replay.fracs)}  (uniform only)')
        print(f'  replay.size:     {config.replay.size}')
        print(f'  replay_context:  {config.replay_context}')
        print(f'  run.train_ratio: {config.run.train_ratio}')
        print(f'  run.envs:        {config.run.envs}')
        print(f'  P2E:             disabled (original Agent has no P2E)')
        print(f'  Eviction:        FIFO (no reservoir sampling)')
        print('=======================================================')

    return config, tag


def train_single(make_env, config, args, env_name=None):
    """Train DreamerV3 on a single environment.

    Args:
        make_env: Callable(seed: int) -> embodied.Env that creates one env instance.
        env_name: If provided, use VectorCraftaxEnv for GPU-vectorised envs.
    """
    np.random.seed(config.seed)
    logdir = elements.Path(config.logdir)

    # Always start fresh: remove old logdir contents (replay, checkpoints)
    import shutil as _shutil_clean
    logdir_str = str(logdir)
    if os.path.exists(logdir_str):
        for sub in ('replay', 'ckpt'):
            sub_path = os.path.join(logdir_str, sub)
            if os.path.exists(sub_path):
                _shutil_clean.rmtree(sub_path, ignore_errors=True)
        print(f'Cleaned old replay/ckpt from {logdir_str} for fresh start.')

    logdir.mkdir()
    config.save(logdir / 'config.yaml')
    print('Logdir:', logdir)

    num_envs = config.run.envs
    use_embedding = getattr(args, 'input_type', 'embedding') == 'embedding'
    embedding_dim = getattr(args, 'embedding_dim', 256)

    if env_name is not None and CRAFTAX_AVAILABLE:
        # === Vectorised path: single process, single CUDA context ===
        print(f'Creating {num_envs} vectorised Craftax envs (env={env_name}).')
        vec_env = VectorCraftaxEnv(
            env_name=env_name, num_envs=num_envs,
            embedding_dim=embedding_dim, use_embedding=use_embedding,
            seed=config.seed)
        driver = VectorDriver(vec_env)
        # Use a lightweight single env for agent space inference
        tmp_env = wrap_env(make_env(config.seed))
        agent = make_agent(config, tmp_env, args)
        tmp_env.close()
    else:
        # === Legacy multiprocessing path ===
        use_parallel = num_envs > 1
        print(f'Creating {num_envs} environments (parallel={use_parallel}).')
        env_fns = [lambda i=i: wrap_env(make_env(config.seed + i)) for i in range(num_envs)]
        driver = embodied.Driver(env_fns, parallel=use_parallel)
        if use_parallel:
            tmp_env = wrap_env(make_env(config.seed))
            agent = make_agent(config, tmp_env, args)
            tmp_env.close()
        else:
            agent = make_agent(config, driver.envs[0], args)
    replay = make_replay(config, logdir / 'replay', args)

    step = elements.Counter()
    logger = make_logger(config, step)
    online = None
    if getattr(args, 'online_metrics', True):
        online = CraftaxMetrics(
            logdir=str(logdir),
            num_tasks=1,
            window_size=100,
            success_threshold=1.0,
        )

    batch_size = config.batch_size
    batch_length = config.batch_length
    batch_steps = batch_size * batch_length
    train_ratio = config.run.train_ratio
    should_train = elements.when.Ratio(train_ratio / batch_steps)

    episodes = collections.defaultdict(elements.Agg)
    episode_achievements = {}  # Track achievements per worker

    # Shared state for training metrics
    training_metrics_state = {'latest': {}, 'log_every': 1000, 'last_log_step': 0}

    def logfn(tran, worker):
        episode = episodes[worker]

        # On episode start, reset achievement tracking
        if tran['is_first']:
            episode.reset()
            episode_achievements[worker] = np.zeros(NUM_CRAFTAX_ACHIEVEMENTS, dtype=np.bool_)

        episode.add('score', tran['reward'], agg='sum')
        episode.add('length', 1, agg='sum')

        # Accumulate achievements throughout episode
        if 'log/achievements' in tran and worker in episode_achievements:
            ach = np.asarray(tran['log/achievements'], dtype=np.bool_)
            if len(ach) == NUM_CRAFTAX_ACHIEVEMENTS:
                episode_achievements[worker] = np.logical_or(
                    episode_achievements[worker], ach)

        if tran['is_last']:
            result = episode.result()
            score = result.pop('score')
            length = result.pop('length')
            logger.add({'score': score, 'length': length}, prefix='episode')

            if online is not None:
                # Get final achievements for this episode
                achievements = episode_achievements.get(worker, np.zeros(NUM_CRAFTAX_ACHIEVEMENTS, dtype=np.bool_))
                achievement_depth = tran.get('log/achievement_depth', -1)
                if isinstance(achievement_depth, np.ndarray):
                    achievement_depth = int(achievement_depth.item())

                online.update(
                    task_id=0,
                    step=int(step.value),
                    score=float(score),
                    length=int(length),
                    achievements=achievements,
                    achievement_depth=int(achievement_depth),
                )
            logger.write()

    def log_training_metrics_fn(mets, step_val):
        """Log training metrics periodically."""
        if online is None:
            return
        current_step = int(step_val)
        if current_step - training_metrics_state['last_log_step'] >= training_metrics_state['log_every']:
            training_metrics_state['last_log_step'] = current_step

            # Extract relevant metrics from training output
            loss_obs = float(mets.get('loss/embedding', mets.get('loss/image', 0.0)))
            loss_rew = float(mets.get('loss/rew', 0.0))
            loss_con = float(mets.get('loss/con', 0.0))
            loss_dyn = float(mets.get('loss/dyn', 0.0))
            loss_rep = float(mets.get('loss/rep', 0.0))
            loss_policy = float(mets.get('loss/policy', 0.0))
            loss_value = float(mets.get('loss/value', 0.0))

            # P2E metrics
            ensemble_disagreement = float(mets.get('loss/disag', 0.0))
            intrinsic_reward = float(mets.get('p2e/intr_rew', 0.0))
            extrinsic_reward = float(mets.get('p2e/extr_rew', 0.0))

            # TD-error approximation from advantage stats
            td_error_mean = float(mets.get('adv', 0.0))
            td_error_max = float(mets.get('adv_mag', 0.0))

            online.log_training_metrics(
                step=current_step,
                loss_obs=loss_obs,
                loss_rew=loss_rew,
                loss_con=loss_con,
                loss_dyn=loss_dyn,
                loss_rep=loss_rep,
                loss_policy=loss_policy,
                loss_value=loss_value,
                td_error_mean=td_error_mean,
                td_error_max=td_error_max,
                ensemble_disagreement=ensemble_disagreement,
                intrinsic_reward=intrinsic_reward,
                extrinsic_reward=extrinsic_reward,
            )

            # Log exploration diagnostics (dream accuracy)
            imagined_value = float(mets.get('val', 0.0))
            actual_value = float(mets.get('ret', 0.0))
            dream_accuracy = 1.0 - abs(imagined_value - actual_value) / max(abs(actual_value), 1e-8) if actual_value != 0 else 0.0

            intr_extr_ratio = 0.0
            if extrinsic_reward > 0:
                intr_extr_ratio = intrinsic_reward / extrinsic_reward

            online.log_exploration_diagnostics(
                step=current_step,
                imagined_value=imagined_value,
                actual_value=actual_value,
                dream_accuracy=dream_accuracy,
                intrinsic_extrinsic_ratio=intr_extr_ratio,
            )

    def log_replay_diagnostics_fn(replay_buffer, step_val):
        """Log replay buffer diagnostics periodically."""
        if online is None:
            return
        current_step = int(step_val)
        if current_step - training_metrics_state['last_log_step'] >= training_metrics_state['log_every']:
            stats = replay_buffer.stats()
            buffer_size = stats.get('total_steps', len(replay_buffer))

            # Compute depth distribution from recent samples if possible
            # This is an approximation - actual implementation depends on buffer structure
            depth_distribution = [0.0] * (NUM_ACHIEVEMENT_TIERS + 1)

            online.log_replay_diagnostics(
                step=current_step,
                buffer_size=buffer_size,
                depth_distribution=depth_distribution,
                mean_td_error=0.0,  # Would need to track priorities
                mean_episode_age=0.0,  # Would need timestamp tracking
            )

    driver.on_step(lambda tran, _: step.increment())
    driver.on_step(replay.add)
    driver.on_step(logfn)

    # Prefill
    prefill = 10000
    if step < prefill:
        print(f'Prefill dataset ({prefill} steps).')
        random_policy = lambda carry, obs: (
            carry,
            {k: np.stack([v.sample() for _ in range(len(obs['is_first']))])
             for k, v in driver.act_space.items() if k != 'reset'},
            {}
        )
        driver.reset()
        driver(random_policy, steps=prefill)

    def make_stream(replay, mode):
        fn = bind(replay.sample, batch_size, mode)
        stream = embodied.streams.Stateless(fn)
        stream = embodied.streams.Consec(
            stream, length=batch_length, consec=1, prefix=config.replay_context,
            strict=(mode == 'train'), contiguous=True)
        return stream

    stream_train = iter(agent.stream(make_stream(replay, 'train')))
    carry_train = [agent.init_train(batch_size)]

    # Always start fresh: remove old checkpoint directory if it exists
    ckpt_dir = logdir / 'ckpt'
    if ckpt_dir.exists():
        import shutil as _shutil
        _shutil.rmtree(str(ckpt_dir), ignore_errors=True)
        print('Removed old checkpoint directory for fresh start.')

    cp = elements.Checkpoint(logdir / 'ckpt')
    cp.step = step
    cp.agent = agent
    cp.replay = replay
    # Never load checkpoint - always start fresh
    cp.save()

    print('Start training loop (fresh start, no checkpoint loaded)')
    policy = lambda carry, obs: agent.policy(carry, obs, mode='train')
    driver.reset(agent.init_policy)

    # Use larger collect steps to reduce collect<->train switching overhead.
    # With N vectorised envs each _step produces N transitions, so
    # collect_steps ~= num_envs * K gives K actual _step calls per collect.
    # For large num_envs (>=64), use bigger K to keep GPU saturated during collection.
    if num_envs >= 64:
        collect_steps = num_envs * 8  # 8 steps per env per collect phase
    else:
        collect_steps = max(num_envs * 4, 64)

    total_steps = int(config.run.steps)
    while step < total_steps:
        driver(policy, steps=collect_steps)
        if len(replay) >= batch_size * batch_length:
            for _ in range(should_train(step)):
                batch = next(stream_train)
                carry_train[0], outs, mets = agent.train(carry_train[0], batch)
                if 'replay' in outs:
                    replay.update(outs['replay'])
                # Log training and exploration metrics
                log_training_metrics_fn(mets, step.value)
                log_replay_diagnostics_fn(replay, step.value)
        if step.value % 10000 < 10:
            cp.save()
        # Periodically clear JAX caches to prevent memory accumulation
        # that causes CUDA_ERROR_ILLEGAL_ADDRESS after ~15k steps
        if step.value % 5000 == 0 and step.value > 0:
            jax.clear_caches()

    cp.save()
    if online is not None:
        online.save_summary()
    driver.close()
    logger.close()


def cl_train_loop(make_envs, config, args, env_names=None):
    """Continual learning training loop for DreamerV3.

    Args:
        make_envs: List of Callable(seed: int) -> embodied.Env, one per task.
        env_names: Optional list of Craftax env name strings for vectorised path.
    """
    np.random.seed(config.seed)

    unbalanced_steps = None
    if args.unbalanced_steps not in [None, 'None', 'none']:
        unbalanced_steps = ast.literal_eval(str(args.unbalanced_steps))

    logdir = elements.Path(config.logdir)

    # Always start fresh: remove old logdir contents (replay, checkpoints)
    import shutil as _shutil_clean
    logdir_str = str(logdir)
    if os.path.exists(logdir_str):
        for sub in ('replay', 'ckpt'):
            sub_path = os.path.join(logdir_str, sub)
            if os.path.exists(sub_path):
                _shutil_clean.rmtree(sub_path, ignore_errors=True)
        print(f'Cleaned old replay/ckpt from {logdir_str} for fresh start.')

    logdir.mkdir()
    config.save(logdir / 'config.yaml')
    print('Logdir:', logdir)

    num_envs = config.run.envs
    env0 = wrap_env(make_envs[0](config.seed))

    agent = make_agent(config, env0, args)
    replay = make_replay(config, logdir / 'replay', args)

    total_step = elements.Counter()
    logger = make_logger(config, total_step)
    online = None
    if getattr(args, 'online_metrics', True):
        online = CraftaxMetrics(
            logdir=str(logdir),
            num_tasks=len(make_envs),
            window_size=100,
            success_threshold=1.0,
        )

    batch_size = config.batch_size
    batch_length = config.batch_length
    batch_steps = batch_size * batch_length
    train_ratio = config.run.train_ratio
    should_train = elements.when.Ratio(train_ratio / batch_steps)

    # Always start fresh: remove old checkpoint directory if it exists
    ckpt_dir = logdir / 'ckpt'
    if ckpt_dir.exists():
        import shutil as _shutil
        _shutil.rmtree(str(ckpt_dir), ignore_errors=True)
        print('Removed old checkpoint directory for fresh start.')

    cp = elements.Checkpoint(logdir / 'ckpt')
    cp.step = total_step
    cp.agent = agent
    cp.replay = replay
    # Never load checkpoint - always start fresh
    cp.save()

    stats = replay.stats()
    total_steps_done = stats.get('total_steps', 0)
    steps_per_task = int(args.steps)
    num_tasks = len(make_envs)

    if unbalanced_steps is not None:
        tot_steps_after_task = np.cumsum(unbalanced_steps)
        task_id = next((i for i, j in enumerate(total_steps_done < tot_steps_after_task) if j), 0)
        rep = int(total_steps_done // np.sum(unbalanced_steps))
    else:
        task_id = int(total_steps_done // steps_per_task) % num_tasks
        rep = int(total_steps_done // (steps_per_task * num_tasks))

    print(f"Starting at Task {task_id}, Rep {rep}")

    def make_stream(replay, mode):
        fn = bind(replay.sample, batch_size, mode)
        stream = embodied.streams.Stateless(fn)
        stream = embodied.streams.Consec(
            stream, length=batch_length, consec=1, prefix=config.replay_context,
            strict=(mode == 'train'), contiguous=True)
        return stream

    use_embedding = getattr(args, 'input_type', 'embedding') == 'embedding'
    embedding_dim = getattr(args, 'embedding_dim', 256)
    use_vectorised = env_names is not None and CRAFTAX_AVAILABLE
    if num_envs >= 64:
        collect_steps = num_envs * 8
    else:
        collect_steps = max(num_envs * 4, 64)

    # Shared state for training metrics logging
    training_metrics_state = {'latest': {}, 'log_every': 1000, 'last_log_step': 0}

    def log_training_metrics_fn(mets, step_val, current_task_id):
        """Log training metrics periodically."""
        if online is None:
            return
        current_step = int(step_val)
        if current_step - training_metrics_state['last_log_step'] >= training_metrics_state['log_every']:
            training_metrics_state['last_log_step'] = current_step

            loss_obs = float(mets.get('loss/embedding', mets.get('loss/image', 0.0)))
            loss_rew = float(mets.get('loss/rew', 0.0))
            loss_con = float(mets.get('loss/con', 0.0))
            loss_dyn = float(mets.get('loss/dyn', 0.0))
            loss_rep = float(mets.get('loss/rep', 0.0))
            loss_policy = float(mets.get('loss/policy', 0.0))
            loss_value = float(mets.get('loss/value', 0.0))

            ensemble_disagreement = float(mets.get('loss/disag', 0.0))
            intrinsic_reward = float(mets.get('p2e/intr_rew', 0.0))
            extrinsic_reward = float(mets.get('p2e/extr_rew', 0.0))

            td_error_mean = float(mets.get('adv', 0.0))
            td_error_max = float(mets.get('adv_mag', 0.0))

            online.log_training_metrics(
                step=current_step,
                loss_obs=loss_obs,
                loss_rew=loss_rew,
                loss_con=loss_con,
                loss_dyn=loss_dyn,
                loss_rep=loss_rep,
                loss_policy=loss_policy,
                loss_value=loss_value,
                td_error_mean=td_error_mean,
                td_error_max=td_error_max,
                ensemble_disagreement=ensemble_disagreement,
                intrinsic_reward=intrinsic_reward,
                extrinsic_reward=extrinsic_reward,
                task_id=current_task_id,
            )

            imagined_value = float(mets.get('val', 0.0))
            actual_value = float(mets.get('ret', 0.0))
            dream_accuracy = 1.0 - abs(imagined_value - actual_value) / max(abs(actual_value), 1e-8) if actual_value != 0 else 0.0
            intr_extr_ratio = intrinsic_reward / extrinsic_reward if extrinsic_reward > 0 else 0.0

            online.log_exploration_diagnostics(
                step=current_step,
                imagined_value=imagined_value,
                actual_value=actual_value,
                dream_accuracy=dream_accuracy,
                intrinsic_extrinsic_ratio=intr_extr_ratio,
            )

    def log_replay_diagnostics_fn(replay_buffer, step_val):
        """Log replay buffer diagnostics periodically."""
        if online is None:
            return
        current_step = int(step_val)
        if current_step - training_metrics_state['last_log_step'] >= training_metrics_state['log_every']:
            stats = replay_buffer.stats()
            buffer_size = stats.get('total_steps', len(replay_buffer))
            depth_distribution = [0.0] * (NUM_ACHIEVEMENT_TIERS + 1)

            online.log_replay_diagnostics(
                step=current_step,
                buffer_size=buffer_size,
                depth_distribution=depth_distribution,
                mean_td_error=0.0,
                mean_episode_age=0.0,
            )

    while rep < args.num_task_repeats:
        while task_id < num_tasks:
            print(f"\n=== Task {task_id + 1}/{num_tasks}, Rep {rep + 1}/{args.num_task_repeats} ===\n")

            make_task_env = make_envs[task_id]
            step = elements.Counter()
            episodes = collections.defaultdict(elements.Agg)
            episode_achievements = {}  # Track achievements per worker
            current_task = task_id  # Capture for closure

            def logfn(tran, worker):
                episode = episodes[worker]

                if tran['is_first']:
                    episode.reset()
                    episode_achievements[worker] = np.zeros(NUM_CRAFTAX_ACHIEVEMENTS, dtype=np.bool_)

                episode.add('score', tran['reward'], agg='sum')
                episode.add('length', 1, agg='sum')

                # Accumulate achievements
                if 'log/achievements' in tran and worker in episode_achievements:
                    ach = np.asarray(tran['log/achievements'], dtype=np.bool_)
                    if len(ach) == NUM_CRAFTAX_ACHIEVEMENTS:
                        episode_achievements[worker] = np.logical_or(
                            episode_achievements[worker], ach)

                if tran['is_last']:
                    result = episode.result()
                    score = result.pop('score')
                    length = result.pop('length')
                    logger.add({'score': score, 'length': length, 'task': current_task}, prefix='episode')

                    if online is not None:
                        achievements = episode_achievements.get(worker, np.zeros(NUM_CRAFTAX_ACHIEVEMENTS, dtype=np.bool_))
                        achievement_depth = tran.get('log/achievement_depth', -1)
                        if isinstance(achievement_depth, np.ndarray):
                            achievement_depth = int(achievement_depth.item())

                        online.update(
                            task_id=current_task,
                            step=int(total_step.value),
                            score=float(score),
                            length=int(length),
                            achievements=achievements,
                            achievement_depth=int(achievement_depth),
                        )
                    logger.write()

            if use_vectorised:
                task_env_name = env_names[task_id]
                print(f'Creating {num_envs} vectorised Craftax envs (env={task_env_name}).')
                vec_env = VectorCraftaxEnv(
                    env_name=task_env_name, num_envs=num_envs,
                    embedding_dim=embedding_dim, use_embedding=use_embedding,
                    seed=config.seed)
                driver = VectorDriver(vec_env)
            else:
                env_fns = [lambda i=i, fn=make_task_env: wrap_env(fn(config.seed + i)) for i in range(num_envs)]
                use_parallel = num_envs > 1
                driver = embodied.Driver(env_fns, parallel=use_parallel)

            driver.on_step(lambda tran, _: total_step.increment())
            driver.on_step(lambda tran, _: step.increment())
            driver.on_step(replay.add)
            driver.on_step(logfn)

            prefill = 10000
            if total_step.value < prefill:
                needed = prefill - total_step.value
                print(f'Prefill dataset ({needed} steps).')
                random_policy = lambda carry, obs: (
                    carry,
                    {k: np.stack([v.sample() for _ in range(len(obs['is_first']))])
                     for k, v in driver.act_space.items() if k != 'reset'},
                    {}
                )
                driver.reset()
                driver(random_policy, steps=int(needed))

            stream_train = iter(agent.stream(make_stream(replay, 'train')))
            carry_train = [agent.init_train(batch_size)]

            policy = lambda carry, obs: agent.policy(carry, obs, mode='train')
            driver.reset(agent.init_policy)

            if unbalanced_steps is not None:
                steps_limit = int(unbalanced_steps[task_id])
            else:
                steps_limit = steps_per_task

            print(f'Training for {steps_limit} steps on task {task_id}')

            while step < steps_limit:
                driver(policy, steps=collect_steps)
                if len(replay) >= batch_size * batch_length:
                    for _ in range(should_train(total_step)):
                        batch = next(stream_train)
                        carry_train[0], outs, mets = agent.train(carry_train[0], batch)
                        if 'replay' in outs:
                            replay.update(outs['replay'])
                        # Log training and exploration metrics
                        log_training_metrics_fn(mets, total_step.value, current_task)
                        log_replay_diagnostics_fn(replay, total_step.value)
                if step.value % 10000 < 10:
                    cp.save()
                # Periodically clear JAX caches to prevent memory accumulation
                if total_step.value % 5000 == 0 and total_step.value > 0:
                    jax.clear_caches()

            driver.close()
            task_id += 1

        task_id = 0
        rep += 1

    cp.save()
    if online is not None:
        online.save_summary()
    logger.close()


def make_craftax(env_name, embedding_dim=256, use_embedding=True, seed=42):
    """Create a Craftax environment with proper wrappers."""
    if not CRAFTAX_AVAILABLE:
        raise ImportError("Craftax is not installed. Install with: pip install craftax")
    
    return CraftaxWrapper(
        env_name=env_name,
        embedding_dim=embedding_dim,
        use_embedding=use_embedding,
        seed=seed,
    )


def run_craftax(args):
    """Main entry point for Craftax training."""
    tag = args.tag + str(args.seed)
    config, tag = load_config(args)

    unbalanced_steps = None
    if args.unbalanced_steps not in [None, 'None', 'none']:
        unbalanced_steps = ast.literal_eval(str(args.unbalanced_steps))

    # Available Craftax environments
    all_envs = [
        'CraftaxSymbolic-v1',       # Symbolic observation (flat vector)
        'CraftaxPixels-v1',          # Pixel observation
        'CraftaxClassicSymbolic-v1', # Classic version symbolic
        'CraftaxClassicPixels-v1',   # Classic version pixels
    ]

    use_embedding = (args.input_type == 'embedding')

    if args.cl:
        # For continual learning, use different configurations
        if args.cl_small:
            env_names = [
                'CraftaxClassicSymbolic-v1',
                'CraftaxSymbolic-v1',
            ]
        elif unbalanced_steps is not None:
            env_names = [
                'CraftaxSymbolic-v1',
                'CraftaxClassicSymbolic-v1',
            ]
        else:
            env_names = [
                'CraftaxSymbolic-v1',
                'CraftaxClassicSymbolic-v1',
                'CraftaxPixels-v1',
                'CraftaxClassicPixels-v1',
            ]

        wandb.init(
            config=dict(config),
            reinit=True,
            resume=False,
            dir=args.wandb_dir,
            mode=getattr(args, 'wandb_mode', 'online'),
            project=args.wandb_proj_name,
            group=args.wandb_group,
            name=f"DreamerV3_craftax_cl-small={args.cl_small}_{tag}",
        )

        make_env_fns = []
        cl_env_names = []
        for i in range(args.num_tasks):
            name = env_names[i % len(env_names)]
            cl_env_names.append(name)
            make_env_fns.append(lambda seed, name=name: make_craftax(
                name,
                embedding_dim=args.embedding_dim,
                use_embedding=use_embedding,
                seed=seed,
            ))
            print(f"Task {i}: env {name}, use_embedding: {use_embedding}")

        cl_train_loop(make_env_fns, config, args, env_names=cl_env_names)
    else:
        env_name = all_envs[args.env % len(all_envs)]

        wandb.init(
            config=dict(config),
            reinit=True,
            resume=False,
            dir=args.wandb_dir,
            mode=getattr(args, 'wandb_mode', 'online'),
            project=args.wandb_proj_name,
            group=args.wandb_group,
            name=f"DreamerV3_craftax_single-env={env_name}_{tag}",
        )

        make_env = lambda seed, name=env_name: make_craftax(
            name,
            embedding_dim=args.embedding_dim,
            use_embedding=use_embedding,
            seed=seed,
        )

        train_single(make_env, config, args, env_name=env_name)

    if args.del_exp_replay:
        shutil.rmtree(os.path.join(config.logdir, 'replay'), ignore_errors=True)

    wandb.finish()


if __name__ == "__main__":
    args = parse_craftax_args()
    run_craftax(args)
