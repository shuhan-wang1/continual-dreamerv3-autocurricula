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
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.70'

# Add dreamerv3 to path
root = pathlib.Path(__file__).parent
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

# Allow implicit host-to-device transfers (needed for passing Python scalars to JIT functions)
jax.config.update("jax_transfer_guard", "allow")

from dreamerv3.agent import Agent
from input_args import parse_craftax_args

import json
from collections import deque


class CraftaxMetrics:
    """Practical metrics tracker for Craftax training.

    Tracks three core metrics over a sliding window:
      1. Episode Return  – mean episode return (= total achievements unlocked).
      2. Success Rate    – fraction of episodes with return > threshold.
      3. Achievement Depth – max return seen (best achievement depth reached).

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
    ) -> None:
        self.logdir = logdir
        os.makedirs(logdir, exist_ok=True)
        self.num_tasks = int(num_tasks)
        self.window_size = int(window_size)
        self.success_threshold = float(success_threshold)
        self.jsonl_path = os.path.join(logdir, jsonl_name)
        self.summary_path = os.path.join(logdir, summary_name)

        # Per-task sliding windows
        self._returns = {i: deque(maxlen=window_size) for i in range(num_tasks)}
        self._lengths = {i: deque(maxlen=window_size) for i in range(num_tasks)}
        # Lifetime accumulators
        self._total_episodes = {i: 0 for i in range(num_tasks)}
        self._total_successes = {i: 0 for i in range(num_tasks)}
        self._max_return = {i: 0.0 for i in range(num_tasks)}
        self._sum_return = {i: 0.0 for i in range(num_tasks)}

    # ---- per-task windowed stats ----
    def _windowed_mean_return(self, task_id: int) -> float:
        buf = self._returns[task_id]
        return float(np.mean(buf)) if buf else 0.0

    def _windowed_success_rate(self, task_id: int) -> float:
        buf = self._returns[task_id]
        if not buf:
            return 0.0
        return float(np.mean([1.0 if r >= self.success_threshold else 0.0 for r in buf]))

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
        """Best single-episode return across all tasks (= deepest achievement)."""
        return float(max(self._max_return.values()))

    def per_task_mean_return(self):
        return [self._windowed_mean_return(i) for i in range(self.num_tasks)]

    def per_task_success_rate(self):
        return [self._windowed_success_rate(i) for i in range(self.num_tasks)]

    def per_task_max_return(self):
        return [float(self._max_return[i]) for i in range(self.num_tasks)]

    def per_task_episodes(self):
        return [self._total_episodes[i] for i in range(self.num_tasks)]

    # ---- update & IO ----
    def update(self, task_id: int, step: int, score: float,
               length: int = 0) -> dict:
        task_id = int(task_id)
        score = float(score)
        self._returns[task_id].append(score)
        self._lengths[task_id].append(int(length))
        self._total_episodes[task_id] += 1
        if score >= self.success_threshold:
            self._total_successes[task_id] += 1
        if score > self._max_return[task_id]:
            self._max_return[task_id] = score
        self._sum_return[task_id] += score

        record = {
            'step': int(step),
            'task': task_id,
            'score': score,
            'length': int(length),
            # Windowed per-task
            'return_mean': self._windowed_mean_return(task_id),
            'success_rate': self._windowed_success_rate(task_id),
            'achievement_depth': float(self._max_return[task_id]),
            # Aggregated across tasks
            'agg_return_mean': self.mean_return(),
            'agg_success_rate': self.mean_success_rate(),
            'agg_achievement_depth': self.max_achievement_depth(),
            # Per-task snapshots
            'per_task_return_mean': self.per_task_mean_return(),
            'per_task_success_rate': self.per_task_success_rate(),
            'per_task_max_return': self.per_task_max_return(),
        }
        with open(self.jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        return record

    def save_summary(self) -> None:
        payload = {
            'mean_return': self.mean_return(),
            'success_rate': self.mean_success_rate(),
            'max_achievement_depth': self.max_achievement_depth(),
            'per_task_return_mean': self.per_task_mean_return(),
            'per_task_success_rate': self.per_task_success_rate(),
            'per_task_max_return': self.per_task_max_return(),
            'per_task_episodes': self.per_task_episodes(),
            'per_task_lifetime_mean': [
                float(self._sum_return[i] / max(1, self._total_episodes[i]))
                for i in range(self.num_tasks)
            ],
            'num_tasks': self.num_tasks,
            'window_size': self.window_size,
            'success_threshold': self.success_threshold,
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
    CRAFTAX_AVAILABLE = True
except ImportError:
    CRAFTAX_AVAILABLE = False
    print("Warning: Craftax not installed. Install with: pip install craftax")


# ============== Craftax Environment Wrapper ==============
class CraftaxWrapper(embodied.Env):
    """Wrapper to convert Craftax JAX environments to embodied interface.
    
    Optimized to minimize GPU<->CPU transfers by keeping data on GPU as long
    as possible and using JIT-compiled processing functions.
    """

    def __init__(self, env_name='CraftaxSymbolic-v1', embedding_dim=256, use_embedding=True, seed=42):
        self._env_name = env_name
        self._embedding_dim = embedding_dim
        self._use_embedding = use_embedding
        self._seed = seed
        self._done = True
        
        # Use a base key and step counter for deterministic key generation
        # Pre-generate a batch of keys to avoid per-step host-to-device transfers
        self._base_key = jax.random.PRNGKey(seed)
        self._step_count = 0
        self._key_batch_size = 1024
        self._key_cache = jax.random.split(self._base_key, self._key_batch_size)
        self._key_idx = 0
        
        # Create the appropriate Craftax environment
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

    def _to_numpy_result(self, obs_jax, reward, is_first, is_last, is_terminal):
        """Single batched transfer from GPU to CPU at the end of step."""
        # Process obs on GPU first
        processed = self._process_obs_jit(self._extract_obs(obs_jax))
        # Single device_get for the processed obs only
        obs_np = np.asarray(processed)
        key = 'embedding' if self._use_embedding else 'image'
        return {
            key: obs_np,
            'reward': np.float32(reward),
            'is_first': is_first,
            'is_last': is_last,
            'is_terminal': is_terminal,
        }

    def step(self, action):
        if action['reset'] or self._done:
            reset_key = self._get_rng_key()
            self._obs, self._state = self._reset_fn(reset_key, self._env_params)
            self._done = False
            return self._to_numpy_result(self._obs, 0.0, True, False, False)
        
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
        return self._to_numpy_result(
            self._obs, reward_val, False, self._done, self._done
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
                 embedding_dim=256, use_embedding=True, seed=42):
        self._env_name = env_name
        self._num_envs = num_envs
        self._embedding_dim = embedding_dim
        self._use_embedding = use_embedding

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
        return {
            key: np.asarray(processed_np, dtype=np.float32),
            'reward': np.asarray(rewards_np, dtype=np.float32),
            'is_first': is_first,
            'is_last': is_last,
            'is_terminal': is_last.copy(),
        }

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
    # Build P2E config from args
    p2e_config = {}
    if args is not None:
        p2e_config = {
            'plan2explore': getattr(args, 'plan2explore', False),
            'disag_models': getattr(args, 'disag_models', 10),
            'disag_target': getattr(args, 'disag_target', 'stoch'),
            'expl_intr_scale': getattr(args, 'expl_intr_scale', 1.0),
            'expl_extr_scale': getattr(args, 'expl_extr_scale', 0.0),
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
    """
    length = config.batch_length + config.replay_context
    capacity = int(config.replay.size)

    # Create selector based on args
    selector = None
    eviction = 'fifo'
    if args is not None:
        selector = make_selector(args, capacity, seed=config.seed)
        # Use reservoir eviction if flag is set
        if getattr(args, 'reservoir_sampling', False):
            eviction = 'reservoir'

    return embodied.replay.Replay(
        length=length,
        capacity=capacity,
        directory=directory,
        online=config.replay.online,
        chunksize=config.replay.chunksize,
        selector=selector,
        eviction=eviction,
    )


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
    """Load DreamerV3 config from YAML and merge with args."""
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
    # Default to 4 envs for JAX-based environments (not 64 from defaults)
    num_envs = int(args.envs) if getattr(args, 'envs', None) is not None else 4
    batch_length = getattr(args, 'batch_length', 32)
    run_overrides = {
        'logdir': f'{args.logdir}/craftax_{tag}',
        'seed': args.seed,
        'batch_size': args.batch_size,
        'batch_length': batch_length,
        'replay_context': 0,  # Disable replay context to avoid needing dyn/deter and dyn/stoch in replay
        'run': {
            'steps': int(args.steps),
            'log_every': 1000,
            'save_every': 10000,
            'report_every': 50000,
            'train_ratio': 64.0,
            'envs': num_envs,
            'debug': False,  # Disable debug mode for performance
        },
        'jax': {
            'prealloc': False,  # Allocate GPU memory on demand, not all at once
        },
        'replay': {
            'size': int(args.replay_capacity),
        },
    }
    if getattr(args, 'eval_envs', None) is not None:
        run_overrides['run']['eval_envs'] = int(args.eval_envs)
    config = config.update(run_overrides)
    return config, tag


def train_single(make_env, config, args, env_name=None):
    """Train DreamerV3 on a single environment.

    Args:
        make_env: Callable(seed: int) -> embodied.Env that creates one env instance.
        env_name: If provided, use VectorCraftaxEnv for GPU-vectorised envs.
    """
    np.random.seed(config.seed)
    logdir = elements.Path(config.logdir)
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

    def logfn(tran, worker):
        episode = episodes[worker]
        tran['is_first'] and episode.reset()
        episode.add('score', tran['reward'], agg='sum')
        episode.add('length', 1, agg='sum')
        if tran['is_last']:
            result = episode.result()
            score = result.pop('score')
            length = result.pop('length')
            logger.add({'score': score, 'length': length}, prefix='episode')
            if online is not None:
                online.update(task_id=0, step=int(step.value), score=float(score), length=int(length))
            logger.write()

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

    cp = elements.Checkpoint(logdir / 'ckpt')
    cp.step = step
    cp.agent = agent
    cp.replay = replay
    cp.load_or_save()

    print('Start training loop')
    policy = lambda carry, obs: agent.policy(carry, obs, mode='train')
    driver.reset(agent.init_policy)

    # Use larger collect steps to reduce collect<->train switching overhead.
    # With N vectorised envs each _step produces N transitions, so
    # collect_steps ~= num_envs * K gives K actual _step calls per collect.
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
        if step.value % 10000 < 10:
            cp.save()

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

    cp = elements.Checkpoint(logdir / 'ckpt')
    cp.step = total_step
    cp.agent = agent
    cp.replay = replay
    cp.load_or_save()

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
    collect_steps = max(num_envs * 4, 64)

    while rep < args.num_task_repeats:
        while task_id < num_tasks:
            print(f"\n=== Task {task_id + 1}/{num_tasks}, Rep {rep + 1}/{args.num_task_repeats} ===\n")

            make_task_env = make_envs[task_id]
            step = elements.Counter()
            episodes = collections.defaultdict(elements.Agg)

            def logfn(tran, worker):
                episode = episodes[worker]
                tran['is_first'] and episode.reset()
                episode.add('score', tran['reward'], agg='sum')
                episode.add('length', 1, agg='sum')
                if tran['is_last']:
                    result = episode.result()
                    score = result.pop('score')
                    length = result.pop('length')
                    logger.add({'score': score, 'length': length, 'task': task_id}, prefix='episode')
                    if online is not None:
                        online.update(task_id=task_id, step=int(total_step.value), score=float(score), length=int(length))
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
                if step.value % 10000 < 10:
                    cp.save()

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
