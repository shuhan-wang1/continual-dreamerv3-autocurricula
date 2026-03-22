"""Episodic intrinsic reward: spatial-counting + craft-novelty.

Implements the design from docs/intrisic_reward_design.md:
  - Spatial reward: 1/sqrt(visit_count) per tile-hash (Eq.1-3)
  - Craft reward: binary novelty on discretized inventory (Eq.4-5)
  - Combined: alpha_spatial * norm(r_spatial) + alpha_craft * norm(r_craft) + alpha_e * r_extr
"""

import math
import numpy as np


# Craftax symbolic observation layout
OBS_ROWS = 9
OBS_COLS = 11
CHANNELS_PER_TILE = 83
K_BLOCK = 37          # one-hot block-type channels per tile
MAP_SIZE = OBS_ROWS * OBS_COLS * CHANNELS_PER_TILE  # 8217
INV_START = MAP_SIZE  # inventory vector starts at index 8217
INV_LEN = 51          # full inventory length


class SpatialNoveltyTracker:
    """Episodic spatial visit-count tracker.

    Hashes the 9x11 tile map by argmax of block-type channels,
    returns 1/sqrt(count) as intrinsic reward.
    """

    def __init__(self, max_entries=50000):
        self._counts = {}
        self._max_entries = max_entries

    def reset(self):
        self._counts.clear()

    def compute(self, raw_obs):
        """Compute spatial intrinsic reward from raw observation.

        Args:
            raw_obs: 1D float array of length >= MAP_SIZE.
        Returns:
            float intrinsic reward.
        """
        tiles = raw_obs[:MAP_SIZE].reshape(OBS_ROWS, OBS_COLS, CHANNELS_PER_TILE)
        # argmax of first K_BLOCK channels per tile -> block-type ID
        block_ids = tuple(int(tiles[r, c, :K_BLOCK].argmax())
                          for r in range(OBS_ROWS)
                          for c in range(OBS_COLS))
        count = self._counts.get(block_ids, 0) + 1
        if len(self._counts) < self._max_entries or block_ids in self._counts:
            self._counts[block_ids] = count
        return 1.0 / math.sqrt(count)


class CraftNoveltyTracker:
    """Episodic craft-novelty tracker.

    Discretizes the inventory vector (round to 1 decimal),
    returns 1.0 on first visit, 0.0 otherwise.
    """

    def __init__(self):
        self._seen = set()

    def reset(self):
        self._seen.clear()

    def compute(self, raw_obs):
        """Compute craft novelty reward from raw observation.

        Args:
            raw_obs: 1D float array of length >= INV_START + INV_LEN.
        Returns:
            float intrinsic reward (0.0 or 1.0).
        """
        inv = raw_obs[INV_START:INV_START + INV_LEN]
        sig = tuple(int(float(v) * 10) for v in inv)
        if sig not in self._seen:
            self._seen.add(sig)
            return 1.0
        return 0.0


class AdaptiveNormalizer:
    """Cross-episode EMA normalizer.

    Normalizes intrinsic reward so that E[norm(r_intr)] ~ E[|r_extr|],
    making alpha weights act as true relative importance.
    """

    def __init__(self, decay=0.99, warmup=100):
        self._decay = decay
        self._warmup = warmup
        self._ema_intr = 0.0
        self._ema_extr = 0.0
        self._steps = 0

    def normalize(self, intr_val, extr_val):
        d = self._decay
        self._ema_intr = d * self._ema_intr + (1 - d) * abs(intr_val)
        self._ema_extr = d * self._ema_extr + (1 - d) * abs(extr_val)
        self._steps += 1
        if self._steps < self._warmup or self._ema_intr < 1e-8:
            return intr_val
        ratio = self._ema_extr / self._ema_intr
        ratio = min(ratio, 100.0)  # prevent extreme scaling
        return intr_val * ratio


class IntrinsicRewardShaper:
    """Per-env episodic intrinsic reward with adaptive normalization.

    Manages independent tracker state for each env index (vectorized envs).

    Combined reward:
        r = alpha_spatial * norm(r_spatial) + alpha_craft * norm(r_craft) + alpha_e * r_extr
    """

    def __init__(self, alpha_spatial=0.1, alpha_craft=0.3, alpha_e=1.0):
        self.alpha_spatial = alpha_spatial
        self.alpha_craft = alpha_craft
        self.alpha_e = alpha_e
        self._spatial = {}   # env_id -> SpatialNoveltyTracker
        self._craft = {}     # env_id -> CraftNoveltyTracker
        self._norm_sp = {}   # env_id -> AdaptiveNormalizer
        self._norm_cr = {}   # env_id -> AdaptiveNormalizer
        # Logging: last computed values per env
        self.last_r_spatial = {}
        self.last_r_craft = {}

    def _ensure_env(self, env_id):
        if env_id not in self._spatial:
            self._spatial[env_id] = SpatialNoveltyTracker()
            self._craft[env_id] = CraftNoveltyTracker()
            self._norm_sp[env_id] = AdaptiveNormalizer()
            self._norm_cr[env_id] = AdaptiveNormalizer()

    def reset_episode(self, env_id):
        self._ensure_env(env_id)
        self._spatial[env_id].reset()
        self._craft[env_id].reset()

    def shape_reward(self, raw_obs, extrinsic_reward, env_id):
        """Compute combined reward from raw observation and extrinsic reward.

        Args:
            raw_obs: 1D numpy array (length >= 8268).
            extrinsic_reward: float, environment reward.
            env_id: int, environment index.
        Returns:
            float, shaped reward.
        """
        self._ensure_env(env_id)
        r_sp = self._spatial[env_id].compute(raw_obs)
        r_cr = self._craft[env_id].compute(raw_obs)
        self.last_r_spatial[env_id] = r_sp
        self.last_r_craft[env_id] = r_cr
        norm_sp = self._norm_sp[env_id].normalize(r_sp, extrinsic_reward)
        norm_cr = self._norm_cr[env_id].normalize(r_cr, extrinsic_reward)
        return (self.alpha_spatial * norm_sp
                + self.alpha_craft * norm_cr
                + self.alpha_e * extrinsic_reward)

    def shape_rewards_batch(self, raw_obs_batch, rewards_np, is_first):
        """Shape rewards for a batch of vectorized envs.

        Args:
            raw_obs_batch: numpy array (N, obs_dim), raw observations.
            rewards_np: numpy array (N,), extrinsic rewards.
            is_first: numpy bool array (N,), episode start flags.
        Returns:
            numpy array (N,), shaped rewards (modifies rewards_np in-place).
        """
        n = len(rewards_np)
        # TODO: vectorize this loop for performance; requires refactoring
        # per-env stateful trackers (spatial counts, craft novelty sets) to
        # operate on batched arrays instead of individual Python dicts/sets.
        for i in range(n):
            if is_first[i]:
                self.reset_episode(i)
            rewards_np[i] = self.shape_reward(
                raw_obs_batch[i].reshape(-1),
                float(rewards_np[i]), i)
        return rewards_np
