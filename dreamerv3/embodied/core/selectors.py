import collections
import threading

import numpy as np


class Fifo:

  def __init__(self):
    self.queue = collections.deque()

  def __call__(self):
    return self.queue[0]

  def __len__(self):
    return len(self.queue)

  def __setitem__(self, key, stepids):
    self.queue.append(key)

  def __delitem__(self, key):
    if self.queue[0] == key:
      self.queue.popleft()
    else:
      # This is very slow but typically not used.
      self.queue.remove(key)


class Reservoir:
  """Reservoir sampling selector - random eviction instead of FIFO.

  Implements Algorithm 2 from https://arxiv.org/pdf/1902.10486.pdf
  Each item has equal probability of being evicted when capacity is reached.
  """

  def __init__(self, capacity, seed=0):
    self.capacity = capacity
    self.indices = {}
    self.keys = []
    self.rng = np.random.default_rng(seed)
    self.lock = threading.Lock()
    self.total_seen = 0

  def __len__(self):
    return len(self.keys)

  def __call__(self):
    with self.lock:
      index = self.rng.integers(0, len(self.keys)).item()
      return self.keys[index]

  def __setitem__(self, key, stepids):
    with self.lock:
      self.total_seen += 1
      if len(self.keys) < self.capacity:
        # Buffer not full yet, just add
        self.indices[key] = len(self.keys)
        self.keys.append(key)
      else:
        # Reservoir sampling: randomly decide whether to include
        j = self.rng.integers(0, self.total_seen)
        if j < self.capacity:
          # Replace item at position j
          old_key = self.keys[j]
          del self.indices[old_key]
          self.keys[j] = key
          self.indices[key] = j

  def __delitem__(self, key):
    with self.lock:
      if key not in self.indices:
        return
      assert 2 <= len(self), len(self)
      index = self.indices.pop(key)
      last = self.keys.pop()
      if index != len(self.keys):
        self.keys[index] = last
        self.indices[last] = index


class Uniform:

  def __init__(self, seed=0):
    self.indices = {}
    self.keys = []
    self.rng = np.random.default_rng(seed)
    self.lock = threading.Lock()

  def __len__(self):
    return len(self.keys)

  def __call__(self):
    with self.lock:
      index = self.rng.integers(0, len(self.keys)).item()
      return self.keys[index]

  def __setitem__(self, key, stepids):
    with self.lock:
      self.indices[key] = len(self.keys)
      self.keys.append(key)

  def __delitem__(self, key):
    with self.lock:
      assert 2 <= len(self), len(self)
      index = self.indices.pop(key)
      last = self.keys.pop()
      if index != len(self.keys):
        self.keys[index] = last
        self.indices[last] = index


class Recency:

  def __init__(self, uprobs, seed=0):
    assert uprobs[0] >= uprobs[-1], uprobs
    self.uprobs = uprobs
    self.tree = self._build(uprobs)
    self.rng = np.random.default_rng(seed)
    self.step = 0
    self.steps = {}
    self.items = {}

  def __len__(self):
    return len(self.items)

  def __call__(self):
    for retry in range(10):
      try:
        age = self._sample(self.tree, self.rng)
        if len(self.items) < len(self.uprobs):
          age = int(age / len(self.uprobs) * len(self.items))
        return self.items[self.step - 1 - age]
      except KeyError:
        # Item might have been deleted very recently.
        if retry < 9:
          import time
          time.sleep(0.01)
        else:
          raise

  def __setitem__(self, key, stepids):
    self.steps[key] = self.step
    self.items[self.step] = key
    self.step += 1

  def __delitem__(self, key):
    step = self.steps.pop(key)
    del self.items[step]

  def _sample(self, tree, rng, bfactor=16):
    path = []
    for level, prob in enumerate(tree):
      p = prob
      for segment in path:
        p = p[segment]
      index = rng.choice(len(segment), p=p)
      path.append(index)
    index = sum(
        index * bfactor ** (len(tree) - level - 1)
        for level, index in enumerate(path))
    return index

  def _build(self, uprobs, bfactor=16):
    assert np.isfinite(uprobs).all(), uprobs
    assert (uprobs >= 0).all(), uprobs
    depth = int(np.ceil(np.log(len(uprobs)) / np.log(bfactor)))
    size = bfactor ** depth
    uprobs = np.concatenate([uprobs, np.zeros(size - len(uprobs))])
    tree = [uprobs]
    for level in reversed(range(depth - 1)):
      tree.insert(0, tree[0].reshape((-1, bfactor)).sum(-1))
    for level, prob in enumerate(tree):
      prob = prob.reshape([bfactor] * (1 + level))
      total = prob.sum(-1, keepdims=True)
      with np.errstate(divide='ignore', invalid='ignore'):
        tree[level] = np.where(total, prob / total, prob)
    return tree


class Prioritized:

  def __init__(
      self, exponent=1.0, initial=1.0, zero_on_sample=False,
      maxfrac=0.0, branching=16, seed=0):
    assert 0 <= maxfrac <= 1, maxfrac
    self.exponent = float(exponent)
    self.initial = float(initial)
    self.zero_on_sample = zero_on_sample
    self.maxfrac = maxfrac
    self.tree = SampleTree(branching, seed)
    self.prios = collections.defaultdict(lambda: self.initial)
    self.stepitems = collections.defaultdict(list)
    self.items = {}

  def prioritize(self, stepids, priorities):
    if not isinstance(stepids[0], bytes):
      stepids = [x.tobytes() for x in stepids]
    for stepid, priority in zip(stepids, priorities):
      try:
        self.prios[stepid] = priority
      except KeyError:
        print('Ignoring priority update for removed time step.')
    items = []
    for stepid in stepids:
      items += self.stepitems[stepid]
    for key in list(set(items)):
      try:
        self.tree.update(key, self._aggregate(key))
      except KeyError:
        print('Ignoring tree update for removed time step.')

  def __len__(self):
    return len(self.items)

  def __call__(self):
    key = self.tree.sample()
    if self.zero_on_sample:
      zeros = [0.0] * len(self.items[key])
      self.prioritize(self.items[key], zeros)
    return key

  def __setitem__(self, key, stepids):
    if not isinstance(stepids[0], bytes):
      stepids = [x.tobytes() for x in stepids]
    self.items[key] = stepids
    [self.stepitems[stepid].append(key) for stepid in stepids]
    self.tree.insert(key, self._aggregate(key))

  def __delitem__(self, key):
    self.tree.remove(key)
    stepids = self.items.pop(key)
    for stepid in stepids:
      stepitems = self.stepitems[stepid]
      stepitems.remove(key)
      if not stepitems:
        del self.stepitems[stepid]
        del self.prios[stepid]

  def _aggregate(self, key):
    # Both list comprehensions in this function are a performance bottleneck
    # because they are called very often.
    prios = [self.prios[stepid] for stepid in self.items[key]]
    if self.exponent != 1.0:
      prios = [x ** self.exponent for x in prios]
    mean = sum(prios) / len(prios)
    if self.maxfrac:
      return self.maxfrac * max(prios) + (1 - self.maxfrac) * mean
    else:
      return mean


class Recent:
  """Samples from the N most recent items uniformly.

  Used for the 50:50 sampling strategy in Continual-Dreamer:
  half from random buffer samples, half from recent experience.
  """

  def __init__(self, window_size=1000, seed=0):
    self.window_size = window_size
    self.indices = {}
    self.keys = []  # All keys in insertion order
    self.rng = np.random.default_rng(seed)
    self.lock = threading.Lock()

  def __len__(self):
    return len(self.keys)

  def __call__(self):
    with self.lock:
      # Sample uniformly from the most recent window_size items
      recent_count = min(len(self.keys), self.window_size)
      if recent_count == 0:
        raise IndexError("Empty selector")
      # Sample from the end of the list (most recent items)
      start_idx = len(self.keys) - recent_count
      index = self.rng.integers(start_idx, len(self.keys)).item()
      return self.keys[index]

  def __setitem__(self, key, stepids):
    with self.lock:
      self.indices[key] = len(self.keys)
      self.keys.append(key)

  def __delitem__(self, key):
    with self.lock:
      if key not in self.indices:
        return
      assert 2 <= len(self), len(self)
      index = self.indices.pop(key)
      last = self.keys.pop()
      if index != len(self.keys):
        self.keys[index] = last
        self.indices[last] = index


class RewardWeighted:
  """Samples episodes weighted by cumulative reward (softmax).

  Higher reward episodes have higher probability of being sampled.
  """

  def __init__(self, temperature=1.0, seed=0):
    self.temperature = temperature
    self.indices = {}
    self.keys = []
    self.rewards = []  # Store cumulative rewards
    self.rng = np.random.default_rng(seed)
    self.lock = threading.Lock()

  def __len__(self):
    return len(self.keys)

  def __call__(self):
    with self.lock:
      if len(self.keys) == 0:
        raise IndexError("Empty selector")
      rewards = np.array(self.rewards)
      # Softmax with temperature
      scaled = rewards / self.temperature
      e_r = np.exp(scaled - np.max(scaled))
      probs = e_r / e_r.sum()
      index = self.rng.choice(len(self.keys), p=probs)
      return self.keys[index]

  def __setitem__(self, key, stepids):
    with self.lock:
      self.indices[key] = len(self.keys)
      self.keys.append(key)
      self.rewards.append(0.0)  # Initialize with 0, will be updated

  def __delitem__(self, key):
    with self.lock:
      if key not in self.indices:
        return
      assert 2 <= len(self), len(self)
      index = self.indices.pop(key)
      last_key = self.keys.pop()
      last_reward = self.rewards.pop()
      if index != len(self.keys):
        self.keys[index] = last_key
        self.rewards[index] = last_reward
        self.indices[last_key] = index

  def update_reward(self, key, reward):
    """Update the cumulative reward for an episode."""
    with self.lock:
      if key in self.indices:
        self.rewards[self.indices[key]] = reward


class PrivilegedNoveltyLearnabilityRecency:
  """Privileged NLR replay selector — uses per-achievement decomposition.

  **WARNING**: This selector uses privileged environment information
  (the 67-dimensional achievement vector) that the agent's policy network
  never observes.  Use ``--nlr_privileged_sampling`` to enable.

  Splits the replay buffer sampling into three pools:
    - **Novel pool** (default 35%): Trajectories containing achievements with
      low success rate.  Priority ∝ 1 / (success_rate + ε) so that rarely-
      accomplished achievements are revisited until they become routine.
    - **Learnable pool** (default 35%): Trajectories whose episodic reward
      exceeds the running mean reward (advantage-based, GRPO-style).
    - **Recent pool** (default 30%): Triangular recency-weighted sampling
      over a sliding window of the most recent items.

  Trajectories can appear in multiple pools (duplication allowed).
  """

  def __init__(
      self,
      novel_frac=0.35,
      learnable_frac=0.35,
      recent_frac=0.30,
      recent_window=1000,
      num_achievements=67,
      reward_ema_decay=0.99,
      novelty_eps=0.01,
      novelty_temp=1.0,
      learnability_temp=1.0,
      seed=0,
  ):
    assert abs(novel_frac + learnable_frac + recent_frac - 1.0) < 1e-6, \
        f'Fractions must sum to 1, got {novel_frac + learnable_frac + recent_frac}'
    self.novel_frac = novel_frac
    self.learnable_frac = learnable_frac
    self.recent_frac = recent_frac
    self.recent_window = recent_window
    self.num_achievements = num_achievements
    self.novelty_eps = novelty_eps
    self.novelty_temp = novelty_temp
    self.learnability_temp = learnability_temp
    self.rng = np.random.default_rng(seed)
    self.lock = threading.Lock()

    # ---- Item storage ----
    self.keys = []           # all item keys in insertion order
    self.indices = {}        # key -> position in self.keys
    self.insertion_order = {}  # key -> monotonic counter for recency

    # ---- Novel pool ----
    self.novel_keys = []     # keys eligible for novelty sampling
    self.novel_scores = []   # unnormalised novelty priority per key

    # ---- Learnable pool ----
    self.learn_keys = []     # keys eligible for learnability sampling
    self.learn_scores = []   # unnormalised learnability priority per key

    # ---- Per-achievement success-rate tracking ----
    self.achievement_counts = np.zeros(num_achievements, dtype=np.float64)
    self.achievement_successes = np.zeros(num_achievements, dtype=np.float64)

    # ---- Reward baseline (exponential moving average) ----
    self.reward_ema_decay = reward_ema_decay
    self.reward_ema = 0.0
    self.reward_ema_initialised = False

    self._insert_counter = 0

  # ------------------------------------------------------------------
  # Public API: update per-episode statistics
  # ------------------------------------------------------------------
  def update_episode_stats(self, key, achievements, reward):
    """Feed per-episode metadata to update novelty and learnability pools.

    Must be called once at the end of every episode for the corresponding
    item *key* that was inserted via ``__setitem__``.

    Parameters
    ----------
    key : int
        Item key returned by the replay buffer's ``_insert``.
    achievements : np.ndarray[bool]  (num_achievements,)
        Binary vector of achievements unlocked in this episode.
    reward : float
        Cumulative episodic reward.
    """
    with self.lock:
      if key not in self.indices:
        return  # item was already evicted

      achievements = np.asarray(achievements, dtype=bool)

      # --- Update achievement success rates ---
      self.achievement_counts += 1  # every episode counts as an attempt
      self.achievement_successes += achievements.astype(np.float64)

      # --- Update reward EMA ---
      if not self.reward_ema_initialised:
        self.reward_ema = float(reward)
        self.reward_ema_initialised = True
      else:
        self.reward_ema = (self.reward_ema_decay * self.reward_ema
                           + (1.0 - self.reward_ema_decay) * float(reward))

      # --- Compute novelty score ---
      # Success rate for each achieved accomplishment
      success_rates = np.where(
          self.achievement_counts > 0,
          self.achievement_successes / self.achievement_counts,
          0.0,
      )
      # Novelty = mean inverse success rate across *achieved* items
      achieved_mask = achievements.astype(bool)
      if achieved_mask.any():
        # The lower the success rate, the higher the novelty
        inv_rates = 1.0 / (success_rates[achieved_mask] + self.novelty_eps)
        novelty_score = float(np.mean(inv_rates))
      else:
        novelty_score = 0.0

      # --- Compute learnability score ---
      # Advantage = reward - baseline.  Only positive advantages qualify.
      advantage = float(reward) - self.reward_ema
      learnability_score = max(0.0, advantage)

      # --- Insert into pools ---
      if novelty_score > 0:
        self.novel_keys.append(key)
        self.novel_scores.append(novelty_score)
      if learnability_score > 0:
        self.learn_keys.append(key)
        self.learn_scores.append(learnability_score)

  def get_achievement_success_rates(self):
    """Return current per-achievement success rates (for logging)."""
    with self.lock:
      mask = self.achievement_counts > 0
      rates = np.zeros(self.num_achievements)
      rates[mask] = self.achievement_successes[mask] / self.achievement_counts[mask]
      return rates

  # ------------------------------------------------------------------
  # Selector interface (called by Replay)
  # ------------------------------------------------------------------
  def __len__(self):
    return len(self.keys)

  def __call__(self):
    """Sample an item key according to the NLR strategy."""
    with self.lock:
      if len(self.keys) == 0:
        raise IndexError('Privileged NLR selector is empty')

      # Choose which pool to sample from
      r = self.rng.random()
      if r < self.novel_frac and len(self.novel_keys) > 0:
        return self._sample_novel()
      elif r < self.novel_frac + self.learnable_frac and len(self.learn_keys) > 0:
        return self._sample_learnable()
      else:
        return self._sample_recent()

  def __setitem__(self, key, stepids):
    """Register a new item (called by Replay._insert)."""
    with self.lock:
      self.indices[key] = len(self.keys)
      self.keys.append(key)
      self._insert_counter += 1
      self.insertion_order[key] = self._insert_counter

  def __delitem__(self, key):
    """Remove an evicted item from all pools."""
    with self.lock:
      if key not in self.indices:
        return
      assert 2 <= len(self), len(self)

      # Remove from main list
      idx = self.indices.pop(key)
      last = self.keys.pop()
      if idx != len(self.keys):
        self.keys[idx] = last
        self.indices[last] = idx

      # Remove from insertion_order
      self.insertion_order.pop(key, None)

      # Remove from novel pool
      self._remove_from_pool(key, self.novel_keys, self.novel_scores)

      # Remove from learnable pool
      self._remove_from_pool(key, self.learn_keys, self.learn_scores)

  # ------------------------------------------------------------------
  # Internal sampling methods
  # ------------------------------------------------------------------
  def _sample_novel(self):
    """Sample from the novel pool, weighted by novelty score."""
    scores = np.array(self.novel_scores, dtype=np.float64)
    # Apply temperature scaling
    scores = scores ** (1.0 / self.novelty_temp)
    total = scores.sum()
    if total <= 0:
      # Fallback to recent if all scores are zero
      return self._sample_recent()
    probs = scores / total
    idx = self.rng.choice(len(self.novel_keys), p=probs)
    return self.novel_keys[idx]

  def _sample_learnable(self):
    """Sample from the learnable pool, weighted by advantage."""
    scores = np.array(self.learn_scores, dtype=np.float64)
    # Apply temperature scaling
    scores = scores ** (1.0 / self.learnability_temp)
    total = scores.sum()
    if total <= 0:
      return self._sample_recent()
    probs = scores / total
    idx = self.rng.choice(len(self.learn_keys), p=probs)
    return self.learn_keys[idx]

  def _sample_recent(self):
    """Sample from recent items with triangular weighting."""
    n = len(self.keys)
    if n == 0:
      raise IndexError('Privileged NLR selector is empty (recent)')
    window = min(n, self.recent_window)
    # Triangular distribution: most recent items get highest weight
    weights = np.linspace(1.0, 0.0, window, endpoint=False)
    total = weights.sum()
    if total <= 0:
      # Uniform fallback
      idx = self.rng.integers(max(0, n - window), n).item()
      return self.keys[idx]
    probs = weights / total
    # Sample within the recent window (end of self.keys)
    start = n - window
    local_idx = self.rng.choice(window, p=probs)
    return self.keys[start + local_idx]

  def _remove_from_pool(self, key, pool_keys, pool_scores):
    """Remove a key from a (keys, scores) pool."""
    try:
      idx = pool_keys.index(key)
      pool_keys.pop(idx)
      pool_scores.pop(idx)
    except ValueError:
      pass  # key not in this pool


class PrivilegedNoveltyLearnabilityUniform(PrivilegedNoveltyLearnabilityRecency):
  """Privileged NLU replay selector — uses per-achievement decomposition.

  **WARNING**: Uses privileged environment information.  See
  ``PrivilegedNoveltyLearnabilityRecency`` docstring.

  Identical to privileged NLR except the third pool samples **uniformly**
  from the entire buffer instead of triangular recency-weighted sampling.
  """

  def _sample_recent(self):
    """Sample uniformly from the entire buffer (overrides NLR triangular)."""
    n = len(self.keys)
    if n == 0:
      raise IndexError('Privileged NLU selector is empty')
    idx = self.rng.integers(0, n).item()
    return self.keys[idx]


# ====================================================================
# Non-privileged NLR / NLU — 2D (reward, length) grid novelty
# ====================================================================

class NoveltyLearnabilityRecency:
  """Non-privileged NLR replay selector using Bayesian 2D grid novelty.

  Replaces the privileged per-achievement novelty scoring with a
  non-privileged approach: a quantile-adaptive 2D histogram over
  (episode_length, episodic_reward) space.  Novelty is defined as the
  inverse bin count weighted by a sigmoid × linear prior that favours
  medium-to-high reward episodes.

  **No privileged information is used.**  The selector only observes
  the same scalar reward and episode length available to the agent.

  Novelty score for trajectory k in bin b:
      score(k) = sigma((R_b - R_min) / beta) * R_b  /  (n_b + eps)

  where R_b is the bin midpoint reward, R_min is the 20th-percentile
  reward, beta is an adaptive sharpness parameter, and n_b is the
  count of episodes in bin b.

  Pools:
    - **Novel pool** (default 35%): Bayesian 2D grid rarity scoring.
    - **Learnable pool** (default 35%): Above-baseline reward (unchanged).
    - **Recent pool** (default 30%): Triangular recency-weighted sampling.
  """

  def __init__(
      self,
      novel_frac=0.35,
      learnable_frac=0.35,
      recent_frac=0.30,
      recent_window=1000,
      reward_ema_decay=0.99,
      novelty_temp=1.0,
      learnability_temp=1.0,
      grid_reward_bins=5,
      grid_length_bins=10,
      grid_recompute_every=500,
      grid_prior_percentile=0.20,
      grid_eps=0.01,
      seed=0,
  ):
    assert abs(novel_frac + learnable_frac + recent_frac - 1.0) < 1e-6, \
        f'Fractions must sum to 1, got {novel_frac + learnable_frac + recent_frac}'
    self.novel_frac = novel_frac
    self.learnable_frac = learnable_frac
    self.recent_frac = recent_frac
    self.recent_window = recent_window
    self.novelty_temp = novelty_temp
    self.learnability_temp = learnability_temp
    self.rng = np.random.default_rng(seed)
    self.lock = threading.Lock()

    # ---- Item storage ----
    self.keys = []           # all item keys in insertion order
    self.indices = {}        # key -> position in self.keys
    self.insertion_order = {}  # key -> monotonic counter for recency

    # ---- Novel pool ----
    self.novel_keys = []     # keys eligible for novelty sampling
    self.novel_scores = []   # unnormalised novelty priority per key

    # ---- Learnable pool ----
    self.learn_keys = []     # keys eligible for learnability sampling
    self.learn_scores = []   # unnormalised learnability priority per key

    # ---- Reward baseline (EMA) ----
    self.reward_ema_decay = reward_ema_decay
    self.reward_ema = 0.0
    self.reward_ema_initialised = False

    # ---- 2D grid novelty ----
    self.grid_reward_bins = grid_reward_bins
    self.grid_length_bins = grid_length_bins
    self.grid_recompute_every = grid_recompute_every
    self.grid_prior_percentile = grid_prior_percentile
    self.grid_eps = grid_eps

    # Per-episode metadata for grid computation
    self._episode_rewards = []   # reward for each key (parallel to self.keys)
    self._episode_lengths = []   # length for each key (parallel to self.keys)
    self._key_to_bin = {}        # key -> (len_bin, rew_bin)

    # Grid state
    self._grid_counts = np.zeros(
        (grid_length_bins, grid_reward_bins), dtype=np.float64)
    self._reward_edges = None    # quantile edges, shape (grid_reward_bins + 1,)
    self._length_edges = None    # quantile edges, shape (grid_length_bins + 1,)
    self._reward_midpoints = None  # shape (grid_reward_bins,)
    self._r_min = 1.0            # prior threshold (adaptive)
    self._beta = 1.0             # prior sharpness (adaptive)
    self._episodes_since_recompute = 0
    self._total_episodes = 0

    self._insert_counter = 0

  # ------------------------------------------------------------------
  # 2D grid management
  # ------------------------------------------------------------------
  def _sigmoid(self, x):
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )

  def _recompute_grid(self):
    """Recompute quantile bin edges from current buffer."""
    rewards = np.array(self._episode_rewards, dtype=np.float64)
    lengths = np.array(self._episode_lengths, dtype=np.float64)
    n = len(rewards)
    if n < 10:
      return  # not enough data to compute meaningful quantiles

    # Reward quantile edges
    r_quantiles = np.linspace(0, 1, self.grid_reward_bins + 1)
    self._reward_edges = np.quantile(rewards, r_quantiles)
    # Ensure strictly increasing edges by adding small perturbation
    for i in range(1, len(self._reward_edges)):
      if self._reward_edges[i] <= self._reward_edges[i - 1]:
        self._reward_edges[i] = self._reward_edges[i - 1] + 1e-6

    # Length quantile edges
    l_quantiles = np.linspace(0, 1, self.grid_length_bins + 1)
    self._length_edges = np.quantile(lengths, l_quantiles)
    for i in range(1, len(self._length_edges)):
      if self._length_edges[i] <= self._length_edges[i - 1]:
        self._length_edges[i] = self._length_edges[i - 1] + 1e-6

    # Compute bin midpoints for reward axis
    self._reward_midpoints = 0.5 * (
        self._reward_edges[:-1] + self._reward_edges[1:])

    # Adaptive prior parameters
    self._r_min = float(np.quantile(rewards, self.grid_prior_percentile))
    r_50 = float(np.quantile(rewards, 0.50))
    self._beta = max(0.1, (r_50 - self._r_min) / 4.0)

    # Rebuild grid counts from scratch
    self._grid_counts = np.zeros(
        (self.grid_length_bins, self.grid_reward_bins), dtype=np.float64)
    self._key_to_bin.clear()

    for i, key in enumerate(self.keys):
      if i < len(self._episode_rewards):
        r = self._episode_rewards[i]
        l = self._episode_lengths[i]
        lb = self._get_length_bin(l)
        rb = self._get_reward_bin(r)
        self._key_to_bin[key] = (lb, rb)
        self._grid_counts[lb, rb] += 1

    # Recompute novel pool scores for all items
    self._rebuild_novel_pool()

  def _get_reward_bin(self, reward):
    """Map reward to bin index."""
    if self._reward_edges is None:
      return 0
    idx = int(np.searchsorted(self._reward_edges[1:], reward, side='right'))
    return min(idx, self.grid_reward_bins - 1)

  def _get_length_bin(self, length):
    """Map length to bin index."""
    if self._length_edges is None:
      return 0
    idx = int(np.searchsorted(self._length_edges[1:], length, side='right'))
    return min(idx, self.grid_length_bins - 1)

  def _compute_novelty_score(self, reward_bin):
    """Compute Bayesian novelty score for a given grid cell.

    score = sigmoid((R_midpoint - R_min) / beta) * R_midpoint / (n_b + eps)
    """
    if self._reward_midpoints is None:
      return 0.0
    r_mid = self._reward_midpoints[reward_bin]
    prior = float(self._sigmoid(
        np.float64((r_mid - self._r_min) / self._beta))) * r_mid
    return prior

  def _compute_novelty_score_for_cell(self, length_bin, reward_bin):
    """Full novelty score for a specific cell: prior / (count + eps)."""
    prior = self._compute_novelty_score(reward_bin)
    count = self._grid_counts[length_bin, reward_bin]
    if prior <= 0:
      return 0.0
    return prior / (count + self.grid_eps)

  def _rebuild_novel_pool(self):
    """Rebuild the entire novel pool from current grid state."""
    self.novel_keys.clear()
    self.novel_scores.clear()
    for key, (lb, rb) in self._key_to_bin.items():
      if key not in self.indices:
        continue
      score = self._compute_novelty_score_for_cell(lb, rb)
      if score > 0:
        self.novel_keys.append(key)
        self.novel_scores.append(score)

  # ------------------------------------------------------------------
  # Public API: update per-episode statistics
  # ------------------------------------------------------------------
  def update_episode_stats(self, key, episode_length, reward):
    """Feed per-episode metadata to update novelty and learnability pools.

    Parameters
    ----------
    key : int
        Item key returned by the replay buffer's ``_insert``.
    episode_length : int
        Number of timesteps in this episode.
    reward : float
        Cumulative episodic reward.
    """
    with self.lock:
      if key not in self.indices:
        return  # item was already evicted

      episode_length = int(episode_length)
      reward = float(reward)

      # Store per-episode metadata (parallel to self.keys)
      key_idx = self.indices[key]
      # Extend lists if needed (items may have been added before metadata arrives)
      while len(self._episode_rewards) <= key_idx:
        self._episode_rewards.append(0.0)
        self._episode_lengths.append(0)
      self._episode_rewards[key_idx] = reward
      self._episode_lengths[key_idx] = episode_length

      # --- Update reward EMA ---
      if not self.reward_ema_initialised:
        self.reward_ema = reward
        self.reward_ema_initialised = True
      else:
        self.reward_ema = (self.reward_ema_decay * self.reward_ema
                           + (1.0 - self.reward_ema_decay) * reward)

      # --- Update episode counter and trigger grid recompute ---
      self._total_episodes += 1
      self._episodes_since_recompute += 1

      if self._episodes_since_recompute >= self.grid_recompute_every:
        self._recompute_grid()
        self._episodes_since_recompute = 0
      else:
        # Incremental update: assign to current grid and update count
        if self._reward_edges is not None:
          lb = self._get_length_bin(episode_length)
          rb = self._get_reward_bin(reward)
          self._key_to_bin[key] = (lb, rb)
          self._grid_counts[lb, rb] += 1

          # Compute novelty score for this item
          score = self._compute_novelty_score_for_cell(lb, rb)
          if score > 0:
            self.novel_keys.append(key)
            self.novel_scores.append(score)

      # --- Compute learnability score ---
      advantage = reward - self.reward_ema
      learnability_score = max(0.0, advantage)
      if learnability_score > 0:
        self.learn_keys.append(key)
        self.learn_scores.append(learnability_score)

  # ------------------------------------------------------------------
  # Selector interface (called by Replay)
  # ------------------------------------------------------------------
  def __len__(self):
    return len(self.keys)

  def __call__(self):
    """Sample an item key according to the non-privileged NLR strategy."""
    with self.lock:
      if len(self.keys) == 0:
        raise IndexError('NLR selector is empty')

      # Choose which pool to sample from
      r = self.rng.random()
      if r < self.novel_frac and len(self.novel_keys) > 0:
        return self._sample_novel()
      elif r < self.novel_frac + self.learnable_frac and len(self.learn_keys) > 0:
        return self._sample_learnable()
      else:
        return self._sample_recent()

  def __setitem__(self, key, stepids):
    """Register a new item (called by Replay._insert)."""
    with self.lock:
      self.indices[key] = len(self.keys)
      self.keys.append(key)
      self._insert_counter += 1
      self.insertion_order[key] = self._insert_counter

  def __delitem__(self, key):
    """Remove an evicted item from all pools."""
    with self.lock:
      if key not in self.indices:
        return
      assert 2 <= len(self), len(self)

      idx = self.indices.pop(key)

      # Update grid counts
      if key in self._key_to_bin:
        lb, rb = self._key_to_bin.pop(key)
        self._grid_counts[lb, rb] = max(0, self._grid_counts[lb, rb] - 1)

      # Remove from main list (swap-and-pop)
      last = self.keys.pop()
      if idx != len(self.keys):
        self.keys[idx] = last
        self.indices[last] = idx
        # Also swap parallel metadata arrays
        if idx < len(self._episode_rewards) and len(self.keys) < len(self._episode_rewards):
          last_meta_idx = len(self.keys)  # was at end before pop
          if last_meta_idx < len(self._episode_rewards):
            self._episode_rewards[idx] = self._episode_rewards[last_meta_idx]
            self._episode_lengths[idx] = self._episode_lengths[last_meta_idx]
          self._episode_rewards.pop()
          self._episode_lengths.pop()
        elif idx < len(self._episode_rewards):
          # Key was the last element, just pop
          if len(self._episode_rewards) > len(self.keys):
            self._episode_rewards.pop()
            self._episode_lengths.pop()
      else:
        # Key was the last element
        if len(self._episode_rewards) > len(self.keys):
          self._episode_rewards.pop()
          self._episode_lengths.pop()

      # Remove from insertion_order
      self.insertion_order.pop(key, None)

      # Remove from novel pool
      self._remove_from_pool(key, self.novel_keys, self.novel_scores)

      # Remove from learnable pool
      self._remove_from_pool(key, self.learn_keys, self.learn_scores)

  # ------------------------------------------------------------------
  # Internal sampling methods
  # ------------------------------------------------------------------
  def _sample_novel(self):
    """Sample from the novel pool, weighted by novelty score."""
    scores = np.array(self.novel_scores, dtype=np.float64)
    scores = scores ** (1.0 / self.novelty_temp)
    total = scores.sum()
    if total <= 0:
      return self._sample_recent()
    probs = scores / total
    idx = self.rng.choice(len(self.novel_keys), p=probs)
    return self.novel_keys[idx]

  def _sample_learnable(self):
    """Sample from the learnable pool, weighted by advantage."""
    scores = np.array(self.learn_scores, dtype=np.float64)
    scores = scores ** (1.0 / self.learnability_temp)
    total = scores.sum()
    if total <= 0:
      return self._sample_recent()
    probs = scores / total
    idx = self.rng.choice(len(self.learn_keys), p=probs)
    return self.learn_keys[idx]

  def _sample_recent(self):
    """Sample from recent items with triangular weighting."""
    n = len(self.keys)
    if n == 0:
      raise IndexError('NLR selector is empty (recent)')
    window = min(n, self.recent_window)
    weights = np.linspace(1.0, 0.0, window, endpoint=False)
    total = weights.sum()
    if total <= 0:
      idx = self.rng.integers(max(0, n - window), n).item()
      return self.keys[idx]
    probs = weights / total
    start = n - window
    local_idx = self.rng.choice(window, p=probs)
    return self.keys[start + local_idx]

  def _remove_from_pool(self, key, pool_keys, pool_scores):
    """Remove a key from a (keys, scores) pool."""
    try:
      idx = pool_keys.index(key)
      pool_keys.pop(idx)
      pool_scores.pop(idx)
    except ValueError:
      pass

  def get_grid_stats(self):
    """Return grid statistics for logging."""
    with self.lock:
      return {
          'grid_counts': self._grid_counts.copy(),
          'reward_edges': self._reward_edges.copy() if self._reward_edges is not None else None,
          'length_edges': self._length_edges.copy() if self._length_edges is not None else None,
          'r_min': self._r_min,
          'beta': self._beta,
          'novel_pool_size': len(self.novel_keys),
          'learn_pool_size': len(self.learn_keys),
          'total_episodes': self._total_episodes,
      }


class NoveltyLearnabilityUniform(NoveltyLearnabilityRecency):
  """Non-privileged NLU replay selector.

  Identical to non-privileged NLR except the third pool samples
  **uniformly** from the entire buffer instead of triangular
  recency-weighted sampling.

  No privileged information is used.
  """

  def _sample_recent(self):
    """Sample uniformly from the entire buffer (overrides NLR triangular)."""
    n = len(self.keys)
    if n == 0:
      raise IndexError('NLU selector is empty')
    idx = self.rng.integers(0, n).item()
    return self.keys[idx]


class Mixture:

  def __init__(self, selectors, fractions, seed=0):
    assert set(selectors.keys()) == set(fractions.keys())
    assert sum(fractions.values()) == 1, fractions
    for key, frac in list(fractions.items()):
      if not frac:
        selectors.pop(key)
        fractions.pop(key)
    keys = sorted(selectors.keys())
    self.selectors = [selectors[key] for key in keys]
    self.fractions = np.array([fractions[key] for key in keys], np.float32)
    self.rng = np.random.default_rng(seed)

  def __len__(self):
    # All selectors have the same items, so return length of first one
    return len(self.selectors[0]) if self.selectors else 0

  def __call__(self):
    return self.rng.choice(self.selectors, p=self.fractions)()

  def __setitem__(self, key, stepids):
    for selector in self.selectors:
      selector[key] = stepids

  def __delitem__(self, key):
    for selector in self.selectors:
      del selector[key]

  def prioritize(self, stepids, priorities):
    for selector in self.selectors:
      if hasattr(selector, 'prioritize'):
        selector.prioritize(stepids, priorities)


class SampleTree:

  def __init__(self, branching=16, seed=0):
    assert 2 <= branching
    self.branching = branching
    self.root = SampleTreeNode()
    self.last = None
    self.entries = {}
    self.rng = np.random.default_rng(seed)

  def __len__(self):
    return len(self.entries)

  def insert(self, key, uprob):
    if not self.last:
      node = self.root
    else:
      ups = 0
      node = self.last.parent
      while node and len(node) >= self.branching:
        node = node.parent
        ups += 1
      if not node:
        node = SampleTreeNode()
        node.append(self.root)
        self.root = node
      for _ in range(ups):
        below = SampleTreeNode()
        node.append(below)
        node = below
    entry = SampleTreeEntry(key, uprob)
    node.append(entry)
    self.entries[key] = entry
    self.last = entry

  def remove(self, key):
    entry = self.entries.pop(key)
    entry_parent = entry.parent
    last_parent = self.last.parent
    entry.parent.remove(entry)
    if entry is not self.last:
      entry_parent.append(self.last)
    node = last_parent
    ups = 0
    while node.parent and not len(node):
      above = node.parent
      above.remove(node)
      node = above
      ups += 1
    if not len(node):
      self.last = None
      return
    while isinstance(node, SampleTreeNode):
      node = node.children[-1]
    self.last = node

  def update(self, key, uprob):
    entry = self.entries[key]
    entry.uprob = uprob
    entry.parent.recompute()

  def sample(self):
    node = self.root
    while isinstance(node, SampleTreeNode):
      uprobs = np.array([x.uprob for x in node.children])
      total = uprobs.sum()
      if not np.isfinite(total):
        finite = np.isinf(uprobs)
        probs = finite / finite.sum()
      elif total == 0:
        probs = np.ones(len(uprobs)) / len(uprobs)
      else:
        probs = uprobs / total
      choice = self.rng.choice(np.arange(len(uprobs)), p=probs)
      node = node.children[choice.item()]
    return node.key


class SampleTreeNode:

  __slots__ = ('parent', 'children', 'uprob')

  def __init__(self, parent=None):
    self.parent = parent
    self.children = []
    self.uprob = 0

  def __repr__(self):
    return (
        f'SampleTreeNode(uprob={self.uprob}, '
        f'children={[x.uprob for x in self.children]})'
    )

  def __len__(self):
    return len(self.children)

  def __bool__(self):
    return True

  def append(self, child):
    if child.parent:
      child.parent.remove(child)
    child.parent = self
    self.children.append(child)
    self.recompute()

  def remove(self, child):
    child.parent = None
    self.children.remove(child)
    self.recompute()

  def recompute(self):
    self.uprob = sum(x.uprob for x in self.children)
    self.parent and self.parent.recompute()


class SampleTreeEntry:

  __slots__ = ('parent', 'key', 'uprob')

  def __init__(self, key=None, uprob=None):
    self.parent = None
    self.key = key
    self.uprob = uprob
