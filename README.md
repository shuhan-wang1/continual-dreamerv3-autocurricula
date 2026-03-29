# Continual DreamerV3 with Autocurricula

This repository extends [DreamerV3](https://github.com/danijar/dreamerv3) with continual learning capabilities and novel replay sampling strategies, evaluated primarily on [Craftax](https://github.com/MichaelTMatthews/Craftax). It is inspired by and builds on the Continual DreamerV2 work: [On The Effectiveness of World Models For Continual Reinforcement Learning](https://arxiv.org/abs/2211.15944).

```bibtex
@article{kessler2022surprising,
  title={The surprising effectiveness of latent world models for continual reinforcement learning},
  author={Kessler, Samuel and Mi{\l}o{\'s}, Piotr and Parker-Holder, Jack and Roberts, Stephen J},
  journal={arXiv preprint arXiv:2211.15944},
  year={2022}
}
```

---

## Installation

Create a conda environment with Python >= 3.9:

```sh
conda create -n continual-dv3 python=3.10
conda activate continual-dv3
pip install -r requirements.txt
```

---

## Codebase Structure & Innovations

```
continual-dreamerv3-autocurricula/
├── train.py                  # Unified entry point (routes to craftax or navix)
├── train_craftax.py          # Main Craftax training script (primary focus)
├── train_navix.py            # NAVIX training script
├── input_args.py             # All argument parsers (craftax, navix)
├── intrinsic_reward.py       # Spatial + Craft novelty reward shaping
├── run_ablation.py           # Ablation experiment runner (14 configs × 3 seeds)
├── plot_ablation.py          # Result visualization & analysis
├── analyze_all_results.py    # Final results aggregation for publication
├── analyze_mask_effect.py    # Action masking analysis
│
├── dreamerv3/                # ENHANCED DreamerV3 (our modified version)
│   ├── dreamerv3/
│   │   ├── agent.py          # Agent with Plan2Explore + action masking
│   │   └── rssm.py           # RSSM encoder/decoder
│   └── embodied/core/
│       ├── selectors.py      # All replay sampling selectors (key innovation)
│       └── replay.py         # Replay buffer with selector integration
├── dreamerv3-main/           # Original DreamerV3 (baseline, --use_original_dreamer)
│
├── craftax_mask/             # Craftax action feasibility masking
│   ├── extractor.py          # Extract mask context from game state
│   ├── mask.py               # Compute logit bias for feasible actions
│   ├── rules.py              # Action feasibility rules + constants
│   └── test_mask.py          # Unit tests
│
├── notebooks/
│   └── metrics.py            # OnlineMetrics class (used by train_navix.py)
│
├── experiment_results/       # Analysis scripts + saved metrics
│   ├── analyze_metrics.py    # Comparative analysis across experiments
│   └── plot_neurips_figures.py  # Publication-quality figure generation
│
├── all_results/              # Pre-computed experiment metrics (JSONL)
│   └── figures/              # Generated figures (output of plot_neurips_figures.py)
│
├── myriad_*.sh               # UCL Myriad HPC job submission scripts
├── autodl_*.sh               # AutoDL GPU cloud submission scripts
├── package_results_*.sh      # Results packaging scripts
│
├── docs/                     # Design documents & meeting notes
└── Report.tex                # LaTeX report
```

### Notation

We consider an agent interacting with a Craftax environment, producing episodes stored in a replay buffer. The following notation is used throughout:

| Symbol | Definition |
|---|---|
| $\mathcal{B} = \{e_1, \ldots, e_n\}$ | Replay buffer containing $n$ stored episodes |
| $R_e \in \mathbb{R}$ | Cumulative undiscounted reward of episode $e$ |
| $L_e \in \mathbb{N}$ | Number of timesteps in episode $e$ |
| $\mathbf{a}_e \in \{0,1\}^{67}$ | Binary achievement vector; $a_{e,i} = 1$ iff achievement $i$ was unlocked |
| $\mathcal{A}_e = \{i : a_{e,i} = 1\}$ | Set of achievements unlocked in episode $e$ |
| $\mathbf{o}_t \in \mathbb{R}^{8268}$ | Raw symbolic observation at timestep $t$ (map tiles + inventory + player state) |
| $\mathbf{c} \in \mathbb{R}^{46}$ | Action mask context vector extracted from game state |
| $\ell_a \in \mathbb{R}$ | Raw policy logit for discrete action $a$ |
| $\text{sg}(\cdot)$ | Stop-gradient operator |

### Our Innovations

#### 1. NLR / NLU Replay Sampling (`dreamerv3/embodied/core/selectors.py`)

We propose a structured replay sampling strategy that partitions each mini-batch across three functionally distinct pools. We provide four variants differing in how novelty is scored and how the third pool samples:

| Variant | Novelty source | Third pool | CLI flag |
|---|---|---|---|
| NLR-privileged | Per-achievement success rates | Triangular recency | `--nlr_privileged_sampling` |
| NLU-privileged | Per-achievement success rates | Uniform | `--nlu_privileged_sampling` |
| NLR | 2-D reward $\times$ length grid | Triangular recency | `--nlr_sampling` |
| NLU | 2-D reward $\times$ length grid | Uniform | `--nlu_sampling` |

**1.1 Pool selection.** Each mini-batch element is independently assigned to pool $p \in \{N, L, R\}$ (Novelty, Learnability, Third) with configurable target fractions $(\omega_N, \omega_L, \omega_R)$ (default 0.35, 0.35, 0.30 via `--nlr_novel_frac`, `--nlr_learnable_frac`, `--nlr_recent_frac`; constrained to sum to 1). When pool $p$ is empty, its weight is redistributed proportionally among non-empty pools:

$$P(\text{pool} = p) = \frac{\omega_p \cdot \mathbb{1}[\,|p| > 0\,]}{\displaystyle\sum_{q \in \{N,L,R\}} \omega_q \cdot \mathbb{1}[\,|q| > 0\,]}$$

where $|p|$ denotes the number of episodes currently in pool $p$.

**1.2 Novelty scoring — privileged variant.** We maintain a per-achievement success rate vector $\mathbf{s} \in [0,1]^{67}$ computed over all episodes currently in $\mathcal{B}$:

$$s_i = \frac{1}{|\mathcal{B}|} \sum_{e \in \mathcal{B}} a_{e,i}$$

i.e., $s_i$ is the fraction of buffered episodes in which achievement $i$ was unlocked. The novelty score of episode $e$ is the mean inverse success rate across its achieved skills:

$$\nu(e) = \begin{cases} \dfrac{1}{|\mathcal{A}_e|} \displaystyle\sum_{i \in \mathcal{A}_e} \dfrac{1}{s_i + \epsilon} & \text{if } \mathcal{A}_e \neq \emptyset \\[6pt] 0 & \text{otherwise} \end{cases}$$

where $\epsilon = 0.01$ (`--nlr_novelty_eps`) prevents division by zero. An episode enters the novelty pool iff $\nu(e) > 0$. Within the pool, episode $e$ is sampled with probability:

$$P(e \mid \text{pool} = N) = \frac{\nu(e)^{1/\tau_N}}{\displaystyle\sum_{e' \in N} \nu(e')^{1/\tau_N}}$$

where $\tau_N > 0$ is the novelty temperature (`--nlr_novelty_temp`, default 1.0). As $\tau_N \to 0^+$, sampling concentrates on the highest-novelty episode; as $\tau_N \to \infty$, sampling becomes uniform over the pool. The success rates $\mathbf{s}$ are recomputed from scratch every `--nlr_recompute_interval` (default 500) episodes to correct for staleness caused by buffer evictions.

**1.3 Novelty scoring — non-privileged variant.** We replace the privileged achievement vector with a 2-D quantile histogram over $(L_e, R_e)$ space that requires no environment-specific information. Let $\{R_j\}_{j=1}^{n}$ and $\{L_j\}_{j=1}^{n}$ be the cumulative rewards and lengths of all $n$ episodes in $\mathcal{B}$. The reward axis is partitioned into $N_R$ quantile bins (`--nlr_grid_reward_bins`, default 5) and the length axis into $N_L$ quantile bins (`--nlr_grid_length_bins`, default 10). Let $n_{l,r}$ denote the count of episodes in grid cell $(l, r)$ and $R_r$ denote the midpoint of reward bin $r$ (the average of the bin's lower and upper edges).

We define two adaptive parameters from the empirical reward distribution. Let $\text{percentile}_q(S)$ denote the value below which fraction $q/100$ of the elements of set $S$ fall (equivalently, `np.quantile(S, q/100)`). Then:

$$R_{\min} = \text{percentile}_{20}\big(\{R_j\}\big), \quad R_{50} = \text{percentile}_{50}\big(\{R_j\}\big), \quad \beta = \max\left(0.1,\; \frac{R_{50} - R_{\min}}{4}\right)$$

The novelty score for an episode assigned to grid cell $(l, r)$ is:

$$\nu(e) = \frac{\sigma\left(\dfrac{R_r - R_{\min}}{\beta}\right) \cdot R_r}{n_{l,r} + \epsilon}$$

where $\sigma(x) = (1 + e^{-x})^{-1}$ is the logistic sigmoid and $\epsilon = 0.01$ (`--nlr_grid_eps`). The sigmoid term gates out low-quality bins: when $R_r \ll R_{\min}$, the argument $(R_r - R_{\min})/\beta$ is large and negative, yielding $\sigma \approx 0$. The $R_r / (n_{l,r} + \epsilon)$ term up-weights bins that are simultaneously high-reward and sparsely populated. Bin edges are recomputed from quantiles every `--nlr_grid_recompute_every` (default 500) episodes to adapt to the evolving reward distribution. Sampling within the pool uses the same temperature-scaled softmax as the privileged variant.

**1.4 Learnability scoring.** We maintain an exponential moving average (EMA) of episodic rewards as a running baseline:

$$\bar{R}_t = \gamma \cdot \bar{R}_{t-1} + (1 - \gamma) \cdot R_t$$

where $\gamma = 0.99$ (`--nlr_reward_ema_decay`) and $\bar{R}_0$ is initialized to the first observed reward. For episode $e$ arriving at time $t$, the learnability score is the positive part of the advantage over the **pre-update** baseline:

$$\ell(e) = \max(0, \; R_e - \bar{R}_{t-1})$$

The use of $\bar{R}_{t-1}$ rather than the post-update $\bar{R}_t$ is critical: since $\bar{R}_t$ partially incorporates $R_e$, using it would systematically attenuate the computed advantage. An episode enters the learnability pool iff $\ell(e) > 0$. Within the pool:

$$P(e \mid \text{pool} = L) = \frac{\ell(e)^{1/\tau_L}}{\displaystyle\sum_{e' \in L} \ell(e')^{1/\tau_L}}$$

where $\tau_L > 0$ is the learnability temperature (`--nlr_learnability_temp`, default 1.0).

**1.5 Third pool.** In NLR variants, the third pool uses triangular recency weighting. Let $e_{(0)}, e_{(1)}, \ldots, e_{(W-1)}$ be the $W$ most recent episodes in $\mathcal{B}$, ordered from newest ($i = 0$) to oldest ($i = W - 1$), where $W$ is the window size (`--nlr_recent_window`, default 1000). The sampling weight decreases linearly:

$$w_i = W - i, \quad P\big(e_{(i)} \mid \text{pool} = R\big) = \frac{w_i}{\displaystyle\sum_{j=0}^{W-1} w_j} = \frac{2(W - i)}{W(W + 1)}$$

In NLU variants, the third pool samples uniformly from the entire buffer: $P(e \mid \text{pool} = R) = 1 / |\mathcal{B}|$.

**Baseline selectors.** For comparison, we also implement reservoir sampling with random eviction (Vitter, 1985), a 50:50 recent/uniform mixture, and reward-weighted sampling via softmax over cumulative rewards.

#### 2. Intrinsic Reward Shaping (`intrinsic_reward.py`)

We introduce environment-level episodic intrinsic rewards with two independently normalized components. All tracker state is **per-episode** (reset at episode boundaries) and **per-environment** (vectorized environments maintain independent trackers).

**2.1 Spatial novelty.** The Craftax symbolic observation $\mathbf{o}_t \in \mathbb{R}^{8268}$ encodes a $9 \times 11$ tile grid where each tile occupies 83 channels. The first $K = 37$ channels per tile are a one-hot encoding of the block type. We hash the visible map by extracting the dominant block-type ID at each position:

$$b_{r,c} = \text{argmax}_{k \in \{0,\ldots,K-1\}} \; \mathbf{o}_t[(r \cdot 11 + c) \cdot 83 + k]$$

$$\mathbf{h} = (b_{0,0},\; b_{0,1},\; \ldots,\; b_{8,10}) \in \{0,\ldots,K-1\}^{99}$$

A per-episode dictionary $N : \{0,\ldots,K-1\}^{99} \to \mathbb{N}$ maps each hash to its visit count, initialized to empty at episode start. At each timestep, $N(\mathbf{h})$ is incremented *before* computing the reward, so the first encounter of any hash $\mathbf{h}$ yields $N(\mathbf{h}) = 1$. The spatial intrinsic reward is:

$$r_{\text{spatial}} = \frac{1}{\sqrt{N(\mathbf{h})}}$$

This decays smoothly with revisitation ($1 \to 1/\sqrt{2} \to 1/\sqrt{3} \to \cdots$), unlike a binary novelty indicator, and is immune to circle-walking: identical tile layouts produce the same hash $\mathbf{h}$ regardless of the path taken.

**2.2 Craft novelty.** Let $\mathbf{inv}_t \in \mathbb{R}^{51}$ denote the inventory sub-vector of $\mathbf{o}_t$, starting at index 8217. We discretize each component by truncating to one decimal place:

$$\text{disc}(\mathbf{inv}_t)_i = \lfloor \mathbf{inv}_{t,i} \times 10 \rfloor \in \mathbb{Z}$$

A per-episode set $\mathcal{H} \subseteq \mathbb{Z}^{51}$, initialized to $\emptyset$ at episode start, accumulates all discretized inventory states observed so far. The craft intrinsic reward is:

$$r_{\text{craft}} = \mathbb{1}\big[\text{disc}(\mathbf{inv}_t) \notin \mathcal{H}\big]$$

where $\mathbb{1}[\cdot]$ is the indicator function (1 if true, 0 otherwise). After computing $r_{\text{craft}}$, the current $\text{disc}(\mathbf{inv}_t)$ is added to $\mathcal{H}$. This fires exactly once per novel inventory configuration, capturing pick-up, craft, and equip events.

**2.3 Adaptive normalization.** Each intrinsic component (spatial, craft) has its own cross-episode EMA normalizer that rescales the intrinsic signal to match the extrinsic reward magnitude. For each component, let $\hat{\mu}_{\text{intr}}$ and $\hat{\mu}_{\text{extr}}$ be running EMAs of absolute values, updated at every timestep:

$$\hat{\mu}_{\text{intr}} \leftarrow \gamma \, \hat{\mu}_{\text{intr}} + (1 - \gamma) \, |r_{\text{intr}}|$$

$$\hat{\mu}_{\text{extr}} \leftarrow \gamma \, \hat{\mu}_{\text{extr}} + (1 - \gamma) \, |r_{\text{extr}}|$$

with decay $\gamma = 0.99$ and both initialized to 0. During a warmup period of 100 steps, $r_{\text{intr}}$ is returned unnormalized. After warmup, the normalized intrinsic reward is:

$$\text{norm}(r_{\text{intr}}) = r_{\text{intr}} \cdot \min\left(\frac{\hat{\mu}_{\text{extr}}}{\hat{\mu}_{\text{intr}}},\; 100\right)$$

The ratio $\hat{\mu}_{\text{extr}} / \hat{\mu}_{\text{intr}}$ rescales the intrinsic signal to match the extrinsic scale. The cap of 100 prevents extreme amplification when $\hat{\mu}_{\text{intr}} \approx 0$ (e.g., when the craft reward fires very rarely).

**2.4 Combined reward.** The final shaped reward at each timestep is:

$$r_t = \alpha_{\text{sp}} \cdot \text{norm}_{\text{sp}}(r_{\text{spatial},t}) + \alpha_{\text{cr}} \cdot \text{norm}_{\text{cr}}(r_{\text{craft},t}) + \alpha_e \cdot r_{\text{extr},t}$$

where $\text{norm}_{\text{sp}}$ and $\text{norm}_{\text{cr}}$ are **independent** normalizers, each maintaining their own $(\hat{\mu}_{\text{intr}}, \hat{\mu}_{\text{extr}})$ pair. Default weights: $\alpha_{\text{sp}} = 0.1$ (`--alpha_spatial`), $\alpha_{\text{cr}} = 0.3$ (`--alpha_craft`), $\alpha_e = 1.0$ (`--alpha_e`).

#### 3. Action Feasibility Masking (`craftax_mask/`, `dreamerv3/dreamerv3/agent.py`)

We introduce a learned action masking system that biases the policy away from infeasible actions using a declarative rule system over a structured context vector.

**3.1 Context extraction.** At each timestep, a context vector $\mathbf{c} \in \mathbb{R}^{46}$ is extracted from the Craftax game state (see `craftax_mask/rules.py` for the full schema):

| Indices | Content | Dimensionality |
|---|---|---|
| 0--13 | Inventory: wood, stone, coal, iron, diamond, sapling, pickaxe, sword, bow, arrows, torches, books, ruby, sapphire | 14 (integer counts) |
| 14--17 | Armour levels (4 slots) | 4 |
| 18--23 | Potion counts (6 types) | 6 |
| 24--28 | Player state: health, energy, mana, XP, level | 5 |
| 29--31 | Attributes: dexterity, strength, intelligence | 3 |
| 32--33 | Learned spells: fireball, iceball | 2 (binary) |
| 34--35 | Proximity: near crafting table, near furnace | 2 (binary) |
| 36--39 | Facing tile: placeable, grass, enchantment table, torch-placeable | 4 (binary) |
| 40--41 | Ladder: on ladder down, on ladder up | 2 (binary) |
| 42 | Level cleared (monsters killed $\geq$ 8) | 1 (binary) |
| 43 | Projectile slots available (active projectiles $<$ 3) | 1 (binary) |
| 44--45 | Facing table type: fire enchantment, ice enchantment | 2 (binary) |

**3.2 Context prediction head.** The world model learns to predict $\hat{\mathbf{c}}$ from RSSM latent features via a dedicated MLP head (2 hidden layers, SiLU activation, RMSNorm), trained jointly with the world model:

$$\mathcal{L}_{\text{mask}} = -\log p_\theta(\mathbf{c} \mid z_t)$$

where $p_\theta$ is a learned diagonal Gaussian parameterized by the head (outputting mean and log-std). This enables mask application during imagination (Section 3.4) without access to the true game state.

**3.3 Deficit computation and logit bias.** Each maskable action $a$ is associated with a set of declarative conditions $\mathcal{C}_a$ defined in `craftax_mask/rules.py`. The **deficit** $\delta_a \geq 0$ quantifies the degree of infeasibility as the sum of individual condition shortfalls:

$$\delta_a = \sum_{c \in \mathcal{C}_a} d_c(\mathbf{c})$$

where $d_c(\mathbf{c})$ depends on the condition type:

| Condition type | Shortfall $d_c(\mathbf{c})$ | Semantics |
|---|---|---|
| `"min"` | $\max(0,\; v - \mathbf{c}[j])$ | Resource $\mathbf{c}[j]$ must be $\geq v$ |
| `"bool"` | $\mathbb{1}[\mathbf{c}[j] \leq 0.5]$ | Binary flag must be true |
| `"below"` | $\mathbb{1}[\mathbf{c}[j] \geq v]$ | Value must be $< v$ |
| `"any_below"` | $\mathbb{1}\big[\min_{j \in [s,e)} \mathbf{c}[j] \geq v\big]$ | At least one slot must be $< v$ |
| `"sum_pos"` | $\mathbb{1}\big[\sum_{j=s}^{e-1} \mathbf{c}[j] = 0\big]$ | Sum over range must be positive |
| `"attr_below_max"` | $\mathbb{1}[\mathbf{c}[j] \geq 5]$ | Attribute must not be maxed |

For example, `PLACE_TABLE` (action 8) requires `("min", WOOD, 2)` and `("bool", FACING_PLACEABLE)`, so its deficit is $\delta = \max(0, 2 - \mathbf{c}[0]) + \mathbb{1}[\mathbf{c}[36] \leq 0.5]$.

The logit bias $b_a$ is computed from $\delta_a$ in one of two modes:

- **Soft** (`--action_mask_mode soft`): $b_a = -\lambda \cdot \delta_a$, where $\lambda = 5.0$ (`--action_mask_lambda_penalty`). The penalty is proportional to the total deficit; partially-satisfied conditions yield a milder bias.
- **Hard** (`--action_mask_mode hard`): $b_a = -M \cdot \mathbb{1}[\delta_a > 0]$, where $M = 10^9$ (`--action_mask_large_negative`). Any non-zero deficit effectively blocks the action.

The adjusted policy is the softmax over biased logits:

$$\pi_{\text{adj}}(a \mid s) = \frac{\exp(\ell_a + b_a)}{\displaystyle\sum_{a'} \exp(\ell_{a'} + b_{a'})}$$

where $\ell_a$ are the raw logits from the actor network. Unmasked actions (those without rules in `ACTION_RULES`) have $b_a = 0$.

**3.4 Imagination-time masking.** During imagination (policy optimization in the RSSM latent space), the true game state is unavailable. Instead, the predicted context $\hat{\mathbf{c}}$ from the learned head (Section 3.2) is used in place of $\mathbf{c}$ to compute $\delta_a$ and $b_a$, allowing the agent to plan with feasibility awareness in the learned world model.

#### 4. Integrated Plan2Explore (`dreamerv3/dreamerv3/agent.py`)

We integrate [Plan2Explore](https://arxiv.org/abs/2005.05960) into DreamerV3 as an ensemble-based epistemic exploration bonus. An ensemble of $K$ prediction heads $\{\hat{\phi}_k\}_{k=1}^{K}$ (each a 3-layer MLP with SiLU activation and RMSNorm) is trained to predict a target representation $\phi(z_{t+1}) \in \mathbb{R}^D$ of the next latent state from the current features $z_t$. The target depends on `--disag_target`:

| `--disag_target` | Target $\phi(z)$ | Dimension $D$ |
|---|---|---|
| `feat` (default) | $[\mathbf{h}_t;\; \text{flatten}(\mathbf{z}_t)]$ | $d_h + d_z \cdot d_c$ |
| `stoch` | $\text{flatten}(\mathbf{z}_t)$ | $d_z \cdot d_c$ |
| `deter` | $\mathbf{h}_t$ | $d_h$ |

where $\mathbf{h}_t$ is the RSSM deterministic state, $\mathbf{z}_t$ is the stochastic state with $d_z$ variables each over $d_c$ classes, and $[\cdot;\cdot]$ denotes concatenation. The ensemble is trained on real transitions via negative log-likelihood loss with stop-gradient on the target.

During imagination, ensemble disagreement serves as an intrinsic reward. For imagined latent state $z_t$, each head produces a prediction $\hat{\phi}_{k,d}(z_t)$ for each target dimension $d$. The intrinsic reward is the mean per-dimension variance:

$$r_{\text{p2e}} = \frac{1}{D} \sum_{d=1}^{D} \frac{1}{K} \sum_{k=1}^{K} \left(\hat{\phi}_{k,d}(z_t) - \bar{\phi}_d(z_t)\right)^2, \quad \bar{\phi}_d(z_t) = \frac{1}{K}\sum_{k=1}^{K} \hat{\phi}_{k,d}(z_t)$$

High disagreement indicates epistemic uncertainty about the dynamics, driving the agent to explore unfamiliar regions. The total imagination reward is:

$$r_{\text{total}} = \alpha_i \cdot \text{sg}(r_{\text{p2e}}) + \alpha_e \cdot r_{\text{extr}}$$

where $\alpha_i = 0.9$ (`--expl_intr_scale`), $\alpha_e = 0.0$ (`--expl_extr_scale`) by default, and $\text{sg}(\cdot)$ stops gradients from flowing through the ensemble into the policy/value optimization. Key parameter: `--disag_models` ($K$, default 10).

#### 5. Online Continual-Learning Metrics (`train_craftax.py`)

We track per-episode metrics across all 67 Craftax achievements, organized into 5 tiers (tier 0 = basic survival, tier 4 = endgame). Let $p_a(t) \in [0,1]$ denote the empirical success rate of achievement $a$ at evaluation time $t$, computed as a running average over recent episodes.

**Per-achievement forgetting.** For each achievement $a \in \{1, \ldots, 67\}$:

$$F_a(t) = \max_{t' < t} \, p_a(t') - p_a(t)$$

This measures how much the current success rate has regressed from its historical peak. $F_a(t) = 0$ indicates no forgetting; $F_a(t) > 0$ indicates the agent has lost proficiency at achievement $a$.

**Aggregate forgetting.** The mean forgetting across all achievements:

$$\bar{F}(t) = \frac{1}{67} \sum_{a=1}^{67} F_a(t)$$

Achievements never encountered contribute $F_a(t) = 0$ since their peak rate $\max_{t'<t} p_a(t') = 0$.

**Achievement depth.** For a single episode, the maximum tier index among all achievements unlocked in that episode:

$$\text{depth}(e) = \max_{i \in \mathcal{A}_e} \text{tier}(i)$$

where $\text{tier}(i) \in \{0, 1, 2, 3, 4\}$ maps achievement $i$ to its difficulty tier. Returns $-1$ if $\mathcal{A}_e = \emptyset$.

**Frontier rate.** Let $d^* = \max_{e' \in \text{history}} \text{depth}(e')$ be the agent's personal-best depth across all episodes and tasks. The frontier rate over a window of recent episodes $\{e_1, \ldots, e_m\}$ is:

$$\text{frontier}= \frac{1}{m} \sum_{j=1}^{m} \mathbb{1}[\text{depth}(e_j) \geq d^*]$$

All metrics are written per-episode to `online_metrics.jsonl` and streamed to W&B.

---

## Embedding Mode (Default)

We consistently use **embedding mode** (`--input_type embedding`, the default) rather than raw pixels for all Craftax experiments. In this mode:

- The Craftax symbolic observation (a flat integer/float vector encoding the local map, inventory, and world state) is treated as a 1-D vector and projected into a fixed-size embedding space of dimension `--embedding_dim` (default 256) by a shallow MLP encoder.
- This bypasses the convolutional encoder that would otherwise be used for pixel observations, giving a much lower-dimensional and semantically dense input to the RSSM.
- Training is significantly faster and more stable than pixel mode: the world model does not need to reconstruct images, making the RSSM focus entirely on dynamics and reward prediction.
- **To use raw pixel observations instead**, pass `--input_type pixel` (not recommended for Craftax — symbolic embeddings consistently outperform pixels on this environment).

---

## Training

### Quick Start (Craftax)

```sh
# Single-task training with default settings
python train.py --env_type craftax \
    --steps 500000 \
    --batch_size 16 \
    --batch_length 64 \
    --envs 16 \
    --model_size 25m \
    --tag my_run \
    --seed 42

# ..with original DreamerV3 (no continual-learning modifications)
python train.py --env_type craftax --steps 500000 --use_original_dreamer

# With intrinsic rewards (spatial + craft novelty)
python train.py --env_type craftax \
    --intrinsic_spatial \
    --alpha_spatial 0.1 --alpha_craft 0.3 \
    --no_plan2explore \
    --tag intrinsic_run

# Continual learning with NLR sampling + intrinsic rewards
python train.py --env_type craftax \
    --cl --num_tasks 2 --num_task_repeats 3 \
    --steps 250000 \
    --nlr_sampling \
    --intrinsic_spatial \
    --no_plan2explore \
    --tag cl_nlr_intrinsic
```

---

## Craftax Training Arguments

All arguments below are parsed by `parse_craftax_args()` in `input_args.py`.

### Continual Learning

| Argument | Description | Default |
|---|---|---|
| `--cl` | Enable continual learning mode (sequential tasks) | `False` |
| `--cl_small` | Use small CL configuration | `False` |
| `--num_tasks` | Number of tasks in the sequence | `1` |
| `--num_task_repeats` | Number of times to cycle through the task sequence | `1` |
| `--unbalanced_steps` | Override per-task step budget (list) | `None` |

### Environment & Run

| Argument | Description | Default |
|---|---|---|
| `--env` | Environment index for single-task training | `0` |
| `--steps` | Training steps per task | `500000` |
| `--seed` | Global random seed | `42` |
| `--tag` | Run identifier suffix (appended to log directory name) | `''` |
| `--logdir` | Root directory for logs and checkpoints | `logs` |
| `--envs` | Number of parallel training environments | config default (64) |
| `--eval_envs` | Number of parallel evaluation environments | config default (4) |
| `--del_exp_replay` | Delete replay buffer after training to save storage | `False` |

### Observation Input

| Argument | Description | Default |
|---|---|---|
| `--input_type` | `embedding` (recommended) or `pixel` | `embedding` |
| `--embedding_dim` | Dimension of the embedding projection | `256` |

### Model & Training

| Argument | Description | Default |
|---|---|---|
| `--model_size` | DreamerV3 model size preset: `1m` `12m` `25m` `50m` `100m` `200m` `400m` | `25m` |
| `--batch_size` | Mini-batch size | `16` |
| `--batch_length` | Sequence length per training batch (reduce to save VRAM) | `64` |
| `--train_ratio` | World-model training steps per environment step | auto |
| `--replay_capacity` | Maximum number of episodes in the replay buffer | `2000000` |
| `--use_original_dreamer` | Use the unmodified DreamerV3 from `dreamerv3-main/` | `False` |

### Intrinsic Reward (Spatial + Craft Novelty)

| Argument | Description | Default |
|---|---|---|
| `--intrinsic_spatial` | Enable spatial + craft novelty intrinsic reward | `False` |
| `--no_intrinsic_spatial` | Disable intrinsic reward | -- |
| `--alpha_spatial` | Spatial novelty weight (independently normalized to match extrinsic scale) | `0.1` |
| `--alpha_craft` | Craft novelty weight (independently normalized; rare events get larger scale) | `0.3` |
| `--alpha_e` | Extrinsic reward weight | `1.0` |

### Replay Sampling Strategy

Choose **one** of the following strategies (they are mutually exclusive):

| Argument | Description | Default |
|---|---|---|
| `--reservoir_eviction` | Use reservoir eviction (random) instead of FIFO | `False` |
| `--reward_sampling` | Reward-weighted episode sampling | `False` |
| `--recency_sampling` | Recency-biased triangular sampling | `False` |
| `--recent_frac` | Fraction of each mini-batch from recent experience (50:50 strategy) | `0.5` |
| `--recent_window` | Window size for recent experience pool | `1000` |
| `--uniform_frac` | Fraction from uniform distribution in Mixture selector | `0.5` |
| `--recency_frac` | Fraction from recency distribution in Mixture selector | `0.0` |

#### NLR / NLU Sampling (Novel Contribution)

| Argument | Description | Default |
|---|---|---|
| `--nlr_privileged_sampling` | NLR with privileged per-achievement success rates | `False` |
| `--nlu_privileged_sampling` | NLU (uniform 3rd pool) with privileged per-achievement info | `False` |
| `--nlr_sampling` | NLR with non-privileged 2-D reward x length grid novelty | `False` |
| `--nlu_sampling` | NLU with non-privileged 2-D grid novelty | `False` |
| `--nlr_novel_frac` | Fraction of mini-batch from the novelty pool | `0.35` |
| `--nlr_learnable_frac` | Fraction of mini-batch from the learnability pool | `0.35` |
| `--nlr_recent_frac` | Fraction of mini-batch from the recency/uniform pool | `0.30` |
| `--nlr_recent_window` | Window size for the recency pool | `1000` |
| `--nlr_reward_ema_decay` | EMA decay for learnability reward baseline | `0.99` |
| `--nlr_novelty_temp` | Temperature for novelty pool softmax | `1.0` |
| `--nlr_learnability_temp` | Temperature for learnability pool softmax | `1.0` |
| `--nlr_novelty_eps` | (Privileged only) Epsilon smoothing for achievement success rates | `0.01` |
| `--nlr_recompute_interval` | (Privileged only) Recompute all pool scores every N episodes | `500` |
| `--nlr_grid_reward_bins` | (Non-privileged) Number of quantile bins on the reward axis | `5` |
| `--nlr_grid_length_bins` | (Non-privileged) Number of quantile bins on the length axis | `10` |
| `--nlr_grid_recompute_every` | (Non-privileged) Recompute grid bin edges every N episodes | `500` |
| `--nlr_grid_prior_percentile` | (Non-privileged) Percentile of reward distribution for prior R_min | `0.20` |
| `--nlr_grid_eps` | (Non-privileged) Smoothing epsilon for bin counts | `0.01` |

### Action Masking

| Argument | Description | Default |
|---|---|---|
| `--action_mask_enabled` | Enable action feasibility masking | `False` |
| `--action_mask_mode` | Masking mode: `soft`, `hard`, or `none` | `soft` |
| `--action_mask_lambda_penalty` | Penalty weight for soft masking | `5.0` |
| `--action_mask_large_negative` | Hard mask blocking magnitude | `1e9` |

### Exploration (Plan2Explore)

| Argument | Description | Default |
|---|---|---|
| `--plan2explore` | Enable Plan2Explore intrinsic exploration | `True` |
| `--no_plan2explore` | Disable Plan2Explore | -- |
| `--disag_models` | Number of ensemble heads for disagreement | `10` |
| `--disag_target` | Prediction target: `feat` (deter+stoch), `stoch`, or `deter` | `feat` |
| `--expl_intr_scale` | Scale of intrinsic (disagreement) reward | `0.9` |
| `--sep_exp_eval_policies` | Use separate exploration and evaluation policies | `False` |
| `--rssm_full_recon` | Reconstruct obs, discount, and rewards (not just obs) | `False` |

### Logging & W&B

| Argument | Description | Default |
|---|---|---|
| `--wandb_proj_name` | W&B project name | `craftax` |
| `--wandb_group` | W&B experiment group | `craftax_experiment` |
| `--wandb_mode` | W&B mode: `online`, `offline`, or `disabled` | `online` |
| `--wandb_dir` | Custom W&B local directory | `None` |
| `--online_metrics` | Enable online CL metrics logging (forgetting, tier rates, etc.) | `True` |
| `--ref_metrics_path` | Path to a reference metrics JSON for forward transfer calculation | `None` |
| `--ref_metrics_dir` | Directory of per-task reference JSON files | `None` |

---

## Memory & Performance Tips

```sh
# Reduce parallelism to fit smaller GPUs
python train.py --env_type craftax --envs 8 --eval_envs 1 --batch_size 8 --batch_length 32

# Disable W&B for offline runs
python train.py --env_type craftax --wandb_mode disabled
```

JAX pre-allocates GPU memory at startup. The default allocation is 70% of VRAM (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.70`). If you are running multiple processes, lower this value in `train_craftax.py`.

---

## Ablation Experiments

A systematic ablation study is provided via `run_ablation.py` (groups A-F, 1M steps) and dedicated shell scripts (group G, 10M steps). All experiments use 3 seeds (1, 4, 42).

### Experiment Groups

| Group | Focus | Experiments | Steps |
|---|---|---|---|
| **A** | Core comparison | A0 50:50 baseline, A1 uniform baseline, A2 P2E, A3 intrinsic, A4 P2E+intrinsic | 1M |
| **B** | Intrinsic component ablation | B1 spatial-only, B2 craft-only | 1M |
| **D** | Replay strategy comparison | D1 NLR, D2 NLU, D3 NLR-privileged, D4 NLU-privileged | 1M |
| **E** | Combined model | E1 NLR + intrinsic (no P2E) | 1M |
| **F** | Action masking | F1 soft mask, F2 hard mask | 1M |
| **G** | Extended training | G1v2 mask+intrinsic+NLU, G2 baseline 10M, G3v3 mask+craft+NLU | 10M |

- **Group A** answers: Does intrinsic reward help? Does P2E help? Do they combine well?
- **Group B** answers: Is each intrinsic component necessary? (spatial-only vs craft-only vs both in A3)
- **Group D** uses pure baseline (no intrinsic) to isolate the replay strategy effect.
- **Group E** combines the best replay strategy (NLR) with intrinsic rewards.
- **Group F** tests learned action masking (soft penalty vs hard blocking of infeasible actions).
- **Group G** runs the most promising configurations for 10M steps to evaluate long-horizon scaling.

### Default Hyperparameters

All experiments use: `--steps 1000000 --batch_size 48 --batch_length 64 --envs 64 --model_size 25m`

### Running Experiments

```sh
# Run all 1M ablation experiments (groups A-F, 14 configs × 3 seeds)
python run_ablation.py

# Dry run - print commands without executing
python run_ablation.py --dry_run

# Run a specific group
python run_ablation.py --only A

# Run specific experiments
python run_ablation.py --only A1_baseline,D3_nlr_priv

# Skip a group
python run_ablation.py --skip D

# Disable W&B logging
python run_ablation.py --wandb_mode disabled

# Select GPU
python run_ablation.py --gpu 0

# Keep replay buffer after each run (default: replay deleted, ckpt kept)
python run_ablation.py --no_cleanup

# Run 10M extended experiments (G-series, via dedicated scripts)
bash autodl_10m_g1.sh   # G1v2: mask + intrinsic + NLU
bash autodl_10m_g2.sh   # G2: baseline @ 10M
bash autodl_10m_g3.sh   # G3v3: mask + craft + NLU
```

### Output Directory Structure

After running, results are organized as:

```
experiment_results/ablation/
├── experiment_manifest.json          # run metadata & status for all experiments
├── A1_baseline_seed1/
│   └── craftax_A1_baseline/          # DreamerV3 logdir
│       ├── config.yaml               # full training config snapshot
│       ├── metrics.jsonl             # DreamerV3 training metrics (loss, reward, etc.)
│       ├── online_metrics.jsonl      # per-episode CL metrics (achievements, forgetting)
│       ├── metrics_summary.json      # aggregated achievement stats
│       ├── nlr_args.yaml             # NLR/NLU config (if enabled)
│       └── ckpt/                     # model weights (saved every 50k steps)
├── A1_baseline_seed4/
│   └── ...
└── ...
```

**Saved artifacts per run:**
- `config.yaml` — Full training config for reproducibility
- `metrics.jsonl` — DreamerV3 training metrics (loss, reward, gradient norms, P2E disagreement, etc.) logged every 1k steps
- `online_metrics.jsonl` — Per-episode continual learning metrics (67 achievement rates, forgetting, frontier rate, tier distribution)
- `metrics_summary.json` — Aggregated achievement statistics
- `ckpt/` — Model weights (saved every 50k steps, kept after run completes)

The replay buffer is deleted after each successful run to save disk space. Use `--no_cleanup` to retain it.

---

## Logging

Training logs are saved to:

- `{logdir}/craftax_{tag}/` — Config YAML, checkpoints, `online_metrics.jsonl`, `metrics_summary.json`
- W&B dashboard — Real-time curves for reward, achievements, forgetting, and replay diagnostics

---

## Results Analysis & Figure Generation

Pre-computed experiment metrics (67 per-achievement success rates, forgetting, depth, etc.) for all seeds are stored as JSONL files in `all_results/`.

### Generating Publication Figures

```sh
# Generate all NeurIPS figures (1M ablation + 10M extended runs)
python experiment_results/plot_neurips_figures.py \
    --results_dir all_results \
    --output_dir all_results/figures
```

This produces:
- **Fig 1-4** — Learning curves: mean achievement rate, # achievements unlocked, aggregate forgetting, achievement depth
- **Fig 5** — Per-achievement success rates (grouped bar chart by tier)
- **Fig 6** — Per-tier success rate breakdown (5 subplots)
- **Fig 7** — Achievement heatmap over training
- **Fig 8** — Summary bar chart (final checkpoint comparison)
- **Fig 9** — Ablation group panels (A, B, D, E+F)
- **Fig 10-12** — Extended 10M training (G-series): learning curves, per-tier, heatmap

### Other Analysis Scripts

```sh
# Comprehensive statistical analysis across all experiments
python analyze_all_results.py --results_dir all_results

# Comparative metrics analysis
python experiment_results/analyze_metrics.py
```

---

## Citation

If you use this code, please cite the original Continual DreamerV2 paper:

```bibtex
@article{kessler2022surprising,
  title={The surprising effectiveness of latent world models for continual reinforcement learning},
  author={Kessler, Samuel and Mi{\l}o{\'s}, Piotr and Parker-Holder, Jack and Roberts, Stephen J},
  journal={arXiv preprint arXiv:2211.15944},
  year={2022}
}
```
