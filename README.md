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

### Our Innovations

#### 1. NLR / NLU Replay Sampling (`dreamerv3/embodied/core/selectors.py`)

Our primary contribution. Each mini-batch element is drawn from one of three pools with probabilities $(\omega_N, \omega_L, \omega_R)$ (default 0.35, 0.35, 0.30). If a pool is empty its weight is redistributed proportionally.

**Novelty pool** — prioritises trajectories exhibiting rare behaviour.

*Privileged variant* (`--nlr_privileged_sampling`): uses the per-achievement success rate vector $\mathbf{s} \in [0,1]^{67}$ maintained across training. For an episode whose achieved set is $\mathcal{A}$:

$$\text{novelty}(e) = \frac{1}{|\mathcal{A}|}\sum_{i \in \mathcal{A}} \frac{1}{s_i + \epsilon}, \quad \epsilon = 0.01$$

Episodes achieving rare skills (low $s_i$) score highest. The pool samples proportionally to $\text{novelty}(e)^{1/\tau_N}$ where $\tau_N$ is the novelty temperature.

*Non-privileged variant* (`--nlr_sampling`): replaces the achievement vector with a 2-D quantile histogram over (episode length, cumulative reward). Bin edges are recomputed from quantiles every 500 episodes. For bin $b$ with count $n_b$ and reward midpoint $R_b$:

$$\text{novelty}(e \in b) = \sigma\left(\frac{R_b - R_{\min}}{\beta}\right) \cdot \frac{R_b}{n_b + \epsilon}$$

where $R_{\min} = Q_{0.20}(\text{rewards})$, $\beta = \max(0.1, \; (Q_{0.50}(\text{rewards}) - R_{\min})/4)$, and $\sigma$ is the sigmoid. The sigmoid gates out low-reward bins; $R_b / n_b$ up-weights rare, high-reward trajectories.

**Learnability pool** — GRPO-style advantage filtering. An EMA baseline tracks expected reward:

$$\bar{R}_t = \gamma_{\text{ema}} \cdot \bar{R}_{t-1} + (1 - \gamma_{\text{ema}}) \cdot R_t, \quad \gamma_{\text{ema}} = 0.99$$

An episode enters the learnable pool iff its advantage is positive, and is sampled proportionally to the advantage magnitude:

$$\text{learnability}(e) = \max(0, \; R_e - \bar{R}_{t-1}), \quad \text{sampled} \propto \text{learnability}(e)^{1/\tau_L}$$

Note: $\bar{R}_{t-1}$ is the EMA *before* the current episode's update, preventing systematic attenuation.

**Third pool** — NLR uses triangular recency weighting over the most recent $W$ episodes ($w_i \propto W - i$); NLU samples uniformly from the entire buffer.

**Baseline selectors.** We also provide reservoir sampling (Vitter 1985), 50:50 recent/uniform mixture, and reward-weighted sampling for comparison.

#### 2. Independently-Normalized Spatial + Craft Intrinsic Reward (`intrinsic_reward.py`)

Environment-level episodic intrinsic rewards, independently normalized so rare craft events receive proportionally larger signal.

**Spatial novelty.** The agent's $9 \times 11$ visible tile grid is hashed by block-type ID. A per-episode visit counter $N(\mathbf{h})$ tracks hash occurrences:

$$r_{\text{spatial}} = \frac{1}{\sqrt{N(\mathbf{h})}}, \quad \mathbf{h} = \left(\operatorname{argmax}_{k} \; \text{tile}_{r,c}[0{:}37]\right)_{r,c}$$

This decays smoothly with revisitation, unlike a binary indicator, and is immune to circle-walking exploitation (same tile layout = same hash).

**Craft novelty.** The 51-dimensional inventory vector is discretized to 1 decimal place. A per-episode set $\mathcal{H}_{\text{inv}}$ tracks seen states:

$$r_{\text{craft}} = \mathbb{1}[\text{disc}(\mathbf{inv}) \notin \mathcal{H}_{\text{inv}}]$$

This fires only on meaningful inventory changes (pick up, craft, equip).

**Adaptive normalization.** Each component has its own cross-episode EMA normalizer that matches intrinsic scale to extrinsic:

$$\hat{\mu}_{\text{intr}} = \gamma \hat{\mu}_{\text{intr}} + (1-\gamma)|r_{\text{intr}}|, \quad \hat{\mu}_{\text{extr}} = \gamma \hat{\mu}_{\text{extr}} + (1-\gamma)|r_{\text{extr}}|, \quad \text{norm}(r) = r \cdot \min\left(\frac{\hat{\mu}_{\text{extr}}}{\hat{\mu}_{\text{intr}}},\; 100\right)$$

This ensures $\alpha_{\text{spatial}}$ and $\alpha_{\text{craft}}$ act as true relative-importance weights regardless of firing frequency.

**Combined reward:**

$$r = \alpha_{\text{spatial}} \cdot \text{norm}_{\text{sp}}(r_{\text{spatial}}) + \alpha_{\text{craft}} \cdot \text{norm}_{\text{cr}}(r_{\text{craft}}) + \alpha_e \cdot r_{\text{extr}}$$

Default: $\alpha_{\text{spatial}} = 0.1$, $\alpha_{\text{craft}} = 0.3$, $\alpha_e = 1.0$.

#### 3. Action Feasibility Masking (`craftax_mask/`, `dreamerv3/dreamerv3/agent.py`)

A learned masking system that biases the policy away from infeasible actions. A 46-dimensional mask context $\mathbf{c}$ is extracted from the Craftax game state encoding action feasibility. The world model learns to predict $\hat{\mathbf{c}}$ from latent features via a 2-layer MLP head, trained with:

$$\mathcal{L}_{\text{mask}} = \| \hat{\mathbf{c}} - \text{sg}(\mathbf{c}) \|^2$$

At action selection, a bias vector $\mathbf{b} \in \mathbb{R}^{|\mathcal{A}|}$ is computed from the predicted context and applied to raw policy logits:

$$\pi_{\text{adj}}(a \mid s) \propto \exp\left(\ell_a + b_a\right)$$

where $\ell_a$ are the raw logits from the policy network. For each action rule, a deficit $\delta_a \geq 0$ is computed as the sum of all condition shortfalls (e.g. missing resources, absent proximity). Two modes:

- **Soft** ($b_a = -\lambda \cdot \delta_a$): penalty proportional to total deficit, $\lambda = 5.0$ by default.
- **Hard** ($b_a = -M \cdot \mathbb{1}[\delta_a > 0]$): large negative $M = 10^9$ blocks any action with non-zero deficit.

During imagination (policy optimisation in latent space), the mask is applied using the *predicted* context $\hat{\mathbf{c}}$, allowing the agent to plan with feasibility awareness without access to the true game state.

#### 4. Integrated Plan2Explore (`dreamerv3/dreamerv3/agent.py`)

[Plan2Explore](https://arxiv.org/abs/2005.05960) integrated into DreamerV3. An ensemble of $K$ MLP heads predicts a target $\phi(z) \in \mathbb{R}^D$ in latent space; the mean per-dimension variance across heads forms the intrinsic reward:

$$r_{\text{p2e}} = \frac{1}{D}\sum_{d=1}^{D} \text{Var}_{k=1}^{K}\left[\hat{\phi}_{k,d}(z)\right]$$

Key parameters: `--disag_models` ($K$, default 10), `--disag_target` (`feat`|`stoch`|`deter`), `--expl_intr_scale` (default 0.9).

#### 5. Online Continual-Learning Metrics (`train_craftax.py`)

Per-episode metrics tracking 67 Craftax achievements across 5 tiers:

- **Per-achievement forgetting**: $F_a(t) = \max_{t' < t} \, p_a(t') - p_a(t)$
- **Aggregate forgetting**: $\bar{F}(t) = \frac{1}{|\mathcal{A}|}\sum_{a \in \mathcal{A}} F_a(t)$ (mean over all 67 achievements; unseen achievements contribute 0)
- **Achievement depth**: mean tier of achieved skills (0 = basic, 4 = endgame)
- **Frontier rate**: fraction of recent episodes reaching the agent's personal-best depth

All metrics written to `online_metrics.jsonl` and streamed to W&B.

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
