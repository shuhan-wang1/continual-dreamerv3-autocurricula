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
├── input_args.py             # All argument parsers
├── dreamerv3/                # ENHANCED DreamerV3 (our modified version)
│   ├── dreamerv3/
│   │   ├── agent.py          # Agent with integrated Plan2Explore
│   │   ├── rssm.py           # RSSM encoder/decoder
│   │   └── expl.py           # Plan2Explore ensemble implementation
│   └── embodied/core/
│       ├── selectors.py      # All replay sampling selectors (key innovation)
│       └── replay.py         # Replay buffer with selector integration
├── dreamerv3-main/           # Original DreamerV3 (baseline, --use_original_dreamer)
├── experiment_results/       # Saved CSVs and analysis scripts
└── notebooks/                # Metric plotting utilities
```

### Our Innovations

#### 1. Advanced Replay Sampling Strategies (`dreamerv3/embodied/core/selectors.py`)

We introduce several new replay buffer sampling selectors designed for continual learning:

- **Reservoir sampling** — Random eviction (Algorithm 2, Vitter 1985). Prevents the buffer from becoming dominated by the most recent task, ensuring uniform coverage across all past experience.
- **50:50 Recent/Uniform mixture** — Half of each mini-batch comes from the most recent `--recent_window` episodes; the other half is uniformly sampled from the full buffer. This balances plasticity (learning the current task) with stability (retaining past experience).
- **Reward-weighted sampling** — Episodes are sampled proportionally to their cumulative return (softmax). This focuses learning on high-quality trajectories.
- **NLR / NLU (Novelty–Learnability–Recency/Uniform)** — Our primary novel contribution. Each mini-batch is split into three pools:
  - **Novelty pool** — Trajectories containing rarely accomplished achievements, prioritising skills the agent has not yet mastered.
  - **Learnability pool** — Trajectories whose reward exceeds a running EMA baseline (GRPO-style advantage), prioritising episodes that are "hard but solvable".
  - **Recency / Uniform pool** — Recent experience (NLR) or uniform random (NLU) to maintain coverage.

  Two variants are provided:
  - `--nlr_privileged_sampling` / `--nlu_privileged_sampling` — Use Craftax per-achievement success rates (privileged info, not available to the policy).
  - `--nlr_sampling` / `--nlu_sampling` — Use a Bayesian 2-D (reward × episode-length) grid to estimate rarity without accessing privileged info.

#### 2. Integrated Plan2Explore (`dreamerv3/dreamerv3/expl.py`, `agent.py`)

[Plan2Explore](https://arxiv.org/abs/2005.05960) is integrated directly into the DreamerV3 agent. An ensemble of `--disag_models` MLP heads predicts a target in the world model's latent space; their disagreement (standard deviation) forms an intrinsic reward. This encourages the agent to visit novel states and supports exploration in the sparse-reward Craftax environment.

Key parameters: `--disag_models`, `--disag_target` (`feat` | `stoch` | `deter`), `--expl_intr_scale`, `--expl_extr_scale`.

#### 3. Spatial-Counting + Craft-Novelty Intrinsic Reward (`train_craftax.py`)

Environment-level intrinsic rewards that shape the reward signal before it enters the world model:

- **Spatial-counting** — `r_spatial = 1/sqrt(N_ep(phi(o_t)))` where `phi` hashes the map tile at the agent's position. Encourages visiting new map locations within each episode.
- **Craft-novelty** — `r_craft = I[psi(o_t) not in H_inv]` where `psi` is the inventory configuration hash. Binary bonus for discovering new inventory states.
- **Combined reward** — `r = alpha_i * norm(r_intr) + alpha_e * r_extrinsic`, where `norm` adaptively scales `r_intr` to match `mean(|r_extr|)` via cross-episode EMA, so `alpha_i / alpha_e` is the true relative weight.

Key parameters: `--intrinsic_spatial`, `--alpha_i` (default 0.1), `--alpha_e` (default 1.0), `--craft_weight` (lambda, default 1.0).

#### 4. Online Continual-Learning Metrics (`train_craftax.py` — `CraftaxMetrics`)

A comprehensive metrics tracker records:
- **Per-achievement success rates** — 67 binary achievements tracked per task per episode.
- **Forgetting** — $F_a = \max_{t' < t} p_a(t') - p_a(t)$, measuring performance degradation on previously mastered achievements.
- **Frontier rate** — Fraction of recent episodes reaching the agent's personal-best exploration depth.
- **Score tier distribution** — Episodes bucketed by achievement tier (0–4).
- **Replay buffer diagnostics** — Buffer composition, sampling statistics.

All metrics are written to `online_metrics.jsonl` and summarised in `metrics_summary.json` under the log directory, and streamed to W&B.

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

# Continual learning with NLR sampling + Plan2Explore
python train.py --env_type craftax \
    --cl --num_tasks 2 --num_task_repeats 3 \
    --steps 250000 \
    --nlr_privileged_sampling \
    --plan2explore --expl_intr_scale 0.9 \
    --tag cl_nlr_p2e
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

### Replay Sampling Strategy

Choose **one** of the following strategies (they are mutually exclusive):

| Argument | Description | Default |
|---|---|---|
| `--reservoir_sampling` | Reservoir sampling (random eviction, uniform coverage) | `True` |
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
| `--nlr_sampling` | NLR with non-privileged 2-D reward×length grid novelty | `False` |
| `--nlu_sampling` | NLU with non-privileged 2-D grid novelty | `False` |
| `--nlr_novel_frac` | Fraction of mini-batch from the novelty pool | `0.35` |
| `--nlr_learnable_frac` | Fraction of mini-batch from the learnability pool | `0.35` |
| `--nlr_recent_frac` | Fraction of mini-batch from the recency/uniform pool | `0.30` |
| `--nlr_recent_window` | Window size for the recency pool | `1000` |
| `--nlr_reward_ema_decay` | EMA decay for learnability reward baseline | `0.99` |
| `--nlr_novelty_temp` | Temperature for novelty pool softmax | `1.0` |
| `--nlr_learnability_temp` | Temperature for learnability pool softmax | `1.0` |
| `--nlr_novelty_eps` | (Privileged only) Epsilon smoothing for achievement success rates | `0.01` |
| `--nlr_grid_reward_bins` | (Non-privileged) Number of quantile bins on the reward axis | `5` |
| `--nlr_grid_length_bins` | (Non-privileged) Number of quantile bins on the length axis | `10` |
| `--nlr_grid_recompute_every` | (Non-privileged) Recompute grid bin edges every N episodes | `500` |
| `--nlr_grid_prior_percentile` | (Non-privileged) Percentile of reward distribution for prior R_min | `0.20` |
| `--nlr_grid_eps` | (Non-privileged) Smoothing epsilon for bin counts | `0.01` |

### Exploration (Plan2Explore)

| Argument | Description | Default |
|---|---|---|
| `--plan2explore` | Enable Plan2Explore intrinsic exploration | `True` |
| `--no_plan2explore` | Disable Plan2Explore | — |
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

A systematic ablation study is provided via `run_ablation.py`. It covers 13 experiment configurations across 4 groups, each run with 3 seeds (39 total runs).

### Experiment Groups

| Group | Focus | Experiments |
|---|---|---|
| **A** | Core comparison | A1 baseline, A2 P2E, A3 intrinsic, A4 P2E+intrinsic |
| **B** | Craft-weight sensitivity | B1 spatial-only, B2 light (0.5), B3 heavy (2.0) |
| **C** | Reward scale (alpha_i sensitivity) | C1 tiny (0.01), C2 high (0.3), C3 equal (1.0) |
| **D** | Replay strategy (NLR interaction) | D1 NLR+P2E, D2 NLR+intrinsic, D3 NLR+P2E+intrinsic |

### Default Hyperparameters

All experiments use: `--steps 1000000 --batch_size 16 --batch_length 64 --envs 16 --model_size 25m`

### Running Experiments

```sh
# Run all 39 experiments (13 configs × 3 seeds)
python run_ablation.py

# Dry run — print commands without executing
python run_ablation.py --dry_run

# Run a specific group
python run_ablation.py --only A

# Run specific experiments
python run_ablation.py --only A1_baseline,D3_nlr_p2e_intrinsic

# Skip a group
python run_ablation.py --skip C,D

# Disable W&B logging
python run_ablation.py --wandb_mode disabled

# Select GPU
python run_ablation.py --gpu 0

# Keep replay buffer after each run (default: replay deleted, ckpt kept)
python run_ablation.py --no_cleanup
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
