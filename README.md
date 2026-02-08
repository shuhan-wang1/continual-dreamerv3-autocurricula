# Continual DreamerV3 Autocurricula

This repository contains code for continual reinforcement learning experiments using DreamerV3 with JAX-based environments (Craftax and NAVIX). Based on the original Continual DreamerV2 work: [On The Effectiveness of World Models For Continual Reinforcement Learning](https://arxiv.org/abs/2211.15944).

```
@article{kessler2022surprising,
  title={The surprising effectiveness of latent world models for continual reinforcement learning},
  author={Kessler, Samuel and Mi{\l}o{\'s}, Piotr and Parker-Holder, Jack and Roberts, Stephen J},
  journal={arXiv preprint arXiv:2211.15944},
  year={2022}
}
```

## Installation

Create a conda environment with Python >= 3.9:

```sh
conda create -n continual-dv3 python=3.10
conda activate continual-dv3
pip install -r requirements.txt
```

## Supported Environments

This codebase now uses **JAX-based environments** for faster training:

### Craftax

[Craftax](https://github.com/MichaelTMatthews/Craftax) is a JAX-based reimplementation of Crafter with ~100x speedup.

Available environments:

- `CraftaxSymbolic-v1` - Symbolic observation (flat vector)
- `CraftaxPixels-v1` - Pixel observation
- `CraftaxClassicSymbolic-v1` - Classic version with symbolic obs
- `CraftaxClassicPixels-v1` - Classic version with pixels

### NAVIX

[NAVIX](https://github.com/epignatelli/navix) is a JAX-based reimplementation of MiniGrid with ~1000x speedup.

Available environments:

- `Navix-Empty-8x8-v0`
- `Navix-DoorKey-6x6-v0`
- `Navix-LavaCrossing-9x9-v0`
- `Navix-KeyCorridor-S3R1-v0`
- `Navix-Dynamic-Obstacles-6x6-v0`
- `Navix-MultiRoom-N2S4-v0`

## Training

### Unified Training Script

Use the unified training script to select environment type:

```sh
# Train on Craftax (single environment)
python train.py --env_type craftax --env 0 --steps 500000 --seed 42

# Train on NAVIX (single environment)
python train.py --env_type navix --env 0 --steps 500000 --seed 42

# Continual learning on Craftax
python train.py --env_type craftax --cl --num_tasks 2 --num_task_repeats 3 --steps 250000

# Continual learning on NAVIX
python train.py --env_type navix --cl --num_tasks 3 --num_task_repeats 2 --steps 200000
```

### Direct Training Scripts

You can also use the environment-specific scripts directly:

## Craftax Training

```sh
# Single environment training
python train_craftax.py --env 0 --steps 500000 --seed 42 --tag craftax_single

# Continual learning (multiple tasks)
python train_craftax.py --cl --num_tasks 2 --num_task_repeats 3 --steps 250000 --tag craftax_cl

# Small configuration for testing
python train_craftax.py --cl --cl_small --num_tasks 2 --steps 100000 --tag craftax_small

python train.py --env_type craftax \
    --cl \
    --steps 500000 \
    --batch_length 64 \
    --batch_size 16 \
    --envs 16 \
    --model_size 25m \
    --tag "4090_optimal"
```

## NAVIX Training (MiniGrid replacement)

```sh
# Single environment training
python train_navix.py --env 0 --steps 500000 --seed 42 --tag navix_single

# Continual learning
python train_navix.py --cl --num_tasks 3 --num_task_repeats 2 --steps 200000 --tag navix_cl

# With Plan2Explore
python train_navix.py --cl --num_tasks 3 --plan2explore --expl_intr_scale 0.9 --steps 300000
```

## Key Arguments

| Argument               | Description                                       | Default       |
| ---------------------- | ------------------------------------------------- | ------------- |
| `--cl`               | Enable continual learning mode                    | False         |
| `--cl_small`         | Use small CL configuration                        | False         |
| `--num_tasks`        | Number of tasks for CL                            | 1             |
| `--num_task_repeats` | Number of times to repeat task sequence           | 1             |
| `--steps`            | Training steps per task                           | 500000        |
| `--seed`             | Random seed                                       | 42            |
| `--env`              | Environment index for single-task training        | 0             |
| `--input_type`       | Observation type: 'embedding' or 'pixel'          | embedding     |
| `--embedding_dim`    | Embedding dimension                               | 256           |
| `--batch_size`       | Mini-batch size                                   | 16            |
| `--replay_capacity`  | Replay buffer size                                | 2000000       |
| `--envs`             | Number of parallel environments (run.envs)        | 64            |
| `--eval_envs`        | Number of evaluation environments (run.eval_envs) | 4             |
| `--plan2explore`     | Enable Plan2Explore exploration                   | False         |
| `--wandb_proj_name`  | W&B project name                                  | craftax/navix |
| `--wandb_group`      | W&B experiment group                              | experiment    |
| `--logdir`           | Log directory                                     | logs          |

### Reducing GPU/CPU Memory Usage

The default DreamerV3 config uses 64 parallel environments. If you see high memory usage or low utilization, reduce parallelism:

```sh
python train_craftax.py --envs 8 --eval_envs 1 ...
python train_navix.py --envs 8 --eval_envs 1 ...
```

## Embedding Mode

By default, all environments output **embeddings** instead of raw pixels for more efficient training:

- Image observations are flattened and projected to a lower-dimensional embedding space
- Set `--input_type pixel` to use raw pixel observations

## Legacy Scripts

The original MiniGrid and MiniHack training scripts are still available but deprecated:

- `train_minigrid.py` - Uses gym-minigrid (CPU-based)
- `train_minihack.py` - Uses MiniHack/NLE (CPU-based)

We recommend using `train_navix.py` (NAVIX) and `train_craftax.py` (Craftax) for significantly faster training with JAX acceleration.

## Logging

Training logs are saved to:

- `{logdir}/{env_type}_{tag}/` - Contains config, metrics, and checkpoints
- W&B: Real-time training visualization (configure with `--wandb_*` args)

## Performance Comparison

JAX-based environments offer significant speedups:

- **NAVIX** (MiniGrid replacement): ~1000x faster than gym-minigrid
- **Craftax** (Crafter replacement): ~100x faster than original CrafterCitation

If you use this code, please cite:

```bibtex
@article{kessler2022surprising,
  title={The surprising effectiveness of latent world models for continual reinforcement learning},
  author={Kessler, Samuel and Mi{\l}o{\'s}, Piotr and Parker-Holder, Jack and Roberts, Stephen J},
  journal={arXiv preprint arXiv:2211.15944},
  year={2022}
}
```
