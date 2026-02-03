import os
import shutil
import wandb
import gym
from gym_minigrid.wrappers import *

# Use DreamerV3 API instead of DreamerV2
import dreamerv3_api as dv3
from input_args import parse_minigrid_args

import jax
# Configure JAX memory allocation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"


class EmbeddingWrapper(gym.ObservationWrapper):
    """
    Wrapper that converts image observations to embeddings.
    For actual use, replace the random projection with a proper encoder.
    """
    def __init__(self, env, embedding_dim=256):
        super().__init__(env)
        self.embedding_dim = embedding_dim
        # Get the original observation space shape
        if hasattr(env.observation_space, 'shape'):
            self.orig_shape = env.observation_space.shape
        else:
            self.orig_shape = (64, 64, 3)  # default

        # Create embedding observation space
        import numpy as np
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(embedding_dim,),
            dtype=np.float32
        )

        # Simple random projection matrix (replace with actual encoder)
        np.random.seed(42)
        flat_dim = int(np.prod(self.orig_shape))
        self.projection = np.random.randn(flat_dim, embedding_dim).astype(np.float32)
        self.projection /= np.sqrt(flat_dim)

    def observation(self, obs):
        import numpy as np
        # Flatten and project to embedding space
        flat = obs.flatten().astype(np.float32) / 255.0
        embedding = np.dot(flat, self.projection)
        return embedding


def run_minigrid(args):
    tag = args.tag + "_" + str(args.seed)

    # Build config using DreamerV3 defaults
    config = dv3.defaults.update({
        'logdir': '{0}/minigrid_{1}'.format(args.logdir, tag),
        'log_every': 1e3,
        'log_every_video': 2e5,
        'train_every': 10,
        'prefill': 1e4,
        'time_limit': 100,
        'steps': args.steps,
        'cl': args.cl,
        'num_tasks': args.num_tasks,
        'num_task_repeats': args.num_task_repeats,
        'seed': args.seed,
        'eval_every': 1e4,
        'eval_steps': 1e3,
        'tag': tag,
        "unbalanced_steps": str(args.unbalanced_steps) if args.unbalanced_steps else 'None',
        'sep_exp_eval_policies': args.sep_exp_eval_policies,
        # Embedding input configuration
        'input_type': args.input_type,
        'embedding_dim': args.embedding_dim,
        # Wandb config
        'wandb': {
            'group': args.wandb_group,
            'name': f"DreamerV3_cl_{tag}" if args.cl else f"DreamerV3_single-env={args.env}_{tag}",
            'project': args.wandb_proj_name,
        },
        # Replay buffer config
        'replay': {
            'capacity': int(args.replay_capacity),
            'reservoir_sampling': args.reservoir_sampling,
            'recent_past_sampl_thres': args.recent_past_sampl_thres,
            'minlen': args.minlen,
        },
        # Agent config for embedding input
        'agent': dv3.defaults.agent.update({
            'enc': {
                'typ': 'simple',
                'simple': {
                    'depth': 64,
                    'mults': [2, 3, 4, 4],
                    'layers': 3,
                    'units': 512,
                    'act': 'silu',
                    'norm': 'rms',
                    'winit': 'trunc_normal_in',
                    'symlog': True,  # Use symlog for embeddings
                    'outer': False,
                    'kernel': 5,
                    'strided': False,
                },
            },
        }),
    })

    # Plan2Explore configuration
    if args.plan2explore:
        config = config.update({
            'expl_behavior': 'Plan2Explore',
            'pred_discount': args.rssm_full_recon,
            'grad_heads': ['decoder', 'reward', 'discount'] if args.rssm_full_recon else ['decoder'],
            'expl_intr_scale': args.expl_intr_scale,
            'expl_extr_scale': args.expl_extr_scale,
            'expl_every': args.expl_every,
            'wandb': {
                'name': f"Plan2Explore_cl_{tag}" if args.cl else f"Plan2Explore_single-env={args.env}_{tag}",
            },
        })

    # Parse flags
    config = config.parse_flags()

    # Initialize wandb
    wandb.init(
        config=dict(config),
        reinit=True,
        resume=False,
        dir=args.wandb_dir,
        **config.wandb,
    )

    if config.cl:
        env_names = [
            'MiniGrid-DoorKey-9x9-v0',
            'MiniGrid-LavaCrossingS9N1-v0',
            'MiniGrid-SimpleCrossingS9N1-v0',
        ]

        envs = []
        for i in range(config.num_tasks):
            name = env_names[i]
            env = gym.make(name)
            env = RGBImgPartialObsWrapper(env)
            if args.state_bonus:
                assert not args.plan2explore, "state bonus versus plan2explore experiment"
                env = StateBonus(env)
            # Apply embedding wrapper for embedding input mode
            if args.input_type == 'embedding':
                env = EmbeddingWrapper(env, embedding_dim=args.embedding_dim)
            envs.append(env)

        if args.eval_skills:
            env_names = [
                'MiniGrid-DoorKey-9x9-v0',
                'MiniGrid-LavaCrossingS9N1-v0',
                'MiniGrid-SimpleCrossingS9N1-v0',
                'MiniGrid-MultiSkill-N2-v0',
            ]

            eval_envs = []
            for i in range(len(env_names)):
                name = env_names[i]
                env = gym.make(name)
                env = RGBImgPartialObsWrapper(env)
                if args.input_type == 'embedding':
                    env = EmbeddingWrapper(env, embedding_dim=args.embedding_dim)
                eval_envs.append(env)
        else:
            eval_envs = []
            for i in range(config.num_tasks):
                name = env_names[i]
                env = gym.make(name)
                env = RGBImgPartialObsWrapper(env)
                if args.input_type == 'embedding':
                    env = EmbeddingWrapper(env, embedding_dim=args.embedding_dim)
                eval_envs.append(env)

        dv3.cl_train_loop(envs, config, eval_envs=eval_envs)

    else:
        env_names = [
            'MiniGrid-DoorKey-9x9-v0',
            'MiniGrid-LavaCrossingS9N1-v0',
            'MiniGrid-SimpleCrossingS9N1-v0',
        ]

        name = env_names[args.env]
        env = gym.make(name)
        env = RGBImgPartialObsWrapper(env)
        if args.input_type == 'embedding':
            env = EmbeddingWrapper(env, embedding_dim=args.embedding_dim)
        dv3.train(env, config)

    if args.del_exp_replay:
        shutil.rmtree(os.path.join(config['logdir'], 'replay'), ignore_errors=True)

    wandb.finish()


if __name__ == "__main__":
    args = parse_minigrid_args()
    run_minigrid(args)
