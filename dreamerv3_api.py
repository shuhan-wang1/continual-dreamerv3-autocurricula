"""
DreamerV3 API wrapper for continual learning experiments.
This module provides a compatible interface with the original DreamerV2 API.
"""
import collections
import os
import pathlib
import sys
from functools import partial as bind

# Add dreamerv3 to path (now nested under this package)
root = pathlib.Path(__file__).parent
sys.path.insert(0, str(root / 'dreamerv3'))
sys.path.insert(0, str(root / 'dreamerv3' / 'dreamerv3'))

import elements
import embodied
import numpy as np
import ruamel.yaml as yaml
import jax

from dreamerv3.agent import Agent


# Load DreamerV3 configs
_configs_path = root / 'dreamerv3' / 'dreamerv3' / 'configs.yaml'
_configs = yaml.YAML(typ='safe').load(_configs_path.read_text())
configs = {k: v for k, v in _configs.items() if k != 'defaults'}
defaults = elements.Config(_configs['defaults'])


class Config(elements.Config):
    """Extended config class with DreamerV2-style interface."""

    def parse_flags(self, argv=None):
        """Parse command line flags and return updated config."""
        return elements.Flags(self).parse(argv)

    def save(self, path):
        """Save config to yaml file."""
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.YAML().dump(dict(self), f)


# Convert defaults to our Config class
defaults = Config(dict(defaults))

# Add continual learning specific defaults
defaults = defaults.update({
    'cl': False,
    'cl_small': False,
    'num_tasks': 1,
    'num_task_repeats': 1,
    'tag': '',
    'eval_steps': 1e4,
    'sep_exp_eval_policies': False,
    'unbalanced_steps': 'None',
    'time_limit': 0,
    'prefill': 10000,
    'pretrain': 1,
    'train_every': 5,
    'train_steps': 1,
    'eval_every': 1e5,
    'log_every': 1e4,
    'log_every_video': 1e4,
    'log_recon_every': 1e5,
    'expl_until': 0,
    'expl_every': 0,
    'expl_behavior': 'greedy',
    'expl_intr_scale': 1.0,
    'expl_extr_scale': 0.0,
    'pred_discount': True,
    'grad_heads': ['decoder', 'reward', 'discount'],
    'skipped_metrics': [],
    # Embedding input config (default to embedding mode)
    'input_type': 'embedding',  # 'embedding' or 'pixel'
    'embedding_dim': 256,  # dimension of input embeddings
    # Wandb config
    'wandb': {
        'mode': 'online',
        'project': 'continual-dreamerv3',
        'entity': 'continual-dv3',
        'name': 'test',
        'group': 'test',
        'tags': None,
        'notes': None,
    },
    # Replay buffer config
    'replay': {
        'capacity': 2e6,
        'minlen': 50,
        'maxlen': 50,
        'prioritize_ends': True,
        'reservoir_sampling': False,
        'recent_past_sampl_thres': 0.0,
    },
    'dataset': {'batch': 16, 'length': 50},
    'log_keys_video': ['image'],
})


class GymWrapper(embodied.Env):
    """Wrapper to convert gym environments to embodied interface."""

    def __init__(self, env, obs_key='image', embedding_key='embedding'):
        self._env = env
        self._obs_key = obs_key
        self._embedding_key = embedding_key
        self._done = True
        self._info = None

        # Check if env has dict observation space
        self._obs_dict = hasattr(env.observation_space, 'spaces')
        self._act_dict = hasattr(env.action_space, 'spaces')

    @property
    def obs_space(self):
        import gym.spaces
        spaces = {}

        if self._obs_dict:
            for k, v in self._env.observation_space.spaces.items():
                spaces[k] = self._convert_space(v)
        else:
            space = self._env.observation_space
            # Determine if this is an image (3D) or embedding (1D/2D)
            if len(space.shape) == 3:
                spaces[self._obs_key] = self._convert_space(space)
            else:
                spaces[self._embedding_key] = self._convert_space(space)

        spaces.update({
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool),
        })
        return spaces

    @property
    def act_space(self):
        if self._act_dict:
            spaces = {k: self._convert_space(v)
                     for k, v in self._env.action_space.spaces.items()}
        else:
            spaces = {'action': self._convert_space(self._env.action_space)}
        spaces['reset'] = elements.Space(bool)
        return spaces

    def _convert_space(self, space):
        import gym.spaces
        if isinstance(space, gym.spaces.Discrete):
            return elements.Space(np.int32, (), 0, space.n)
        elif isinstance(space, gym.spaces.Box):
            return elements.Space(space.dtype, space.shape, space.low, space.high)
        else:
            raise NotImplementedError(f"Space type {type(space)} not supported")

    def step(self, action):
        if action['reset'] or self._done:
            self._done = False
            obs = self._env.reset()
            return self._obs(obs, 0.0, is_first=True)

        if self._act_dict:
            act = action
        else:
            act = action['action']
            # Handle discrete actions
            if hasattr(self._env.action_space, 'n'):
                if isinstance(act, np.ndarray):
                    act = int(act.item() if act.ndim == 0 else np.argmax(act))

        obs, reward, done, info = self._env.step(act)
        self._done = done
        self._info = info

        return self._obs(
            obs, reward,
            is_last=bool(done),
            is_terminal=bool(info.get('is_terminal', done)))

    def _obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
        if self._obs_dict:
            result = {k: np.asarray(v) for k, v in obs.items()}
        else:
            obs = np.asarray(obs)
            if len(obs.shape) == 3:
                result = {self._obs_key: obs}
            else:
                result = {self._embedding_key: obs}

        result.update(
            reward=np.float32(reward),
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal)
        return result

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass


class EmbeddingEnvWrapper(embodied.Env):
    """Wrapper that extracts embeddings from observations using a provided encoder."""

    def __init__(self, env, encoder_fn=None, embedding_dim=256):
        self._env = env
        self._encoder_fn = encoder_fn
        self._embedding_dim = embedding_dim
        self._done = True

    @property
    def obs_space(self):
        spaces = dict(self._env.obs_space)
        # Replace image space with embedding space
        if 'image' in spaces:
            del spaces['image']
        spaces['embedding'] = elements.Space(np.float32, (self._embedding_dim,))
        return spaces

    @property
    def act_space(self):
        return self._env.act_space

    def step(self, action):
        obs = self._env.step(action)
        if self._encoder_fn is not None and 'image' in obs:
            obs['embedding'] = self._encoder_fn(obs['image'])
            del obs['image']
        return obs

    def close(self):
        self._env.close()


def make_env_wrapper(gym_env, config):
    """Create an embodied environment from a gym environment."""
    env = GymWrapper(gym_env)

    # Apply wrappers based on config
    for name, space in env.act_space.items():
        if name != 'reset' and not space.discrete:
            env = embodied.wrappers.NormalizeAction(env, name)

    env = embodied.wrappers.UnifyDtypes(env)
    env = embodied.wrappers.CheckSpaces(env)

    return env


def make_agent(config, env):
    """Create a DreamerV3 agent."""
    notlog = lambda k: not k.startswith('log/')
    obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
    act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}

    agent_config = elements.Config(
        **config.agent,
        logdir=config.logdir,
        seed=config.seed,
        jax=config.jax,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        replay_context=config.get('replay_context', 1),
        report_length=config.get('report_length', 32),
        replica=config.get('replica', 0),
        replicas=config.get('replicas', 1),
    )

    return Agent(obs_space, act_space, agent_config)


def make_replay(config, directory):
    """Create a replay buffer."""
    length = config.dataset['length'] + config.get('replay_context', 1)
    capacity = int(config.replay.get('capacity', 2e6))

    return embodied.replay.Replay(
        length=length,
        capacity=capacity,
        directory=directory,
        online=True,
        chunksize=1024,
    )


def make_logger(config, step):
    """Create a logger."""
    logdir = config.logdir
    outputs = [
        elements.logger.TerminalOutput('.*', 'Agent'),
        elements.logger.JSONLOutput(logdir, 'metrics.jsonl'),
    ]

    multiplier = 1
    return elements.Logger(step, outputs, multiplier)


def train(gym_env, config, outputs=None):
    """
    Train DreamerV3 on a single environment.
    Compatible with DreamerV2 API.
    """
    # Set random seeds
    np.random.seed(config.seed)

    # Setup logging
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    config.save(logdir / 'config.yaml')
    print(config, '\n')
    print('Logdir', logdir)

    # Create environment wrapper
    env = make_env_wrapper(gym_env, config)

    # Create agent
    agent = make_agent(config, env)

    # Create replay buffer
    replay = make_replay(config, logdir / 'replay')

    # Create logger
    step = elements.Counter()
    logger = make_logger(config, step)

    # Create driver
    driver = embodied.Driver([lambda: env], parallel=False)
    driver.on_step(lambda tran, _: step.increment())
    driver.on_step(replay.add)

    # Prefill
    if config.prefill > 0:
        print(f'Prefill dataset ({config.prefill} steps).')
        random_policy = lambda carry, obs: (
            carry,
            {k: np.stack([v.sample() for _ in range(len(obs['is_first']))])
             for k, v in env.act_space.items() if k != 'reset'},
            {}
        )
        driver.reset()
        driver(random_policy, steps=config.prefill)

    # Setup training
    batch_size = config.dataset.get('batch', 16)
    batch_length = config.dataset.get('length', 50)

    def make_stream(replay, mode):
        fn = bind(replay.sample, batch_size, mode)
        stream = embodied.streams.Stateless(fn)
        stream = embodied.streams.Consec(
            stream,
            length=batch_length,
            consec=1,
            prefix=config.get('replay_context', 1),
            strict=(mode == 'train'),
            contiguous=True)
        return stream

    stream_train = iter(agent.stream(make_stream(replay, 'train')))
    carry_train = [agent.init_train(batch_size)]

    # Training loop
    print('Start training loop')
    policy = lambda carry, obs: agent.policy(carry, obs, mode='train')
    driver.reset(agent.init_policy)

    train_ratio = config.get('run', {}).get('train_ratio', 32.0)
    batch_steps = batch_size * batch_length
    should_train = elements.when.Ratio(train_ratio / batch_steps)

    while step < config.steps:
        driver(policy, steps=10)

        if len(replay) >= batch_size * batch_length:
            for _ in range(should_train(step)):
                batch = next(stream_train)
                carry_train[0], outs, mets = agent.train(carry_train[0], batch)
                if 'replay' in outs:
                    replay.update(outs['replay'])

    # Cleanup
    driver.close()
    logger.close()


def cl_train_loop(envs, config, outputs=None, eval_envs=None):
    """
    Continual learning training loop for DreamerV3.
    Compatible with DreamerV2 API.

    Args:
        envs: List of gym environments for training
        config: Configuration object
        outputs: Optional list of output handlers
        eval_envs: Optional list of environments for evaluation
    """
    import ast

    # Set random seeds
    np.random.seed(config.seed)

    # Parse unbalanced steps
    unbalanced_steps = None
    if hasattr(config, 'unbalanced_steps') and config.unbalanced_steps not in [None, 'None', 'none']:
        if isinstance(config.unbalanced_steps, str):
            unbalanced_steps = ast.literal_eval(config.unbalanced_steps)
        else:
            unbalanced_steps = config.unbalanced_steps

    # Setup logging
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    config.save(logdir / 'config.yaml')
    print(config, '\n')
    print('Logdir', logdir)

    # Create wrapped environments
    wrapped_envs = [make_env_wrapper(env, config) for env in envs]
    if eval_envs is None:
        eval_envs = envs
    wrapped_eval_envs = [make_env_wrapper(env, config) for env in eval_envs]

    # Use first environment to create agent
    env = wrapped_envs[0]

    # Create replay buffer
    replay = make_replay(config, logdir / 'replay')

    # Create logger
    total_step = elements.Counter()
    logger = make_logger(config, total_step)

    # Training state
    metrics = collections.defaultdict(list)

    # Determine starting point from replay buffer stats
    stats = replay.stats()
    total_steps = stats.get('total_steps', 0)

    if unbalanced_steps is not None:
        tot_steps_after_task = np.cumsum(unbalanced_steps)
        task_id = next((i for i, j in enumerate(total_steps < tot_steps_after_task) if j), 0)
        rep = int(total_steps // np.sum(unbalanced_steps))
        restart_step = (unbalanced_steps - tot_steps_after_task + total_steps)[task_id] if task_id < len(unbalanced_steps) else 0
    else:
        task_id = int(total_steps // config.steps) if config.steps > 0 else 0
        rep = int(total_steps // (config.steps * config.num_tasks)) if config.steps > 0 and config.num_tasks > 0 else 0
        restart_step = int(total_steps % config.steps) if config.steps > 0 else 0

    print(f"Task {task_id}, Rep {rep}, Restart step: {restart_step}")
    restart = restart_step > 0

    # Create agent (shared across tasks for continual learning)
    agent = make_agent(config, env)

    # Checkpoint
    cp = elements.Checkpoint(logdir / 'ckpt')
    cp.step = total_step
    cp.agent = agent
    cp.replay = replay
    cp.load_or_save()

    # Training parameters
    batch_size = config.dataset.get('batch', 16)
    batch_length = config.dataset.get('length', 50)
    train_ratio = config.get('run', {}).get('train_ratio', 32.0)
    batch_steps = batch_size * batch_length
    should_train = elements.when.Ratio(train_ratio / batch_steps)

    def make_stream(replay, mode):
        fn = bind(replay.sample, batch_size, mode)
        stream = embodied.streams.Stateless(fn)
        stream = embodied.streams.Consec(
            stream,
            length=batch_length,
            consec=1,
            prefix=config.get('replay_context', 1),
            strict=(mode == 'train'),
            contiguous=True)
        return stream

    # Continual learning loop
    while rep < config.num_task_repeats:
        while task_id < len(wrapped_envs):
            print(f"\n\t Task {task_id + 1} Rep {rep + 1} \n")

            current_env = wrapped_envs[task_id]

            if restart:
                start_step = restart_step
                restart = False
            else:
                start_step = 0

            step = elements.Counter(start_step)

            # Create driver for current task
            driver = embodied.Driver([lambda e=current_env: e], parallel=False)
            driver.on_step(lambda tran, _: total_step.increment())
            driver.on_step(lambda tran, _: step.increment())
            driver.on_step(replay.add)

            # Episode logging
            episodes = collections.defaultdict(elements.Agg)

            def logfn(tran, worker):
                episode = episodes[worker]
                tran['is_first'] and episode.reset()
                episode.add('score', tran['reward'], agg='sum')
                episode.add('length', 1, agg='sum')
                if tran['is_last']:
                    result = episode.result()
                    logger.add({
                        'score': result.pop('score'),
                        'length': result.pop('length'),
                        'task': task_id,
                    }, prefix='episode')
                    logger.write()

            driver.on_step(logfn)

            # Prefill on first task
            if total_step.value < config.prefill:
                prefill = config.prefill - total_step.value
                print(f'Prefill dataset ({prefill} steps).')
                random_policy = lambda carry, obs: (
                    carry,
                    {k: np.stack([v.sample() for _ in range(len(obs['is_first']))])
                     for k, v in current_env.act_space.items() if k != 'reset'},
                    {}
                )
                driver.reset()
                driver(random_policy, steps=int(prefill))

            # Setup training streams
            stream_train = iter(agent.stream(make_stream(replay, 'train')))
            carry_train = [agent.init_train(batch_size)]

            # Training policy
            policy = lambda carry, obs: agent.policy(carry, obs, mode='train')
            driver.reset(agent.init_policy)

            # Determine steps limit for this task
            if unbalanced_steps is not None:
                steps_limit = int(unbalanced_steps[task_id])
            else:
                steps_limit = int(config.steps)

            print(f'Training for {steps_limit} steps on task {task_id}')

            # Training loop for current task
            while step < steps_limit:
                driver(policy, steps=10)

                if len(replay) >= batch_size * batch_length:
                    for _ in range(should_train(total_step)):
                        batch = next(stream_train)
                        carry_train[0], outs, mets = agent.train(carry_train[0], batch)
                        if 'replay' in outs:
                            replay.update(outs['replay'])
                        for key, value in mets.items():
                            metrics[key].append(value)

                # Periodic evaluation
                if step.value % config.eval_every < 10:
                    # Evaluate on all eval environments
                    for eval_idx, eval_env in enumerate(wrapped_eval_envs):
                        eval_driver = embodied.Driver([lambda e=eval_env: e], parallel=False)
                        eval_returns = []

                        def eval_logfn(tran, worker):
                            if tran['is_last']:
                                eval_returns.append(tran['reward'])

                        eval_driver.on_step(eval_logfn)
                        eval_policy = lambda carry, obs: agent.policy(carry, obs, mode='eval')
                        eval_driver.reset(agent.init_policy)
                        eval_driver(eval_policy, steps=int(config.eval_steps))
                        eval_driver.close()

                        if eval_returns:
                            logger.add({f'eval_return_{eval_idx}': np.mean(eval_returns)})

                    logger.write()

                # Save checkpoint
                if step.value % 10000 < 10:
                    cp.save()

            driver.close()
            task_id += 1

        # Reset for next repetition
        task_id = 0
        rep += 1

    # Final save
    cp.save()
    logger.close()
