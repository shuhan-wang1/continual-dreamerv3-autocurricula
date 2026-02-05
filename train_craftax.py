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
from notebooks.metrics import OnlineMetrics, save_ref_metrics

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
    """Wrapper to convert Craftax JAX environments to embodied interface."""

    def __init__(self, env_name='CraftaxSymbolic-v1', embedding_dim=256, use_embedding=True, seed=42):
        self._env_name = env_name
        self._embedding_dim = embedding_dim
        self._use_embedding = use_embedding
        self._seed = seed
        self._done = True
        
        # Use a base key and step counter for deterministic key generation
        self._base_key = jax.random.PRNGKey(seed)
        self._step_count = 0
        
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
        
        # JIT compile the key generation function to avoid host-to-device transfers
        @jax.jit
        def make_key(base_key, step):
            return jax.random.fold_in(base_key, step)
        self._make_key = make_key
        
        # Initialize state for space inference - use jit compiled reset
        self._reset_fn = jax.jit(self._env.reset)
        self._step_fn = jax.jit(self._env.step)
        
        # Get initial observation to determine shape
        init_key = self._make_key(self._base_key, 0)
        obs, self._state = self._reset_fn(init_key, self._env_params)
        self._step_count = 1
        self._obs = obs
        
        # Determine observation shape from the observation
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
            # Fallback for other observation types
            obs_array = np.asarray(obs)
            self._obs_shape = obs_array.shape
        
        # Setup embedding projection if needed
        if use_embedding:
            np.random.seed(42)
            flat_dim = int(np.prod(self._obs_shape))
            self._projection = np.random.randn(flat_dim, embedding_dim).astype(np.float32)
            self._projection /= np.sqrt(flat_dim)
        
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
        """Get next random key using JIT-compiled fold_in to avoid device transfers."""
        step_on_device = jax.device_put(self._step_count)
        key = self._make_key(self._base_key, step_on_device)
        self._step_count += 1
        return key

    def _process_obs(self, obs):
        """Process observation: convert to numpy and optionally embed."""
        # Handle different observation types
        if hasattr(obs, 'shape'):
            obs_array = np.asarray(obs)
        elif isinstance(obs, dict):
            if 'obs' in obs:
                obs_array = np.asarray(obs['obs'])
            elif 'pixels' in obs:
                obs_array = np.asarray(obs['pixels'])
            else:
                obs_array = np.asarray(jax.tree_util.tree_leaves(obs)[0])
        else:
            obs_array = np.asarray(obs)
        
        # Normalize to [0, 1] if needed
        if obs_array.dtype == np.uint8:
            obs_array = obs_array.astype(np.float32) / 255.0
        else:
            obs_array = obs_array.astype(np.float32)
        
        if self._use_embedding:
            flat = obs_array.flatten()
            embedding = np.dot(flat, self._projection)
            return embedding
        else:
            return obs_array

    def step(self, action):
        if action['reset'] or self._done:
            reset_key = self._get_rng_key()
            self._obs, self._state = self._reset_fn(reset_key, self._env_params)
            self._done = False
            obs_processed = self._process_obs(self._obs)
            result = {'embedding': obs_processed} if self._use_embedding else {'image': obs_processed}
            result.update(reward=np.float32(0.0), is_first=True, is_last=False, is_terminal=False)
            return result
        
        # Get action
        act = action['action']
        if isinstance(act, np.ndarray):
            act = int(act.item() if act.ndim == 0 else act[0])
        
        # Step environment using jitted step function
        step_key = self._get_rng_key()
        self._obs, self._state, reward, done, info = self._step_fn(
            step_key, self._state, act, self._env_params
        )
        
        self._done = bool(done)
        obs_processed = self._process_obs(self._obs)
        result = {'embedding': obs_processed} if self._use_embedding else {'image': obs_processed}
        result.update(
            reward=np.float32(float(reward)),
            is_first=False,
            is_last=self._done,
            is_terminal=self._done,
        )
        return result

    def close(self):
        pass


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
        replay_context=config.replay_context,
        report_length=config.report_length,
        replica=config.replica,
        replicas=config.replicas,
    )
    return Agent(obs_space, act_space, agent_config)


def make_selector(args, capacity, seed=0):
    """Create a replay selector based on sampling strategy args.

    Supports:
    - Uniform sampling (default)
    - Reservoir sampling (random eviction) - use with eviction='reservoir' in make_replay
    - Recency-biased sampling
    - 50:50 sampling (half random, half recent) - Continual-Dreamer strategy
    - Mixture of multiple strategies
    """
    from embodied.core import selectors

    # Check if using 50:50 sampling (Continual-Dreamer strategy)
    # This is the recommended setup for continual learning with 8+ tasks
    recent_frac = getattr(args, 'recent_frac', 0.0)
    if recent_frac > 0:
        # Mixture of uniform (random from buffer) and recent (recent experience)
        window_size = getattr(args, 'recent_window', 1000)
        selector_dict = {
            'uniform': selectors.Uniform(seed=seed),
            'recent': selectors.Recent(window_size=window_size, seed=seed + 1),
        }
        fractions = {
            'uniform': 1.0 - recent_frac,
            'recent': recent_frac,
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


def load_config(args):
    """Load DreamerV3 config from YAML and merge with args."""
    configs_path = root / 'dreamerv3' / 'dreamerv3' / 'configs.yaml'
    configs = yaml.YAML(typ='safe').load(configs_path.read_text())
    config = elements.Config(configs['defaults'])
    
    # Apply size preset if specified
    if args.cl_small and 'small' in configs:
        config = config.update(configs['small'])
    
    tag = args.tag + str(args.seed)
    
    # Build overrides from args
    overrides = {
        'logdir': f'{args.logdir}/craftax_{tag}',
        'seed': args.seed,
        'batch_size': args.batch_size,
        'replay_context': 0,  # Disable replay context to avoid needing dyn/deter and dyn/stoch in replay
        'run': {
            'steps': int(args.steps),
            'log_every': 1000,
            'save_every': 10000,
            'report_every': 50000,
            'train_ratio': 64.0,
        },
        'replay': {
            'size': int(args.replay_capacity),
        },
    }
    config = config.update(overrides)
    return config, tag


def train_single(make_env, config, args):
    """Train DreamerV3 on a single environment.

    Args:
        make_env: Callable(seed: int) -> embodied.Env that creates one env instance.
    """
    np.random.seed(config.seed)
    logdir = elements.Path(config.logdir)
    logdir.mkdir()
    config.save(logdir / 'config.yaml')
    print('Logdir:', logdir)

    num_envs = config.run.envs
    print(f'Creating {num_envs} parallel environments.')
    env_fns = [lambda i=i: wrap_env(make_env(config.seed + i)) for i in range(num_envs)]
    driver = embodied.Driver(env_fns, parallel=False)

    agent = make_agent(config, driver.envs[0])
    replay = make_replay(config, logdir / 'replay', args)

    step = elements.Counter()
    logger = make_logger(config, step)
    online = None
    if getattr(args, 'online_metrics', True):
        steps_per_task = int(config.run.steps)
        online = OnlineMetrics(
            logdir=str(logdir),
            num_tasks=1,
            steps_per_task=steps_per_task,
            ref_metrics_path=getattr(args, 'ref_metrics_path', None),
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

    total_steps = int(config.run.steps)
    while step < total_steps:
        driver(policy, steps=10)
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
        online.mark_task_end(0)
        online.save_summary()
        ref_path = os.path.join(str(logdir), 'ref_metrics.json')
        ref_auc = {'0': online.auc_norm_list()[0]}
        save_ref_metrics(ref_path, ref_auc, steps_per_task)
    driver.close()
    logger.close()


def cl_train_loop(make_envs, config, args):
    """Continual learning training loop for DreamerV3.

    Args:
        make_envs: List of Callable(seed: int) -> embodied.Env, one per task.
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

    agent = make_agent(config, env0)
    replay = make_replay(config, logdir / 'replay', args)

    total_step = elements.Counter()
    logger = make_logger(config, total_step)
    online = None
    if getattr(args, 'online_metrics', True):
        steps_per_task = unbalanced_steps if unbalanced_steps is not None else int(args.steps)
        online = OnlineMetrics(
            logdir=str(logdir),
            num_tasks=len(make_envs),
            steps_per_task=steps_per_task,
            ref_metrics_path=getattr(args, 'ref_metrics_path', None),
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

    while rep < args.num_task_repeats:
        while task_id < num_tasks:
            print(f"\n=== Task {task_id + 1}/{num_tasks}, Rep {rep + 1}/{args.num_task_repeats} ===\n")

            make_task_env = make_envs[task_id]
            env_fns = [lambda i=i, fn=make_task_env: wrap_env(fn(config.seed + i)) for i in range(num_envs)]
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

            driver = embodied.Driver(env_fns, parallel=False)
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
                driver(policy, steps=10)
                if len(replay) >= batch_size * batch_length:
                    for _ in range(should_train(total_step)):
                        batch = next(stream_train)
                        carry_train[0], outs, mets = agent.train(carry_train[0], batch)
                        if 'replay' in outs:
                            replay.update(outs['replay'])
                if step.value % 10000 < 10:
                    cp.save()

            driver.close()
            if online is not None:
                online.mark_task_end(task_id)
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
        for i in range(args.num_tasks):
            name = env_names[i % len(env_names)]
            make_env_fns.append(lambda seed, name=name: make_craftax(
                name,
                embedding_dim=args.embedding_dim,
                use_embedding=use_embedding,
                seed=seed,
            ))
            print(f"Task {i}: env {name}, use_embedding: {use_embedding}")

        cl_train_loop(make_env_fns, config, args)
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

        train_single(make_env, config, args)

    if args.del_exp_replay:
        shutil.rmtree(os.path.join(config.logdir, 'replay'), ignore_errors=True)

    wandb.finish()


if __name__ == "__main__":
    args = parse_craftax_args()
    run_craftax(args)
