import argparse

def parse_minigrid_args(args=None):
    parser = argparse.ArgumentParser(description="Continual DV2 Minigrid")

    parser.add_argument('--cl', default=False, action='store_true',
                        help='Flag for continual learning loop.')
    parser.add_argument('--num_tasks', type=int, default=1)
    parser.add_argument('--num_task_repeats', type=int, default=1)
    parser.add_argument('--steps', type=int, default=5e5)
    parser.add_argument('--env', type=int, default=0, help='picks the env for the single task dv2.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tag', type=str, default='', help='unique str to tag tb.')
    parser.add_argument('--del_exp_replay', default=False, action='store_true',
                        help='Flag to delete the training episodes after running the script to save storage space.')
    parser.add_argument('--logdir', type=str, default='logs', help='directory for the tb logs and exp replay episodes.')
    parser.add_argument('--eval_skills', default=False, action='store_true',
                        help='Flag evaluating our model on the multiskill envs.')
    parser.add_argument('--wandb_group', type=str, default='experiment', help='name of the gruop in wandb')
    parser.add_argument('--wandb_proj_name', type=str, default='minihack',
                        help='unique str for wandb projs.')
    parser.add_argument('--wandb_dir', type=str, default=None,
                        help='unique str for wandb directory.')
    parser.add_argument('--state_bonus', default=False, action='store_true',
                    help='Flag to decide whether to use a state bonus.')
    # Input type
    parser.add_argument('--input_type', type=str, default='embedding', choices=['embedding', 'pixel'],
                        help='Observation input type. Default is embedding.')
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Dimension of the embedding input.')
    # DV2
    parser.add_argument('--replay_capacity', type=int, default=2e6)
    parser.add_argument('--reservoir_sampling', default=True, action='store_true',
                        help='Flag for using reservoir sampling.') 
    parser.add_argument('--recent_past_sampl_thres', type=float, default=0.5,
                        help="probability of triangle distribution, expected to be > 0 and <= 1. 0 denotes taking episodes always from uniform distribution.")
    parser.add_argument('--minlen', type=int, default=50,
                        help="minimal episode length of episodes stored in the replay buffer")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="mini-batch size")
    parser.add_argument('--unbalanced_steps', type=list, default=None,
                        help="number of steps per each task")

    # exploration
    parser.add_argument('--plan2explore', default=False, action='store_true',
                            help='Enable plan2explore exploration strategy.')
    parser.add_argument('--expl_intr_scale', type=float, default=1.0,
                        help="scale of the intrinsic reward needs to be > 0 and <= 1.")
    parser.add_argument('--expl_extr_scale', type=float, default=0.0,
                        help="scale of the extrinsic reward needs to be > 0 and <= 1.")
    parser.add_argument('--expl_every', type=int, default=0, 
                        help="how often to run the exploration strategy.")
    parser.add_argument('--sep_exp_eval_policies', default=False, action='store_true',
                        help='Whether to use separate exploration and evaluation polcies.')
    parser.add_argument('--rssm_full_recon', default=False, action='store_true',
                        help='Whether to have the WM reconstruct the obs, discount and rewards rather than the obs only for p2e only.')

    args = parser.parse_known_args(args=args)[0]
    return args

def parse_minihack_args(args=None):
    parser = argparse.ArgumentParser(description="Continual Dv2")

    parser.add_argument('--cl', default=False, action='store_true',
                        help='Flag for continual learning loop.')
    parser.add_argument('--cl_small', default=False, action='store_true',
                        help='Flag for continual learning loop.')                        
    parser.add_argument('--del_exp_replay', default=False, action='store_true',
                        help='Flag to delete the training episodes after running the script to save storage space.')
    parser.add_argument('--num_tasks', type=int, default=1)
    parser.add_argument('--num_task_repeats', type=int, default=1)
    parser.add_argument('--steps', type=int, default=5e5)
    parser.add_argument('--train_every', type=int, default=10, help="")

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--env', type=int, default=0, help='picks the env for the single task dv2.')
    parser.add_argument('--tag', type=str, default='', help='unique str to tag tb.')
    parser.add_argument('--logdir', type=str, default='logs', help='directory for the tb logs and exp replay episodes.')
    # Interference
    parser.add_argument("--env_seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument('--proc', default=False, action='store_true',
                    help='Flag decide whether to train on 2 interfereing tasks strictly or on procedurally generated envs.')

    # wandb
    parser.add_argument('--wandb_group', type=str, default='experiment', help='name of the gruop in wandb')
    parser.add_argument('--wandb_proj_name', type=str, default='minihack',
                        help='unique str for wandb projs.')
    parser.add_argument('--wandb_dir', type=str, default=None,
                        help='unique str for wandb directory.')

    # Input type
    parser.add_argument('--input_type', type=str, default='embedding', choices=['embedding', 'pixel'],
                        help='Observation input type. Default is embedding.')
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Dimension of the embedding input.')


    # DV2
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--eta', type=float, default=3e-3)
    parser.add_argument('--replay_capacity', type=int, default=2e6)
    parser.add_argument('--rssm_stoch', type=int, default=32,
                        help="number of different stochastic latent variables in the wm")
    parser.add_argument('--rssm_discrete', type=int, default=32,
                        help="number of different classes per stochastic latent variable")
    parser.add_argument('--actor_ent', type=float, default=2e-3,
                        help="entropy coeeficient")
    parser.add_argument('--discount', type=float, default= 0.99,
                        help="discount factor")
    parser.add_argument('--reservoir_sampling', default=False, action='store_true',
                        help='Flag for using reservoir sampling.')  
    parser.add_argument('--uncertainty_sampling', default=False, action='store_true',
                        help='Flag for using uncertainty sampling.')
    parser.add_argument('--recent_past_sampl_thres', type=float, default=0.,
                        help="probability of triangle distribution, expected to be > 0 and <= 1. 0 denotes taking episodes always from uniform distribution.")
    parser.add_argument('--reward_sampling', default=False, action='store_true',
                        help='Flag for using reward sampling.')
    parser.add_argument('--coverage_sampling', default=False, action='store_true',
                        help='Flag for using coverage maximization.')
    parser.add_argument('--coverage_sampling_args', default={"filters": 64, 
                            "kernel_size": [3,3], 
                            "number_of_comparisons": 1000, 
                            "normalize_lstm_state": True, 
                            "distance": "euclid"}, action='store_true',
                        help='Coverage maximization arguments.')
    parser.add_argument('--minlen', type=int, default=50,
                        help="minimal episode length of episodes stored in the replay buffer")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="mini-batch size")
    parser.add_argument('--unbalanced_steps', type=list, default=None,
                        help="number of steps per each task")
    parser.add_argument('--sep_ac', default=False, action='store_true',
                        help='Flag for using separate Actor-Critics per task.')

    # expl
    parser.add_argument('--plan2explore', default=False, action='store_true',
                        help='Enable plan2explore exploration strategy.')
    parser.add_argument('--expl_intr_scale', type=float, default=1.0,
                        help="scale of the intrinsic reward needs to be > 0 and <= 1.")
    parser.add_argument('--expl_extr_scale', type=float, default=0.0,
                        help="scale of the extrinsic reward needs to be > 0 and <= 1.")
    parser.add_argument('--expl_every', type=int, default=0, 
                        help="how often to run the exploration strategy.")
    parser.add_argument('--sep_exp_eval_policies', default=False, action='store_true',
                        help='Whether to use separate exploration and evaluation polcies.')
    parser.add_argument('--rssm_full_recon', default=False, action='store_true',
                        help='Whether to have the WM reconstruct the obs, discount and rewards rather than the obs only for p2e only.')
                        
    args = parser.parse_known_args(args=args)[0]
    return args


def parse_craftax_args(args=None):
    """Parse arguments for Craftax training."""
    parser = argparse.ArgumentParser(description="Continual DreamerV3 Craftax")

    parser.add_argument('--cl', default=False, action='store_true',
                        help='Flag for continual learning loop.')
    parser.add_argument('--cl_small', default=False, action='store_true',
                        help='Flag for small continual learning configuration.')
    parser.add_argument('--del_exp_replay', default=False, action='store_true',
                        help='Flag to delete the training episodes after running the script to save storage space.')
    parser.add_argument('--num_tasks', type=int, default=1)
    parser.add_argument('--num_task_repeats', type=int, default=1)
    parser.add_argument('--steps', type=int, default=5e5)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--env', type=int, default=0, help='picks the env for the single task training.')
    parser.add_argument('--tag', type=str, default='', help='unique str to tag runs.')
    parser.add_argument('--logdir', type=str, default='logs', help='directory for the logs and exp replay episodes.')

    # wandb
    parser.add_argument('--wandb_group', type=str, default='craftax_experiment', help='name of the group in wandb')
    parser.add_argument('--wandb_proj_name', type=str, default='craftax',
                        help='unique str for wandb projs.')
    parser.add_argument('--wandb_dir', type=str, default=None,
                        help='unique str for wandb directory.')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'],
                        help='wandb logging mode.')

    # Input type
    parser.add_argument('--input_type', type=str, default='embedding', choices=['embedding', 'pixel'],
                        help='Observation input type. Default is embedding.')
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Dimension of the embedding input.')

    # Training
    parser.add_argument('--replay_capacity', type=int, default=2e6)
    parser.add_argument('--batch_size', type=int, default=16,
                        help="mini-batch size (default 16, increase for better GPU utilization)")
    parser.add_argument('--batch_length', type=int, default=64,
                        help="sequence length for training batches (default 64, reduce to save VRAM)")
    parser.add_argument('--train_ratio', type=float, default=None,
                        help="Training steps per env step (default: auto-scaled based on envs/batch_length)")
    parser.add_argument('--model_size', type=str, default='25m',
                        choices=['1m', '12m', '25m', '50m', '100m', '200m', '400m'],
                        help="Model size preset (default 12m for Craftax). "
                             "12m~2GB, 25m~4GB, 50m~8GB, 100m~16GB, 200m~32GB+")
    parser.add_argument('--envs', type=int, default=None,
                        help='Number of parallel environments (override config run.envs).')
    parser.add_argument('--eval_envs', type=int, default=None,
                        help='Number of evaluation environments (override config run.eval_envs).')
    parser.add_argument('--unbalanced_steps', type=list, default=None,
                        help="number of steps per each task")

    # Replay sampling strategies (PDF Section 4.1)
    parser.add_argument('--reservoir_sampling', default=True, action='store_true',
                        help='Flag for using reservoir sampling (random eviction). PDF Section 4.1: Vitter 1985.')
    parser.add_argument('--reward_sampling', default=False, action='store_true',
                        help='Flag for using reward-weighted sampling.')
    parser.add_argument('--recency_sampling', default=False, action='store_true',
                        help='Flag for using recency-biased sampling.')
    parser.add_argument('--uniform_frac', type=float, default=0.5,
                        help='Fraction of samples from uniform distribution (for Mixture selector).')
    parser.add_argument('--recency_frac', type=float, default=0.0,
                        help='Fraction of samples from recency distribution (for Mixture selector).')
    # 50:50 sampling strategy (Continual-Dreamer paper, PDF Section 4.1)
    parser.add_argument('--recent_frac', type=float, default=0.5,
                        help='Fraction of mini-batch from recent experience (PDF: 50:50 sampling).')
    parser.add_argument('--recent_window', type=int, default=1000,
                        help='Window size for recent experience sampling.')

    # Novelty-Learnability-Recency (NLR) replay sampling
    parser.add_argument('--nlr_sampling', default=False, action='store_true',
                        help='Enable NLR replay sampling (overrides recent_frac/50:50 strategy). '
                             'Splits buffer into novel (low success-rate achievements), '
                             'learnable (above-baseline reward), and recent (triangular recency) pools.')
    # Novelty-Learnability-Uniform (NLU) replay sampling (ablation variant)
    parser.add_argument('--nlu_sampling', default=False, action='store_true',
                        help='Enable NLU replay sampling (like NLR but the third pool is uniform '
                             'over the entire buffer instead of triangular recency). '
                             'Use for ablation to test recency bias vs uniform sampling.')
    parser.add_argument('--nlr_novel_frac', type=float, default=0.35,
                        help='NLR: fraction of samples from the novelty pool (default 0.35).')
    parser.add_argument('--nlr_learnable_frac', type=float, default=0.35,
                        help='NLR: fraction of samples from the learnability pool (default 0.35).')
    parser.add_argument('--nlr_recent_frac', type=float, default=0.30,
                        help='NLR: fraction of samples from the recent pool (default 0.30).')
    parser.add_argument('--nlr_recent_window', type=int, default=1000,
                        help='NLR: window size for the recent pool triangular sampling.')
    parser.add_argument('--nlr_reward_ema_decay', type=float, default=0.99,
                        help='NLR: EMA decay for reward baseline in learnability scoring.')
    parser.add_argument('--nlr_novelty_eps', type=float, default=0.01,
                        help='NLR: epsilon added to success rate in novelty scoring (1/(rate+eps)).')
    parser.add_argument('--nlr_novelty_temp', type=float, default=1.0,
                        help='NLR: temperature for novelty pool sampling distribution.')
    parser.add_argument('--nlr_learnability_temp', type=float, default=1.0,
                        help='NLR: temperature for learnability pool sampling distribution.')

    # Exploration
    parser.add_argument('--plan2explore', default=True, action='store_true',
                        help='Enable plan2explore exploration strategy (default: enabled).')
    parser.add_argument('--no_plan2explore', dest='plan2explore', action='store_false',
                        help='Disable plan2explore exploration strategy.')
    parser.add_argument('--disag_models', type=int, default=10,
                        help='Number of ensemble models for Plan2Explore.')
    parser.add_argument('--disag_target', type=str, default='feat', choices=['stoch', 'deter', 'feat'],
                        help='Target for ensemble disagreement prediction (feat = deter + stoch).')
    parser.add_argument('--expl_intr_scale', type=float, default=0.9,
                        help="scale of the intrinsic reward (PDF Section 5.1: α_i=0.9).")
    parser.add_argument('--expl_extr_scale', type=float, default=0.9,
                        help="scale of the extrinsic reward (PDF Section 5.1: α_e=0.9).")
    parser.add_argument('--sep_exp_eval_policies', default=False, action='store_true',
                        help='Whether to use separate exploration and evaluation policies.')
    parser.add_argument('--rssm_full_recon', default=False, action='store_true',
                        help='Whether to have the WM reconstruct the obs, discount and rewards.')

    # Online metrics
    parser.add_argument('--online_metrics', default=True, action='store_true',
                        help='Enable online continual-learning metrics logging.')
    parser.add_argument('--ref_metrics_path', type=str, default=None,
                        help='Path to reference metrics JSON for forward transfer.')
    parser.add_argument('--ref_metrics_dir', type=str, default=None,
                        help='Directory containing per-task ref_metrics JSON files.')

    # DreamerV3 version selection
    parser.add_argument('--use_original_dreamer', default=False, action='store_true',
                        help='Use original DreamerV3 from dreamerv3-main folder instead of continuous enhanced version.')

    args = parser.parse_known_args(args=args)[0]
    return args


def parse_navix_args(args=None):
    """Parse arguments for NAVIX training."""
    parser = argparse.ArgumentParser(description="Continual DreamerV3 NAVIX")

    parser.add_argument('--cl', default=False, action='store_true',
                        help='Flag for continual learning loop.')
    parser.add_argument('--cl_small', default=False, action='store_true',
                        help='Flag for small continual learning configuration.')
    parser.add_argument('--del_exp_replay', default=False, action='store_true',
                        help='Flag to delete the training episodes after running the script to save storage space.')
    parser.add_argument('--num_tasks', type=int, default=1)
    parser.add_argument('--num_task_repeats', type=int, default=1)
    parser.add_argument('--steps', type=int, default=5e5)
    parser.add_argument('--max_steps', type=int, default=500,
                        help='Maximum steps per episode in NAVIX environment.')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--env', type=int, default=0, help='picks the env for the single task training.')
    parser.add_argument('--tag', type=str, default='', help='unique str to tag runs.')
    parser.add_argument('--logdir', type=str, default='logs', help='directory for the logs and exp replay episodes.')

    # wandb
    parser.add_argument('--wandb_group', type=str, default='navix_experiment', help='name of the group in wandb')
    parser.add_argument('--wandb_proj_name', type=str, default='navix',
                        help='unique str for wandb projs.')
    parser.add_argument('--wandb_dir', type=str, default=None,
                        help='unique str for wandb directory.')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'],
                        help='wandb logging mode.')

    # Parallelism
    parser.add_argument('--envs', type=int, default=None,
                        help='Number of parallel environments (override config run.envs).')
    parser.add_argument('--eval_envs', type=int, default=None,
                        help='Number of evaluation environments (override config run.eval_envs).')

    # Online metrics
    parser.add_argument('--online_metrics', default=True, action='store_true',
                        help='Enable online continual-learning metrics logging.')
    parser.add_argument('--ref_metrics_path', type=str, default=None,
                        help='Path to reference metrics JSON for forward transfer.')
    parser.add_argument('--ref_metrics_dir', type=str, default=None,
                        help='Directory containing per-task ref_metrics JSON files.')

    # Input type
    parser.add_argument('--input_type', type=str, default='embedding', choices=['embedding', 'pixel'],
                        help='Observation input type. Default is embedding.')
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Dimension of the embedding input.')

    # Training
    parser.add_argument('--replay_capacity', type=int, default=2e6)
    parser.add_argument('--batch_size', type=int, default=16,
                        help="mini-batch size")
    parser.add_argument('--unbalanced_steps', type=list, default=None,
                        help="number of steps per each task")

    # Replay sampling strategies
    parser.add_argument('--reservoir_sampling', default=False, action='store_true',
                        help='Flag for using reservoir sampling (random eviction).')
    parser.add_argument('--reward_sampling', default=False, action='store_true',
                        help='Flag for using reward-weighted sampling.')
    parser.add_argument('--recency_sampling', default=False, action='store_true',
                        help='Flag for using recency-biased sampling.')
    parser.add_argument('--uniform_frac', type=float, default=1.0,
                        help='Fraction of samples from uniform distribution (for Mixture selector).')
    parser.add_argument('--recency_frac', type=float, default=0.0,
                        help='Fraction of samples from recency distribution (for Mixture selector).')
    # 50:50 sampling strategy (Continual-Dreamer paper)
    parser.add_argument('--recent_frac', type=float, default=0.0,
                        help='Fraction of mini-batch from recent experience (0.5 for 50:50 sampling).')
    parser.add_argument('--recent_window', type=int, default=1000,
                        help='Window size for recent experience sampling.')

    # Exploration
    parser.add_argument('--plan2explore', default=False, action='store_true',
                        help='Enable plan2explore exploration strategy.')
    parser.add_argument('--disag_models', type=int, default=10,
                        help='Number of ensemble models for Plan2Explore.')
    parser.add_argument('--disag_target', type=str, default='stoch', choices=['stoch', 'deter', 'feat'],
                        help='Target for ensemble disagreement prediction.')
    parser.add_argument('--expl_intr_scale', type=float, default=1.0,
                        help="scale of the intrinsic reward.")
    parser.add_argument('--expl_extr_scale', type=float, default=1.0,
                        help="scale of the extrinsic reward.")
    parser.add_argument('--sep_exp_eval_policies', default=False, action='store_true',
                        help='Whether to use separate exploration and evaluation policies.')
    parser.add_argument('--rssm_full_recon', default=False, action='store_true',
                        help='Whether to have the WM reconstruct the obs, discount and rewards.')
    parser.add_argument('--state_bonus', default=False, action='store_true',
                        help='Flag to decide whether to use a state bonus.')
    parser.add_argument('--eval_skills', default=False, action='store_true',
                        help='Flag evaluating our model on the multiskill envs.')

    args = parser.parse_known_args(args=args)[0]
    return args
