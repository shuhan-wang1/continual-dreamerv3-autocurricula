"""
Unified training script for Continual DreamerV3 with JAX-based environments.
Supports both Craftax and NAVIX environments.

Usage:
    python train.py --env_type craftax [other args]
    python train.py --env_type navix [other args]
"""
import argparse
import sys


def main():
    # Parse only the env_type argument first
    parser = argparse.ArgumentParser(description="Continual DreamerV3 Training", add_help=False)
    parser.add_argument('--env_type', type=str, default='craftax', 
                        choices=['craftax', 'navix'],
                        help='Type of environment to use: craftax or navix')
    
    args, remaining_args = parser.parse_known_args()
    
    # Route to the appropriate training script
    if args.env_type == 'craftax':
        from train_craftax import run_craftax
        from input_args import parse_craftax_args
        
        training_args = parse_craftax_args(remaining_args)
        run_craftax(training_args)
    
    elif args.env_type == 'navix':
        from train_navix import run_navix
        from input_args import parse_navix_args
        
        training_args = parse_navix_args(remaining_args)
        run_navix(training_args)
    
    else:
        print(f"Unknown environment type: {args.env_type}")
        print("Available options: craftax, navix")
        sys.exit(1)


if __name__ == "__main__":
    main()
