"""Main training script for EdgeSim-RL."""

import argparse
import yaml
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms.trainer import RLTrainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle defaults if specified
    if 'defaults' in config:
        base_configs = config['defaults']
        if not isinstance(base_configs, list):
            base_configs = [base_configs]
        
        # Load base configurations
        merged_config = {}
        config_dir = Path(config_path).parent
        
        for base_config_name in base_configs:
            base_path = config_dir / f"{base_config_name}.yaml"
            if base_path.exists():
                with open(base_path, 'r') as f:
                    base_config = yaml.safe_load(f)
                    merged_config = deep_merge(merged_config, base_config)
        
        # Merge with current config (current config overrides base)
        config = deep_merge(merged_config, config)
        
        # Remove defaults key
        config.pop('defaults', None)
    
    return config


def deep_merge(dict1: dict, dict2: dict) -> dict:
    """Deep merge two dictionaries.
    
    Args:
        dict1: Base dictionary
        dict2: Dictionary to merge (overrides dict1)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train RL agent for EdgeSim environment')
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/dqn_config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=None,
        help='Number of training episodes (overrides config)'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Number of parallel workers (overrides config)'
    )
    
    parser.add_argument(
        '--num-gpus',
        type=int,
        default=None,
        help='Number of GPUs to use (overrides config)'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Run evaluation instead of training'
    )
    
    parser.add_argument(
        '--eval-episodes',
        type=int,
        default=10,
        help='Number of evaluation episodes'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.num_episodes is not None:
        config['training']['num_episodes'] = args.num_episodes
    
    if args.num_workers is not None:
        config['resources']['num_workers'] = args.num_workers
    
    if args.num_gpus is not None:
        config['resources']['num_gpus'] = args.num_gpus
    
    # Print configuration
    print("Configuration:")
    print("-" * 50)
    print(f"Algorithm: {config['algorithm']['name']}")
    print(f"Environment: {config['env']['name']}")
    print(f"Dataset: {config['env']['dataset_path']}")
    print(f"Episodes: {config['training']['num_episodes']}")
    print(f"Workers: {config['resources']['num_workers']}")
    print(f"GPUs: {config['resources']['num_gpus']}")
    print("-" * 50)
    
    try:
        # Create trainer
        trainer = RLTrainer(config)
        
        if args.evaluate:
            # Run evaluation
            checkpoint_path = args.checkpoint or os.path.join(
                config['training']['checkpoint_dir'], 'best'
            )
            trainer.evaluate(
                checkpoint_path=checkpoint_path,
                num_episodes=args.eval_episodes
            )
        else:
            # Run training
            if args.checkpoint:
                trainer.trainer.restore(args.checkpoint)
                print(f"Resumed from checkpoint: {args.checkpoint}")
            
            trainer.train()
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if 'trainer' in locals():
            trainer.close()
        print("\nTraining session ended")


if __name__ == "__main__":
    main()