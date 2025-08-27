"""Simplified training script for EdgeSim-RL with better error handling."""

import argparse
import yaml
import sys
import os
import warnings
from pathlib import Path

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_environment():
    """Test if the environment can be created and reset."""
    print("Testing environment creation...")
    try:
        from src.envs.edge_env import EdgeEnv
        
        # Test with minimal config
        config = {
            "dataset_path": "datasets/sample_dataset.json",
            "max_steps": 100,
        }
        
        env = EdgeEnv(config)
        obs, info = env.reset()
        print(f"✓ Environment created successfully")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Action space: {env.action_space}")
        
        # Test a few steps
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"✓ Environment steps executed successfully")
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def simple_train_dqn(config):
    """Simple DQN training without complex configurations."""
    import ray
    from ray.rllib.algorithms.dqn import DQNConfig
    from src.envs.edge_env import EdgeEnv
    
    # Initialize Ray with minimal resources
    if not ray.is_initialized():
        ray.init(num_cpus=2, num_gpus=0, ignore_reinit_error=True)
    
    print("Building DQN algorithm...")
    
    try:
        # Create simple DQN configuration
        algo_config = (
            DQNConfig()
            .environment(
                env=EdgeEnv,
                env_config=config.get('env', {}),
                disable_env_checking=True
            )
            .framework("torch")
            .rollouts(
                num_rollout_workers=1,
                num_envs_per_worker=1,
                rollout_fragment_length=32
            )
            .resources(
                num_gpus=0
            )
            .training(
                lr=1e-4,
                gamma=0.99,
                train_batch_size=32,
                replay_buffer_config={
                    "type": "MultiAgentReplayBuffer",
                    "capacity": 10000,
                },
                target_network_update_freq=100
            )
            .exploration(
                explore=True,
                exploration_config={
                    "type": "EpsilonGreedy",
                    "initial_epsilon": 1.0,
                    "final_epsilon": 0.1,
                    "epsilon_timesteps": 1000,
                }
            )
            .debugging(seed=42, log_level="ERROR")
        )
        
        # Build algorithm
        algo = algo_config.build()
        print("✓ Algorithm built successfully")
        
        # Training loop
        num_episodes = config.get('training', {}).get('num_episodes', 100)
        print(f"\nStarting training for {num_episodes} episodes...")
        print("=" * 50)
        
        best_reward = float('-inf')
        
        for episode in range(num_episodes):
            try:
                # Train one iteration
                result = algo.train()
                
                # Extract reward (try multiple keys)
                reward = (
                    result.get("episode_reward_mean", 0) or
                    result.get("env_runners", {}).get("episode_reward_mean", 0) or
                    0
                )
                
                # Print progress
                if episode % 10 == 0:
                    print(f"Episode {episode}: Reward = {reward:.2f}")
                
                # Track best reward
                if reward > best_reward:
                    best_reward = reward
                    
            except KeyboardInterrupt:
                print("\nTraining interrupted")
                break
            except Exception as e:
                print(f"Error in episode {episode}: {e}")
                continue
        
        print(f"\nTraining complete! Best reward: {best_reward:.2f}")
        
        # Cleanup
        algo.stop()
        ray.shutdown()
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        ray.shutdown()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
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
        
        # Merge with current config
        config = deep_merge(merged_config, config)
        config.pop('defaults', None)
    
    return config


def deep_merge(dict1: dict, dict2: dict) -> dict:
    """Deep merge two dictionaries."""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def main():
    """Main function with simplified training."""
    parser = argparse.ArgumentParser(description='Train RL agent for EdgeSim environment')
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/dqn_config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--test-env',
        action='store_true',
        help='Test environment before training'
    )
    
    parser.add_argument(
        '--simple',
        action='store_true',
        help='Use simplified training (recommended for debugging)'
    )
    
    args = parser.parse_args()
    
    # Test environment if requested
    if args.test_env:
        if not test_environment():
            print("Environment test failed. Please fix the issues before training.")
            return
        print()
    
    # Load configuration
    config = load_config(args.config)
    
    print("Configuration loaded:")
    print(f"  Dataset: {config['env']['dataset_path']}")
    print(f"  Episodes: {config['training']['num_episodes']}")
    print()
    
    # Check if dataset file exists
    dataset_path = config['env']['dataset_path']
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found: {dataset_path}")
        print("Please ensure the dataset file exists.")
        return
    
    if args.simple or True:  # Always use simple training for now
        print("Using simplified training...")
        simple_train_dqn(config)
    else:
        print("Using advanced training...")
        from src.algorithms.trainer import RLTrainer
        
        try:
            trainer = RLTrainer(config)
            trainer.train()
        except Exception as e:
            print(f"Training failed: {e}")
            print("\nTry running with --simple flag for simplified training")
        finally:
            if 'trainer' in locals():
                trainer.close()


if __name__ == "__main__":
    main()
