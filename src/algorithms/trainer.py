"""RLlib algorithm trainer wrapper."""

import os
from typing import Dict, Any, Optional
import numpy as np
import ray
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks


class EdgeSimCallbacks(DefaultCallbacks):
    """Custom callbacks for monitoring EdgeSim training."""
    
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        """Called at the end of each episode."""
        # Get custom metrics from environment
        try:
            env = base_env.get_unwrapped()[0]
            if hasattr(env, 'get_info'):
                info = env.get_info()
                for key, value in info.items():
                    episode.custom_metrics[key] = value
        except:
            pass


class RLTrainer:
    """Wrapper class for RLlib algorithm training.
    
    This class provides a unified interface for training
    different RL algorithms with EdgeSim environments.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer.
        
        Args:
            config: Complete configuration dictionary
        """
        self.config = config
        self.algorithm_name = config['algorithm']['name']
        self.env_config = config.get('env', {})
        self.training_config = config.get('training', {})
        self.resource_config = config.get('resources', {})
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                num_cpus=self.resource_config.get('num_workers', 2) + 1,
                num_gpus=self.resource_config.get('num_gpus', 0)
            )
        
        # Setup algorithm
        self.trainer = None
        self.setup_algorithm()
        
    def setup_algorithm(self):
        """Setup the RL algorithm based on configuration."""
        if self.algorithm_name == "DQN":
            self.trainer = self._setup_dqn()
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm_name}")
    
    def _setup_dqn(self):
        """Setup DQN algorithm.
        
        Returns:
            Configured DQN algorithm instance
        """
        # Get DQN-specific configuration
        dqn_cfg_dict = self.config.get('algorithm', {}).get('hyperparameters', {})
        
        # Import env here to avoid registration timing issues
        from ..envs.edge_env import EdgeEnv
        
        # Build RLlib DQN config using the modern API
        cfg = DQNConfig()
        
        # Environment configuration
        cfg = cfg.environment(
            env=EdgeEnv,
            env_config=self.env_config,
            disable_env_checking=True  # Disable checking to avoid issues with custom env
        )
        
        # Framework configuration
        cfg = cfg.framework(framework="torch")
        
        # Resources configuration
        cfg = cfg.resources(
            num_gpus=self.resource_config.get('num_gpus', 0),
            num_cpus_for_driver=1
        )
        
        # Rollout workers configuration
        cfg = cfg.rollouts(
            num_rollout_workers=self.resource_config.get('num_workers', 1),
            num_envs_per_worker=1,
            rollout_fragment_length=32,
            batch_mode="truncate_episodes"
        )
        
        # Training configuration
        cfg = cfg.training(
            lr=float(dqn_cfg_dict.get('lr', 1e-4)),
            gamma=float(dqn_cfg_dict.get('gamma', 0.99)),
            double_q=dqn_cfg_dict.get('double_q', True),
            dueling=dqn_cfg_dict.get('dueling', True),
            n_step=dqn_cfg_dict.get('n_step', 1),
            train_batch_size=dqn_cfg_dict.get('train_batch_size', 32),
            target_network_update_freq=dqn_cfg_dict.get('target_network_update_freq', 500),
            replay_buffer_config={
                "type": "MultiAgentPrioritizedReplayBuffer",
                "capacity": dqn_cfg_dict.get('replay_buffer_config', {}).get('capacity', 50000),
                "prioritized_replay": True,
                "prioritized_replay_alpha": dqn_cfg_dict.get('replay_buffer_config', {}).get('prioritized_replay_alpha', 0.6),
                "prioritized_replay_beta": dqn_cfg_dict.get('replay_buffer_config', {}).get('prioritized_replay_beta', 0.4),
                "prioritized_replay_eps": dqn_cfg_dict.get('replay_buffer_config', {}).get('prioritized_replay_eps', 1e-6),
            },
            adam_epsilon=float(dqn_cfg_dict.get('adam_epsilon', 1e-8)),
            grad_clip=dqn_cfg_dict.get('grad_clip', 40)
        )
        
        # Model configuration - using rl_module API
        model_config = dqn_cfg_dict.get('model', {})
        cfg = cfg.rl_module(
            rl_module_spec={
                "module_class": "ray.rllib.algorithms.dqn.dqn_catalog.DQNTorchRLModule",
                "model_config_dict": {
                    "fcnet_hiddens": model_config.get("fcnet_hiddens", [256, 256]),
                    "fcnet_activation": model_config.get("fcnet_activation", "relu"),
                }
            }
        )
        
        # Exploration configuration for new API
        exploration_config = dqn_cfg_dict.get('exploration_config', {})
        initial_epsilon = exploration_config.get('initial_epsilon', 1.0)
        final_epsilon = exploration_config.get('final_epsilon', 0.01)
        epsilon_timesteps = exploration_config.get('epsilon_timesteps', 10000)
        
        cfg = cfg.exploration(
            explore=True,
            exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": initial_epsilon,
                "final_epsilon": final_epsilon,
                "epsilon_timesteps": epsilon_timesteps,
            }
        )
        
        # Callbacks configuration
        cfg = cfg.callbacks(callbacks_class=EdgeSimCallbacks)
        
        # Debugging configuration
        cfg = cfg.debugging(
            seed=self.training_config.get('seed', 42),
            log_level="WARN"
        )
        
        # Build the Algorithm instance
        try:
            algo = cfg.build()
            return algo
        except Exception as e:
            print(f"Error building algorithm: {e}")
            print("Falling back to simpler configuration...")
            
            # Fallback to simpler configuration if advanced features fail
            cfg = (
                DQNConfig()
                .environment(env=EdgeEnv, env_config=self.env_config, disable_env_checking=True)
                .framework("torch")
                .resources(num_gpus=0)
                .rollouts(num_rollout_workers=1)
                .training(
                    lr=1e-4,
                    gamma=0.99,
                    train_batch_size=32,
                    replay_buffer_config={
                        "type": "MultiAgentReplayBuffer",
                        "capacity": 50000,
                    }
                )
                .debugging(seed=42)
            )
            
            return cfg.build()
    
    def train(self):
        """Run training loop."""
        # Setup directories
        checkpoint_dir = self.training_config.get('checkpoint_dir', 'checkpoints/')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Training loop
        num_episodes = self.training_config.get('num_episodes', 1000)
        checkpoint_freq = self.training_config.get('checkpoint_freq', 100)
        
        print(f"Starting {self.algorithm_name} training for {num_episodes} episodes...")
        print("=" * 50)
        
        best_reward = float('-inf')
        episode_count = 0
        
        try:
            while episode_count < num_episodes:
                # Train one iteration
                result = self.trainer.train()
                
                # Get episode count from result
                episodes_total = result.get("episodes_total", 0)
                if episodes_total > 0:
                    episode_count = episodes_total
                else:
                    episode_count += 1
                
                # Extract metrics (support both legacy and newer keys)
                episode_reward = (
                    result.get("episode_reward_mean")
                    or result.get("episode_return_mean") 
                    or result.get("env_runners", {}).get("episode_reward_mean")
                    or 0
                )
                
                episode_len = (
                    result.get("episode_len_mean")
                    or result.get("episode_length_mean")
                    or result.get("env_runners", {}).get("episode_len_mean")
                    or 0
                )
                
                # Custom metrics
                custom_metrics = result.get("custom_metrics", {})
                
                # Print progress
                if episode_count % 10 == 0 or episode_count == 1:
                    print(f"\nEpisode {episode_count}/{num_episodes}")
                    print(f"  Reward: {episode_reward:.2f}")
                    print(f"  Length: {episode_len:.0f}")
                    
                    if custom_metrics:
                        print("  Custom Metrics:")
                        for key, value in custom_metrics.items():
                            if isinstance(value, (int, float)):
                                print(f"    {key}: {value:.2f}")
                
                # Save checkpoint
                if episode_count % checkpoint_freq == 0 and episode_count > 0:
                    checkpoint = self.trainer.save(checkpoint_dir)
                    print(f"  Checkpoint saved: {checkpoint}")
                
                # Save best model
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    if self.training_config.get('evaluation', {}).get('save_best_model', True):
                        best_checkpoint = self.trainer.save(os.path.join(checkpoint_dir, "best"))
                        print(f"  New best model saved with reward: {best_reward:.2f}")
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"\nError during training: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 50)
        print(f"Training completed! Best reward: {best_reward:.2f}")
        
        return self.trainer
    
    def evaluate(self, checkpoint_path: Optional[str] = None, num_episodes: int = 10):
        """Evaluate trained model.
        
        Args:
            checkpoint_path: Path to checkpoint to load
            num_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation results
        """
        # Load checkpoint if provided
        if checkpoint_path:
            try:
                self.trainer.restore(checkpoint_path)
                print(f"Loaded checkpoint from: {checkpoint_path}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
        
        print(f"\nEvaluating for {num_episodes} episodes...")
        
        # Run evaluation
        results = []
        for episode in range(num_episodes):
            try:
                result = self.trainer.evaluate()
                episode_reward = (
                    result.get("evaluation", {}).get("episode_reward_mean")
                    or result.get("evaluation", {}).get("episode_return_mean")
                    or result.get("evaluation", {}).get("env_runners", {}).get("episode_reward_mean")
                    or 0
                )
                results.append(episode_reward)
                print(f"  Episode {episode + 1}: Reward = {episode_reward:.2f}")
            except Exception as e:
                print(f"  Episode {episode + 1}: Error - {e}")
                results.append(0)
        
        # Calculate statistics
        if results:
            mean_reward = np.mean(results)
            std_reward = np.std(results)
        else:
            mean_reward = 0
            std_reward = 0
        
        print(f"\nEvaluation Results:")
        print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        
        return {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "episodes": results
        }
    
    def close(self):
        """Clean up resources."""
        if self.trainer:
            try:
                self.trainer.stop()
            except:
                pass
        try:
            ray.shutdown()
        except:
            pass
