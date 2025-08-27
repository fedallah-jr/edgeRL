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
        env = base_env.get_unwrapped()[0]
        if hasattr(env, 'get_info'):
            info = env.get_info()
            for key, value in info.items():
                episode.custom_metrics[key] = value


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
        self.env_config = config['env']
        self.training_config = config['training']
        self.resource_config = config['resources']
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
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
        dqn_cfg_dict = self.config['algorithm']['hyperparameters']
        
        # Import env here and pass a creator to RLlib to avoid registration timing issues
        from ..envs.edge_env import EdgeEnv

        # Build RLlib DQN config using the modern API
        cfg = (
            DQNConfig()
            .environment(env=EdgeEnv, env_config=self.env_config)
            .framework("torch")
            .resources(
                num_gpus=self.resource_config.get('num_gpus', 0),
            )
            .env_runners(
                num_env_runners=self.resource_config.get('num_workers', 2),
                num_cpus_per_env_runner=self.resource_config.get('num_cpus_per_worker', 1),
            )
            .training(
                lr=dqn_cfg_dict.get('lr', 1e-4),
                gamma=dqn_cfg_dict.get('gamma', 0.99),
                double_q=dqn_cfg_dict.get('double_q', True),
                dueling=dqn_cfg_dict.get('dueling', True),
                n_step=dqn_cfg_dict.get('n_step', 1),
                train_batch_size=dqn_cfg_dict.get('train_batch_size', 32),
                target_network_update_freq=dqn_cfg_dict.get('target_network_update_freq', 500),
                adam_epsilon=dqn_cfg_dict.get('adam_epsilon', 1e-8),
                grad_clip=dqn_cfg_dict.get('grad_clip', 40),
            )
            .rl_module(
                model_config=dqn_cfg_dict.get('model', {
                    "fcnet_hiddens": [256, 256],
                    "fcnet_activation": "relu",
                })
            )
            .callbacks(callbacks_class=EdgeSimCallbacks)
            .debugging(seed=self.training_config.get('seed', 42))
        )
        
        # Set random seed for reproducibility (added above via .debugging())
        
        # Do not set exploration_config when RLModule (new API stack) is enabled.
        # RLlib's new API expects exploration to be handled inside the RLModule. If
        # you need custom exploration, implement forward_exploration in a custom module.
        
        # Replay buffer capacity (optional). If provided, map to new-style config.
        rb = dqn_cfg_dict.get('replay_buffer_config', {})
        if rb:
            # Prefer new-style replay buffer configuration if available in this RLlib version.
            # Will be ignored if unsupported.
            cfg = cfg.training(replay_buffer_config={
                "type": "PrioritizedReplayBuffer",
                "capacity": rb.get('capacity', 50000),
                "alpha": rb.get('prioritized_replay_alpha', 0.6),
                "beta": rb.get('prioritized_replay_beta', 0.4),
                "eps": rb.get('prioritized_replay_eps', 1e-6),
            })
        
        # Build the Algorithm instance
        return cfg.build_algo()
    
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
        
        for episode in range(num_episodes):
            # Train one iteration
            result = self.trainer.train()
            
            # Extract metrics (support both legacy and newer keys)
            episode_reward = (
                result.get("episode_reward_mean")
                or result.get("episode_return_mean")
                or 0
            )
            episode_len = (
                result.get("episode_len_mean")
                or result.get("episode_length_mean")
                or 0
            )
            
            # Custom metrics
            custom_metrics = result.get("custom_metrics", {})
            
            # Print progress
            if episode % 10 == 0:
                print(f"\nEpisode {episode}/{num_episodes}")
                print(f"  Reward: {episode_reward:.2f}")
                print(f"  Length: {episode_len:.0f}")
                
                if custom_metrics:
                    print("  Custom Metrics:")
                    for key, value in custom_metrics.items():
                        if isinstance(value, (int, float)):
                            print(f"    {key}: {value:.2f}")
            
            # Save checkpoint
            if episode % checkpoint_freq == 0 and episode > 0:
                checkpoint = self.trainer.save(checkpoint_dir)
                print(f"  Checkpoint saved: {checkpoint}")
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                if self.training_config.get('evaluation', {}).get('save_best_model', True):
                    best_checkpoint = self.trainer.save(os.path.join(checkpoint_dir, "best"))
                    print(f"  New best model saved with reward: {best_reward:.2f}")
        
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
            self.trainer.restore(checkpoint_path)
            print(f"Loaded checkpoint from: {checkpoint_path}")
        
        print(f"\nEvaluating for {num_episodes} episodes...")
        
        # Run evaluation
        results = []
        for episode in range(num_episodes):
            result = self.trainer.evaluate()
            episode_reward = (
                result.get("evaluation", {}).get("episode_reward_mean")
                or result.get("evaluation", {}).get("episode_return_mean")
                or 0
            )
            results.append(episode_reward)
            print(f"  Episode {episode + 1}: Reward = {episode_reward:.2f}")
        
        # Calculate statistics
        mean_reward = np.mean(results)
        std_reward = np.std(results)
        
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
            self.trainer.stop()
        ray.shutdown()
