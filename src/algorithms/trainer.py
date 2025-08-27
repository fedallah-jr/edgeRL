"""RLlib algorithm trainer wrapper."""

import os
from typing import Dict, Any, Optional
import ray
from ray import tune
from ray.rllib.agents import dqn
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.tune.logger import pretty_print


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
            Configured DQN trainer
        """
        # Get DQN-specific configuration
        dqn_config = self.config['algorithm']['hyperparameters']
        
        # Create RLlib configuration
        config = dqn.DEFAULT_CONFIG.copy()
        
        # Update with our configuration
        config.update({
            # Environment
            "env": "EdgeEnv",
            "env_config": self.env_config,
            
            # Resources
            "num_workers": self.resource_config.get('num_workers', 2),
            "num_gpus": self.resource_config.get('num_gpus', 0),
            "num_cpus_per_worker": self.resource_config.get('num_cpus_per_worker', 1),
            
            # DQN specific
            "lr": dqn_config.get('lr', 0.0001),
            "gamma": dqn_config.get('gamma', 0.99),
            "double_q": dqn_config.get('double_q', True),
            "dueling": dqn_config.get('dueling', True),
            "noisy": dqn_config.get('noisy', False),
            "n_step": dqn_config.get('n_step', 1),
            
            # Exploration
            "exploration_config": dqn_config.get('exploration_config', {
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.01,
                "epsilon_timesteps": 10000,
            }),
            
            # Network
            "model": dqn_config.get('model', {
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            }),
            
            # Replay buffer
            "buffer_size": dqn_config.get('replay_buffer_config', {}).get('capacity', 50000),
            "prioritized_replay": True,
            "prioritized_replay_alpha": dqn_config.get('replay_buffer_config', {}).get('prioritized_replay_alpha', 0.6),
            "prioritized_replay_beta": dqn_config.get('replay_buffer_config', {}).get('prioritized_replay_beta', 0.4),
            "prioritized_replay_eps": dqn_config.get('replay_buffer_config', {}).get('prioritized_replay_eps', 1e-6),
            
            # Training
            "train_batch_size": dqn_config.get('train_batch_size', 32),
            "target_network_update_freq": dqn_config.get('target_network_update_freq', 500),
            "training_intensity": dqn_config.get('training_intensity', 1),
            
            # Optimization
            "adam_epsilon": dqn_config.get('adam_epsilon', 1e-8),
            "grad_clip": dqn_config.get('grad_clip', 40),
            
            # Callbacks
            "callbacks": EdgeSimCallbacks,
            
            # Framework
            "framework": "torch",
            
            # Seed for reproducibility
            "seed": self.training_config.get('seed', 42),
        })
        
        # Create trainer
        return dqn.DQNTrainer(config=config)
    
    def train(self):
        """Run training loop."""
        # Register environment
        from ray.tune.registry import register_env
        from ..envs.edge_env import EdgeEnv
        
        register_env("EdgeEnv", lambda config: EdgeEnv(config))
        
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
            
            # Extract metrics
            episode_reward = result.get("episode_reward_mean", 0)
            episode_len = result.get("episode_len_mean", 0)
            
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
            episode_reward = result.get("evaluation", {}).get("episode_reward_mean", 0)
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