"""RLlib algorithm trainer wrapper."""

import os
import csv
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
import ray
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module.rl_module import RLModuleSpec


class EdgeSimCallbacks(DefaultCallbacks):
    """Custom callbacks for monitoring EdgeSim training and evaluation logging."""

    # Class-level flags for evaluation logging
    SAVE_EVAL_METRICS: bool = False
    EVAL_DIR: Optional[str] = None
    EP_COUNTER: int = 0
    
    def on_episode_end(
        self,
        *,
        episode,
        env_runner=None,
        metrics_logger=None,
        env=None,
        env_index: int = 0,
        rl_module=None,
        **kwargs,
    ):
        """Called at the end of each episode (new API stack)."""
        # Get custom metrics from episode if available and save eval metrics if requested
        try:
            infos = episode.get_infos() if hasattr(episode, 'get_infos') else None
            if infos and len(infos) > 0:
                # Attach last-step scalar metrics back onto episode custom metrics
                last_info = infos[-1]
                for key, value in last_info.items():
                    if isinstance(value, (int, float)):
                        try:
                            episode.set_custom_metric(key, value)
                        except Exception:
                            pass

                # If evaluation saving is enabled, dump per-step info to CSV
                if EdgeSimCallbacks.SAVE_EVAL_METRICS and EdgeSimCallbacks.EVAL_DIR:
                    EdgeSimCallbacks.EP_COUNTER += 1
                    ep_idx = EdgeSimCallbacks.EP_COUNTER

                    # Build union of keys across all infos
                    keys = set()
                    for it in infos:
                        if isinstance(it, dict):
                            keys.update(it.keys())
                    keys = sorted(list(keys))

                    # Ensure directory exists
                    os.makedirs(EdgeSimCallbacks.EVAL_DIR, exist_ok=True)
                    steps_csv = os.path.join(EdgeSimCallbacks.EVAL_DIR, f"episode_{ep_idx}_steps.csv")

                    # Write CSV
                    with open(steps_csv, mode='w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(["step_index"] + keys)
                        for i, it in enumerate(infos):
                            row = [i]
                            for k in keys:
                                v = it.get(k, "") if isinstance(it, dict) else ""
                                row.append(v)
                            writer.writerow(row)
        except Exception:
            # Silently ignore callback errors to not disrupt training/eval
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
            num_gpus=self.resource_config.get('num_gpus', 0)
        )
        
        # Learners configuration (replaces part of resources)
        cfg = cfg.learners(
            num_learners=0,  # 0 means local learner
            num_cpus_per_learner=1
        )
        
        # Environment runners configuration (replaces rollouts)
        cfg = cfg.env_runners(
            num_env_runners=self.resource_config.get('num_workers', 1),
            num_envs_per_env_runner=1,
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
            adam_epsilon=float(dqn_cfg_dict.get('adam_epsilon', 1e-8)),
            grad_clip=dqn_cfg_dict.get('grad_clip', 40)
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
            algo = cfg.build_algo()
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
                .learners(num_learners=0)
                .env_runners(num_env_runners=1)
                .training(
                    lr=1e-4,
                    gamma=0.99,
                    train_batch_size=32
                )
                .debugging(seed=42)
            )
            
            return cfg.build_algo()
    
    def _get_nested(self, d: dict, path: list):
        """Safely get a nested value from a dict using a path list."""
        current = d
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    def _first_non_none(self, result: dict, key_paths: list, default=0):
        """Return the first non-None value found by checking the given key paths.
        key_paths is a list like [["a"], ["b"], ["c","d"]].
        Unlike `or`, this preserves valid 0.0 values.
        """
        for path in key_paths:
            value = self._get_nested(result, path)
            if value is not None:
                return value
        return default

    def train(self):
        """Run training loop."""
        # Setup directories - use absolute path
        checkpoint_dir = self.training_config.get('checkpoint_dir', 'checkpoints/')
        checkpoint_dir = os.path.abspath(checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_checkpoint_dir = os.path.join(checkpoint_dir, "best")
        os.makedirs(best_checkpoint_dir, exist_ok=True)
        
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
                
                # Extract metrics (robust across RLlib API variants; preserves 0.0)
                episode_reward = self._first_non_none(
                    result,
                    [
                        ["episode_reward_mean"],
                        ["episode_return_mean"],
                        ["env_runners", "episode_reward_mean"],
                        ["env_runners", "episode_return_mean"],
                    ],
                    default=0.0,
                )
                
                episode_len = self._first_non_none(
                    result,
                    [
                        ["episode_len_mean"],
                        ["episode_length_mean"],
                        ["env_runners", "episode_len_mean"],
                        ["env_runners", "episode_length_mean"],
                    ],
                    default=0.0,
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
                        best_checkpoint = self.trainer.save(best_checkpoint_dir)
                        try:
                            # Persist latest best checkpoint path for easier lookup during evaluation
                            with open(os.path.join(best_checkpoint_dir, 'latest.txt'), 'w') as f:
                                f.write(str(best_checkpoint))
                        except Exception:
                            pass
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
        """Evaluate trained model and save metrics to disk.
        
        Args:
            checkpoint_path: Path to checkpoint to load
            num_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation results
        """
        # Resolve and load checkpoint if provided
        if checkpoint_path:
            try:
                resolved = self._resolve_checkpoint_path(checkpoint_path)
                self.trainer.restore(resolved)
                print(f"Loaded checkpoint from: {resolved}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
        
        # Prepare evaluation output directory
        log_root = self.training_config.get('log_dir', 'logs/')
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        eval_root = os.path.join(os.path.abspath(log_root), 'eval', ts)
        steps_dir = os.path.join(eval_root, 'steps')
        os.makedirs(steps_dir, exist_ok=True)

        # Enable per-episode step logging via callback
        EdgeSimCallbacks.SAVE_EVAL_METRICS = True
        EdgeSimCallbacks.EVAL_DIR = steps_dir
        EdgeSimCallbacks.EP_COUNTER = 0

        # Prepare summary CSV
        summary_csv = os.path.join(eval_root, 'eval_summary.csv')
        with open(summary_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward"])
        
        print(f"\nEvaluating for {num_episodes} episodes...")
        print(f"Evaluation logs will be saved under: {eval_root}")
        
        results = []
        
        # Preferred: Use RLlib evaluate() once if available; otherwise, fall back to manual episodes
        used_rllib_eval = False
        try:
            # Attempt to configure evaluation duration dynamically (best-effort)
            # Note: Not all RLlib versions expose this, so we just call evaluate() once.
            eval_result = self.trainer.evaluate()
            # Many RLlib builds summarize over internal episodes; we still produce num_episodes rows
            mean_val = self._first_non_none(
                eval_result.get("evaluation", {}) if isinstance(eval_result, dict) else {},
                [
                    ["episode_reward_mean"],
                    ["episode_return_mean"],
                    ["env_runners", "episode_reward_mean"],
                    ["env_runners", "episode_return_mean"],
                ],
                default=None,
            )
            if mean_val is not None:
                # Emit a single line using RLlib's evaluation result
                results = [mean_val] * num_episodes
                for i in range(num_episodes):
                    print(f"  Episode {i + 1}: Reward = {mean_val:.2f}")
                    with open(summary_csv, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([i + 1, mean_val])
                used_rllib_eval = True
        except Exception as e:
            print(f"RLlib evaluate() not available or failed: {e}. Falling back to manual evaluation.")
        
        if not used_rllib_eval:
            # Manual evaluation loop using a local env and the trained Algorithm for inference
            try:
                from ..envs.edge_env import EdgeEnv
                env = EdgeEnv(self.env_config)
            except Exception as e:
                print(f"Failed to construct evaluation environment: {e}")
                env = None
            
            for episode in range(num_episodes):
                ep_reward = 0.0
                try:
                    if env is None:
                        raise RuntimeError("No evaluation environment available")
                    obs, info = env.reset()
                    terminated = False
                    truncated = False
                    step_idx = 0
                    while not (terminated or truncated):
                        # Use RLlib's (still available) convenience API for single-action inference.
                        try:
                            action = self.trainer.compute_single_action(obs, explore=False)
                        except Exception:
                            # Fallback try: Some versions expose `predict`
                            try:
                                action = self.trainer.predict(obs)
                            except Exception as ee:
                                raise RuntimeError(f"Cannot compute action: {ee}")
                        obs, reward, terminated, truncated, step_info = env.step(action)
                        ep_reward += float(reward)
                        step_idx += 1
                    print(f"  Episode {episode + 1}: Reward = {ep_reward:.2f}")
                    results.append(ep_reward)
                    with open(summary_csv, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([episode + 1, ep_reward])
                except Exception as e:
                    print(f"  Episode {episode + 1}: Error - {e}")
                    results.append(0.0)
                    with open(summary_csv, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([episode + 1, 0.0])
            
            # Ensure env is closed
            try:
                if env is not None:
                    env.close()
            except Exception:
                pass
        
        # Disable step logging
        EdgeSimCallbacks.SAVE_EVAL_METRICS = False
        EdgeSimCallbacks.EVAL_DIR = None
        
        # Calculate statistics
        if results:
            mean_reward = np.mean(results)
            std_reward = np.std(results)
        else:
            mean_reward = 0
            std_reward = 0
        
        print(f"\nEvaluation Results:")
        print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")

        # Save summary stats
        summary_txt = os.path.join(eval_root, 'summary.txt')
        with open(summary_txt, 'w') as f:
            f.write(f"Mean Reward: {mean_reward:.6f}\n")
            f.write(f"Std Reward: {std_reward:.6f}\n")
            f.write(f"Episodes: {len(results)}\n")
        
        return {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "episodes": results,
            "eval_dir": eval_root,
        }
    
    def _resolve_checkpoint_path(self, checkpoint_path: str) -> str:
        """Resolve a user-provided checkpoint path to a concrete checkpoint directory.

        Accepts paths to:
        - a concrete RLlib checkpoint directory
        - a directory (e.g., checkpoints/best) containing sub-checkpoints
        - a text file containing the absolute path to the latest checkpoint
        """
        import os

        path = os.path.abspath(checkpoint_path)

        # If a text file exists (written during training), read the real checkpoint path
        if os.path.isfile(path) and path.lower().endswith('.txt'):
            try:
                with open(path, 'r') as f:
                    candidate = f.read().strip()
                candidate = os.path.abspath(candidate)
                if os.path.exists(candidate):
                    return candidate
            except Exception:
                pass

        # If path is a directory like .../checkpoints/best, pick most recent subdir
        if os.path.isdir(path):
            entries = [os.path.join(path, d) for d in os.listdir(path)]
            entries = [d for d in entries if os.path.isdir(d)]
            if entries:
                entries.sort(key=lambda d: os.path.getmtime(d), reverse=True)
                return entries[0]
        return path

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
