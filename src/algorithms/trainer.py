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
        
        # Evaluation configuration (use config values if provided)
        try:
            eval_cfg = self.config.get('evaluation', {})
            cfg = cfg.evaluation(
                evaluation_num_env_runners=eval_cfg.get('num_env_runners', 1),
                evaluation_duration=eval_cfg.get('eval_episodes', 10),
                evaluation_duration_unit="episodes",
            )
        except Exception:
            # If RLlib version doesn't support these, skip silently
            pass
        
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

    def _resolve_checkpoint_path(self, checkpoint_path: str) -> Optional[str]:
        """Resolve checkpoint path to a valid RLlib checkpoint directory.
        If a directory like "checkpoints/best" is provided, select the latest
        subdirectory containing a checkpoint file.
        """
        try:
            if not checkpoint_path:
                return None
            path = os.path.abspath(checkpoint_path)
            if os.path.isfile(path):
                return path
            if os.path.isdir(path):
                candidates = []
                for root, dirs, files in os.walk(path):
                    # Only search at most 2 levels deep under the given path
                    depth = root[len(path):].count(os.sep)
                    if depth > 2:
                        continue
                    if (
                        "algorithm_state.pkl" in files
                        or "params.pkl" in files
                        or "checkpoint.pkl" in files
                    ):
                        # Use mtime as ranking signal
                        try:
                            if "algorithm_state.pkl" in files:
                                mtime = os.path.getmtime(os.path.join(root, "algorithm_state.pkl"))
                            else:
                                mtime = os.path.getmtime(root)
                        except Exception:
                            mtime = 0
                        candidates.append((mtime, root))
                if candidates:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    return candidates[0][1]
            return path
        except Exception:
            return checkpoint_path

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
            checkpoint_path: Path to checkpoint to load (file or directory)
            num_episodes: Number of evaluation episodes (best-effort)
            
        Returns:
            Evaluation results
        """
        # Load checkpoint if provided (resolve directories like checkpoints/best)
        if checkpoint_path:
            resolved = self._resolve_checkpoint_path(checkpoint_path)
            try:
                self.trainer.restore(resolved)
                print(f"Loaded checkpoint from: {resolved}")
            except Exception as e:
                print(f"Error loading checkpoint from '{checkpoint_path}': {e}")
        
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

        # Try to set evaluation duration dynamically (if supported by RLlib version)
        try:
            self.trainer.config = self.trainer.config.evaluation(
                evaluation_duration=num_episodes,
                evaluation_duration_unit="episodes",
            )
        except Exception:
            pass
        
        print(f"\nEvaluating for {num_episodes} episodes...")
        print(f"Evaluation logs will be saved under: {eval_root}")
        
        # Run a single evaluation call; RLlib manages episodes internally
        try:
            result = self.trainer.evaluate()
            eval_dict = result.get("evaluation", {}) if isinstance(result, dict) else {}
            
            # Parse aggregated metrics
            mean_reward = self._first_non_none(
                eval_dict,
                [
                    ["episode_reward_mean"],
                    ["episode_return_mean"],
                    ["env_runners", "episode_reward_mean"],
                    ["env_runners", "episode_return_mean"],
                ],
                default=0.0,
            )
            episode_len = self._first_non_none(
                eval_dict,
                [
                    ["episode_len_mean"],
                    ["episode_length_mean"],
                    ["env_runners", "episode_len_mean"],
                    ["env_runners", "episode_length_mean"],
                ],
                default=0.0,
            )
            
            # If per-episode rewards are available in hist_stats, capture them
            hist = eval_dict.get("hist_stats", {}) if isinstance(eval_dict, dict) else {}
            ep_rewards = None
            for k in ["episode_reward", "episode_return", "return"]:
                if isinstance(hist, dict) and k in hist and isinstance(hist[k], (list, tuple)):
                    ep_rewards = list(hist[k])
                    break
            if ep_rewards is None:
                ep_rewards = []
            std_reward = float(np.std(ep_rewards)) if ep_rewards else 0.0

            # Summarize step CSVs and write enhanced summary files
            from src.utils.eval_utils import summarize_steps_dir, write_summary_files
            per_ep_metrics, aggregates = summarize_steps_dir(steps_dir)
            write_summary_files(eval_root, per_ep_metrics, ep_rewards)
            
            # Disable step logging
            EdgeSimCallbacks.SAVE_EVAL_METRICS = False
            EdgeSimCallbacks.EVAL_DIR = None
            
            print("Evaluation Results:")
            print(f"  Mean Reward: {mean_reward:.2f}")
            print(f"  Std Reward: {std_reward:.2f}")
            print(f"  Mean Episode Length: {episode_len:.2f}")
            print(f"  Total Migrations: {aggregates.get('migrations', aggregates.get('total_migrations', 0.0)):.2f}")
            print(f"  Mean Migrations/Episode: {aggregates.get('mean_migrations', 0.0):.2f}")
            print(f"  Total Valid Actions: {aggregates.get('valid_actions', aggregates.get('total_valid_actions', 0.0)):.2f}")
            print(f"  Mean Valid Actions/Episode: {aggregates.get('mean_valid_actions', 0.0):.2f}")
            print(f"  Total Latency: {aggregates.get('total_latency', 0.0):.2f}")
            print(f"  Mean Latency/Episode: {aggregates.get('mean_total_latency', aggregates.get('mean_latency', 0.0)):.2f}")
            
            return {
                "mean_reward": float(mean_reward),
                "std_reward": float(std_reward),
                "mean_episode_len": float(episode_len),
                "eval_dir": eval_root,
                "raw": result,
                "aggregates": aggregates,
            }
        except Exception as e:
            # Disable step logging on error
            EdgeSimCallbacks.SAVE_EVAL_METRICS = False
            EdgeSimCallbacks.EVAL_DIR = None
            print(f"Evaluation error: {e}")
            return {
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "mean_episode_len": 0.0,
                "eval_dir": eval_root,
                "raw": {},
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
