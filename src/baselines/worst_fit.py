"""Worst-Fit baseline policy and evaluator for EdgeEnv.

This baseline mimics EdgeAISIM's worst-fit migration: for each service,
choose the edge server with the largest free resources (geometric mean)
that can host the service and is different from its current host.
"""

from typing import Optional, Dict, Any
import os
import csv
from datetime import datetime
import numpy as np

from edge_sim_py.components import EdgeServer


class WorstFitBaseline:
    def __init__(self, allow_no_migration: bool = True):
        self.allow_no_migration = allow_no_migration

    def select_action(self, env) -> int:
        """Select target server index for the current service using worst-fit.

        - Ranks edge servers by free resource capacity (geometric mean).
        - Picks the first server with enough capacity and different from current host.
        - Falls back to no-migration (if allowed) or current-host index.
        """
        servers = EdgeServer.all()
        service = env.get_current_service()

        # If no current service, no-op.
        if service is None or not servers:
            return env.num_servers if getattr(env, "allow_no_migration", True) else 0

        # Compute free capacity with geometric mean of free CPU, MEM, DISK
        def free_score(s):
            free_cpu = max(0, s.cpu - s.cpu_demand)
            free_mem = max(0, s.memory - s.memory_demand)
            free_disk = max(0, s.disk - s.disk_demand)
            return (free_cpu * free_mem * free_disk) ** (1 / 3) if free_cpu * free_mem * free_disk > 0 else 0

        sorted_servers = sorted(servers, key=lambda s: free_score(s), reverse=True)

        # Try to find a feasible new host
        for target in sorted_servers:
            if service.server is not None and target == service.server:
                continue
            try:
                if target.has_capacity_to_host(service):
                    return servers.index(target)
            except Exception:
                continue

        # Fallbacks
        if getattr(env, "allow_no_migration", True):
            return env.num_servers  # no-migration action index

        # Last resort: stay on current server
        try:
            if service.server in servers:
                return servers.index(service.server)
        except Exception:
            pass
        return 0


def evaluate_worst_fit(env_class, env_config: Dict[str, Any], num_episodes: int = 10, log_root: str = "logs/") -> Dict[str, Any]:
    """Evaluate the Worst-Fit baseline on EdgeEnv and save metrics to disk.

    Creates logs under {log_root}/eval_baseline/{timestamp}/ with:
    - steps/episode_<n>_steps.csv: per-step metrics (info keys + reward)
    - eval_summary.csv: per-episode reward plus migrations, valid actions, unique actions, total latency
    - summary.txt: mean/std rewards, totals and means for the extra metrics
    """
    # Lazy import to avoid circulars
    env = env_class(env_config)
    policy = WorstFitBaseline(allow_no_migration=getattr(env, "allow_no_migration", True))

    abs_log_root = os.path.abspath(log_root)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    eval_root = os.path.join(abs_log_root, 'eval_baseline', ts)
    steps_dir = os.path.join(eval_root, 'steps')
    os.makedirs(steps_dir, exist_ok=True)

    episode_returns = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_return = 0.0

        # Collect step infos for CSV; include reward per step
        infos = []
        step_idx = 0

        while not done and not truncated:
            action = policy.select_action(env)
            obs, reward, done, truncated, step_info = env.step(action)
            ep_return += float(reward)

            # Merge reward into info for logging
            if isinstance(step_info, dict):
                row = dict(step_info)
                row['reward'] = float(reward)
            else:
                row = {'reward': float(reward)}
            row['step_index'] = step_idx
            infos.append(row)
            step_idx += 1

        # Write per-episode steps CSV
        keys = set()
        for it in infos:
            keys.update(it.keys())
        keys = sorted(list(keys))
        steps_csv = os.path.join(steps_dir, f"episode_{ep + 1}_steps.csv")
        with open(steps_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            for it in infos:
                writer.writerow([it.get(k, "") for k in keys])

        print(f"Baseline (Worst-Fit) Episode {ep + 1}: Reward = {ep_return:.4f}")
        episode_returns.append(ep_return)

    # Save enhanced summary files using shared utilities
    mean_reward = float(np.mean(episode_returns)) if episode_returns else 0.0
    std_reward = float(np.std(episode_returns)) if episode_returns else 0.0

    try:
        from src.utils.eval_utils import summarize_steps_dir, write_summary_files
        per_ep_metrics, aggregates = summarize_steps_dir(steps_dir)
        write_summary_files(eval_root, per_ep_metrics, episode_returns)
    except Exception as e:
        # As a fallback, write minimal summary.txt if utils are unavailable
        print(f"Enhanced summary generation failed: {e}")
        summary_txt = os.path.join(eval_root, 'summary.txt')
        with open(summary_txt, 'w') as f:
            f.write(f"Mean Reward: {mean_reward:.6f}\n")
            f.write(f"Std Reward: {std_reward:.6f}\n")
            f.write(f"Episodes: {len(episode_returns)}\n")
        aggregates = {
            'episodes': float(len(episode_returns)),
            'total_migrations': 0.0,
            'mean_migrations': 0.0,
            'total_valid_actions': 0.0,
            'mean_valid_actions': 0.0,
            'total_latency': 0.0,
            'mean_latency': 0.0,
        }

    # Close the env
    try:
        env.close()
    except Exception:
        pass

    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'episodes': episode_returns,
        'eval_dir': eval_root,
        'aggregates': aggregates,
    }

