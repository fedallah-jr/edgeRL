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

    # Summary CSV for per-episode aggregates
    summary_csv = os.path.join(eval_root, 'eval_summary.csv')
    with open(summary_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "mean_power", "mean_latency", "unique_actions"])

    episode_returns = []
    episode_mean_powers = []
    episode_mean_latencies = []
    episode_unique_actions = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_return = 0.0

        # Collect step infos for CSV; include reward per step
        infos = []
        step_idx = 0
        powers = []
        latencies = []

        # Prepare pattern tracking
        try:
            from edge_sim_py.components import Service as _Service
            service_ids_sorted = sorted([s.id for s in _Service.all()])
        except Exception:
            service_ids_sorted = []
        id_to_pos = {sid: i for i, sid in enumerate(service_ids_sorted)}
        current_pattern = [-1] * len(service_ids_sorted)
        unique_patterns = set()

        while not done and not truncated:
            action = policy.select_action(env)
            obs, reward, done, truncated, step_info = env.step(action)
            ep_return += float(reward)

            # Merge reward and action into info for logging
            if isinstance(step_info, dict):
                row = dict(step_info)
                row['reward'] = float(reward)
                row['action'] = int(action)
                row['step_index'] = step_idx

                # Track power and latency if present
                if 'total_power' in step_info and step_info['total_power'] is not None:
                    try:
                        powers.append(float(step_info['total_power']))
                    except Exception:
                        pass
                for k in ('mean_latency', 'avg_latency', 'latency', 'total_latency'):
                    if k in step_info and step_info[k] is not None:
                        try:
                            latencies.append(float(step_info[k]))
                        except Exception:
                            pass
                        break

                # Build pattern vector using service_id/pattern_choice when available
                try:
                    sid = step_info.get('service_id', None)
                    pch = step_info.get('pattern_choice', None)
                    if sid is not None and sid in id_to_pos and pch is not None:
                        current_pattern[id_to_pos[sid]] = int(pch)
                except Exception:
                    pass

                # Finalize pattern at the end of a scheduling round
                try:
                    if int(step_info.get('services_to_migrate', 0)) == 0 and len(current_pattern) > 0:
                        unique_patterns.add(tuple(current_pattern))
                        current_pattern = [-1] * len(service_ids_sorted)
                except Exception:
                    pass

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

        mean_power = float(np.mean(powers)) if powers else 0.0
        mean_latency = float(np.mean(latencies)) if latencies else 0.0
        unique_actions = int(len(unique_patterns))

        print(f"Baseline (Worst-Fit) Episode {ep + 1}: Reward = {ep_return:.4f}, MeanPower = {mean_power:.4f}, MeanLatency = {mean_latency:.4f}, UniqueActions = {unique_actions}")
        episode_returns.append(ep_return)
        episode_mean_powers.append(mean_power)
        episode_mean_latencies.append(mean_latency)
        episode_unique_actions.append(unique_actions)

        # Append row to summary CSV
        with open(summary_csv, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ep + 1, ep_return, mean_power, mean_latency, unique_actions])

    # Save enhanced summary files using shared utilities
    mean_reward = float(np.mean(episode_returns)) if episode_returns else 0.0
    std_reward = float(np.std(episode_returns)) if episode_returns else 0.0
    mean_mean_power = float(np.mean(episode_mean_powers)) if episode_mean_powers else 0.0
    mean_mean_latency = float(np.mean(episode_mean_latencies)) if episode_mean_latencies else 0.0
    mean_unique_actions = float(np.mean(episode_unique_actions)) if episode_unique_actions else 0.0

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
            f.write(f"Mean of Mean Powers: {mean_mean_power:.6f}\n")
            f.write(f"Mean of Mean Latencies: {mean_mean_latency:.6f}\n")
            f.write(f"Mean of Unique Actions: {mean_unique_actions:.6f}\n")
        aggregates = {
            'episodes': float(len(episode_returns)),
            'total_migrations': 0.0,
            'mean_migrations': 0.0,
            'total_valid_actions': 0.0,
            'mean_valid_actions': 0.0,
            'total_latency': 0.0,
            'mean_latency': mean_mean_latency,
            'mean_power': mean_mean_power,
            'mean_unique_actions': mean_unique_actions,
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

