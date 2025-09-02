"""Random-Fit baseline policy and evaluator for EdgeEnv.

This baseline selects uniformly at random among the feasible target servers
that can host the current service (and differ from current host).
Falls back to staying on the current host (no migration) when no feasible target exists.
"""

from typing import Dict, Any
import os
import csv
from datetime import datetime
import random
import numpy as np

from edge_sim_py.components import EdgeServer


class RandomFitBaseline:
    def __init__(self):
        pass

    def select_action(self, env) -> int:
        """Select a random feasible target server for the current service.

        - Builds the set of servers that can host the service and differ from the current host.
        - Returns the index of a random choice.
        - Falls back to the current host index (no migration) when no feasible target exists.
        """
        servers = EdgeServer.all()
        service = env.get_current_service()

        if service is None or not servers:
            # No current service: return any valid index (0 is safe as action space is [0, num_servers-1])
            return 0

        try:
            candidates = []
            for s in servers:
                if service.server is not None and s == service.server:
                    continue
                try:
                    if s.has_capacity_to_host(service):
                        candidates.append(s)
                except Exception:
                    continue
            if candidates:
                chosen = random.choice(candidates)
                return servers.index(chosen)
        except Exception:
            pass

        # Fallbacks
        try:
            if service and service.server in servers:
                return servers.index(service.server)
        except Exception:
            pass
        # Last resort: pick index 0
        return 0


def evaluate_random_fit(env_class, env_config: Dict[str, Any], num_episodes: int = 10, log_root: str = "logs/") -> Dict[str, Any]:
    """Evaluate the Random-Fit baseline on EdgeEnv and save metrics to disk.

    Creates logs under {log_root}/eval_baseline/{timestamp}/ with:
    - steps/episode_<n>_steps.csv: per-step metrics (info keys + reward)
    - eval_summary.csv: per-episode reward plus mean power, mean latency, unique offloading patterns
    - summary.txt: mean/std rewards (best-effort if utils unavailable)
    """
    env = env_class(env_config)
    policy = RandomFitBaseline()

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

        infos = []
        step_idx = 0
        powers = []
        latencies = []
        tick_powers = []  # one value per simulation tick

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

            if isinstance(step_info, dict):
                row = dict(step_info)
                row['reward'] = float(reward)
                row['action'] = int(action)
                row['step_index'] = step_idx
                infos.append(row)

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

                # Build and finalize offloading pattern
                try:
                    sid = step_info.get('service_id', None)
                    pch = step_info.get('pattern_choice', None)
                    if sid is not None and sid in id_to_pos and pch is not None:
                        current_pattern[id_to_pos[sid]] = int(pch)
                except Exception:
                    pass
                # Capture one power value per simulation tick
                try:
                    if step_info.get('no_services', False) and 'tick_power' in step_info and step_info['tick_power'] is not None:
                        tick_powers.append(float(step_info['tick_power']))
                except Exception:
                    pass
                try:
                    if int(step_info.get('services_to_migrate', 0)) == 0 and len(current_pattern) > 0:
                        unique_patterns.add(tuple(current_pattern))
                        current_pattern = [-1] * len(service_ids_sorted)
                except Exception:
                    pass

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

        print(f"Baseline (Random-Fit) Episode {ep + 1}: Reward = {ep_return:.4f}, MeanPower = {mean_power:.4f}, MeanLatency = {mean_latency:.4f}, UniqueActions = {unique_actions}")
        episode_returns.append(ep_return)
        episode_mean_powers.append(mean_power)
        episode_mean_latencies.append(mean_latency)
        episode_unique_actions.append(unique_actions)

        with open(summary_csv, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ep + 1, ep_return, mean_power, mean_latency, unique_actions])

        # Save per-simulation-tick power series for this episode
        try:
            pl_path = os.path.join(eval_root, f"power_list_episode_{ep + 1}.csv")
            with open(pl_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Power"])  # one row per simulation tick
                for v in tick_powers:
                    writer.writerow([v])
        except Exception as e:
            print(f"    Warning: failed to write power_list CSV for episode {ep + 1}: {e}")

    # Final summaries
    mean_reward = float(np.mean(episode_returns)) if episode_returns else 0.0
    std_reward = float(np.std(episode_returns)) if episode_returns else 0.0

    try:
        from src.utils.eval_utils import summarize_steps_dir, write_summary_files
        per_ep_metrics, aggregates = summarize_steps_dir(steps_dir)
        write_summary_files(eval_root, per_ep_metrics, episode_returns)
    except Exception as e:
        print(f"Enhanced summary generation failed: {e}")
        summary_txt = os.path.join(eval_root, 'summary.txt')
        with open(summary_txt, 'w') as f:
            f.write(f"Mean Reward: {mean_reward:.6f}\n")
            f.write(f"Std Reward: {std_reward:.6f}\n")
            f.write(f"Episodes: {len(episode_returns)}\n")
        aggregates = {
            'episodes': float(len(episode_returns)),
        }

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

