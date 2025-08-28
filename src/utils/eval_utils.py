"""Evaluation utilities for summarizing per-episode step metrics.

This module parses the step-level CSVs saved by EdgeSimCallbacks during
evaluation and computes per-episode and aggregated metrics, including:
- migrations (count of True values)
- valid_actions (count of True values)
- unique_actions (number of distinct actions taken)
- total_latency (sum of latency values if available)

It also provides helpers for writing a consolidated eval_summary.csv and
an enriched summary.txt in the evaluation output directory.
"""

import os
import csv
from typing import Dict, List, Tuple, Optional


def _parse_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val != 0
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("true", "1", "yes", "y"):  # treat these as True
            return True
        if s in ("false", "0", "no", "n", ""):  # treat these as False
            return False
    return False


def _parse_float(val) -> float:
    try:
        if val is None:
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        s = str(val).strip()
        if not s:
            return 0.0
        return float(s)
    except Exception:
        return 0.0


def _extract_episode_index(filename: str) -> Optional[int]:
    # Expected pattern: episode_<N>_steps.csv
    base = os.path.basename(filename)
    if not base.startswith("episode_") or not base.endswith("_steps.csv"):
        return None
    middle = base[len("episode_") : -len("_steps.csv")]
    try:
        return int(middle)
    except Exception:
        return None


def list_step_csvs(steps_dir: str) -> List[str]:
    files = []
    if not os.path.isdir(steps_dir):
        return files
    for name in os.listdir(steps_dir):
        if name.startswith("episode_") and name.endswith("_steps.csv"):
            files.append(os.path.join(steps_dir, name))
    # Sort by episode index if possible
    files.sort(key=lambda p: (_extract_episode_index(p) or 0))
    return files


def compute_episode_metrics_from_csv(csv_path: str) -> Dict[str, float]:
    migrations = 0
    valid_actions = 0
    unique_actions = set()
    total_latency = 0.0

    with open(csv_path, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        # Identify candidate latency columns
        latency_cols = [
            c for c in reader.fieldnames or [] if c.lower() in ("latency", "total_latency", "latency_total")
        ]
        for row in reader:
            if "migration" in row:
                if _parse_bool(row["migration"]):
                    migrations += 1
            if "valid_action" in row:
                if _parse_bool(row["valid_action"]):
                    valid_actions += 1
            if "action" in row:
                a = row["action"]
                if a is not None and str(a) != "":
                    try:
                        unique_actions.add(int(float(a)))
                    except Exception:
                        # Fall back to string representation
                        unique_actions.add(str(a))
            # Sum any available latency-like columns
            for col in latency_cols:
                total_latency += _parse_float(row.get(col, 0.0))

    ep_index = _extract_episode_index(csv_path) or 0
    return {
        "episode": ep_index,
        "migrations": float(migrations),
        "valid_actions": float(valid_actions),
        "unique_actions": float(len(unique_actions)),
        "total_latency": float(total_latency),
    }


essential_metric_keys = ["migrations", "valid_actions", "unique_actions", "total_latency"]


def summarize_steps_dir(steps_dir: str) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """Summarize all episode step CSVs in a directory.

    Returns:
        (per_episode_metrics_list, aggregates_dict)
    """
    per_episode: List[Dict[str, float]] = []
    for csv_path in list_step_csvs(steps_dir):
        try:
            per_episode.append(compute_episode_metrics_from_csv(csv_path))
        except Exception:
            # Skip malformed CSV
            continue
    per_episode.sort(key=lambda d: d.get("episode", 0))

    n = len(per_episode)
    totals = {k: 0.0 for k in essential_metric_keys}
    for m in per_episode:
        for k in essential_metric_keys:
            totals[k] += float(m.get(k, 0.0))

    means = {f"mean_{k}": (totals[k] / n if n > 0 else 0.0) for k in essential_metric_keys}
    aggregates = {**totals, **means, "episodes": float(n)}
    return per_episode, aggregates


def write_summary_files(
    eval_root: str,
    per_episode_metrics: List[Dict[str, float]],
    episode_rewards: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Write eval_summary.csv and summary.txt with enhanced metrics.

    Returns the aggregates dict for convenience.
    """
    # Build CSV rows aligned by episode index (1..max_count)
    max_count = max(len(per_episode_metrics), len(episode_rewards or []))
    # Map episode->metrics for quick lookup
    metrics_map = {int(m.get("episode", i + 1)): m for i, m in enumerate(per_episode_metrics)}

    # Prepare output paths
    summary_csv = os.path.join(eval_root, "eval_summary.csv")
    summary_txt = os.path.join(eval_root, "summary.txt")

    # Write CSV with extended columns
    with open(summary_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode",
            "reward",
            "migrations",
            "valid_actions",
            "unique_actions",
            "total_latency",
        ])
        for i in range(1, max_count + 1):
            m = metrics_map.get(i, {})
            r = (episode_rewards[i - 1] if episode_rewards and i - 1 < len(episode_rewards) else "")
            writer.writerow([
                i,
                r if r != "" else "",
                m.get("migrations", ""),
                m.get("valid_actions", ""),
                m.get("unique_actions", ""),
                m.get("total_latency", ""),
            ])

    # Compute aggregates for summary.txt
    n = max_count if max_count > 0 else 0
    total_migrations = sum(float(metrics_map.get(i, {}).get("migrations", 0.0)) for i in range(1, max_count + 1))
    total_valid_actions = sum(float(metrics_map.get(i, {}).get("valid_actions", 0.0)) for i in range(1, max_count + 1))
    total_latency = sum(float(metrics_map.get(i, {}).get("total_latency", 0.0)) for i in range(1, max_count + 1))
    mean_migrations = (total_migrations / n) if n > 0 else 0.0
    mean_valid_actions = (total_valid_actions / n) if n > 0 else 0.0
    mean_latency = (total_latency / n) if n > 0 else 0.0

    # Write enriched summary.txt
    with open(summary_txt, "w") as f:
        if episode_rewards:
            import numpy as _np
            f.write(f"Mean Reward: {_np.mean(episode_rewards):.6f}\n")
            f.write(f"Std Reward: {_np.std(episode_rewards):.6f}\n")
        f.write(f"Episodes: {n}\n")
        f.write(f"Total Migrations: {total_migrations:.6f}\n")
        f.write(f"Mean Migrations per Episode: {mean_migrations:.6f}\n")
        f.write(f"Total Valid Actions: {total_valid_actions:.6f}\n")
        f.write(f"Mean Valid Actions per Episode: {mean_valid_actions:.6f}\n")
        f.write(f"Total Latency: {total_latency:.6f}\n")
        f.write(f"Mean Latency per Episode: {mean_latency:.6f}\n")
        # Also include per-episode unique actions counts
        f.write("Unique Actions per Episode:\n")
        for i in range(1, max_count + 1):
            ua = metrics_map.get(i, {}).get("unique_actions", 0.0)
            f.write(f"  Episode {i}: {ua}\n")

    return {
        "episodes": float(n),
        "total_migrations": float(total_migrations),
        "mean_migrations": float(mean_migrations),
        "total_valid_actions": float(total_valid_actions),
        "mean_valid_actions": float(mean_valid_actions),
        "total_latency": float(total_latency),
        "mean_latency": float(mean_latency),
    }

