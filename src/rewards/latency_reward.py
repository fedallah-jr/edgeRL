"""Latency-based reward function.

This reward encourages lower latency by returning the negative latency.
It supports an optional invalid-action penalty and simple normalization.
"""

from typing import Dict, Any
import numpy as np

try:
    # Optional: used for fallback latency computation if not provided in info
    from edge_sim_py.components.user import User  # type: ignore
except Exception:  # pragma: no cover - if import fails, we'll rely on info
    User = None  # type: ignore


class LatencyReward:
    """Reward calculator based on end-to-end latency.

    Config options:
    - use_total_latency: bool (default False). If True, use total latency; else mean latency.
    - normalize: bool (default True). If True, scale reward roughly to [-1, 1].
      Uses `latency_scale` as divisor before clipping.
    - latency_scale: float (default 100.0). Scale used when normalize is True.
    - penalty_invalid_action: float (default -10.0). Penalty used when info['valid_action'] is False.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.use_total = bool(self.config.get("use_total_latency", False))
        self.normalize = bool(self.config.get("normalize", True))
        self.scale = float(self.config.get("latency_scale", 100.0))
        self.penalty_invalid = float(self.config.get("penalty_invalid_action", -10.0))

    def calculate(self,
                  state,
                  action: int,
                  next_state,
                  info: Dict[str, Any]) -> float:
        # Invalid action penalty if requested
        if not info.get('valid_action', True):
            return float(self.penalty_invalid)

        # Prefer latency from info (Environment should populate these before reward calculation)
        latency = None
        if self.use_total:
            for k in ("total_latency", "latency_total", "sum_latency"):
                if k in info and info[k] is not None:
                    try:
                        latency = float(info[k])
                        break
                    except Exception:
                        pass
        if latency is None:
            for k in ("mean_latency", "avg_latency", "latency"):
                if k in info and info[k] is not None:
                    try:
                        latency = float(info[k])
                        break
                    except Exception:
                        pass

        # Fallback: compute latency directly from simulator if possible
        if latency is None and User is not None:
            try:
                latencies = []
                for user in User.all():  # type: ignore[attr-defined]
                    for app in getattr(user, 'applications', []) or []:
                        d = getattr(user, 'delays', {}).get(str(getattr(app, 'id', '')), None)
                        if d is not None and not (isinstance(d, float) and (np.isinf(d) or np.isnan(d))):
                            latencies.append(float(d))
                if latencies:
                    if self.use_total:
                        latency = float(np.sum(latencies))
                    else:
                        latency = float(np.mean(latencies))
            except Exception:
                pass

        # If still unavailable, return a small neutral negative reward to avoid zeros
        if latency is None:
            return -0.01

        # Reward is negative latency (we want to minimize latency)
        reward = -float(latency)

        if self.normalize:
            denom = self.scale if self.scale > 0 else 1.0
            reward = float(np.clip(reward / denom, -1.0, 1.0))

        return reward

