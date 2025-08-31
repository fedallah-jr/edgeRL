"""Power consumption-based reward functions.

This reward uses the instantaneous total power (not the change/difference)
across all edge servers, mirroring the style of the latency reward where the
reward is the negative of the current metric.
"""

import numpy as np
from typing import Dict, Any
from edge_sim_py.components.edge_server import EdgeServer


class PowerReward:
    """Reward calculator based on instantaneous total power consumption.

    Config options:
    - power_weight: float (default 1.0). Multiplies the negative total power.
    - normalize: bool (default True). If True, scales the reward using power_scale and clips to [-1, 1].
    - power_scale: float (default 1000.0). Denominator for normalization when normalize=True.
    - penalty_invalid_action: float (default -10.0). Penalty added when info['valid_action'] is False
      (applied after normalization and clipped if normalize=True).
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize power reward calculator.

        Args:
            config: Reward configuration dictionary
        """
        self.config = config or {}
        self.weight = float(self.config.get("power_weight", 1.0))
        self.normalize = bool(self.config.get("normalize", True))
        self.scale = float(self.config.get("power_scale", 1000.0))
        self.penalty_invalid = float(self.config.get("penalty_invalid_action", -10.0))

    def calculate(self,
                  state: np.ndarray,
                  action: int,
                  next_state: np.ndarray,
                  info: Dict[str, Any]) -> float:
        """Calculate reward based on instantaneous total power consumption."""
        # Track invalid action but do not early-return; we will add the penalty to the power-based reward
        is_invalid = not info.get('valid_action', True)

        # Prefer total power from info (environment populates it before reward calc)
        total_power = None
        for k in ("total_power", "power_total", "sum_power"):
            if k in info and info[k] is not None:
                try:
                    total_power = float(info[k])
                    break
                except Exception:
                    pass
        if total_power is None:
            # Fallback: compute from servers directly
            total_power = float(self._calculate_total_power())

        # Base reward: negative instantaneous total power (we want to minimize it)
        reward = -total_power * self.weight

        if self.normalize:
            denom = self.scale if self.scale > 0 else 1.0
            reward = float(np.clip(reward / denom, -1.0, 1.0))


        # Add invalid-action penalty (instead of replacing the reward)
        if is_invalid:
            try:
                reward += float(self.penalty_invalid)
                if self.normalize:
                    reward = float(np.clip(reward, -1.0, 1.0))
            except Exception:
                pass

        return float(reward)

    def _calculate_total_power(self) -> float:
        """Calculate total power consumption across all servers."""
        total_power = 0.0
        try:
            total_power = sum(
                server.get_power_consumption()
                for server in EdgeServer.all()
            )
        except Exception:
            total_power = 0.0
        return float(total_power)

    def reset(self):
        """No state to reset for instantaneous power reward."""
        return None
