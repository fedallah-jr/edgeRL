"""Inverse power-based reward function.

This reward computes a value based on the inverse of server power readings.
Two modes are supported:
- sum_of_inverses (default): reward = weight * sum_i 1 / max(P_i, epsilon)
- inverse_of_sum:            reward = weight * 1 / max(sum_i P_i, epsilon)

To encourage lower power consumption, this reward is naturally positive and
increases as powers decrease. Normalization and clipping to [-1, 1] is
supported for stability, similar to other reward implementations.

Config options (all optional):
- mode: "sum_of_inverses" (default) or "inverse_of_sum".
- normalize: bool (default True). If True, divide by scale and clip to [-1,1].
- inv_power_scale / inverse_power_scale / power_scale: float scale for normalization (default 1.0).
- inverse_power_weight / inv_power_weight / power_weight: multiplicative weight (default 1.0).
- penalty_invalid_action: float penalty added if info['valid_action'] is False (default -10.0).
- epsilon: small positive value to avoid division-by-zero (default 1e-6).

"""

from typing import Dict, Any, List
import numpy as np

try:
    from edge_sim_py.components.edge_server import EdgeServer  # type: ignore
except Exception:  # pragma: no cover
    EdgeServer = None  # type: ignore


class InversePowerReward:
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.mode = str(self.config.get("mode", "sum_of_inverses")).lower()
        # Accept common aliases
        if self.mode in ("inverse_sum", "inv_sum", "inverse-of-sum", "inverseofsum"):
            self.mode = "inverse_of_sum"
        elif self.mode in ("sum_inverse", "sum-of-inverses", "suminverse"):
            self.mode = "sum_of_inverses"
        self.normalize = bool(self.config.get("normalize", True))
        # Scale and weight with backwards-compatible fallbacks
        self.scale = float(
            self.config.get(
                "inv_power_scale",
                self.config.get(
                    "inverse_power_scale",
                    self.config.get("power_scale", 1.0),
                ),
            )
        )
        self.weight = float(
            self.config.get(
                "inverse_power_weight",
                self.config.get(
                    "inv_power_weight",
                    self.config.get("power_weight", 1.0),
                ),
            )
        )
        self.penalty_invalid = float(self.config.get("penalty_invalid_action", -10.0))
        self.eps = float(self.config.get("epsilon", 1e-6))

    def calculate(self, state, action: int, next_state, info: Dict[str, Any]) -> float:
        is_invalid = not info.get("valid_action", True)

        powers = self._get_server_powers()
        if not powers:
            # If no powers available, return a small neutral reward so the signal isn't zeroed out
            base = 0.0
        else:
            if self.mode == "inverse_of_sum":
                denom = max(float(np.sum(powers)), self.eps)
                base = 1.0 / denom
            else:  # sum_of_inverses (default)
                invs = [1.0 / max(float(p), self.eps) for p in powers]
                base = float(np.sum(invs))

        reward = base * self.weight

        if self.normalize:
            denom = self.scale if self.scale > 0 else 1.0
            reward = float(np.clip(reward / denom, -1.0, 1.0))

        if is_invalid:
            try:
                reward += float(self.penalty_invalid)
                if self.normalize:
                    reward = float(np.clip(reward, -1.0, 1.0))
            except Exception:
                pass

        return float(reward)

    def _get_server_powers(self) -> List[float]:
        values: List[float] = []
        try:
            servers = EdgeServer.all() if EdgeServer is not None else []
            for s in servers:
                try:
                    p = float(s.get_power_consumption())
                    # Guard against negative or nan/inf values
                    if not (np.isnan(p) or np.isinf(p)):
                        values.append(max(p, 0.0))
                except Exception:
                    continue
        except Exception:
            pass
        return values

    def reset(self):
        return None

