"""EdgeAISIM-style reward.

This reward reproduces the immediate (snapshot) reward used in the EdgeAISIM
repository: After applying an action, the reward is computed as the sum over
all edge servers of the inverse of their current power consumption.

Note:
- No normalization/scaling is applied by default (to match EdgeAISIM).
- No explicit penalty for invalid actions (to match EdgeAISIM). You may
  enable optional penalties via config if desired.
"""

from typing import Dict, Any
from edge_sim_py.components.edge_server import EdgeServer


class EdgeAISIMReward:
    """EdgeAISIM-equivalent reward: sum(1 / power_i) over all servers.

    Config options:
    - epsilon: float (default: 1e-9). Small stabilizer to avoid division-by-zero.
    - penalty_invalid_action: Optional[float]. If set, apply this penalty when
      info.get('valid_action') is False. Default: None (no penalty) to mirror
      EdgeAISIM.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.epsilon = float(self.config.get("epsilon", 1e-9))
        self.penalty_invalid = self.config.get("penalty_invalid_action", None)

    def calculate(self,
                  state,
                  action: int,
                  next_state,
                  info: Dict[str, Any]) -> float:
        # Optional penalty for invalid actions (disabled by default to match EdgeAISIM)
        if self.penalty_invalid is not None and not info.get('valid_action', True):
            return float(self.penalty_invalid)

        total = 0.0
        for server in EdgeServer.all():
            try:
                p = server.get_power_consumption()
                # Avoid division by zero; in EdgeAISIM, power is assumed > 0
                denom = p if p > 0 else self.epsilon
                total += 1.0 / denom
            except Exception:
                # If power retrieval fails, skip this server
                continue
        return float(total)
