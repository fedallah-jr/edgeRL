"""Power consumption-based reward functions."""

import numpy as np
from typing import Dict, Any
from edge_sim_py.components.edge_server import EdgeServer


class PowerReward:
    """Reward calculator based on power consumption.
    
    Supports different modes. By default, aligns with EdgeAISIM by using
    the sum of inverse server power as reward.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize power reward calculator.
        
        Args:
            config: Reward configuration dictionary
        """
        self.config = config
        # Modes: "inverse_sum" (EdgeAISIM default), "diff_total_power" (original behavior)
        self.mode = config.get("mode", "inverse_sum")
        self.weight = config.get("power_weight", 1.0)
        self.normalize = config.get("normalize", False if self.mode == "inverse_sum" else True)
        self.penalty_invalid = config.get("penalty_invalid_action", -10.0)
        self.epsilon = float(config.get("epsilon", 1e-6))
        
        # Track previous power consumption for difference reward
        self.previous_power = None
        
    def calculate(self, 
                  state: np.ndarray,
                  action: int,
                  next_state: np.ndarray,
                  info: Dict[str, Any]) -> float:
        """Calculate reward based on power consumption.
        
        Args:
            state: State before action
            action: Action taken
            next_state: State after action
            info: Additional information from environment
            
        Returns:
            Calculated reward value
        """
        # Check for invalid action
        if not info.get('valid_action', True):
            return self.penalty_invalid
        
        # Compute per-server powers
        powers = [server.get_power_consumption() for server in EdgeServer.all()]
        total_power = float(sum(powers))
        
        # Debug prints
        if self.config.get('debug', False):
            if self.mode == "inverse_sum":
                inv_sum = sum(1.0 / max(p, self.epsilon) for p in powers)
                print(f"Total power: {total_power}, Inverse-sum reward (pre-weight): {inv_sum}")
            else:
                print(f"Current power: {total_power}, Previous: {self.previous_power}")
        
        if self.mode == "inverse_sum":
            # EdgeAISIM-style reward: sum of inverse server powers
            reward = sum(1.0 / max(p, self.epsilon) for p in powers) * self.weight
            # Optional normalization (off by default)
            if self.normalize and len(powers) > 0:
                reward = reward / len(powers)
            return float(reward)
        
        # Original behavior: difference in total power across steps (minimize)
        current_power = total_power
        reward = -current_power * self.weight
        
        if self.previous_power is not None:
            power_diff = current_power - self.previous_power
            reward = -power_diff * self.weight
            if power_diff < 0:
                reward += abs(power_diff) * 0.5
        
        self.previous_power = current_power
        
        if self.normalize:
            reward = np.clip(reward / 100.0, -1.0, 1.0)
        
        if info.get('migration', False):
            reward += 0.1
        
        if abs(reward) < 0.001:
            reward = -0.01
        
        return float(reward)
    
    def _calculate_total_power(self) -> float:
        """Calculate total power consumption across all servers.
        
        Returns:
            Total power consumption
        """
        total_power = sum(
            server.get_power_consumption() 
            for server in EdgeServer.all()
        )
        return total_power
    
    def reset(self):
        """Reset reward calculator state."""
        self.previous_power = self._calculate_total_power()


class CompositeReward:
    """Composite reward combining multiple objectives.
    
    This class can be extended to combine power, latency,
    and other metrics for multi-objective optimization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize composite reward calculator.
        
        Args:
            config: Reward configuration dictionary
        """
        self.config = config
        self.power_calculator = PowerReward(config)
        
        # Weights for different components
        self.power_weight = config.get("power_weight", 0.6)
        self.latency_weight = config.get("latency_weight", 0.3)
        self.balance_weight = config.get("balance_weight", 0.1)
        
    def calculate(self,
                  state: np.ndarray,
                  action: int,
                  next_state: np.ndarray,
                  info: Dict[str, Any]) -> float:
        """Calculate composite reward.
        
        Args:
            state: State before action
            action: Action taken
            next_state: State after action
            info: Additional information from environment
            
        Returns:
            Calculated composite reward
        """
        # Power component
        power_reward = self.power_calculator.calculate(
            state, action, next_state, info
        )
        
        # Load balancing component
        balance_reward = self._calculate_balance_reward()
        
        # Combine components
        total_reward = (
            self.power_weight * power_reward +
            self.balance_weight * balance_reward
        )
        
        return float(total_reward)
    
    def _calculate_balance_reward(self) -> float:
        """Calculate reward for load balancing.
        
        Returns:
            Balance reward based on resource utilization variance
        """
        utilizations = []
        
        for server in EdgeServer.all():
            if server.cpu > 0:
                cpu_util = server.cpu_demand / server.cpu
                utilizations.append(cpu_util)
        
        if len(utilizations) > 1:
            # Reward low variance (better balance)
            variance = np.var(utilizations)
            balance_reward = -variance  # Negative because we minimize variance
        else:
            balance_reward = 0
        
        return balance_reward
    
    def reset(self):
        """Reset reward calculator state."""
        self.power_calculator.reset()