"""Power consumption-based reward functions."""

import numpy as np
from typing import Dict, Any
from edge_sim_py.components.edge_server import EdgeServer


class PowerReward:
    """Reward calculator based on power consumption.
    
    This class implements reward calculation focused on minimizing
    total power consumption across edge servers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize power reward calculator.
        
        Args:
            config: Reward configuration dictionary
        """
        self.config = config
        self.weight = config.get("power_weight", 1.0)
        self.normalize = config.get("normalize", True)
        self.penalty_invalid = config.get("penalty_invalid_action", -10.0)
        
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
        
        # Calculate current total power consumption
        current_power = self._calculate_total_power()
        
        # Debug: print power consumption
        if self.config.get('debug', False):
            print(f"Current power: {current_power}, Previous: {self.previous_power}")
        
        # Base reward: negative power consumption (minimize)
        reward = -current_power * self.weight
        
        # If we have previous power, use difference as reward
        if self.previous_power is not None:
            power_diff = current_power - self.previous_power
            reward = -power_diff * self.weight  # Negative because we want to minimize
            
            # Bonus for reducing power
            if power_diff < 0:
                reward += abs(power_diff) * 0.5  # Extra reward for improvement
        
        # Update previous power
        self.previous_power = current_power
        
        # Normalize reward if configured
        if self.normalize:
            # Normalize to roughly [-1, 1] range
            # Assuming max power change is around 100W
            reward = np.clip(reward / 100.0, -1.0, 1.0)
        
        # Add small bonus for successful migration
        if info.get('migration', False):
            reward += 0.1
        
        # Provide a small non-zero reward even if power is zero
        if abs(reward) < 0.001:
            # Give small negative reward to encourage exploration
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
        self.previous_power = None


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