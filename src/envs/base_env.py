"""Base environment class for EdgeSim-RL."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import gym
import numpy as np


class BaseEdgeEnv(gym.Env, ABC):
    """Abstract base class for EdgeSim environments.
    
    This class provides the interface between EdgeSimPy simulations
    and RLlib training algorithms.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the base environment.
        
        Args:
            config: Environment configuration dictionary
        """
        super().__init__()
        self.config = config
        self.current_step = 0
        self.max_steps = config.get("max_steps", 1000)
        
        # These should be defined in subclasses
        self.observation_space = None
        self.action_space = None
        
    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state.
        
        Returns:
            Initial observation
        """
        self.current_step = 0
        
    @abstractmethod
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one environment step.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        self.current_step += 1
        done = self.current_step >= self.max_steps
        return None, 0.0, done, {}
    
    @abstractmethod
    def get_state(self) -> np.ndarray:
        """Get current environment state.
        
        Returns:
            State observation as numpy array
        """
        pass
    
    @abstractmethod
    def is_valid_action(self, action: Any) -> bool:
        """Check if an action is valid in current state.
        
        Args:
            action: Action to validate
            
        Returns:
            True if action is valid
        """
        pass
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the environment.
        
        Args:
            mode: Rendering mode
            
        Returns:
            Rendered frame if mode is 'rgb_array'
        """
        if mode == 'human':
            print(f"Step: {self.current_step}/{self.max_steps}")
        return None
    
    def close(self) -> None:
        """Clean up environment resources."""
        pass
    
    def seed(self, seed: Optional[int] = None) -> list:
        """Set random seed for reproducibility.
        
        Args:
            seed: Random seed value
            
        Returns:
            List containing the seed
        """
        np.random.seed(seed)
        return [seed]