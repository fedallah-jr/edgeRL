# src/__init__.py
"""EdgeSim-RL: Reinforcement Learning Framework for Edge Computing."""

__version__ = "0.1.0"

# ===================================

# src/envs/__init__.py
"""Environment modules for EdgeSim-RL."""

from .base_env import BaseEdgeEnv
from .edge_env import EdgeEnv

__all__ = ['BaseEdgeEnv', 'EdgeEnv']

# ===================================

# src/rewards/__init__.py
"""Reward functions for EdgeSim-RL."""

from .power_reward import PowerReward, CompositeReward

__all__ = ['PowerReward', 'CompositeReward']

# ===================================

# src/algorithms/__init__.py
"""RL algorithm trainers."""

from .trainer import RLTrainer

__all__ = ['RLTrainer']

# ===================================

# src/utils/__init__.py
"""Utility functions and classes."""

from .metrics import MetricsCollector, PowerAnalyzer, create_training_report

__all__ = ['MetricsCollector', 'PowerAnalyzer', 'create_training_report']

# ===================================

# configs/__init__.py
"""Configuration module for EdgeSim-RL."""