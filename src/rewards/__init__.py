
"""Reward functions for EdgeSim-RL."""

from .power_reward import PowerReward
from .latency_reward import LatencyReward

__all__ = ['PowerReward', 'LatencyReward']
