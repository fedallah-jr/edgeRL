"""Baselines package for edgeRL."""

from .worst_fit import WorstFitBaseline, evaluate_worst_fit
from .random_fit import RandomFitBaseline, evaluate_random_fit

__all__ = [
    "WorstFitBaseline",
    "evaluate_worst_fit",
    "RandomFitBaseline",
    "evaluate_random_fit",
]

