"""Baselines package for edgeRL."""

from .worst_fit import WorstFitBaseline, evaluate_worst_fit

__all__ = [
    "WorstFitBaseline",
    "evaluate_worst_fit",
]

