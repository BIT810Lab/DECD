"""Confidence-aware dispatcher (DECD Stage II) and safe inference."""

from .abstention import abstention_threshold, selective_predictions
from .calibration import fit_temperature, temperature_nll
from .model import ConfidenceAwareDispatcher, fused_probability, routed_confidence
from .signals import reliability_vector

__all__ = [
    "ConfidenceAwareDispatcher",
    "abstention_threshold",
    "fit_temperature",
    "fused_probability",
    "reliability_vector",
    "routed_confidence",
    "selective_predictions",
    "temperature_nll",
]
