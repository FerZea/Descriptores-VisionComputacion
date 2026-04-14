"""
descriptors/ — Paquete con los tres matchers de descriptores.

Expone: ORBMatcher, SIFTMatcher, AKAZEMatcher.
Todos implementan la misma interfaz: precompute() + score_frame() + threshold + class_names.
"""

from .orb_matcher import ORBMatcher
from .sift_matcher import SIFTMatcher
from .akaze_matcher import AKAZEMatcher

__all__ = ["ORBMatcher", "SIFTMatcher", "AKAZEMatcher"]
