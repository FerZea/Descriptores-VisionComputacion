"""
descriptors/ — Paquete con los tres matchers de descriptores.

Expone: ORBMatcher, SIFTMatcher, CannyMatcher.
Todos implementan la misma interfaz: precompute() + score_frame() + threshold + class_names.
"""

from .orb_matcher import ORBMatcher
from .sift_matcher import SIFTMatcher
from .canny_matcher import CannyMatcher

__all__ = ["ORBMatcher", "SIFTMatcher", "CannyMatcher"]
