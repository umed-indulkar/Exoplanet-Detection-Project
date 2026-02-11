"""
Feature Extraction Module
========================

Handles all feature extraction from light curves.
Independent module - feature extraction team only needs to work here.

Main functions:
- extract_basic_features: Simple statistical features
- extract_tsfresh_features: Advanced TSFresh features
"""

from .basic_extractor import BasicFeatureExtractor, extract_basic_features
try:
    from .tsfresh_extractor import extract_tsfresh_features
    _HAS_TSFRESH = True
except ImportError:
    extract_tsfresh_features = None
    _HAS_TSFRESH = False

__all__ = [
    'BasicFeatureExtractor',
    'extract_basic_features',
    'extract_tsfresh_features',
    '_HAS_TSFRESH'
]
