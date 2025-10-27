"""
Feature Extraction Module
==========================

Three-tier feature extraction system:
- Fast: 100+ features in <1s per curve (basic statistics and simple features)
- Standard: 150+ features in 2-5s per curve (includes ML-optimized features)
- Comprehensive: 500+ features in 10-30s per curve (full TSFresh feature set)
"""

from .basic_extractor import BasicFeatureExtractor, extract_basic_features

__all__ = [
    # Extractors
    'BasicFeatureExtractor',
    
    # Convenience functions
    'extract_basic_features',
]
