"""
Exoplanet Detection System - Unified Package
==============================================

A comprehensive toolkit for detecting exoplanets from light curve data combining:
- Advanced feature extraction (100-800+ features)
- Machine learning models (Siamese networks, ensembles)
- High-performance parallel processing
- Interactive visualization and dashboards

Copyright (c) 2025 Exoplanet Detection Team
License: MIT
"""

from .core.data_loader import load_lightcurve, load_batch_lightcurves
from .core.preprocessing import PreprocessingPipeline, preprocess_lightcurve
from .core.config import Config
from .features.basic_extractor import BasicFeatureExtractor, extract_basic_features
from .__version__ import (
    __version__,
    __author__,
    __email__,
    __description__
)

__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__email__',
    '__description__',
    
    # Core functionality
    'load_lightcurve',
    'load_batch_lightcurves',
    'PreprocessingPipeline',
    'preprocess_lightcurve',
    'Config',
    
    # Feature extraction
    'BasicFeatureExtractor',
    'extract_basic_features',
]

# Package metadata
__title__ = 'exodet'
__license__ = 'MIT'
__copyright__ = 'Copyright 2025 Exoplanet Detection Team'
