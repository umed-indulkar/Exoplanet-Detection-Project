"""
Exoplanet Detection System - Modular Package
==============================================

A modular toolkit for detecting exoplanets from light curve data.
Each module is independent - teams can work separately:

- preprocessing/: Data loading and cleaning
- feature_extraction/: Feature extraction from light curves  
- models/: ML models (baseline and Siamese)
- visualization/: Visualization and plotting tools
- pipeline/: Integration and workflows
- cli/: Command line interface

Copyright (c) 2025 Exoplanet Detection Team
License: MIT
"""

from .preprocessing import load_lightcurve, load_batch_lightcurves, preprocess_lightcurve
from .feature_extraction import extract_basic_features, extract_tsfresh_features
from .models import train_baseline, evaluate_baseline, train_siamese_from_csv, evaluate_siamese_from_csv
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
    
    # Preprocessing module
    'load_lightcurve',
    'load_batch_lightcurves',
    'preprocess_lightcurve',
    
    # Feature extraction module
    'extract_basic_features',
    'extract_tsfresh_features',
    
    # Models module
    'train_baseline',
    'evaluate_baseline',
    'train_siamese_from_csv',
    'evaluate_siamese_from_csv',
]

# Package metadata
__title__ = 'exodet'
__license__ = 'MIT'
__copyright__ = 'Copyright 2025 Exoplanet Detection Team'
