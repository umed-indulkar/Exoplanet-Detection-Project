"""
Core Module
===========

Core functionality for the Exoplanet Detection System including:
- Data loading from multiple formats (NPZ, CSV, FITS)
- Advanced preprocessing pipeline
- Configuration management
- Custom exceptions
"""

from .data_loader import (
    load_lightcurve,
    load_batch_lightcurves,
    LightCurve,
    UniversalDataLoader
)
from .preprocessing import PreprocessingPipeline, preprocess_lightcurve
from .config import Config, load_config
from .exceptions import (
    ExoDetError,
    DataLoadError,
    PreprocessingError,
    ConfigError,
    FeatureExtractionError
)

__all__ = [
    # Data loading
    'load_lightcurve',
    'load_batch_lightcurves',
    'LightCurve',
    'UniversalDataLoader',
    
    # Preprocessing
    'PreprocessingPipeline',
    'preprocess_lightcurve',
    
    # Configuration
    'Config',
    'load_config',
    
    # Exceptions
    'ExoDetError',
    'DataLoadError',
    'PreprocessingError',
    'ConfigError',
    'FeatureExtractionError',
]
