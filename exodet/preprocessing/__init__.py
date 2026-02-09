"""
Preprocessing Module
===================

Handles all data loading, cleaning, and preparation.
Independent module - preprocessing team only needs to work here.

Main functions:
- load_lightcurve: Load raw light curve files
- preprocess_lightcurve: Clean and normalize data
- load_batch_lightcurves: Process multiple files
"""

from ..core.data_loader import load_lightcurve, load_batch_lightcurves
from ..core.preprocessing import preprocess_lightcurve, PreprocessingPipeline
from ..core.config import Config

__all__ = [
    'load_lightcurve',
    'load_batch_lightcurves', 
    'preprocess_lightcurve',
    'PreprocessingPipeline',
    'Config'
]
