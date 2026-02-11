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

from .data_loader import load_lightcurve, load_batch_lightcurves
from .preprocessing import preprocess_lightcurve, PreprocessingPipeline

# Define Config locally since core was removed
class Config:
    """Default configuration for preprocessing."""
    def __init__(self):
        self.remove_nans = True
        self.detrend = True
        self.sigma_clip = 3.0
        self.normalize = True
        self.fold_period = None
        self.bin_size = None

__all__ = [
    'load_lightcurve',
    'load_batch_lightcurves', 
    'preprocess_lightcurve',
    'PreprocessingPipeline',
    'Config'
]
