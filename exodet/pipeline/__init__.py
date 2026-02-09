"""
Pipeline Module - Integration Team
===================================

Coordinates all modules together for end-to-end workflows.
This is the only module that imports from all other modules.

Main functions:
- run_full_pipeline: Complete workflow from data to results
- organize_dataset: Data preparation and splitting
"""

# Import from all modules for integration
from ..preprocessing import load_lightcurve, preprocess_lightcurve
from ..feature_extraction import extract_basic_features, extract_tsfresh_features
from ..models import train_baseline, evaluate_baseline, train_siamese_from_csv, evaluate_siamese_from_csv

__all__ = [
    'load_lightcurve',
    'preprocess_lightcurve', 
    'extract_basic_features',
    'extract_tsfresh_features',
    'train_baseline',
    'evaluate_baseline',
    'train_siamese_from_csv',
    'evaluate_siamese_from_csv'
]
