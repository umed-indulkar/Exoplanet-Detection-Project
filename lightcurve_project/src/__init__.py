"""
Light Curve Analysis Package

A comprehensive toolkit for processing, visualizing, and extracting features 
from astronomical light curves stored in .npz format.
"""

__version__ = "1.0.0"
__author__ = "Light Curve Analysis Project"

from .data_loader import load_npz_curve, preprocess_lightcurve
from .visualization import plot_lightcurve, plot_folded_curve, plot_feature_distribution
from .feature_extraction import extract_features
from .feature_pruning import manual_prune

__all__ = [
    'load_npz_curve',
    'preprocess_lightcurve', 
    'plot_lightcurve',
    'plot_folded_curve',
    'plot_feature_distribution',
    'extract_features',
    'manual_prune'
]