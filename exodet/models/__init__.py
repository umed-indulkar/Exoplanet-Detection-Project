"""
Machine Learning Models Module
==============================

Handles all ML model training and evaluation.
Independent module - ML team only needs to work here.

Main functions:
- train_baseline: Train Random Forest/Logistic Regression
- evaluate_baseline: Evaluate traditional models
- train_siamese: Train Siamese Neural Network
- evaluate_siamese: Evaluate Siamese model
"""

from .baseline_models import train_baseline, evaluate_baseline, load_model, save_model, predict_on_features
try:
    from .siamese_model import train_siamese_from_csv, evaluate_siamese_from_csv
    _HAS_SIAMESE = True
except ImportError:
    train_siamese_from_csv = None
    evaluate_siamese_from_csv = None
    _HAS_SIAMESE = False

__all__ = [
    'train_baseline',
    'evaluate_baseline', 
    'load_model',
    'save_model',
    'predict_on_features',
    'train_siamese_from_csv',
    'evaluate_siamese_from_csv',
    '_HAS_SIAMESE'
]
