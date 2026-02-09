"""
TSFresh Feature Extraction - Feature Extraction Team
======================================================

Moved from features/ to feature_extraction/ for modular structure.
Original file: exodet/features/tsfresh_extractor.py
"""

from importlib import import_module
import sys
import os

# Add the original path to import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'features'))

# Import the original module
original_module = import_module('tsfresh_extractor')

# Export everything from original module
globals().update(vars(original_module))
