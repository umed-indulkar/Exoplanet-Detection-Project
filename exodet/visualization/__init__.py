"""
Visualization Module
====================

Comprehensive plotting tools for light curves and features.
"""

from .plots import (
    plot_lightcurve,
    plot_folded_lightcurve,
    plot_feature_distributions,
    plot_comparison
)

__all__ = [
    'plot_lightcurve',
    'plot_folded_lightcurve',
    'plot_feature_distributions',
    'plot_comparison',
]
