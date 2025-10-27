"""
Plotting Functions for Light Curves and Features
=================================================

Extracted and enhanced from lightcurve_project/src/visualization.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Optional, Union, Tuple

from ..core.data_loader import LightCurve

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_lightcurve(
    lc: Union[LightCurve, tuple],
    title: str = "Light Curve",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    show_errors: bool = True
):
    """
    Plot a light curve.
    
    Args:
        lc: LightCurve object or tuple of (time, flux, flux_err)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        show_errors: Whether to show error bars
        
    Returns:
        Figure and axes objects
    """
    # Handle input
    if isinstance(lc, LightCurve):
        time = lc.time
        flux = lc.flux
        flux_err = lc.flux_err if len(lc.flux_err) > 0 else None
    else:
        time, flux = lc[0], lc[1]
        flux_err = lc[2] if len(lc) > 2 else None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if flux_err is not None and show_errors:
        ax.errorbar(time, flux, yerr=flux_err, fmt='o', alpha=0.6,
                   markersize=2, capsize=1, label='Data')
    else:
        ax.plot(time, flux, 'o', alpha=0.6, markersize=2, label='Data')
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Flux', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = (
        f'Points: {len(time)}\n'
        f'Mean: {np.mean(flux):.4f}\n'
        f'Std: {np.std(flux):.4f}\n'
        f'Range: {np.max(flux) - np.min(flux):.4f}'
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig, ax


def plot_folded_lightcurve(
    lc: LightCurve,
    period: float,
    epoch: float = 0.0,
    title: str = "Folded Light Curve",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot a phase-folded light curve.
    
    Args:
        lc: LightCurve object
        period: Folding period
        epoch: Epoch time
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Figure and axes objects
    """
    # Calculate phase
    phase = ((lc.time - epoch) / period) % 1.0
    
    # Sort by phase
    sort_idx = np.argsort(phase)
    phase_sorted = phase[sort_idx]
    flux_sorted = lc.flux[sort_idx]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot folded curve
    ax.plot(phase_sorted, flux_sorted, 'o', alpha=0.5, markersize=2)
    
    # Plot again shifted by 1 for continuity
    ax.plot(phase_sorted + 1, flux_sorted, 'o', alpha=0.5, markersize=2)
    
    ax.set_xlabel('Phase', fontsize=12)
    ax.set_ylabel('Flux', fontsize=12)
    ax.set_title(f'{title} (Period: {period:.6f})', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 2)
    ax.grid(True, alpha=0.3)
    
    # Add vertical line at phase 0.5 and 1.5
    ax.axvline(0.5, color='red', linestyle='--', alpha=0.3, label='Phase 0.5')
    ax.axvline(1.5, color='red', linestyle='--', alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig, ax


def plot_feature_distributions(
    features_df: pd.DataFrame,
    features_to_plot: Optional[list] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Plot distributions of extracted features.
    
    Args:
        features_df: DataFrame with features
        features_to_plot: List of feature names to plot (None = plot top 12)
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Figure object
    """
    if features_to_plot is None:
        # Plot first 12 numeric features
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        features_to_plot = list(numeric_cols[:12])
    
    n_features = len(features_to_plot)
    n_rows = (n_features + 2) // 3
    n_cols = min(3, n_features)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for i, feature in enumerate(features_to_plot):
        if i >= len(axes):
            break
            
        ax = axes[i]
        data = features_df[feature].dropna()
        
        if len(data) > 0:
            ax.hist(data, bins=30, alpha=0.7, edgecolor='black')
            ax.set_xlabel(feature, fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.set_title(f'{feature}\nMean: {data.mean():.3f}, Std: {data.std():.3f}',
                        fontsize=9)
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(features_to_plot), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Feature Distributions', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_comparison(
    lc_original: LightCurve,
    lc_processed: LightCurve,
    title: str = "Before vs After Preprocessing",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6)
):
    """
    Plot original vs processed light curves side by side.
    
    Args:
        lc_original: Original LightCurve
        lc_processed: Processed LightCurve
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Figure and axes objects
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Original
    ax1.plot(lc_original.time, lc_original.flux, 'o', alpha=0.5, markersize=2)
    ax1.set_xlabel('Time', fontsize=11)
    ax1.set_ylabel('Flux', fontsize=11)
    ax1.set_title('Original', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    stats1 = (
        f'Points: {len(lc_original)}\n'
        f'Mean: {np.mean(lc_original.flux):.4f}\n'
        f'Std: {np.std(lc_original.flux):.4f}'
    )
    ax1.text(0.02, 0.98, stats1, transform=ax1.transAxes,
            verticalalignment='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Processed
    ax2.plot(lc_processed.time, lc_processed.flux, 'o', alpha=0.5, markersize=2, color='orange')
    ax2.set_xlabel('Time', fontsize=11)
    ax2.set_ylabel('Flux', fontsize=11)
    ax2.set_title('Processed', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    stats2 = (
        f'Points: {len(lc_processed)}\n'
        f'Mean: {np.mean(lc_processed.flux):.4f}\n'
        f'Std: {np.std(lc_processed.flux):.4f}\n'
        f'Removed: {len(lc_original) - len(lc_processed)}'
    )
    ax2.text(0.02, 0.98, stats2, transform=ax2.transAxes,
            verticalalignment='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig, (ax1, ax2)
