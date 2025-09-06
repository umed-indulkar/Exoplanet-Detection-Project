"""
Visualization Module

This module provides comprehensive plotting capabilities for light curves,
folded curves, and feature distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.gridspec import GridSpec
import warnings

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_lightcurve(time, flux, flux_err=None, title="Light Curve", 
                   save_path=None, figsize=(12, 6), show_errors=True):
    """
    Plot a basic light curve with optional error bars.
    
    Parameters:
    -----------
    time : array-like
        Time values
    flux : array-like
        Flux values
    flux_err : array-like, optional
        Flux error values
    title : str, default="Light Curve"
        Plot title
    save_path : str, optional
        Path to save the figure
    figsize : tuple, default=(12, 6)
        Figure size (width, height)
    show_errors : bool, default=True
        Whether to show error bars if flux_err is provided
    
    Returns:
    --------
    tuple
        (figure, axes) objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if flux_err is not None and show_errors and len(flux_err) == len(flux):
        ax.errorbar(time, flux, yerr=flux_err, fmt='o', alpha=0.7, 
                   markersize=3, capsize=2, capthick=1)
    else:
        ax.scatter(time, flux, alpha=0.7, s=10)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Flux')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Points: {len(time)}\nMean: {np.mean(flux):.4f}\nStd: {np.std(flux):.4f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, ax


def plot_folded_curve(time, flux, period, epoch=0, flux_err=None, 
                     title="Folded Light Curve", save_path=None, 
                     figsize=(10, 6), phase_bins=100):
    """
    Plot a folded light curve with optional binning.
    
    Parameters:
    -----------
    time : array-like
        Time values
    flux : array-like
        Flux values
    period : float
        Period for folding
    epoch : float, default=0
        Epoch time for folding
    flux_err : array-like, optional
        Flux error values
    title : str, default="Folded Light Curve"
        Plot title
    save_path : str, optional
        Path to save the figure
    figsize : tuple, default=(10, 6)
        Figure size
    phase_bins : int, default=100
        Number of phase bins for binned curve
        
    Returns:
    --------
    tuple
        (figure, axes) objects
    """
    # Calculate phase
    phase = ((time - epoch) / period) % 1.0
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                   gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot folded curve
    if flux_err is not None:
        ax1.errorbar(phase, flux, yerr=flux_err, fmt='o', alpha=0.6, 
                    markersize=2, capsize=1)
    else:
        ax1.scatter(phase, flux, alpha=0.6, s=8)
    
    # Plot binned version if we have enough data
    if len(phase) > phase_bins:
        phase_binned, flux_binned, flux_err_binned = bin_phase_curve(
            phase, flux, flux_err, phase_bins
        )
        ax1.plot(phase_binned, flux_binned, 'r-', linewidth=2, 
                label=f'Binned ({phase_bins} bins)')
        ax1.legend()
    
    ax1.set_ylabel('Flux')
    ax1.set_title(f'{title} (Period: {period:.6f})')
    ax1.grid(True, alpha=0.3)
    
    # Plot phase histogram
    ax2.hist(phase, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Phase')
    ax2.set_ylabel('Count')
    ax2.set_title('Phase Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Ensure phase limits are [0, 1]
    for ax in [ax1, ax2]:
        ax.set_xlim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, (ax1, ax2)


def plot_feature_distribution(features_df, max_features=20, save_path=None, 
                             figsize=(15, 10)):
    """
    Plot distributions of extracted features using histograms and boxplots.
    
    Parameters:
    -----------
    features_df : pandas.DataFrame
        DataFrame containing feature values
    max_features : int, default=20
        Maximum number of features to plot
    save_path : str, optional
        Path to save the figure
    figsize : tuple, default=(15, 10)
        Figure size
        
    Returns:
    --------
    tuple
        (figure, axes) objects
    """
    if features_df.empty:
        print("No features to plot")
        return None, None
    
    # Select features to plot (skip non-numeric or constant features)
    numeric_features = features_df.select_dtypes(include=[np.number]).columns
    varying_features = []
    
    for col in numeric_features:
        if features_df[col].std() > 1e-10:  # Not constant
            varying_features.append(col)
    
    if len(varying_features) == 0:
        print("No varying numeric features found")
        return None, None
    
    # Limit number of features
    features_to_plot = varying_features[:max_features]
    n_features = len(features_to_plot)
    
    # Calculate subplot layout
    n_cols = min(4, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_features == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each feature
    for i, feature in enumerate(features_to_plot):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        data = features_df[feature].dropna()
        
        if len(data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(feature)
            continue
        
        # Plot histogram
        ax.hist(data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title(f'{feature}\n(μ={data.mean():.3f}, σ={data.std():.3f})')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add vertical line for mean
        ax.axvline(data.mean(), color='red', linestyle='--', alpha=0.8, label='Mean')
        ax.legend()
    
    # Hide empty subplots
    for i in range(n_features, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows > 1:
            axes[row, col].set_visible(False)
        else:
            axes[col].set_visible(False)
    
    plt.suptitle(f'Feature Distributions ({n_features} features)', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, axes


def plot_feature_correlation_matrix(features_df, max_features=50, save_path=None, 
                                   figsize=(12, 10)):
    """
    Plot correlation matrix of features.
    
    Parameters:
    -----------
    features_df : pandas.DataFrame
        DataFrame containing feature values
    max_features : int, default=50
        Maximum number of features to include
    save_path : str, optional
        Path to save the figure
    figsize : tuple, default=(12, 10)
        Figure size
        
    Returns:
    --------
    tuple
        (figure, axes) objects
    """
    if features_df.empty:
        print("No features to plot")
        return None, None
    
    # Select numeric features
    numeric_features = features_df.select_dtypes(include=[np.number])
    
    # Remove constant features
    varying_cols = []
    for col in numeric_features.columns:
        if numeric_features[col].std() > 1e-10:
            varying_cols.append(col)
    
    if len(varying_cols) == 0:
        print("No varying features found")
        return None, None
    
    # Limit features
    features_subset = numeric_features[varying_cols[:max_features]]
    
    # Calculate correlation matrix
    corr_matrix = features_subset.corr()
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
    
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                center=0, square=True, ax=ax, cbar_kws={'shrink': 0.8})
    
    ax.set_title(f'Feature Correlation Matrix ({len(features_subset.columns)} features)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, ax


def bin_phase_curve(phase, flux, flux_err, n_bins):
    """
    Bin a phase-folded light curve.
    
    Parameters:
    -----------
    phase : array-like
        Phase values (0 to 1)
    flux : array-like
        Flux values
    flux_err : array-like or None
        Flux error values
    n_bins : int
        Number of phase bins
        
    Returns:
    --------
    tuple
        Binned (phase, flux, flux_err) arrays
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    binned_flux = []
    binned_flux_err = []
    
    for i in range(n_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
        
        if np.sum(mask) == 0:
            binned_flux.append(np.nan)
            binned_flux_err.append(np.nan)
            continue
        
        bin_flux = flux[mask]
        
        if flux_err is not None:
            bin_err = flux_err[mask]
            # Weighted average
            weights = 1.0 / (bin_err**2)
            weighted_flux = np.average(bin_flux, weights=weights)
            weighted_err = 1.0 / np.sqrt(np.sum(weights))
            
            binned_flux.append(weighted_flux)
            binned_flux_err.append(weighted_err)
        else:
            binned_flux.append(np.mean(bin_flux))
            binned_flux_err.append(np.std(bin_flux) / np.sqrt(len(bin_flux)))
    
    return bin_centers, np.array(binned_flux), np.array(binned_flux_err)


def plot_comprehensive_analysis(lc_data, features_df=None, period=None, 
                               save_path=None, figsize=(16, 12)):
    """
    Create a comprehensive multi-panel plot showing various aspects of the light curve.
    
    Parameters:
    -----------
    lc_data : dict
        Light curve data with 'time', 'flux', 'flux_err'
    features_df : pandas.DataFrame, optional
        Extracted features
    period : float, optional
        Period for folding
    save_path : str, optional
        Path to save the figure
    figsize : tuple, default=(16, 12)
        Figure size
        
    Returns:
    --------
    tuple
        (figure, axes) objects
    """
    time = lc_data['time']
    flux = lc_data['flux']
    flux_err = lc_data['flux_err']
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 3, figure=fig)
    
    # Original light curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.scatter(time, flux, alpha=0.7, s=8)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Flux')
    ax1.set_title('Light Curve')
    ax1.grid(True, alpha=0.3)
    
    # Folded curve (if period provided)
    if period is not None:
        ax2 = fig.add_subplot(gs[1, :2])
        phase = ((time - np.min(time)) / period) % 1.0
        ax2.scatter(phase, flux, alpha=0.7, s=8)
        ax2.set_xlabel('Phase')
        ax2.set_ylabel('Flux')
        ax2.set_title(f'Folded Light Curve (P={period:.6f})')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)
        
        # Phase histogram
        ax3 = fig.add_subplot(gs[1, 2])
        ax3.hist(phase, bins=30, alpha=0.7, orientation='horizontal')
        ax3.set_ylabel('Phase')
        ax3.set_xlabel('Count')
        ax3.set_title('Phase Dist.')
        ax3.set_ylim(0, 1)
    
    # Flux histogram
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(flux, bins=30, alpha=0.7, orientation='horizontal')
    ax4.set_ylabel('Flux')
    ax4.set_xlabel('Count')
    ax4.set_title('Flux Distribution')
    
    # Flux vs error
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.scatter(flux_err, flux, alpha=0.7, s=8)
    ax5.set_xlabel('Flux Error')
    ax5.set_ylabel('Flux')
    ax5.set_title('Flux vs Error')
    ax5.grid(True, alpha=0.3)
    
    # Feature summary (if provided)
    ax6 = fig.add_subplot(gs[2, 2])
    if features_df is not None and not features_df.empty:
        # Show top 10 features by value
        numeric_features = features_df.select_dtypes(include=[np.number])
        if len(numeric_features.columns) > 0:
            feature_values = numeric_features.iloc[0].sort_values(ascending=False)[:10]
            ax6.barh(range(len(feature_values)), feature_values.values)
            ax6.set_yticks(range(len(feature_values)))
            ax6.set_yticklabels([f[:15] + '...' if len(f) > 15 else f 
                                for f in feature_values.index])
            ax6.set_xlabel('Feature Value')
            ax6.set_title('Top Features')
        else:
            ax6.text(0.5, 0.5, 'No numeric\nfeatures', ha='center', va='center')
            ax6.set_title('Features')
    else:
        ax6.text(0.5, 0.5, 'No features\nprovided', ha='center', va='center')
        ax6.set_title('Features')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comprehensive analysis saved to {save_path}")
    
    return fig, [ax1, ax2 if period else None, ax3 if period else None, 
                ax4, ax5, ax6]