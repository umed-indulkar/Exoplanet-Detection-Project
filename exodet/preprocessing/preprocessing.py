"""
Preprocessing Module - Preprocessing Team
=======================================

Advanced preprocessing pipeline for light curves.
This is the ACTUAL preprocessing code - no more duplication!

Features:
- NaN removal and validation
- Polynomial/Savitzky-Golay detrending  
- Sigma clipping for outlier removal
- Multiple normalization methods
- Period folding
- Time binning
- Configurable pipeline steps
"""

import numpy as np
from scipy import signal, stats
from typing import Optional, Union, Dict, Tuple, List
import warnings

# Import data loader from the same module
from .data_loader import LightCurve

# Define exceptions locally to avoid core dependency
class PreprocessingError(Exception):
    """Preprocessing failed."""
    pass

class InsufficientDataError(Exception):
    """Not enough data points after preprocessing."""
    pass


class PreprocessingPipeline:
    """
    Configurable preprocessing pipeline for light curves.
    
    Combines best practices with flexible configuration.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize preprocessing pipeline.
        
        Args:
            config: Configuration dictionary. If None, uses defaults.
        """
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        
        self.history = []  # Track preprocessing steps applied
    
    @staticmethod
    def _get_default_config() -> Dict:
        """Get default preprocessing configuration."""
        return {
            'remove_nans': True,
            'detrend': {
                'enabled': True,
                'method': 'polynomial',  # 'polynomial', 'savgol', 'median', 'none'
                'order': 3,  # For polynomial or savgol
                'window_length': 51,  # For savgol or median
            },
            'sigma_clip': {
                'enabled': True,
                'sigma': 3.0,
                'iterations': 3,
                'method': 'iterative',  # 'iterative' or 'mad'
            },
            'normalize': {
                'enabled': True,
                'method': 'zscore',  # 'zscore', 'minmax', 'robust', 'median'
            },
            'fold': {
                'enabled': False,
                'period': None,
                'epoch': 0.0,
            },
            'bin': {
                'enabled': False,
                'bin_size': 0.01,  # In phase units if folded, time units otherwise
                'method': 'weighted',  # 'weighted', 'mean', 'median'
            },
            'quality_mask': {
                'enabled': True,
                'mad_threshold': 10.0,  # Median Absolute Deviation threshold
            },
            'min_points': 10,  # Minimum points after preprocessing
        }
    
    def process(self, lc: LightCurve, inplace: bool = False) -> LightCurve:
        """
        Apply full preprocessing pipeline to a light curve.
        
        Args:
            lc: Input LightCurve object
            inplace: If True, modify input object; if False, create new object
            
        Returns:
            Preprocessed LightCurve object
            
        Raises:
            PreprocessingError: If preprocessing fails
            InsufficientDataError: If too few points remain after preprocessing
        """
        # Copy data if not inplace
        if not inplace:
            lc = LightCurve(
                time=lc.time.copy(),
                flux=lc.flux.copy(),
                flux_err=lc.flux_err.copy() if lc.flux_err is not None else None,
                metadata=lc.metadata.copy() if lc.metadata else None,
                source_file=lc.source_file,
                format=lc.format
            )
        
        self.history = []
        
        # Step 1: Remove NaN values
        if self.config['remove_nans']:
            lc = self._remove_nans(lc)
            self.history.append('remove_nans')
        
        # Step 2: Quality masking
        if self.config['quality_mask']['enabled']:
            lc = self._apply_quality_mask(lc)
            self.history.append('quality_mask')
        
        # Step 3: Detrending
        if self.config['detrend']['enabled']:
            lc = self._detrend(lc)
            self.history.append('detrend')
        
        # Step 4: Sigma clipping
        if self.config['sigma_clip']['enabled']:
            lc = self._sigma_clip(lc)
            self.history.append('sigma_clip')
        
        # Step 5: Period folding
        if self.config['fold']['enabled']:
            lc = self._fold(lc)
            self.history.append('fold')
        
        # Step 6: Binning
        if self.config['bin']['enabled']:
            lc = self._bin(lc)
            self.history.append('bin')
        
        # Step 7: Normalization
        if self.config['normalize']['enabled']:
            lc = self._normalize(lc)
            self.history.append('normalize')
        
        # Final validation
        if len(lc.time) < self.config['min_points']:
            raise InsufficientDataError(
                f"Only {len(lc.time)} points remain after preprocessing "
                f"(minimum: {self.config['min_points']})"
            )
        
        return lc
    
    def _remove_nans(self, lc: LightCurve) -> LightCurve:
        """Remove NaN values from light curve."""
        mask = ~np.isnan(lc.flux)
        if not np.any(mask):
            raise PreprocessingError("All flux values are NaN")
        
        lc.time = lc.time[mask]
        lc.flux = lc.flux[mask]
        if lc.flux_err is not None:
            lc.flux_err = lc.flux_err[mask]
        
        return lc
    
    def _apply_quality_mask(self, lc: LightCurve) -> LightCurve:
        """Apply quality masking using Median Absolute Deviation."""
        flux_median = np.median(lc.flux)
        flux_mad = np.median(np.abs(lc.flux - flux_median))
        
        # Mask points far from median
        threshold = self.config['quality_mask']['mad_threshold'] * flux_mad
        mask = np.abs(lc.flux - flux_median) < threshold
        
        if not np.any(mask):
            raise PreprocessingError("Quality masking removed all points")
        
        lc.time = lc.time[mask]
        lc.flux = lc.flux[mask]
        if lc.flux_err is not None:
            lc.flux_err = lc.flux_err[mask]
        
        return lc
    
    def _detrend(self, lc: LightCurve) -> LightCurve:
        """Remove trends from light curve."""
        method = self.config['detrend']['method']
        
        if method == 'none':
            return lc
        
        if method == 'polynomial':
            order = self.config['detrend']['order']
            coeffs = np.polyfit(lc.time, lc.flux, order)
            trend = np.polyval(coeffs, lc.time)
            lc.flux = lc.flux - trend
            
        elif method == 'savgol':
            window_length = self.config['detrend']['window_length']
            order = self.config['detrend']['order']
            # Ensure window_length is odd and >= order+1
            if window_length % 2 == 0:
                window_length += 1
            window_length = max(window_length, order + 1)
            trend = signal.savgol_filter(lc.flux, window_length, order)
            lc.flux = lc.flux - trend
            
        elif method == 'median':
            window_length = self.config['detrend']['window_length']
            trend = signal.medfilt(lc.flux, kernel_size=window_length)
            lc.flux = lc.flux - trend
        
        return lc
    
    def _sigma_clip(self, lc: LightCurve) -> LightCurve:
        """Remove outliers using sigma clipping."""
        sigma = self.config['sigma_clip']['sigma']
        iterations = self.config['sigma_clip']['iterations']
        method = self.config['sigma_clip']['method']
        
        for _ in range(iterations):
            if method == 'iterative':
                flux_mean = np.mean(lc.flux)
                flux_std = np.std(lc.flux)
                mask = np.abs(lc.flux - flux_mean) < sigma * flux_std
            else:  # mad
                flux_median = np.median(lc.flux)
                flux_mad = np.median(np.abs(lc.flux - flux_median))
                mask = np.abs(lc.flux - flux_median) < sigma * flux_mad
            
            if not np.any(mask):
                break
            
            lc.time = lc.time[mask]
            lc.flux = lc.flux[mask]
            if lc.flux_err is not None:
                lc.flux_err = lc.flux_err[mask]
        
        return lc
    
    def _fold(self, lc: LightCurve) -> LightCurve:
        """Fold light curve on given period."""
        period = self.config['fold']['period']
        epoch = self.config['fold']['epoch']
        
        if period is None:
            raise PreprocessingError("Period folding enabled but no period specified")
        
        # Convert to phase
        lc.time = ((lc.time - epoch) / period) % 1.0
        
        # Sort by phase
        sort_idx = np.argsort(lc.time)
        lc.time = lc.time[sort_idx]
        lc.flux = lc.flux[sort_idx]
        if lc.flux_err is not None:
            lc.flux_err = lc.flux_err[sort_idx]
        
        return lc
    
    def _bin(self, lc: LightCurve) -> LightCurve:
        """Bin light curve."""
        bin_size = self.config['bin']['bin_size']
        method = self.config['bin']['method']
        
        # Create bins
        bins = np.arange(lc.time.min(), lc.time.max() + bin_size, bin_size)
        bin_indices = np.digitize(lc.time, bins) - 1
        
        # Aggregate in each bin
        unique_bins = np.unique(bin_indices)
        binned_time = []
        binned_flux = []
        binned_flux_err = []
        
        for bin_idx in unique_bins:
            mask = bin_indices == bin_idx
            if not np.any(mask):
                continue
            
            if method == 'weighted':
                weights = 1.0 / (lc.flux_err[mask]**2) if lc.flux_err is not None else None
                binned_flux.append(np.average(lc.flux[mask], weights=weights))
                binned_time.append(bins[bin_idx])
                if weights is not None:
                    binned_flux_err.append(np.sqrt(1.0 / np.sum(weights)))
                else:
                    binned_flux_err.append(np.std(lc.flux[mask]))
            else:
                binned_flux.append(np.mean(lc.flux[mask]) if method == 'mean' else np.median(lc.flux[mask]))
                binned_time.append(bins[bin_idx])
                binned_flux_err.append(np.std(lc.flux[mask]))
        
        lc.time = np.array(binned_time)
        lc.flux = np.array(binned_flux)
        lc.flux_err = np.array(binned_flux_err)
        
        return lc
    
    def _normalize(self, lc: LightCurve) -> LightCurve:
        """Normalize light curve."""
        method = self.config['normalize']['method']
        
        if method == 'zscore':
            flux_mean = np.mean(lc.flux)
            flux_std = np.std(lc.flux)
            lc.flux = (lc.flux - flux_mean) / flux_std
            
        elif method == 'minmax':
            flux_min, flux_max = lc.flux.min(), lc.flux.max()
            lc.flux = (lc.flux - flux_min) / (flux_max - flux_min)
            
        elif method == 'robust':
            flux_median = np.median(lc.flux)
            flux_mad = np.median(np.abs(lc.flux - flux_median))
            lc.flux = (lc.flux - flux_median) / flux_mad
            
        elif method == 'median':
            flux_median = np.median(lc.flux)
            lc.flux = lc.flux / flux_median
        
        return lc


def preprocess_lightcurve(lc: LightCurve, config: Optional[Dict] = None) -> LightCurve:
    """
    Convenience function to preprocess a light curve with default settings.
    
    Args:
        lc: Input LightCurve object
        config: Optional configuration dictionary
        
    Returns:
        Preprocessed LightCurve object
    """
    pipeline = PreprocessingPipeline(config)
    return pipeline.process(lc)
