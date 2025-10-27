"""
Preprocessing Pipeline Module
==============================

Advanced preprocessing pipeline combining best practices from all branches:
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

from .data_loader import LightCurve
from .exceptions import PreprocessingError, InsufficientDataError


class PreprocessingPipeline:
    """
    Configurable preprocessing pipeline for light curves.
    
    Combines best practices from all branches with flexible configuration.
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
                flux_err=lc.flux_err.copy() if len(lc.flux_err) > 0 else np.array([]),
                metadata=lc.metadata.copy(),
                source_file=lc.source_file,
                format=lc.format
            )
        
        self.history = []
        
        try:
            # Step 1: Remove NaNs and infinite values
            if self.config['remove_nans']:
                lc = self._remove_invalid_points(lc)
                self.history.append('remove_nans')
            
            # Step 2: Quality masking (remove extreme outliers first)
            if self.config['quality_mask']['enabled']:
                lc = self._apply_quality_mask(lc)
                self.history.append('quality_mask')
            
            # Step 3: Detrending
            if self.config['detrend']['enabled']:
                lc = self._detrend(lc)
                self.history.append(f"detrend_{self.config['detrend']['method']}")
            
            # Step 4: Sigma clipping
            if self.config['sigma_clip']['enabled']:
                lc = self._sigma_clip(lc)
                self.history.append('sigma_clip')
            
            # Step 5: Period folding (if requested)
            if self.config['fold']['enabled'] and self.config['fold']['period']:
                lc = self._fold(lc)
                self.history.append('fold')
            
            # Step 6: Binning (if requested)
            if self.config['bin']['enabled']:
                lc = self._bin(lc)
                self.history.append('bin')
            
            # Step 7: Normalization
            if self.config['normalize']['enabled']:
                lc = self._normalize(lc)
                self.history.append(f"normalize_{self.config['normalize']['method']}")
            
            # Final validation
            if len(lc) < self.config['min_points']:
                raise InsufficientDataError(
                    f"Only {len(lc)} points remain after preprocessing "
                    f"(minimum: {self.config['min_points']})"
                )
            
            # Add preprocessing info to metadata
            lc.metadata['preprocessing'] = {
                'steps': self.history,
                'config': self.config
            }
            
            return lc
            
        except Exception as e:
            if isinstance(e, (PreprocessingError, InsufficientDataError)):
                raise
            raise PreprocessingError(f"Preprocessing failed: {str(e)}")
    
    def _remove_invalid_points(self, lc: LightCurve) -> LightCurve:
        """Remove NaN and infinite values."""
        valid_mask = np.isfinite(lc.time) & np.isfinite(lc.flux)
        
        if len(lc.flux_err) > 0:
            valid_mask &= np.isfinite(lc.flux_err) & (lc.flux_err > 0)
        
        n_removed = np.sum(~valid_mask)
        
        if n_removed > 0:
            lc.time = lc.time[valid_mask]
            lc.flux = lc.flux[valid_mask]
            if len(lc.flux_err) > 0:
                lc.flux_err = lc.flux_err[valid_mask]
        
        return lc
    
    def _apply_quality_mask(self, lc: LightCurve) -> LightCurve:
        """Remove extreme outliers based on MAD threshold."""
        median_flux = np.median(lc.flux)
        mad = np.median(np.abs(lc.flux - median_flux))
        threshold = self.config['quality_mask']['mad_threshold']
        
        quality_mask = np.abs(lc.flux - median_flux) < threshold * mad
        
        n_removed = np.sum(~quality_mask)
        
        if n_removed > 0:
            lc.time = lc.time[quality_mask]
            lc.flux = lc.flux[quality_mask]
            if len(lc.flux_err) > 0:
                lc.flux_err = lc.flux_err[quality_mask]
        
        return lc
    
    def _detrend(self, lc: LightCurve) -> LightCurve:
        """Remove systematic trends from flux."""
        method = self.config['detrend'].get('method', 'polynomial')
        
        if method == 'polynomial':
            # Polynomial detrending (from main branch)
            order = self.config['detrend'].get('order', 3)
            try:
                coeffs = np.polyfit(lc.time, lc.flux, deg=order)
                trend = np.polyval(coeffs, lc.time)
                lc.flux = lc.flux / trend
            except Exception as e:
                warnings.warn(f"Polynomial detrending failed: {e}")
        
        elif method == 'savgol':
            # Savitzky-Golay filter
            window_length = self.config['detrend'].get('window_length', 51)
            order = self.config['detrend'].get('order', 3)
            
            # Ensure window_length is odd and valid
            window_length = min(window_length, len(lc.flux))
            if window_length % 2 == 0:
                window_length -= 1
            window_length = max(order + 2, window_length)
            
            try:
                trend = signal.savgol_filter(lc.flux, window_length, order)
                lc.flux = lc.flux / trend
            except Exception as e:
                warnings.warn(f"Savitzky-Golay detrending failed: {e}")
        
        elif method == 'median':
            # Median filtering
            window_length = self.config['detrend'].get('window_length', 51)
            try:
                trend = signal.medfilt(lc.flux, kernel_size=window_length)
                lc.flux = lc.flux / trend
            except Exception as e:
                warnings.warn(f"Median detrending failed: {e}")
        
        return lc
    
    def _sigma_clip(self, lc: LightCurve) -> LightCurve:
        """Iterative sigma clipping to remove outliers."""
        sigma = self.config['sigma_clip'].get('sigma', 3.0)
        iterations = self.config['sigma_clip'].get('iterations', 3)
        method = self.config['sigma_clip'].get('method', 'iterative')
        
        time = lc.time
        flux = lc.flux
        flux_err = lc.flux_err if len(lc.flux_err) > 0 else None
        
        for i in range(iterations):
            if method == 'iterative':
                # Standard iterative sigma clipping
                mean = np.mean(flux)
                std = np.std(flux)
                mask = np.abs(flux - mean) < sigma * std
            else:  # 'mad'
                # MAD-based sigma clipping (more robust)
                median = np.median(flux)
                mad = np.median(np.abs(flux - median))
                mask = np.abs(flux - median) < sigma * mad
            
            n_clipped = np.sum(~mask)
            if n_clipped == 0:
                break
            
            time = time[mask]
            flux = flux[mask]
            if flux_err is not None:
                flux_err = flux_err[mask]
        
        lc.time = time
        lc.flux = flux
        if flux_err is not None:
            lc.flux_err = flux_err
        
        return lc
    
    def _fold(self, lc: LightCurve) -> LightCurve:
        """Fold light curve by period."""
        period = self.config['fold'].get('period')
        epoch = self.config['fold'].get('epoch', 0.0)
        
        # Calculate phase
        phase = ((lc.time - epoch) / period) % 1.0
        
        # Sort by phase
        sort_idx = np.argsort(phase)
        lc.time = phase[sort_idx]
        lc.flux = lc.flux[sort_idx]
        if len(lc.flux_err) > 0:
            lc.flux_err = lc.flux_err[sort_idx]
        
        # Store folding parameters in metadata
        lc.metadata['folded'] = True
        lc.metadata['fold_period'] = period
        lc.metadata['fold_epoch'] = epoch
        
        return lc
    
    def _bin(self, lc: LightCurve) -> LightCurve:
        """Bin light curve data."""
        bin_size = self.config['bin'].get('bin_size', 0.01)
        method = self.config['bin'].get('method', 'weighted')
        
        time_min = np.min(lc.time)
        time_max = np.max(lc.time)
        
        # Create bin edges
        bins = np.arange(time_min, time_max + bin_size, bin_size)
        
        # Bin indices
        bin_indices = np.digitize(lc.time, bins) - 1
        
        # Binned arrays
        time_binned = []
        flux_binned = []
        flux_err_binned = []
        
        for i in range(len(bins) - 1):
            mask = bin_indices == i
            if np.sum(mask) == 0:
                continue
            
            time_bin = lc.time[mask]
            flux_bin = lc.flux[mask]
            
            # Bin center
            time_binned.append(np.mean(time_bin))
            
            # Bin flux based on method
            if method == 'weighted' and len(lc.flux_err) > 0:
                # Weighted average
                err_bin = lc.flux_err[mask]
                weights = 1.0 / (err_bin ** 2)
                flux_binned.append(np.sum(flux_bin * weights) / np.sum(weights))
                flux_err_binned.append(1.0 / np.sqrt(np.sum(weights)))
            elif method == 'median':
                flux_binned.append(np.median(flux_bin))
                if len(lc.flux_err) > 0:
                    flux_err_binned.append(np.median(lc.flux_err[mask]))
            else:  # 'mean'
                flux_binned.append(np.mean(flux_bin))
                if len(lc.flux_err) > 0:
                    flux_err_binned.append(np.sqrt(np.sum(lc.flux_err[mask]**2)) / np.sum(mask))
        
        lc.time = np.array(time_binned)
        lc.flux = np.array(flux_binned)
        if len(flux_err_binned) > 0:
            lc.flux_err = np.array(flux_err_binned)
        
        return lc
    
    def _normalize(self, lc: LightCurve) -> LightCurve:
        """Normalize flux values."""
        method = self.config['normalize'].get('method', 'zscore')
        
        if method == 'zscore':
            # Zero mean, unit variance
            mean = np.mean(lc.flux)
            std = np.std(lc.flux)
            if std > 0:
                lc.flux = (lc.flux - mean) / std
                if len(lc.flux_err) > 0:
                    lc.flux_err = lc.flux_err / std
        
        elif method == 'minmax':
            # Scale to [0, 1]
            min_val = np.min(lc.flux)
            max_val = np.max(lc.flux)
            if max_val > min_val:
                lc.flux = (lc.flux - min_val) / (max_val - min_val)
                if len(lc.flux_err) > 0:
                    lc.flux_err = lc.flux_err / (max_val - min_val)
        
        elif method == 'robust':
            # Robust scaling using median and IQR
            median = np.median(lc.flux)
            q75, q25 = np.percentile(lc.flux, [75, 25])
            iqr = q75 - q25
            if iqr > 0:
                lc.flux = (lc.flux - median) / iqr
                if len(lc.flux_err) > 0:
                    lc.flux_err = lc.flux_err / iqr
        
        elif method == 'median':
            # Divide by median
            median = np.median(lc.flux)
            if median != 0:
                lc.flux = lc.flux / median
                if len(lc.flux_err) > 0:
                    lc.flux_err = lc.flux_err / median
        
        return lc


# Convenience function
def preprocess_lightcurve(lc: LightCurve, 
                         config: Optional[Dict] = None,
                         **kwargs) -> LightCurve:
    """
    Preprocess a light curve with optional configuration.
    
    Args:
        lc: Input LightCurve object
        config: Preprocessing configuration dictionary
        **kwargs: Override specific config parameters
        
    Returns:
        Preprocessed LightCurve object
        
    Example:
        >>> lc_clean = preprocess_lightcurve(lc, detrend={'enabled': True, 'method': 'polynomial'})
    """
    if config is None:
        config = {}
    
    # Allow kwargs to override config
    if kwargs:
        for key, value in kwargs.items():
            if isinstance(value, dict):
                if key not in config:
                    config[key] = {}
                config[key].update(value)
            else:
                # Simple key-value, create nested structure if needed
                if '.' in key:
                    parts = key.split('.')
                    if parts[0] not in config:
                        config[parts[0]] = {}
                    config[parts[0]][parts[1]] = value
                else:
                    config[key] = value
    
    pipeline = PreprocessingPipeline(config)
    return pipeline.process(lc, inplace=False)
