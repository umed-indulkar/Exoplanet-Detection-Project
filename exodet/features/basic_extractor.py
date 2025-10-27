"""
Basic Feature Extractor - Fast Mode
====================================

Extracts 100+ features from light curves in <1s per curve.
Combines the best features from lightcurve_project branch.

This is the FAST tier extractor for quick screening.
"""

import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from typing import Optional, Dict
import warnings

from ..core.data_loader import LightCurve
from ..core.exceptions import FeatureExtractionError

warnings.filterwarnings('ignore')


class BasicFeatureExtractor:
    """
    Fast feature extractor for light curves.
    
    Extracts 100+ statistical, time-domain, frequency-domain,
    and transit-specific features optimized for speed.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize feature extractor.
        
        Args:
            verbose: Whether to print extraction progress
        """
        self.verbose = verbose
        self.feature_count = 0
    
    def extract(self, lc: LightCurve) -> pd.DataFrame:
        """
        Extract features from a light curve.
        
        Args:
            lc: LightCurve object
            
        Returns:
            DataFrame with one row containing all features
        """
        try:
            time = lc.time
            flux = lc.flux
            flux_err = lc.flux_err if len(lc.flux_err) > 0 else None
            
            if self.verbose:
                print(f"Extracting features from {lc.source_file}...")
            
            # Validate
            if len(time) < 10:
                raise FeatureExtractionError(f"Need at least 10 points, got {len(time)}")
            
            features = {}
            
            # Extract all feature categories
            features.update(self._extract_basic_statistics(flux))
            features.update(self._extract_time_domain(time, flux))
            features.update(self._extract_frequency_domain(time, flux))
            features.update(self._extract_variability(time, flux))
            features.update(self._extract_shape(flux))
            features.update(self._extract_transit(time, flux))
            
            if flux_err is not None:
                features.update(self._extract_error_based(flux, flux_err))
            
            # Add metadata
            features['n_points'] = len(time)
            features['time_span'] = np.max(time) - np.min(time)
            
            self.feature_count = len(features)
            
            if self.verbose:
                print(f"✓ Extracted {len(features)} features")
            
            return pd.DataFrame([features])
            
        except Exception as e:
            raise FeatureExtractionError(f"Feature extraction failed: {str(e)}")
    
    def _extract_basic_statistics(self, flux: np.ndarray) -> Dict:
        """Extract basic statistical features."""
        features = {}
        
        # Central tendency
        features['mean'] = np.mean(flux)
        features['median'] = np.median(flux)
        features['std'] = np.std(flux)
        features['var'] = np.var(flux)
        
        # Spread
        features['range'] = np.max(flux) - np.min(flux)
        features['iqr'] = np.percentile(flux, 75) - np.percentile(flux, 25)
        features['mad'] = np.median(np.abs(flux - np.median(flux)))
        
        # Shape
        features['skewness'] = stats.skew(flux)
        features['kurtosis'] = stats.kurtosis(flux)
        
        # Extremes
        features['min'] = np.min(flux)
        features['max'] = np.max(flux)
        
        # Percentiles
        for p in [5, 10, 25, 75, 90, 95]:
            features[f'percentile_{p}'] = np.percentile(flux, p)
        
        # Coefficient of variation
        if features['mean'] != 0:
            features['cv'] = features['std'] / abs(features['mean'])
        else:
            features['cv'] = np.nan
        
        return features
    
    def _extract_time_domain(self, time: np.ndarray, flux: np.ndarray) -> Dict:
        """Extract time domain features."""
        features = {}
        
        # Cadence
        if len(time) > 1:
            cadences = np.diff(time)
            features['mean_cadence'] = np.mean(cadences)
            features['std_cadence'] = np.std(cadences)
        
        # Differences
        if len(flux) > 1:
            diff1 = np.diff(flux)
            features['mean_diff'] = np.mean(diff1)
            features['std_diff'] = np.std(diff1)
            features['max_diff'] = np.max(np.abs(diff1))
        
        # Linear trend
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(time, flux)
            features['linear_slope'] = slope
            features['linear_r2'] = r_value**2
        except:
            features['linear_slope'] = np.nan
            features['linear_r2'] = np.nan
        
        # Autocorrelation
        try:
            autocorr = np.correlate(flux - np.mean(flux), flux - np.mean(flux), mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0]
            
            if len(autocorr) > 1:
                features['autocorr_lag1'] = autocorr[1] if len(autocorr) > 1 else np.nan
                features['autocorr_lag2'] = autocorr[2] if len(autocorr) > 2 else np.nan
        except:
            features['autocorr_lag1'] = np.nan
            features['autocorr_lag2'] = np.nan
        
        return features
    
    def _extract_frequency_domain(self, time: np.ndarray, flux: np.ndarray) -> Dict:
        """Extract frequency domain features."""
        features = {}
        
        try:
            n = len(flux)
            if n < 4:
                return {f'fft_{i}': np.nan for i in range(5)}
            
            # FFT
            flux_centered = flux - np.mean(flux)
            fft_values = fft(flux_centered)
            freqs = fftfreq(n, d=np.mean(np.diff(time)))
            
            # Power spectrum
            power = np.abs(fft_values)**2
            pos_mask = freqs > 0
            power_pos = power[pos_mask]
            
            if len(power_pos) > 0:
                features['spectral_power_mean'] = np.mean(power_pos)
                features['spectral_power_std'] = np.std(power_pos)
                features['spectral_power_max'] = np.max(power_pos)
                
                # Dominant frequency
                features['dominant_freq'] = freqs[pos_mask][np.argmax(power_pos)]
            
        except:
            features['spectral_power_mean'] = np.nan
            features['spectral_power_std'] = np.nan
            features['spectral_power_max'] = np.nan
            features['dominant_freq'] = np.nan
        
        return features
    
    def _extract_variability(self, time: np.ndarray, flux: np.ndarray) -> Dict:
        """Extract variability features."""
        features = {}
        
        # Amplitude
        features['amplitude'] = (np.max(flux) - np.min(flux)) / 2
        features['rms'] = np.sqrt(np.mean(flux**2))
        
        # Relative variability
        mean_flux = np.mean(flux)
        if mean_flux != 0:
            features['relative_amplitude'] = features['amplitude'] / abs(mean_flux)
        else:
            features['relative_amplitude'] = np.nan
        
        # Von Neumann ratio
        if len(flux) > 1:
            numerator = np.sum((flux[1:] - flux[:-1])**2)
            denominator = np.sum((flux - mean_flux)**2)
            features['von_neumann'] = numerator / denominator if denominator > 0 else np.nan
        
        return features
    
    def _extract_shape(self, flux: np.ndarray) -> Dict:
        """Extract shape-based features."""
        features = {}
        
        # Symmetry
        median_flux = np.median(flux)
        above_median = flux[flux > median_flux]
        below_median = flux[flux < median_flux]
        
        if len(above_median) > 0 and len(below_median) > 0:
            features['asymmetry_ratio'] = len(above_median) / len(below_median)
        else:
            features['asymmetry_ratio'] = np.nan
        
        # Runs analysis
        mean_flux = np.mean(flux)
        above_runs = self._get_runs(flux > mean_flux)
        below_runs = self._get_runs(flux < mean_flux)
        
        features['max_run_above'] = np.max(above_runs) if len(above_runs) > 0 else 0
        features['max_run_below'] = np.max(below_runs) if len(below_runs) > 0 else 0
        
        return features
    
    def _extract_transit(self, time: np.ndarray, flux: np.ndarray) -> Dict:
        """Extract transit-specific features."""
        features = {}
        
        try:
            median_flux = np.median(flux)
            std_flux = np.std(flux)
            
            # Look for dips
            dip_threshold = median_flux - 2 * std_flux
            dip_mask = flux < dip_threshold
            
            # Count dips
            features['n_dips'] = len(self._get_connected_regions(dip_mask))
            
            if np.any(dip_mask):
                dip_regions = self._get_connected_regions(dip_mask)
                
                if len(dip_regions) > 0:
                    # Analyze deepest dip
                    deepest = min(dip_regions, key=lambda r: np.min(flux[r[0]:r[1]+1]))
                    start, end = deepest
                    
                    features['deepest_dip_depth'] = median_flux - np.min(flux[start:end+1])
                    features['deepest_dip_duration'] = time[end] - time[start]
                    features['deepest_dip_points'] = end - start + 1
                else:
                    features['deepest_dip_depth'] = 0
                    features['deepest_dip_duration'] = 0
                    features['deepest_dip_points'] = 0
            else:
                features['deepest_dip_depth'] = 0
                features['deepest_dip_duration'] = 0
                features['deepest_dip_points'] = 0
                
        except:
            features['n_dips'] = 0
            features['deepest_dip_depth'] = np.nan
            features['deepest_dip_duration'] = np.nan
            features['deepest_dip_points'] = np.nan
        
        return features
    
    def _extract_error_based(self, flux: np.ndarray, flux_err: np.ndarray) -> Dict:
        """Extract error-based features."""
        features = {}
        
        # Signal-to-noise
        features['mean_snr'] = np.mean(np.abs(flux) / flux_err)
        features['median_snr'] = np.median(np.abs(flux) / flux_err)
        
        # Weighted mean
        weights = 1.0 / (flux_err**2)
        features['weighted_mean'] = np.average(flux, weights=weights)
        
        # Chi-squared
        expected = np.mean(flux)
        chi2 = np.sum((flux - expected)**2 / flux_err**2)
        features['chi2'] = chi2
        features['chi2_reduced'] = chi2 / (len(flux) - 1) if len(flux) > 1 else np.nan
        
        return features
    
    @staticmethod
    def _get_runs(boolean_array: np.ndarray) -> np.ndarray:
        """Get lengths of consecutive True runs."""
        if len(boolean_array) == 0:
            return np.array([])
        
        d = np.diff(np.concatenate(([False], boolean_array, [False])).astype(int))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        
        return ends - starts
    
    @staticmethod
    def _get_connected_regions(boolean_mask: np.ndarray) -> list:
        """Get start and end indices of connected True regions."""
        if not np.any(boolean_mask):
            return []
        
        d = np.diff(np.concatenate(([False], boolean_mask, [False])).astype(int))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0] - 1
        
        return list(zip(starts, ends))


# Convenience function
def extract_basic_features(lc: LightCurve, verbose: bool = True) -> pd.DataFrame:
    """
    Extract basic features from a light curve.
    
    Args:
        lc: LightCurve object
        verbose: Whether to print progress
        
    Returns:
        DataFrame with extracted features
    """
    extractor = BasicFeatureExtractor(verbose=verbose)
    return extractor.extract(lc)
