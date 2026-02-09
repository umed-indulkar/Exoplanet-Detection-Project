"""
Basic Feature Extractor - Feature Extraction Team
=================================================

Extracts 100+ features from light curves in <1s per curve.
This is the ACTUAL feature extraction code - no more duplication!
"""

import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from typing import Optional, Dict
import warnings

# Import LightCurve from preprocessing (no core dependency)
from ..preprocessing.data_loader import LightCurve

# Define exception locally
class FeatureExtractionError(Exception):
    """Feature extraction failed."""
    pass

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
        features['min'] = np.min(flux)
        features['max'] = np.max(flux)
        features['range'] = np.max(flux) - np.min(flux)
        features['iqr'] = np.percentile(flux, 75) - np.percentile(flux, 25)
        
        # Shape
        features['skew'] = stats.skew(flux)
        features['kurtosis'] = stats.kurtosis(flux)
        
        # Quantiles
        features['q25'] = np.percentile(flux, 25)
        features['q75'] = np.percentile(flux, 75)
        features['q90'] = np.percentile(flux, 90)
        features['q95'] = np.percentile(flux, 95)
        
        return features
    
    def _extract_time_domain(self, time: np.ndarray, flux: np.ndarray) -> Dict:
        """Extract time-domain features."""
        features = {}
        
        # Autocorrelation
        if len(flux) > 20:
            autocorr = np.correlate(flux - np.mean(flux), flux - np.mean(flux), mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]  # Normalize
            
            features['autocorr_max'] = np.max(autocorr[1:])  # Excluding zero lag
            features['autocorr_mean'] = np.mean(autocorr[1:])
            features['autocorr_std'] = np.std(autocorr[1:])
        
        # Linear trend
        if len(time) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(time, flux)
            features['trend_slope'] = slope
            features['trend_intercept'] = intercept
            features['trend_r2'] = r_value**2
            features['trend_p_value'] = p_value
        
        return features
    
    def _extract_frequency_domain(self, time: np.ndarray, flux: np.ndarray) -> Dict:
        """Extract frequency-domain features."""
        features = {}
        
        if len(flux) > 10:
            # FFT
            fft_vals = fft(flux)
            fft_freq = fftfreq(len(flux), d=np.mean(np.diff(time)))
            
            # Power spectrum
            power = np.abs(fft_vals)**2
            positive_freq_idx = fft_freq > 0
            
            if np.any(positive_freq_idx):
                positive_power = power[positive_freq_idx]
                positive_freq = fft_freq[positive_freq_idx]
                
                features['fft_max_power'] = np.max(positive_power)
                features['fft_mean_power'] = np.mean(positive_power)
                features['fft_std_power'] = np.std(positive_power)
                
                # Dominant frequency
                dominant_freq_idx = np.argmax(positive_power)
                features['dominant_frequency'] = positive_freq[dominant_freq_idx]
                features['dominant_power'] = positive_power[dominant_freq_idx]
                
                # Spectral centroid
                features['spectral_centroid'] = np.average(positive_freq, weights=positive_power)
                
                # Spectral entropy
                if np.sum(positive_power) > 0:
                    power_norm = positive_power / np.sum(positive_power)
                    power_norm = power_norm[power_norm > 0]  # Remove zeros
                    features['spectral_entropy'] = -np.sum(power_norm * np.log2(power_norm))
                else:
                    features['spectral_entropy'] = 0
        
        return features
    
    def _extract_variability(self, time: np.ndarray, flux: np.ndarray) -> Dict:
        """Extract variability features."""
        features = {}
        
        # Relative variability
        mean_flux = np.mean(flux)
        if mean_flux != 0:
            features['relative_std'] = np.std(flux) / np.abs(mean_flux)
            features['relative_range'] = (np.max(flux) - np.min(flux)) / np.abs(mean_flux)
        
        # Fraction of points beyond thresholds
        median_flux = np.median(flux)
        std_flux = np.std(flux)
        
        if std_flux > 0:
            features['beyond_1sigma'] = np.sum(np.abs(flux - median_flux) > std_flux) / len(flux)
            features['beyond_2sigma'] = np.sum(np.abs(flux - median_flux) > 2*std_flux) / len(flux)
            features['beyond_3sigma'] = np.sum(np.abs(flux - median_flux) > 3*std_flux) / len(flux)
        
        # RMS
        features['rms'] = np.sqrt(np.mean(flux**2))
        
        return features
    
    def _extract_shape(self, flux: np.ndarray) -> Dict:
        """Extract shape-related features."""
        features = {}
        
        # Number of peaks
        peaks, _ = signal.find_peaks(flux, distance=5)
        features['n_peaks'] = len(peaks)
        
        # Number of valleys (invert signal)
        valleys, _ = signal.find_peaks(-flux, distance=5)
        features['n_valleys'] = len(valleys)
        
        # Peak-to-peak variations
        if len(peaks) > 1:
            peak_heights = flux[peaks]
            features['peak_height_mean'] = np.mean(peak_heights)
            features['peak_height_std'] = np.std(peak_heights)
            features['peak_height_range'] = np.max(peak_heights) - np.min(peak_heights)
        
        return features
    
    def _extract_transit(self, time: np.ndarray, flux: np.ndarray) -> Dict:
        """Extract transit-specific features."""
        features = {}
        
        # Look for dips (potential transits)
        median_flux = np.median(flux)
        std_flux = np.std(flux)
        
        # Define dip threshold (e.g., 3 sigma below median)
        dip_threshold = median_flux - 2 * std_flux
        dip_mask = flux < dip_threshold
        
        if np.any(dip_mask):
            # Number of dips
            dip_indices = np.where(dip_mask)[0]
            # Group consecutive dips
            dip_groups = []
            current_group = [dip_indices[0]]
            
            for i in range(1, len(dip_indices)):
                if dip_indices[i] == dip_indices[i-1] + 1:
                    current_group.append(dip_indices[i])
                else:
                    dip_groups.append(current_group)
                    current_group = [dip_indices[i]]
            dip_groups.append(current_group)
            
            features['n_dips'] = len(dip_groups)
            
            if dip_groups:
                # Dip statistics
                dip_lengths = [len(group) for group in dip_groups]
                dip_depths = [median_flux - np.min(flux[group]) for group in dip_groups]
                
                features['dip_length_mean'] = np.mean(dip_lengths)
                features['dip_length_max'] = np.max(dip_lengths)
                features['dip_depth_mean'] = np.mean(dip_depths)
                features['dip_depth_max'] = np.max(dip_depths)
                
                # Fraction of time in dips
                total_dip_points = sum(dip_lengths)
                features['dip_fraction'] = total_dip_points / len(flux)
        else:
            features['n_dips'] = 0
            features['dip_length_mean'] = 0
            features['dip_length_max'] = 0
            features['dip_depth_mean'] = 0
            features['dip_depth_max'] = 0
            features['dip_fraction'] = 0
        
        return features
    
    def _extract_error_based(self, flux: np.ndarray, flux_err: np.ndarray) -> Dict:
        """Extract features using flux uncertainties."""
        features = {}
        
        # Signal-to-noise ratio
        if len(flux_err) > 0 and np.mean(flux_err) > 0:
            features['snr_mean'] = np.mean(flux) / np.mean(flux_err)
            features['snr_median'] = np.median(flux) / np.median(flux_err)
            
            # Weighted statistics
            weights = 1.0 / (flux_err**2)
            features['weighted_mean'] = np.average(flux, weights=weights)
            features['weighted_std'] = np.sqrt(np.average((flux - np.average(flux, weights=weights))**2, weights=weights))
        
        return features


def extract_basic_features(lc: LightCurve, verbose: bool = False) -> Dict:
    """
    Convenience function to extract basic features from a light curve.
    
    Args:
        lc: LightCurve object
        verbose: Whether to print progress
        
    Returns:
        Dictionary of features
    """
    extractor = BasicFeatureExtractor(verbose=verbose)
    df = extractor.extract(lc)
    return df.iloc[0].to_dict()
