"""
Feature Extraction Module for Light Curves
Extracts statistical and shape-based features from flux time series
"""

import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from typing import Dict, List, Optional, Tuple
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')


class FeatureExtractor:
    """Extract features from light curve time series"""
    
    def __init__(self):
        """Initialize feature extractor"""
        self.feature_names = []
        
    def extract_statistical_features(self, flux: np.ndarray) -> Dict[str, float]:
        """
        Extract statistical features from flux array
        
        Args:
            flux: Array of flux measurements
            
        Returns:
            Dictionary of statistical features
        """
        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(flux)
        features['std'] = np.std(flux)
        features['median'] = np.median(flux)
        features['min'] = np.min(flux)
        features['max'] = np.max(flux)
        features['range'] = features['max'] - features['min']
        
        # Percentiles
        features['q1'] = np.percentile(flux, 25)
        features['q3'] = np.percentile(flux, 75)
        features['iqr'] = features['q3'] - features['q1']
        features['percentile_10'] = np.percentile(flux, 10)
        features['percentile_90'] = np.percentile(flux, 90)
        
        # Higher moments
        features['skewness'] = stats.skew(flux)
        features['kurtosis'] = stats.kurtosis(flux)
        
        # Variability metrics
        features['cv'] = features['std'] / (np.abs(features['mean']) + 1e-10)  # Coefficient of variation
        features['mad'] = np.median(np.abs(flux - features['median']))  # Median absolute deviation
        
        return features
    
    def extract_shape_features(self, flux: np.ndarray) -> Dict[str, float]:
        """
        Extract shape-based features from flux array
        
        Args:
            flux: Array of flux measurements
            
        Returns:
            Dictionary of shape features
        """
        features = {}
        
        # Detect peaks and troughs
        peaks, peak_properties = signal.find_peaks(flux, prominence=0.1)
        troughs, trough_properties = signal.find_peaks(-flux, prominence=0.1)
        
        features['num_peaks'] = len(peaks)
        features['num_troughs'] = len(troughs)
        
        if len(peaks) > 0:
            features['mean_peak_height'] = np.mean(flux[peaks])
            features['max_peak_height'] = np.max(flux[peaks])
            if 'prominences' in peak_properties:
                features['mean_peak_prominence'] = np.mean(peak_properties['prominences'])
        else:
            features['mean_peak_height'] = 0
            features['max_peak_height'] = 0
            features['mean_peak_prominence'] = 0
        
        if len(troughs) > 0:
            features['mean_trough_depth'] = np.mean(flux[troughs])
            features['min_trough_depth'] = np.min(flux[troughs])
        else:
            features['mean_trough_depth'] = 0
            features['min_trough_depth'] = 0
        
        # Transit-like features
        features['transit_depth'] = features['min_trough_depth'] if len(troughs) > 0 else 0
        
        # Smoothness
        features['roughness'] = np.mean(np.abs(np.diff(flux)))
        features['smoothness'] = 1 / (1 + features['roughness'])
        
        return features
    
    def extract_frequency_features(self, flux: np.ndarray) -> Dict[str, float]:
        """
        Extract frequency domain features using FFT
        
        Args:
            flux: Array of flux measurements
            
        Returns:
            Dictionary of frequency features
        """
        features = {}
        
        # Compute FFT
        n = len(flux)
        fft_vals = fft(flux)
        fft_power = np.abs(fft_vals[:n//2]) ** 2
        fft_freq = fftfreq(n, 1.0)[:n//2]
        
        # Dominant frequency
        if len(fft_power) > 1:
            dominant_freq_idx = np.argmax(fft_power[1:]) + 1  # Skip DC component
            features['dominant_frequency'] = fft_freq[dominant_freq_idx]
            features['dominant_frequency_power'] = fft_power[dominant_freq_idx]
        else:
            features['dominant_frequency'] = 0
            features['dominant_frequency_power'] = 0
        
        # Spectral features
        features['spectral_centroid'] = np.sum(fft_freq * fft_power) / (np.sum(fft_power) + 1e-10)
        features['spectral_spread'] = np.sqrt(np.sum(((fft_freq - features['spectral_centroid']) ** 2) * fft_power) / (np.sum(fft_power) + 1e-10))
        features['spectral_energy'] = np.sum(fft_power)
        
        # Band power ratios
        total_power = np.sum(fft_power)
        if total_power > 0:
            low_freq_power = np.sum(fft_power[fft_freq < 0.1])
            mid_freq_power = np.sum(fft_power[(fft_freq >= 0.1) & (fft_freq < 0.3)])
            high_freq_power = np.sum(fft_power[fft_freq >= 0.3])
            
            features['low_freq_ratio'] = low_freq_power / total_power
            features['mid_freq_ratio'] = mid_freq_power / total_power
            features['high_freq_ratio'] = high_freq_power / total_power
        else:
            features['low_freq_ratio'] = 0
            features['mid_freq_ratio'] = 0
            features['high_freq_ratio'] = 0
        
        return features
    
    def extract_time_series_features(self, flux: np.ndarray) -> Dict[str, float]:
        """
        Extract time series specific features
        
        Args:
            flux: Array of flux measurements
            
        Returns:
            Dictionary of time series features
        """
        features = {}
        
        # Autocorrelation
        if len(flux) > 1:
            autocorr = np.correlate(flux - np.mean(flux), flux - np.mean(flux), mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / (autocorr[0] + 1e-10)
            
            features['autocorr_lag1'] = autocorr[1] if len(autocorr) > 1 else 0
            features['autocorr_lag5'] = autocorr[5] if len(autocorr) > 5 else 0
            features['autocorr_lag10'] = autocorr[10] if len(autocorr) > 10 else 0
        else:
            features['autocorr_lag1'] = 0
            features['autocorr_lag5'] = 0
            features['autocorr_lag10'] = 0
        
        # Trend features
        time_indices = np.arange(len(flux))
        slope, intercept, r_value, _, _ = stats.linregress(time_indices, flux)
        features['trend_slope'] = slope
        features['trend_strength'] = r_value ** 2
        
        # Detrended variance
        detrended = flux - (slope * time_indices + intercept)
        features['detrended_variance'] = np.var(detrended)
        
        # Zero crossing rate
        zero_mean = flux - np.mean(flux)
        features['zero_crossing_rate'] = np.sum(np.diff(np.sign(zero_mean)) != 0) / len(flux)
        
        # Entropy
        hist, _ = np.histogram(flux, bins=20)
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]
        features['entropy'] = -np.sum(hist * np.log(hist + 1e-10))
        
        return features
    
    def extract_transit_specific_features(self, flux: np.ndarray) -> Dict[str, float]:
        """
        Extract features specific to exoplanet transits
        
        Args:
            flux: Array of flux measurements
            
        Returns:
            Dictionary of transit-specific features
        """
        features = {}
        
        # Find potential transit events (significant dips)
        median_flux = np.median(flux)
        std_flux = np.std(flux)
        threshold = median_flux - 2 * std_flux
        
        # Identify transit points
        transit_mask = flux < threshold
        features['transit_points_ratio'] = np.sum(transit_mask) / len(flux)
        
        # Transit depth estimation
        if np.sum(transit_mask) > 0:
            features['estimated_transit_depth'] = median_flux - np.mean(flux[transit_mask])
            features['max_transit_depth'] = median_flux - np.min(flux)
        else:
            features['estimated_transit_depth'] = 0
            features['max_transit_depth'] = 0
        
        # Find consecutive transit segments
        transit_segments = []
        in_transit = False
        start_idx = 0
        
        for i, is_transit in enumerate(transit_mask):
            if is_transit and not in_transit:
                start_idx = i
                in_transit = True
            elif not is_transit and in_transit:
                transit_segments.append((start_idx, i))
                in_transit = False
        
        if in_transit:
            transit_segments.append((start_idx, len(flux)))
        
        features['num_transit_events'] = len(transit_segments)
        
        if len(transit_segments) > 0:
            transit_durations = [end - start for start, end in transit_segments]
            features['mean_transit_duration'] = np.mean(transit_durations)
            features['max_transit_duration'] = np.max(transit_durations)
            
            # Period estimation (if multiple transits)
            if len(transit_segments) > 1:
                transit_centers = [(start + end) / 2 for start, end in transit_segments]
                periods = np.diff(transit_centers)
                features['estimated_period'] = np.median(periods) if len(periods) > 0 else 0
                features['period_variance'] = np.var(periods) if len(periods) > 0 else 0
            else:
                features['estimated_period'] = 0
                features['period_variance'] = 0
        else:
            features['mean_transit_duration'] = 0
            features['max_transit_duration'] = 0
            features['estimated_period'] = 0
            features['period_variance'] = 0
        
        # Signal-to-noise ratio for transit
        if features['estimated_transit_depth'] > 0:
            features['transit_snr'] = features['estimated_transit_depth'] / (std_flux + 1e-10)
        else:
            features['transit_snr'] = 0
        
        return features
    
    def extract_all_features(self, flux: np.ndarray) -> Dict[str, float]:
        """
        Extract all features from flux array
        
        Args:
            flux: Array of flux measurements
            
        Returns:
            Dictionary of all features
        """
        all_features = {}
        
        # Extract different feature groups
        all_features.update(self.extract_statistical_features(flux))
        all_features.update(self.extract_shape_features(flux))
        all_features.update(self.extract_frequency_features(flux))
        all_features.update(self.extract_time_series_features(flux))
        all_features.update(self.extract_transit_specific_features(flux))
        
        return all_features
    
    def extract_features_from_dataframe(self, df: pd.DataFrame, 
                                       flux_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Extract features from all light curves in a DataFrame
        
        Args:
            df: DataFrame with flux measurements
            flux_columns: List of flux column names
            
        Returns:
            DataFrame with extracted features
        """
        if flux_columns is None:
            flux_columns = [col for col in df.columns if col.startswith('FLUX')]
        
        feature_list = []
        
        for idx in tqdm(range(len(df)), desc="Extracting features"):
            flux = df.iloc[idx][flux_columns].values
            features = self.extract_all_features(flux)
            
            # Add label if present
            if 'LABEL' in df.columns:
                features['LABEL'] = df.iloc[idx]['LABEL']
            
            feature_list.append(features)
        
        feature_df = pd.DataFrame(feature_list)
        
        # Store feature names
        self.feature_names = [col for col in feature_df.columns if col != 'LABEL']
        
        print(f"Extracted {len(self.feature_names)} features from {len(df)} light curves")
        
        return feature_df
    
    def save_features(self, feature_df: pd.DataFrame, filepath: str):
        """Save extracted features to CSV"""
        feature_df.to_csv(filepath, index=False)
        print(f"Saved features to {filepath}")


def extract_and_save_features(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Extract features from processed data and save
    
    Args:
        input_path: Path to processed data CSV
        output_path: Path to save features CSV
        
    Returns:
        DataFrame with extracted features
    """
    # Load data
    df = pd.read_csv(input_path)
    
    # Initialize extractor
    extractor = FeatureExtractor()
    
    # Extract features
    feature_df = extractor.extract_features_from_dataframe(df)
    
    # Save features
    extractor.save_features(feature_df, output_path)
    
    return feature_df


if __name__ == "__main__":
    # Example usage
    processed_file = "../data/processed/train_processed.csv"
    features_file = "../data/features/train_features.csv"
    
    features = extract_and_save_features(processed_file, features_file)
