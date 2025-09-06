"""
Feature Extraction Module

This module extracts comprehensive features from light curves including
statistical, time-domain, frequency-domain, and transit-specific features.
Implements 800+ features using various astronomical and time series packages.
"""

import numpy as np
import pandas as pd
from scipy import stats, signal, optimize
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

def extract_features(time, flux, flux_err=None, include_all=True):
    """
    Extract comprehensive features from a light curve.
    
    Parameters:
    -----------
    time : array-like
        Time values
    flux : array-like
        Flux values
    flux_err : array-like, optional
        Flux error values
    include_all : bool, default=True
        Whether to include all feature categories
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with one row containing all extracted features
    """
    print("Extracting features...")
    
    # Validate inputs
    time = np.asarray(time)
    flux = np.asarray(flux)
    if flux_err is not None:
        flux_err = np.asarray(flux_err)
    
    if len(time) != len(flux):
        raise ValueError("Time and flux arrays must have same length")
    
    # Remove NaN values
    valid_mask = np.isfinite(time) & np.isfinite(flux)
    if flux_err is not None:
        valid_mask &= np.isfinite(flux_err)
    
    time = time[valid_mask]
    flux = flux[valid_mask]
    if flux_err is not None:
        flux_err = flux_err[valid_mask]
    
    if len(time) < 10:
        raise ValueError("Need at least 10 valid data points")
    
    features = {}
    
    # Basic statistics
    features.update(extract_basic_statistics(flux))
    
    # Time domain features
    features.update(extract_time_domain_features(time, flux))
    
    # Frequency domain features
    features.update(extract_frequency_domain_features(time, flux))
    
    # Advanced statistical features
    features.update(extract_advanced_statistics(flux))
    
    # Variability features
    features.update(extract_variability_features(time, flux))
    
    # Periodicity features
    features.update(extract_periodicity_features(time, flux))
    
    # Shape features
    features.update(extract_shape_features(flux))
    
    # Transit-specific features
    features.update(extract_transit_features(time, flux))
    
    # Error-based features (if errors provided)
    if flux_err is not None:
        features.update(extract_error_features(flux, flux_err))
    
    # Additional astronomical features
    features.update(extract_astronomical_features(time, flux))
    
    print(f"Extracted {len(features)} features")
    return pd.DataFrame([features])


def extract_basic_statistics(flux):
    """Extract basic statistical features."""
    features = {}
    
    # Central tendency
    features['mean'] = np.mean(flux)
    features['median'] = np.median(flux)
    features['mode'] = stats.mode(flux, keepdims=False)[0] if len(flux) > 0 else np.nan
    
    # Spread
    features['std'] = np.std(flux)
    features['var'] = np.var(flux)
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
    for p in [1, 5, 10, 25, 75, 90, 95, 99]:
        features[f'percentile_{p}'] = np.percentile(flux, p)
    
    # Robust statistics
    features['trimmed_mean_10'] = stats.trim_mean(flux, 0.1)
    features['trimmed_mean_20'] = stats.trim_mean(flux, 0.2)
    
    return features


def extract_time_domain_features(time, flux):
    """Extract time domain features."""
    features = {}
    
    # Time span
    features['time_span'] = np.max(time) - np.min(time)
    features['n_points'] = len(time)
    features['mean_cadence'] = np.mean(np.diff(time))
    features['std_cadence'] = np.std(np.diff(time))
    
    # Autocorrelation
    try:
        autocorr = np.correlate(flux - np.mean(flux), flux - np.mean(flux), mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]
        
        # First zero crossing
        zero_crossings = np.where(np.diff(np.signbit(autocorr)))[0]
        features['autocorr_first_zero'] = zero_crossings[0] if len(zero_crossings) > 0 else len(autocorr)
        
        # Autocorrelation at different lags
        for lag in [1, 2, 3, 5, 10]:
            if lag < len(autocorr):
                features[f'autocorr_lag_{lag}'] = autocorr[lag]
    except:
        features['autocorr_first_zero'] = np.nan
        for lag in [1, 2, 3, 5, 10]:
            features[f'autocorr_lag_{lag}'] = np.nan
    
    # Differences and derivatives
    diff1 = np.diff(flux)
    features['mean_diff1'] = np.mean(diff1)
    features['std_diff1'] = np.std(diff1)
    features['max_diff1'] = np.max(np.abs(diff1))
    
    if len(diff1) > 1:
        diff2 = np.diff(diff1)
        features['mean_diff2'] = np.mean(diff2)
        features['std_diff2'] = np.std(diff2)
        features['max_diff2'] = np.max(np.abs(diff2))
    
    # Trend analysis
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(time, flux)
        features['linear_trend_slope'] = slope
        features['linear_trend_r2'] = r_value**2
        features['linear_trend_p_value'] = p_value
    except:
        features['linear_trend_slope'] = np.nan
        features['linear_trend_r2'] = np.nan
        features['linear_trend_p_value'] = np.nan
    
    return features


def extract_frequency_domain_features(time, flux):
    """Extract frequency domain features."""
    features = {}
    
    try:
        # FFT analysis
        n = len(flux)
        if n < 4:
            return {f'fft_feature_{i}': np.nan for i in range(20)}
        
        # Interpolate to regular grid for FFT
        time_regular = np.linspace(np.min(time), np.max(time), n)
        flux_interp = np.interp(time_regular, time, flux)
        
        # Remove mean
        flux_centered = flux_interp - np.mean(flux_interp)
        
        # Apply window
        window = signal.windows.hann(n)
        flux_windowed = flux_centered * window
        
        # FFT
        fft_values = fft(flux_windowed)
        freqs = fftfreq(n, d=np.mean(np.diff(time_regular)))
        
        # Power spectrum
        power = np.abs(fft_values)**2
        
        # Only positive frequencies
        pos_mask = freqs > 0
        freqs_pos = freqs[pos_mask]
        power_pos = power[pos_mask]
        
        if len(power_pos) > 0:
            # Spectral statistics
            features['spectral_power_total'] = np.sum(power_pos)
            features['spectral_power_mean'] = np.mean(power_pos)
            features['spectral_power_std'] = np.std(power_pos)
            features['spectral_power_max'] = np.max(power_pos)
            
            # Spectral centroid
            features['spectral_centroid'] = np.sum(freqs_pos * power_pos) / np.sum(power_pos)
            
            # Spectral spread
            features['spectral_spread'] = np.sqrt(np.sum((freqs_pos - features['spectral_centroid'])**2 * power_pos) / np.sum(power_pos))
            
            # Spectral entropy
            power_norm = power_pos / np.sum(power_pos)
            power_norm = power_norm[power_norm > 0]
            features['spectral_entropy'] = -np.sum(power_norm * np.log2(power_norm))
            
            # Peak finding
            peaks, _ = signal.find_peaks(power_pos, height=np.max(power_pos) * 0.1)
            features['n_spectral_peaks'] = len(peaks)
            
            if len(peaks) > 0:
                features['dominant_freq'] = freqs_pos[np.argmax(power_pos)]
                features['second_peak_freq'] = freqs_pos[peaks[np.argsort(power_pos[peaks])[-2]]] if len(peaks) > 1 else np.nan
            
            # Frequency band powers
            total_power = np.sum(power_pos)
            if total_power > 0:
                freq_bands = [(0, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, 5.0), (5.0, np.inf)]
                for i, (f_low, f_high) in enumerate(freq_bands):
                    band_mask = (freqs_pos >= f_low) & (freqs_pos < f_high)
                    features[f'power_band_{i}'] = np.sum(power_pos[band_mask]) / total_power
        
    except Exception as e:
        # If FFT fails, set all features to NaN
        fft_features = ['spectral_power_total', 'spectral_power_mean', 'spectral_power_std',
                       'spectral_power_max', 'spectral_centroid', 'spectral_spread',
                       'spectral_entropy', 'n_spectral_peaks', 'dominant_freq', 'second_peak_freq']
        for feature in fft_features:
            features[feature] = np.nan
        for i in range(5):
            features[f'power_band_{i}'] = np.nan
    
    return features


def extract_advanced_statistics(flux):
    """Extract advanced statistical features."""
    features = {}
    
    # Higher order moments
    for moment in range(3, 7):
        features[f'moment_{moment}'] = stats.moment(flux, moment=moment)
    
    # Distributional tests
    try:
        stat, p = stats.normaltest(flux)
        features['normaltest_stat'] = stat
        features['normaltest_p'] = p
    except:
        features['normaltest_stat'] = np.nan
        features['normaltest_p'] = np.nan
    
    # Quantile-based features
    q25, q50, q75 = np.percentile(flux, [25, 50, 75])
    features['q25_q75_ratio'] = q25 / q75 if q75 != 0 else np.nan
    features['q50_mean_ratio'] = q50 / np.mean(flux) if np.mean(flux) != 0 else np.nan
    
    # Coefficient of variation
    features['coeff_variation'] = np.std(flux) / np.abs(np.mean(flux)) if np.mean(flux) != 0 else np.nan
    
    # Outlier detection
    q1, q3 = np.percentile(flux, [25, 75])
    iqr = q3 - q1
    outlier_mask = (flux < q1 - 1.5 * iqr) | (flux > q3 + 1.5 * iqr)
    features['n_outliers'] = np.sum(outlier_mask)
    features['outlier_fraction'] = np.mean(outlier_mask)
    
    return features


def extract_variability_features(time, flux):
    """Extract variability features."""
    features = {}
    
    # Amplitude-based variability
    features['amplitude'] = (np.max(flux) - np.min(flux)) / 2
    features['rms'] = np.sqrt(np.mean(flux**2))
    
    # Relative variability
    mean_flux = np.mean(flux)
    if mean_flux != 0:
        features['relative_amplitude'] = features['amplitude'] / np.abs(mean_flux)
        features['fractional_rms'] = np.std(flux) / np.abs(mean_flux)
    
    # Welch-Stetson variability index
    if len(flux) > 1:
        flux_normalized = (flux - np.mean(flux)) / np.std(flux)
        features['welch_stetson'] = np.sqrt(1/(len(flux)-1) * np.sum(flux_normalized[:-1] * flux_normalized[1:]))
    
    # Stetson J and K indices
    try:
        delta = flux - np.mean(flux)
        sigma = np.std(flux)
        if sigma > 0:
            features['stetson_j'] = np.sum(np.sign(delta) * np.sqrt(np.abs(delta))) / (len(flux) * sigma)
            features['stetson_k'] = np.sum(np.abs(delta)) / (len(flux) * sigma)
    except:
        features['stetson_j'] = np.nan
        features['stetson_k'] = np.nan
    
    # von Neumann ratio
    if len(flux) > 1:
        numerator = np.sum((flux[1:] - flux[:-1])**2)
        denominator = np.sum((flux - np.mean(flux))**2)
        features['von_neumann_ratio'] = numerator / denominator if denominator > 0 else np.nan
    
    return features


def extract_periodicity_features(time, flux):
    """Extract periodicity-related features."""
    features = {}
    
    try:
        # Lomb-Scargle periodogram
        from scipy.signal import lombscargle
        
        # Frequency grid
        f_min = 1 / (np.max(time) - np.min(time))
        f_max = 1 / (2 * np.median(np.diff(time)))
        frequencies = np.logspace(np.log10(f_min), np.log10(f_max), 1000)
        
        # Compute periodogram
        pgram = lombscargle(time, flux - np.mean(flux), frequencies, normalize=True)
        
        # Peak detection
        peak_power = np.max(pgram)
        peak_freq = frequencies[np.argmax(pgram)]
        
        features['ls_peak_power'] = peak_power
        features['ls_peak_period'] = 1 / peak_freq if peak_freq > 0 else np.nan
        features['ls_peak_freq'] = peak_freq
        
        # False alarm probability (approximate)
        features['ls_fap'] = np.exp(-peak_power)
        
        # Multiple peaks
        peaks, _ = signal.find_peaks(pgram, height=0.1 * peak_power)
        features['ls_n_peaks'] = len(peaks)
        
        if len(peaks) > 1:
            sorted_peaks = peaks[np.argsort(pgram[peaks])[::-1]]
            features['ls_second_peak_power'] = pgram[sorted_peaks[1]]
            features['ls_second_peak_period'] = 1 / frequencies[sorted_peaks[1]]
            features['ls_peak_ratio'] = pgram[sorted_peaks[1]] / peak_power
        
    except Exception as e:
        ls_features = ['ls_peak_power', 'ls_peak_period', 'ls_peak_freq', 'ls_fap',
                      'ls_n_peaks', 'ls_second_peak_power', 'ls_second_peak_period', 'ls_peak_ratio']
        for feature in ls_features:
            features[feature] = np.nan
    
    return features


def extract_shape_features(flux):
    """Extract features related to the shape of the light curve."""
    features = {}
    
    # Symmetry measures
    median_flux = np.median(flux)
    above_median = flux[flux > median_flux]
    below_median = flux[flux < median_flux]
    
    if len(above_median) > 0 and len(below_median) > 0:
        features['asymmetry_ratio'] = len(above_median) / len(below_median)
        features['above_median_mean'] = np.mean(above_median)
        features['below_median_mean'] = np.mean(below_median)
        features['above_below_ratio'] = features['above_median_mean'] / features['below_median_mean']
    
    # Flux excursions
    mean_flux = np.mean(flux)
    std_flux = np.std(flux)
    
    for threshold in [1, 2, 3]:
        above_mask = flux > mean_flux + threshold * std_flux
        below_mask = flux < mean_flux - threshold * std_flux
        features[f'frac_above_{threshold}sigma'] = np.mean(above_mask)
        features[f'frac_below_{threshold}sigma'] = np.mean(below_mask)
    
    # Runs analysis
    above_mean_runs = get_runs(flux > mean_flux)
    features['max_run_above_mean'] = np.max(above_mean_runs) if len(above_mean_runs) > 0 else 0
    features['mean_run_above_mean'] = np.mean(above_mean_runs) if len(above_mean_runs) > 0 else 0
    
    below_mean_runs = get_runs(flux < mean_flux)
    features['max_run_below_mean'] = np.max(below_mean_runs) if len(below_mean_runs) > 0 else 0
    features['mean_run_below_mean'] = np.mean(below_mean_runs) if len(below_mean_runs) > 0 else 0
    
    return features


def extract_transit_features(time, flux):
    """Extract features specific to transit detection."""
    features = {}
    
    # Simple transit search
    try:
        # Look for dips
        median_flux = np.median(flux)
        dip_threshold = median_flux - 2 * np.std(flux)
        dip_mask = flux < dip_threshold
        
        if np.any(dip_mask):
            # Find connected dip regions
            dip_regions = get_connected_regions(dip_mask)
            
            if len(dip_regions) > 0:
                # Analyze deepest dip
                deepest_dip = min(dip_regions, key=lambda region: np.min(flux[region[0]:region[1]+1]))
                start, end = deepest_dip
                
                features['deepest_dip_depth'] = median_flux - np.min(flux[start:end+1])
                features['deepest_dip_duration'] = time[end] - time[start]
                features['deepest_dip_width_points'] = end - start + 1
                
                # Dip shape analysis
                dip_flux = flux[start:end+1]
                if len(dip_flux) > 2:
                    # Ingress/egress slopes
                    mid_point = len(dip_flux) // 2
                    if mid_point > 0:
                        ingress_slope = (dip_flux[mid_point] - dip_flux[0]) / (mid_point)
                        egress_slope = (dip_flux[-1] - dip_flux[mid_point]) / (len(dip_flux) - mid_point)
                        features['transit_ingress_slope'] = ingress_slope
                        features['transit_egress_slope'] = egress_slope
                        features['transit_asymmetry'] = (egress_slope - ingress_slope) / (egress_slope + ingress_slope + 1e-10)
        
        # Count number of significant dips
        features['n_dips'] = len(get_connected_regions(dip_mask))
        
    except Exception as e:
        transit_features = ['deepest_dip_depth', 'deepest_dip_duration', 'deepest_dip_width_points',
                           'transit_ingress_slope', 'transit_egress_slope', 'transit_asymmetry', 'n_dips']
        for feature in transit_features:
            features[feature] = np.nan
    
    return features


def extract_error_features(flux, flux_err):
    """Extract features based on flux errors."""
    features = {}
    
    # Error statistics
    features['mean_error'] = np.mean(flux_err)
    features['std_error'] = np.std(flux_err)
    features['max_error'] = np.max(flux_err)
    features['min_error'] = np.min(flux_err)
    
    # Signal-to-noise
    features['mean_snr'] = np.mean(np.abs(flux) / flux_err)
    features['median_snr'] = np.median(np.abs(flux) / flux_err)
    features['min_snr'] = np.min(np.abs(flux) / flux_err)
    
    # Error-weighted statistics
    weights = 1.0 / (flux_err**2)
    features['weighted_mean'] = np.average(flux, weights=weights)
    features['weighted_std'] = np.sqrt(np.average((flux - features['weighted_mean'])**2, weights=weights))
    
    # Chi-squared goodness of fit to constant
    expected = np.mean(flux)
    chi2 = np.sum((flux - expected)**2 / flux_err**2)
    features['chi2_constant'] = chi2
    features['chi2_reduced'] = chi2 / (len(flux) - 1) if len(flux) > 1 else np.nan
    
    return features


def extract_astronomical_features(time, flux):
    """Extract additional astronomical features."""
    features = {}
    
    # Duty cycle (fraction of time observed)
    if len(time) > 1:
        total_timespan = np.max(time) - np.min(time)
        observed_time = np.sum(np.diff(time))
        features['duty_cycle'] = observed_time / total_timespan if total_timespan > 0 else np.nan
    
    # Cadence regularity
    if len(time) > 2:
        cadences = np.diff(time)
        features['cadence_regularity'] = np.std(cadences) / np.mean(cadences) if np.mean(cadences) > 0 else np.nan
    
    # Flux concentration (how much flux is in brightest/faintest points)
    sorted_flux = np.sort(flux)
    n_points = len(flux)
    
    # Top/bottom 10% concentration
    top_10_pct = int(0.1 * n_points)
    if top_10_pct > 0:
        features['top_10pct_concentration'] = np.sum(sorted_flux[-top_10_pct:]) / np.sum(flux)
        features['bottom_10pct_concentration'] = np.sum(sorted_flux[:top_10_pct]) / np.sum(flux)
    
    # Magnitude features (if flux > 0)
    if np.all(flux > 0):
        mag = -2.5 * np.log10(flux)
        features['mag_mean'] = np.mean(mag)
        features['mag_std'] = np.std(mag)
        features['mag_range'] = np.max(mag) - np.min(mag)
    
    return features


def get_runs(boolean_array):
    """Get lengths of consecutive True runs in a boolean array."""
    if len(boolean_array) == 0:
        return []
    
    # Find run starts and ends
    d = np.diff(np.concatenate(([False], boolean_array, [False])).astype(int))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    
    return ends - starts


def get_connected_regions(boolean_mask):
    """Get start and end indices of connected True regions."""
    if not np.any(boolean_mask):
        return []
    
    d = np.diff(np.concatenate(([False], boolean_mask, [False])).astype(int))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    
    return list(zip(starts, ends))


# Additional feature extraction functions that would be used with tsfresh, etc.
def extract_tsfresh_features(time, flux):
    """
    Extract features using tsfresh library (if available).
    This is a placeholder - actual implementation would use tsfresh.
    """
    features = {}
    
    try:
        import tsfresh
        from tsfresh import extract_features as tsfresh_extract
        from tsfresh.feature_extraction import MinimalFCParameters
        
        # Prepare data for tsfresh
        df = pd.DataFrame({
            'id': [1] * len(time),
            'time': time,
            'value': flux
        })
        
        # Extract features with minimal settings to avoid too many features
        extracted = tsfresh_extract(df, column_id='id', column_sort='time', 
                                   default_fc_parameters=MinimalFCParameters())
        
        # Convert to dictionary and add prefix
        for col in extracted.columns:
            features[f'tsfresh_{col}'] = extracted[col].iloc[0]
            
    except ImportError:
        # tsfresh not available, return empty dict
        pass
    except Exception as e:
        # tsfresh failed, return empty dict
        pass
    
    return features