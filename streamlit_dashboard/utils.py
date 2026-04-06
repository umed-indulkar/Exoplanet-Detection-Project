"""
Advanced Analytics: SNR, transit depth, period estimation, dip detection
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import median_abs_deviation


def calculate_snr(flux):
    """Signal-to-Noise Ratio."""
    signal = np.median(flux)
    noise = median_abs_deviation(flux, scale='normal')
    if noise == 0:
        noise = np.std(flux)
    snr = abs(signal / noise) if noise > 0 else 0
    return {
        'snr': snr,
        'signal': signal,
        'noise': noise,
        'snr_db': 20 * np.log10(snr) if snr > 0 else 0,
    }


def estimate_transit_depth(flux):
    """Estimate transit depth and planet-to-star radius ratio."""
    baseline = np.median(flux)
    threshold = baseline - 2 * np.std(flux)
    dip_mask = flux < threshold
    if not np.any(dip_mask):
        threshold = baseline - np.std(flux)
        dip_mask = flux < threshold

    min_val = np.min(flux[dip_mask]) if np.any(dip_mask) else np.min(flux)
    depth = baseline - min_val
    rel = (depth / baseline) * 100 if baseline != 0 else 0
    rr = np.sqrt(abs(depth / baseline)) if baseline != 0 else 0
    return {
        'depth_absolute': depth,
        'depth_pct': rel,
        'baseline': baseline,
        'min_flux': min_val,
        'radius_ratio': rr,
        'n_dip_points': int(np.sum(dip_mask)),
    }


def estimate_period(flux):
    """Estimate orbital period via autocorrelation."""
    detrended = flux - np.median(flux)
    n = len(detrended)
    ac = np.correlate(detrended, detrended, mode='full')[n - 1:]
    ac = ac / (ac[0] if ac[0] != 0 else 1)
    min_lag = max(10, n // 20)
    peaks, _ = find_peaks(ac[min_lag:], height=0.1, distance=min_lag)
    if len(peaks) > 0:
        pi = peaks[0] + min_lag
        return {'period_idx': pi, 'strength': ac[pi], 'n_periods': n / pi, 'autocorr': ac}
    return {'period_idx': None, 'strength': 0, 'n_periods': 0, 'autocorr': ac}


def find_transit_dips(flux, sigma=2.0):
    """Locate transit dips in the light curve."""
    baseline = np.median(flux)
    threshold = baseline - sigma * np.std(flux)
    inv = -flux
    peaks, _ = find_peaks(inv, distance=max(10, len(flux) // 50), prominence=np.std(flux) * 0.5)
    return [{'index': int(p), 'value': float(flux[p]), 'depth': float(baseline - flux[p])}
            for p in peaks if flux[p] < threshold]
