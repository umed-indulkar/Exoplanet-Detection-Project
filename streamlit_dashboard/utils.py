"""
Advanced Analytics: SNR, transit depth, period estimation, dip detection
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import median_abs_deviation


def calculate_snr(flux):
    """Transit SNR = transit_depth / out-of-transit noise.

    Uses the minimum flux point as the transit bottom and MAD as the
    noise estimate.  This gives a physically meaningful number (how
    many noise-sigmas deep the dip is), unlike median/MAD which would
    return ~1000 for any normalised light curve.
    """
    flux = np.asarray(flux, dtype=float)
    baseline = float(np.median(flux))
    noise = float(median_abs_deviation(flux, scale='normal'))
    if noise == 0:
        noise = float(np.std(flux))
    if noise == 0:
        return {'snr': 0.0, 'signal': baseline, 'noise': 0.0, 'snr_db': 0.0}

    transit_depth = float(baseline - np.min(flux))   # positive for a dip
    snr = max(transit_depth / noise, 0.0)
    snr_db = float(20 * np.log10(snr)) if snr > 0 else 0.0

    return {
        'snr':    snr,
        'signal': baseline,
        'noise':  noise,
        'snr_db': snr_db,
    }


def estimate_transit_depth(flux):
    """Estimate transit depth and planet-to-star radius ratio.

    Uses the minimum flux point so that the depth is never
    underestimated.  Rp/Rs = sqrt(delta_F) where delta_F is the
    fractional flux deficit.  Returns zeros when the baseline is
    degenerate (e.g. z-score flux with median ≈ 0).
    """
    flux = np.asarray(flux, dtype=float)
    baseline = float(np.median(flux))

    # Guard: only compute physical ratios when flux is ratio-normalised
    # (baseline close to 1).  For z-score flux the formula is meaningless.
    if abs(baseline) < 0.1:
        return {
            'depth_absolute': 0.0,
            'depth_pct':      0.0,
            'baseline':       baseline,
            'min_flux':       float(np.min(flux)),
            'dip_level':      float(np.min(flux)),
            'radius_ratio':   0.0,
            'n_dip_points_2s': 0,
            'n_dip_points_1s': 0,
            'n_dip_points':   0,
        }

    min_flux   = float(np.min(flux))
    depth      = float(baseline - min_flux)           # absolute depth
    depth_pct  = float(depth / baseline * 100)        # percentage
    delta_f    = float(depth / baseline)              # fractional depth
    rr         = float(np.sqrt(max(delta_f, 0.0)))   # Rp/Rs

    # Dip masks
    std_f           = float(np.std(flux))
    threshold_2s    = baseline - 2 * std_f
    threshold_1s    = baseline - std_f
    n_dip_2s        = int(np.sum(flux < threshold_2s))
    n_dip_1s        = int(np.sum(flux < threshold_1s))

    return {
        'depth_absolute':  depth,
        'depth_pct':       depth_pct,
        'baseline':        baseline,
        'min_flux':        min_flux,
        'dip_level':       min_flux,
        'radius_ratio':    rr,
        'n_dip_points_2s': n_dip_2s,
        'n_dip_points_1s': n_dip_1s,
        'n_dip_points':    n_dip_2s,
    }


def estimate_period(flux):
    """Estimate orbital period via autocorrelation of detrended flux."""
    detrended = flux - np.median(flux)
    n = len(detrended)
    if n < 30:
        return {'period_idx': None, 'strength': 0, 'n_periods': 0, 'autocorr': None}

    ac = np.correlate(detrended, detrended, mode='full')[n - 1:]
    ac = ac / (ac[0] if ac[0] != 0 else 1)

    min_lag = max(10, n // 20)
    peaks, props = find_peaks(ac[min_lag:], height=0.1, distance=min_lag)

    if len(peaks) > 0:
        pi = int(peaks[0]) + min_lag
        return {
            'period_idx': pi,
            'strength':   float(ac[pi]),
            'n_periods':  float(n / pi),
            'autocorr':   ac,
        }
    return {'period_idx': None, 'strength': 0.0, 'n_periods': 0.0, 'autocorr': ac}


def find_transit_dips(flux, sigma=2.0, min_depth_fraction=0.3):
    """Locate transit dips in the light curve (z-score units)."""
    baseline  = float(np.median(flux))
    std_flux  = float(np.std(flux))
    threshold = baseline - sigma * std_flux

    inv      = -flux
    min_dist = max(10, len(flux) // 50)
    min_prom = std_flux * 0.3

    peaks, props = find_peaks(inv, distance=min_dist, prominence=min_prom)

    dips = []
    for p in peaks:
        depth = baseline - flux[p]
        if flux[p] < threshold and depth > 0:
            dips.append({
                'index':     int(p),
                'value':     float(flux[p]),
                'depth':     float(depth),
                'depth_pct': float(depth / abs(baseline) * 100) if baseline != 0 else 0.0,
                'prominence': float(props['prominences'][list(peaks).index(p)]),
            })

    if dips:
        max_depth = max(d['depth'] for d in dips)
        dips = [d for d in dips if d['depth'] >= max_depth * min_depth_fraction]

    dips.sort(key=lambda d: d['depth'], reverse=True)
    return dips
