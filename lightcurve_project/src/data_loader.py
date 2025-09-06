"""
Data Loading and Preprocessing Module

This module handles loading .npz light curve files and provides comprehensive
preprocessing capabilities including cleaning, folding, and normalization.
"""

import numpy as np
from scipy import stats
import warnings

def load_npz_curve(file_path):
    """
    Load a light curve from an .npz file.
    
    Parameters:
    -----------
    file_path : str
        Path to the .npz file containing light curve data
        
    Returns:
    --------
    dict
        Dictionary containing 'time', 'flux', and 'flux_err' arrays
        
    Raises:
    -------
    FileNotFoundError
        If the specified file doesn't exist
    KeyError
        If required keys are missing from the .npz file
    """
    try:
        data = np.load(file_path)
        
        # Check for required keys (flexible naming)
        time_key = None
        flux_key = None
        flux_err_key = None
        
        for key in data.files:
            key_lower = key.lower()
            if 'time' in key_lower or 't' == key_lower:
                time_key = key
            elif 'flux' in key_lower and 'err' not in key_lower:
                flux_key = key
            elif 'err' in key_lower or 'error' in key_lower:
                flux_err_key = key
                
        if time_key is None or flux_key is None:
            raise KeyError(f"Required keys not found. Available keys: {list(data.files)}")
            
        result = {
            'time': data[time_key],
            'flux': data[flux_key],
            'flux_err': data[flux_err_key] if flux_err_key else np.ones_like(data[flux_key]) * 0.001
        }
        
        print(f"Loaded light curve with {len(result['time'])} data points")
        return result
        
    except Exception as e:
        raise Exception(f"Error loading {file_path}: {str(e)}")


def preprocess_lightcurve(lc, period=None, epoch_time=None, apply_preprocessing=True):
    """
    Ultra-clean processing for light curves.
    
    Parameters:
    -----------
    lc : dict
        Light curve data with 'time', 'flux', 'flux_err' keys
    period : float, optional
        Period for folding the light curve (in same units as time)
    epoch_time : float, optional  
        Epoch time for folding (in same units as time)
    apply_preprocessing : bool, default=True
        Whether to apply full preprocessing pipeline
        
    Returns:
    --------
    dict
        Processed light curve data
    """
    if not apply_preprocessing:
        return lc.copy()
        
    time = lc['time'].copy()
    flux = lc['flux'].copy()
    flux_err = lc['flux_err'].copy()
    
    print("Starting preprocessing pipeline...")
    
    # 1. Remove NaNs and infinite values
    valid_mask = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
    valid_mask &= (flux_err > 0)  # Positive errors only
    
    time = time[valid_mask]
    flux = flux[valid_mask]
    flux_err = flux_err[valid_mask]
    print(f"Step 1: Removed {np.sum(~valid_mask)} invalid points")
    
    if len(time) == 0:
        raise ValueError("No valid data points after removing NaNs")
    
    # 2. Flatten stellar variability (detrend with polynomial)
    try:
        # Fit 3rd order polynomial to remove long-term trends
        poly_coeffs = np.polyfit(time, flux, deg=3)
        trend = np.polyval(poly_coeffs, time)
        flux = flux / trend
        print("Step 2: Removed polynomial trend")
    except:
        print("Step 2: Trend removal failed, skipping")
    
    # 3. Mask good quality points (remove extreme outliers first)
    median_flux = np.median(flux)
    mad_flux = np.median(np.abs(flux - median_flux))
    quality_mask = np.abs(flux - median_flux) < 10 * mad_flux
    
    time = time[quality_mask]
    flux = flux[quality_mask]
    flux_err = flux_err[quality_mask]
    print(f"Step 3: Removed {np.sum(~quality_mask)} extreme outliers")
    
    # 4. Sigma-clip outliers (iterative)
    for iteration in range(3):
        mean_flux = np.mean(flux)
        std_flux = np.std(flux)
        sigma_mask = np.abs(flux - mean_flux) < 3 * std_flux
        
        if np.sum(~sigma_mask) == 0:
            break
            
        time = time[sigma_mask]
        flux = flux[sigma_mask]
        flux_err = flux_err[sigma_mask]
        
    print(f"Step 4: Sigma clipping completed in {iteration + 1} iterations")
    
    # 5. Fold if period and epoch provided
    if period is not None and epoch_time is not None:
        phase = ((time - epoch_time) / period) % 1.0
        # Sort by phase for cleaner plotting
        sort_idx = np.argsort(phase)
        time = phase[sort_idx]
        flux = flux[sort_idx]
        flux_err = flux_err[sort_idx]
        print(f"Step 5: Folded light curve with period {period}")
    else:
        print("Step 5: Skipping folding (no period/epoch provided)")
    
    # 6. Bin data (5 minute bins if time is in days)
    if len(time) > 1000:  # Only bin if we have lots of data
        try:
            bin_size = 5.0 / (24 * 60)  # 5 minutes in days
            if period is not None:  # Adjust bin size for folded data
                bin_size = 0.01  # 1% of phase
                
            time_binned, flux_binned, flux_err_binned = bin_lightcurve(
                time, flux, flux_err, bin_size
            )
            print(f"Step 6: Binned from {len(time)} to {len(time_binned)} points")
            time, flux, flux_err = time_binned, flux_binned, flux_err_binned
        except:
            print("Step 6: Binning failed, keeping original data")
    else:
        print("Step 6: Skipping binning (insufficient data)")
    
    # 7. Normalize flux (zero mean, unit variance)
    flux_mean = np.mean(flux)
    flux_std = np.std(flux)
    if flux_std > 0:
        flux = (flux - flux_mean) / flux_std
        flux_err = flux_err / flux_std
        print("Step 7: Normalized flux to zero mean, unit variance")
    else:
        print("Step 7: Cannot normalize (zero variance)")
    
    processed_lc = {
        'time': time,
        'flux': flux,
        'flux_err': flux_err,
        'preprocessing_applied': True
    }
    
    print(f"Preprocessing complete: {len(time)} final data points")
    return processed_lc


def bin_lightcurve(time, flux, flux_err, bin_size):
    """
    Bin light curve data using weighted averages.
    
    Parameters:
    -----------
    time, flux, flux_err : array-like
        Light curve data
    bin_size : float
        Size of bins in same units as time
        
    Returns:
    --------
    tuple
        Binned (time, flux, flux_err) arrays
    """
    if len(time) == 0:
        return time, flux, flux_err
        
    time_min, time_max = np.min(time), np.max(time)
    n_bins = int(np.ceil((time_max - time_min) / bin_size))
    
    if n_bins <= 1:
        return time, flux, flux_err
    
    bin_edges = np.linspace(time_min, time_max, n_bins + 1)
    bin_centers = []
    binned_flux = []
    binned_flux_err = []
    
    for i in range(n_bins):
        mask = (time >= bin_edges[i]) & (time < bin_edges[i + 1])
        
        if np.sum(mask) == 0:
            continue
            
        bin_time = time[mask]
        bin_flux = flux[mask]
        bin_err = flux_err[mask]
        
        # Weighted average
        weights = 1.0 / (bin_err**2)
        weighted_flux = np.average(bin_flux, weights=weights)
        weighted_err = 1.0 / np.sqrt(np.sum(weights))
        
        bin_centers.append(np.mean(bin_time))
        binned_flux.append(weighted_flux)
        binned_flux_err.append(weighted_err)
    
    return np.array(bin_centers), np.array(binned_flux), np.array(binned_flux_err)