"""
Generate test CSV files for the Exoplanet Detection Dashboard.
Creates realistic synthetic light curves with and without transits.
"""

import numpy as np
import pandas as pd
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "light_curves_csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_stellar_variability(time, amplitude=0.002, periods=None):
    """Simulate slow stellar variability (starspots, rotation)."""
    if periods is None:
        periods = [0.3, 0.7, 1.5]
    variability = np.zeros_like(time)
    for p in periods:
        phase = np.random.uniform(0, 2 * np.pi)
        variability += amplitude * np.random.uniform(0.3, 1.0) * np.sin(2 * np.pi * time / (p * len(time)) + phase)
    return variability


def make_transit(time, period, depth, duration, epoch):
    """Create a box-shaped transit signal."""
    flux_dip = np.zeros_like(time)
    phase = ((time - epoch) % period) / period
    # Transit centered at phase=0 (wraps around 1.0)
    half_dur = (duration / period) / 2
    in_transit = (phase < half_dur) | (phase > 1.0 - half_dur)
    
    # Smooth ingress/egress with a simple model
    for i in np.where(in_transit)[0]:
        p = phase[i]
        if p > 0.5:
            p = p - 1.0  # wrap to negative for phase near 1.0
        dist_from_center = abs(p)
        # Trapezoidal shape
        if dist_from_center < half_dur * 0.7:
            flux_dip[i] = -depth
        else:
            # Linear ingress/egress
            frac = (half_dur - dist_from_center) / (half_dur * 0.3)
            flux_dip[i] = -depth * max(0, frac)
    
    return flux_dip


def generate_planet_transit_csv(filename, n_points=500, noise_level=0.0008,
                                 period=50, depth=0.008, duration=3, 
                                 description="Planet with clear transit"):
    """Generate a light curve CSV with planetary transit."""
    np.random.seed(42)
    
    time = np.linspace(120.0, 130.0 + n_points * 0.02, n_points)
    
    # Base flux = 1.0
    flux = np.ones(n_points)
    
    # Add stellar variability
    flux += make_stellar_variability(time, amplitude=0.001)
    
    # Add transit dips
    epoch = time[0] + period * 0.3
    transit = make_transit(time, period=period, depth=depth, 
                          duration=duration, epoch=epoch)
    flux += transit
    
    # Add Gaussian noise
    flux += np.random.normal(0, noise_level, n_points)
    
    # Add a couple of random gaps (missing data) like real Kepler data
    gap_indices = np.random.choice(n_points, size=5, replace=False)
    
    df = pd.DataFrame({"time": time, "flux": flux})
    # Remove gap points
    df = df.drop(gap_indices).reset_index(drop=True)
    
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"✅ Created {filename}: {description} ({len(df)} points, depth={depth*100:.2f}%)")
    return filepath


def generate_no_planet_csv(filename, n_points=500, noise_level=0.001):
    """Generate a light curve CSV with NO planet (just stellar noise)."""
    np.random.seed(123)
    
    time = np.linspace(200.0, 210.0 + n_points * 0.02, n_points)
    
    flux = np.ones(n_points)
    flux += make_stellar_variability(time, amplitude=0.0015,
                                      periods=[0.2, 0.5, 1.2])
    flux += np.random.normal(0, noise_level, n_points)
    
    # Add a couple of flare-like spikes (positive outliers)
    flare_idx = np.random.choice(n_points, size=3, replace=False)
    flux[flare_idx] += np.random.uniform(0.005, 0.012, 3)
    
    df = pd.DataFrame({"time": time, "flux": flux})
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"✅ Created {filename}: No planet, noise only ({len(df)} points)")
    return filepath


def generate_deep_transit_csv(filename, n_points=800, noise_level=0.0005):
    """Generate a very clear hot Jupiter transit (deep, periodic)."""
    np.random.seed(77)
    
    time = np.linspace(100.0, 140.0, n_points)
    
    flux = np.ones(n_points)
    flux += make_stellar_variability(time, amplitude=0.0008, periods=[0.4, 1.0])
    
    # Hot Jupiter: deep transit, short period
    period_points = 160  # period in time-index units
    period_time = (time[-1] - time[0]) * period_points / n_points
    transit = make_transit(time, period=period_time, depth=0.015,
                          duration=period_time * 0.06, epoch=time[0] + period_time * 0.2)
    flux += transit
    
    flux += np.random.normal(0, noise_level, n_points)
    
    df = pd.DataFrame({"time": time, "flux": flux})
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"✅ Created {filename}: Hot Jupiter deep transit ({len(df)} points, depth=1.5%)")
    return filepath


def generate_multi_transit_csv(filename, n_points=1000, noise_level=0.0006):
    """Generate a light curve with multiple visible transits (long baseline)."""
    np.random.seed(55)
    
    time = np.linspace(100.0, 200.0, n_points)
    
    flux = np.ones(n_points)
    flux += make_stellar_variability(time, amplitude=0.001, periods=[0.15, 0.4, 0.8])
    
    # Earth-like: moderate depth, multiple transits visible
    period_time = 25.0  # days
    transit = make_transit(time, period=period_time, depth=0.005,
                          duration=1.5, epoch=time[0] + 5.0)
    flux += transit
    
    flux += np.random.normal(0, noise_level, n_points)
    
    df = pd.DataFrame({"time": time, "flux": flux})
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"✅ Created {filename}: Multiple transits ({len(df)} points, period=25d)")
    return filepath


def generate_wide_format_csv(filename, n_stars=20, n_bins=500):
    """Generate a wide-format CSV (like Kepler binned data).
    
    Each row is a star. First column is label (1=planet, 0=no planet),
    remaining columns are flux bins.
    """
    np.random.seed(99)
    
    rows = []
    labels = []
    
    for i in range(n_stars):
        has_planet = i < n_stars // 2  # first half have planets
        
        t = np.linspace(0, 1, n_bins)
        flux = np.ones(n_bins)
        flux += np.random.normal(0, np.random.uniform(0.0005, 0.002), n_bins)
        flux += 0.001 * np.sin(2 * np.pi * np.random.uniform(0.5, 3) * t)
        
        if has_planet:
            # Add 1-3 transit dips
            for _ in range(np.random.randint(1, 4)):
                center = np.random.uniform(0.1, 0.9)
                depth = np.random.uniform(0.004, 0.02)
                width = np.random.uniform(0.01, 0.04)
                gaussian_dip = depth * np.exp(-0.5 * ((t - center) / width) ** 2)
                flux -= gaussian_dip
        
        labels.append(1 if has_planet else 0)
        rows.append(flux)
    
    data = np.column_stack([np.array(labels).reshape(-1, 1), np.array(rows)])
    columns = ['LABEL'] + [f'FLUX_{i}' for i in range(n_bins)]
    df = pd.DataFrame(data, columns=columns)
    
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"✅ Created {filename}: Wide-format ({n_stars} stars × {n_bins} bins)")
    return filepath


def generate_subtle_transit_csv(filename, n_points=600, noise_level=0.0012):
    """Generate a subtle, hard-to-detect transit (small planet)."""
    np.random.seed(200)
    
    time = np.linspace(150.0, 175.0, n_points)
    
    flux = np.ones(n_points)
    flux += make_stellar_variability(time, amplitude=0.0012, periods=[0.3, 0.6])
    
    # Very subtle transit — small planet
    period_time = 8.0
    transit = make_transit(time, period=period_time, depth=0.002,
                          duration=0.8, epoch=time[0] + 2.0)
    flux += transit
    
    flux += np.random.normal(0, noise_level, n_points)
    
    df = pd.DataFrame({"time": time, "flux": flux})
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"✅ Created {filename}: Subtle transit ({len(df)} points, depth=0.2%)")
    return filepath


if __name__ == "__main__":
    print("=" * 60)
    print("Generating test CSV files for Exoplanet Detection Dashboard")
    print("=" * 60)
    print()
    
    generate_planet_transit_csv("test_planet_clear.csv",
                                n_points=500, depth=0.008,
                                description="Clear single transit")
    
    generate_deep_transit_csv("test_hot_jupiter.csv")
    
    generate_multi_transit_csv("test_multi_transit.csv")
    
    generate_subtle_transit_csv("test_subtle_planet.csv")
    
    generate_no_planet_csv("test_no_planet.csv")
    
    generate_wide_format_csv("test_wide_format.csv", n_stars=20, n_bins=500)
    
    print()
    print("=" * 60)
    print("All test files created in:", OUTPUT_DIR)
    print()
    print("Recommended testing order:")
    print("  1. test_hot_jupiter.csv      — Deep, obvious dips (should detect planet)")
    print("  2. test_multi_transit.csv     — Multiple periodic transits")
    print("  3. test_planet_clear.csv      — Single clear transit")
    print("  4. test_subtle_planet.csv     — Subtle dip (harder to detect)")
    print("  5. test_no_planet.csv         — No planet (should say No Planet)")
    print("  6. test_wide_format.csv       — Wide-format, test row selector")
    print("=" * 60)
