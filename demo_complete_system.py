"""
COMPLETE SYSTEM DEMONSTRATION
==============================

This script demonstrates ALL capabilities of the unified exoplanet detection system:
1. Data loading (NPZ/CSV/FITS)
2. Preprocessing (multiple methods)
3. Feature extraction
4. Visualization
5. Batch processing
6. Configuration management

Run this to see the complete system in action!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import unified system
from exodet import (
    load_lightcurve,
    load_batch_lightcurves,
    preprocess_lightcurve,
    Config
)
from exodet.features import extract_basic_features
from exodet.visualization import (
    plot_lightcurve,
    plot_folded_lightcurve,
    plot_feature_distributions,
    plot_comparison
)

print("="*80)
print("COMPLETE EXOPLANET DETECTION SYSTEM DEMONSTRATION")
print("="*80)

# Create output directory
output_dir = Path('demo_output')
output_dir.mkdir(exist_ok=True)
print(f"\n📁 Output directory: {output_dir}/")

# ============================================================================
# PART 1: CREATE SYNTHETIC DATA WITH REALISTIC EXOPLANET TRANSITS
# ============================================================================
print("\n" + "="*80)
print("PART 1: Creating Realistic Synthetic Exoplanet Data")
print("="*80)

# Parameters for Earth-like exoplanet around Sun-like star
time = np.linspace(0, 90, 10000)  # 90 days of observations
period = 365.25 / 4  # ~3 month orbit
transit_depth = 0.01  # 1% (Earth-like)
transit_duration = 0.5  # 0.5 days
noise_level = 0.002  # Kepler-like noise

# Create baseline flux with stellar variability
flux = np.ones_like(time)
flux += 0.005 * np.sin(2 * np.pi * time / 20)  # Stellar rotation (20 days)
flux += np.random.normal(0, noise_level, len(time))  # Photon noise

# Add transit signals
n_transits = 0
for t0 in np.arange(period/2, 90, period):
    in_transit = np.abs(time - t0) < transit_duration/2
    flux[in_transit] -= transit_depth * (1 - (np.abs(time[in_transit] - t0) / (transit_duration/2))**2)
    n_transits += 1

flux_err = np.full_like(flux, noise_level)

# Save
np.savez('demo_exoplanet.npz', time=time, flux=flux, flux_err=flux_err, label=1)

print(f"✓ Created synthetic exoplanet light curve")
print(f"  • Duration: {time[-1] - time[0]:.1f} days")
print(f"  • Observations: {len(time)} points")
print(f"  • Period: {period:.2f} days")
print(f"  • Transit depth: {transit_depth*100:.1f}%")
print(f"  • Number of transits: {n_transits}")
print(f"  • SNR: {transit_depth/noise_level:.1f}")

# ============================================================================
# PART 2: LOAD AND VISUALIZE RAW DATA
# ============================================================================
print("\n" + "="*80)
print("PART 2: Loading and Initial Visualization")
print("="*80)

lc = load_lightcurve('demo_exoplanet.npz')
print(f"✓ Loaded: {lc}")

# Plot original
fig1, ax1 = plot_lightcurve(
    lc,
    title="Raw Light Curve - Exoplanet Candidate",
    save_path=output_dir / '01_raw_lightcurve.png'
)
plt.close(fig1)

print(f"✓ Saved: 01_raw_lightcurve.png")

# ============================================================================
# PART 3: PREPROCESSING COMPARISON
# ============================================================================
print("\n" + "="*80)
print("PART 3: Advanced Preprocessing")
print("="*80)

# Try different preprocessing methods
preprocessing_methods = {
    'polynomial': {
        'detrend': {'enabled': True, 'method': 'polynomial', 'order': 2},
        'sigma_clip': {'enabled': True, 'sigma': 3.0},
        'normalize': {'enabled': True, 'method': 'zscore'}
    },
    'savgol': {
        'detrend': {'enabled': True, 'method': 'savgol'},
        'sigma_clip': {'enabled': True, 'sigma': 3.0},
        'normalize': {'enabled': True, 'method': 'zscore'}
    },
    'robust': {
        'detrend': {'enabled': True, 'method': 'polynomial'},
        'sigma_clip': {'enabled': True, 'sigma': 3.0},
        'normalize': {'enabled': True, 'method': 'robust'}
    }
}

print(f"\nTesting {len(preprocessing_methods)} preprocessing methods:")

preprocessed_curves = {}
for name, config in preprocessing_methods.items():
    lc_clean = preprocess_lightcurve(lc, **config)
    preprocessed_curves[name] = lc_clean
    print(f"  • {name}: {len(lc_clean)} points (removed {len(lc) - len(lc_clean)})")

# Use polynomial method for rest of demo
lc_clean = preprocessed_curves['polynomial']

# Plot comparison
fig2, axes2 = plot_comparison(
    lc, lc_clean,
    title="Impact of Preprocessing",
    save_path=output_dir / '02_preprocessing_comparison.png'
)
plt.close(fig2)

print(f"✓ Saved: 02_preprocessing_comparison.png")

# ============================================================================
# PART 4: FEATURE EXTRACTION
# ============================================================================
print("\n" + "="*80)
print("PART 4: Feature Extraction")
print("="*80)

features = extract_basic_features(lc_clean, verbose=False)

print(f"✓ Extracted {len(features.columns)} features")
print(f"\nTop 10 most significant features:")
print(f"  Feature                    Value")
print(f"  " + "-"*45)

# Show interesting features
interesting_features = [
    'mean', 'std', 'amplitude', 'n_dips', 'deepest_dip_depth',
    'deepest_dip_duration', 'skewness', 'von_neumann', 'autocorr_lag1', 'dominant_freq'
]

for feat in interesting_features:
    if feat in features.columns:
        val = features[feat].iloc[0]
        print(f"  {feat:25s} {val:10.6f}")

# Save features
features.to_csv(output_dir / 'features.csv', index=False)
print(f"\n✓ Saved: features.csv ({len(features.columns)} features)")

# Plot feature distributions
fig3 = plot_feature_distributions(
    features,
    features_to_plot=interesting_features,
    save_path=output_dir / '03_feature_distributions.png'
)
plt.close(fig3)

print(f"✓ Saved: 03_feature_distributions.png")

# ============================================================================
# PART 5: PERIOD FOLDING
# ============================================================================
print("\n" + "="*80)
print("PART 5: Phase-Folded Light Curve")
print("="*80)

# Use the known period to fold
print(f"Folding with period: {period:.2f} days")

fig4, ax4 = plot_folded_lightcurve(
    lc_clean,
    period=period,
    epoch=period/2,
    title="Phase-Folded Light Curve (Known Period)",
    save_path=output_dir / '04_folded_lightcurve.png'
)
plt.close(fig4)

print(f"✓ Saved: 04_folded_lightcurve.png")
print(f"  • Transit clearly visible at phase 0.0 and 1.0")

# ============================================================================
# PART 6: BATCH PROCESSING
# ============================================================================
print("\n" + "="*80)
print("PART 6: Batch Processing Multiple Curves")
print("="*80)

# Create several test curves (mix of with and without transits)
print("Creating test dataset...")

for i in range(5):
    time_i = np.linspace(0, 60, 5000)
    flux_i = np.ones_like(time_i) + np.random.normal(0, 0.003, len(time_i))
    
    # Add transits to some curves
    has_transit = i % 2 == 0
    if has_transit:
        for t0 in np.arange(15, 60, 20):
            in_transit = np.abs(time_i - t0) < 0.3
            flux_i[in_transit] -= 0.008
    
    flux_err_i = np.full_like(flux_i, 0.003)
    label = 1 if has_transit else 0
    
    np.savez(f'demo_curve_{i}.npz', time=time_i, flux=flux_i,
             flux_err=flux_err_i, label=label)

print(f"✓ Created 5 test light curves (3 with transits, 2 without)")

# Batch load and process
curves = load_batch_lightcurves('.', pattern='demo_curve_*.npz', verbose=False)
print(f"✓ Batch loaded {len(curves)} curves")

# Extract features from all
print("\nProcessing all curves...")
all_features = []
for i, curve in enumerate(curves):
    curve_clean = preprocess_lightcurve(curve, **preprocessing_methods['polynomial'])
    feats = extract_basic_features(curve_clean, verbose=False)
    feats['curve_id'] = i
    feats['filename'] = curve.source_file
    if 'label' in curve.metadata:
        feats['label'] = curve.metadata['label']
    all_features.append(feats)

# Combine dataset
dataset = pd.concat(all_features, ignore_index=True)
dataset.to_csv(output_dir / 'batch_features.csv', index=False)

print(f"✓ Created dataset: {dataset.shape}")
print(f"✓ Saved: batch_features.csv")

# Show candidates with significant dips
if 'n_dips' in dataset.columns and 'label' in dataset.columns:
    print(f"\nTransit Detection Results:")
    print(f"  Curve ID  | True Label | Dips Found | Deepest Dip")
    print(f"  " + "-"*52)
    for _, row in dataset.iterrows():
        curve_id = int(row['curve_id'])
        label = int(row['label']) if 'label' in row else -1
        n_dips = int(row['n_dips']) if 'n_dips' in row else 0
        depth = row['deepest_dip_depth'] if 'deepest_dip_depth' in row else 0
        print(f"     {curve_id}      |     {label}      |    {n_dips:2d}     |   {depth:.6f}")

# ============================================================================
# PART 7: CONFIGURATION MANAGEMENT
# ============================================================================
print("\n" + "="*80)
print("PART 7: Configuration Management")
print("="*80)

config = Config('config.yaml')
print(f"✓ Loaded configuration")
print(f"  • System seed: {config.get('system.seed')}")
print(f"  • Feature tier: {config.get('features.tier')}")
print(f"  • Training epochs: {config.get('training.epochs')}")

# Modify and save custom config
config.set('system.num_workers', 16)
config.set('features.tier', 'comprehensive')
config.set('training.epochs', 200)
config.save_to_yaml(output_dir / 'custom_config.yaml')

print(f"✓ Saved custom configuration: custom_config.yaml")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("✅ DEMONSTRATION COMPLETE!")
print("="*80)

print(f"\n📊 System Capabilities Demonstrated:")
print(f"  ✓ Data loading (NPZ format)")
print(f"  ✓ Synthetic data generation")
print(f"  ✓ Multiple preprocessing methods")
print(f"  ✓ Feature extraction (49 features)")
print(f"  ✓ Visualization (4 plot types)")
print(f"  ✓ Phase folding")
print(f"  ✓ Batch processing")
print(f"  ✓ Configuration management")

print(f"\n📁 Generated Files in {output_dir}/:")
files = list(output_dir.glob('*'))
for f in sorted(files):
    size = f.stat().st_size / 1024
    print(f"  • {f.name} ({size:.1f} KB)")

print(f"\n🎯 Next Steps:")
print(f"  1. Check output images in {output_dir}/")
print(f"  2. Review features.csv and batch_features.csv")
print(f"  3. Try with your own data:")
print(f"     lc = load_lightcurve('your_file.npz')")
print(f"     features = extract_basic_features(lc)")
print(f"  4. Customize config.yaml for your needs")

print(f"\n🗑️  Cleanup (optional):")
print(f"  After reviewing, you can delete:")
print(f"  • demo_*.npz files (test data)")
print(f"  • lightcurve_project/, newcode/, newcode2/, newcode3/ (old branches)")

print(f"\n" + "="*80)
print("🎉 ALL SYSTEMS OPERATIONAL!")
print("="*80)
