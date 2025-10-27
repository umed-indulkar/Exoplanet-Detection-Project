"""
PROCESS YOUR ACTUAL DATA
=========================

This script processes YOUR real light curve data from the old branches.
Automatically finds and processes all NPZ/CSV files.

Just run: python process_your_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from exodet import load_batch_lightcurves, preprocess_lightcurve, Config
from exodet.features import extract_basic_features
from exodet.visualization import plot_lightcurve, plot_comparison

print("="*70)
print("PROCESSING YOUR EXOPLANET DATA")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directories to search for data
DATA_DIRECTORIES = [
    'lightcurve_project/data',
    'newcode/data',
    'newcode2/data',
    # Add more directories if needed
]

# Output directory
OUTPUT_DIR = Path('my_results')
OUTPUT_DIR.mkdir(exist_ok=True)

# Load configuration
config = Config('config.yaml')

print(f"\n📁 Output directory: {OUTPUT_DIR}/")
print(f"⚙️  Using configuration: config.yaml")

# ============================================================================
# FIND AND LOAD DATA
# ============================================================================

print(f"\n🔍 Searching for data files...")

all_curves = []
for data_dir in DATA_DIRECTORIES:
    dir_path = Path(data_dir)
    if dir_path.exists():
        print(f"  Checking: {data_dir}/")
        
        # Try NPZ files
        npz_curves = load_batch_lightcurves(
            data_dir,
            pattern='*.npz',
            verbose=False
        )
        if npz_curves:
            print(f"    ✓ Found {len(npz_curves)} NPZ files")
            all_curves.extend(npz_curves)
        
        # Try CSV files
        csv_curves = load_batch_lightcurves(
            data_dir,
            pattern='*.csv',
            verbose=False
        )
        if csv_curves:
            print(f"    ✓ Found {len(csv_curves)} CSV files")
            all_curves.extend(csv_curves)
    else:
        print(f"    ✗ Directory not found: {data_dir}/")

if not all_curves:
    print(f"\n⚠️  No data files found!")
    print(f"\n💡 To use this script:")
    print(f"  1. Edit DATA_DIRECTORIES in this file")
    print(f"  2. Point to folders containing your .npz or .csv files")
    print(f"  3. Run again")
    print(f"\n  Or try the demo: python demo_complete_system.py")
    exit(0)

print(f"\n✓ Total curves found: {len(all_curves)}")

# ============================================================================
# PROCESS ALL CURVES
# ============================================================================

print(f"\n⚙️  Processing {len(all_curves)} light curves...")
print(f"  (This may take a few minutes)")

all_features = []
processed_count = 0
failed_count = 0

for i, lc in enumerate(all_curves):
    try:
        # Preprocess
        lc_clean = preprocess_lightcurve(lc, config=config['preprocessing'])
        
        # Extract features
        features = extract_basic_features(lc_clean, verbose=False)
        
        # Add metadata
        features['source_file'] = lc.source_file
        features['original_points'] = len(lc)
        features['processed_points'] = len(lc_clean)
        
        if 'label' in lc.metadata:
            features['label'] = lc.metadata['label']
        
        all_features.append(features)
        processed_count += 1
        
        # Progress indicator
        if (i + 1) % 10 == 0 or i == len(all_curves) - 1:
            print(f"  Progress: {i+1}/{len(all_curves)} ({100*(i+1)/len(all_curves):.0f}%)")
        
    except Exception as e:
        print(f"  ✗ Failed: {lc.source_file} - {str(e)}")
        failed_count += 1

# ============================================================================
# SAVE RESULTS
# ============================================================================

if all_features:
    print(f"\n💾 Saving results...")
    
    # Combine all features
    dataset = pd.concat(all_features, ignore_index=True)
    
    # Save to CSV
    output_file = OUTPUT_DIR / 'all_features.csv'
    dataset.to_csv(output_file, index=False)
    
    print(f"✓ Saved: {output_file}")
    print(f"  • Shape: {dataset.shape}")
    print(f"  • Features per curve: {len(dataset.columns)}")
    
    # Save summary statistics
    summary = dataset.describe()
    summary.to_csv(OUTPUT_DIR / 'feature_summary.csv')
    print(f"✓ Saved: feature_summary.csv")
    
    # If labels exist, show distribution
    if 'label' in dataset.columns:
        label_counts = dataset['label'].value_counts()
        print(f"\n📊 Label distribution:")
        for label, count in label_counts.items():
            print(f"  Label {label}: {count} curves")
        
        # Identify potential candidates
        if 'n_dips' in dataset.columns:
            candidates = dataset[dataset['n_dips'] > 3].copy()
            candidates = candidates.sort_values('deepest_dip_depth', ascending=False)
            
            if len(candidates) > 0:
                print(f"\n🎯 Top 10 transit candidates (most significant dips):")
                print(f"  File                                    Dips  Depth      Label")
                print(f"  " + "-"*65)
                
                for idx, row in candidates.head(10).iterrows():
                    filename = Path(row['source_file']).name
                    n_dips = int(row['n_dips'])
                    depth = row['deepest_dip_depth']
                    label = int(row['label']) if 'label' in row and pd.notna(row['label']) else '?'
                    print(f"  {filename:35s}  {n_dips:4d}  {depth:8.5f}   {label}")
                
                # Save candidates
                candidates.to_csv(OUTPUT_DIR / 'transit_candidates.csv', index=False)
                print(f"\n✓ Saved: transit_candidates.csv ({len(candidates)} candidates)")

# ============================================================================
# GENERATE SAMPLE PLOTS
# ============================================================================

print(f"\n📊 Generating sample visualizations...")

# Plot first few curves
n_plots = min(5, len(all_curves))

for i in range(n_plots):
    lc = all_curves[i]
    filename = Path(lc.source_file).stem
    
    try:
        # Preprocess
        lc_clean = preprocess_lightcurve(lc, config=config['preprocessing'])
        
        # Plot comparison
        plot_comparison(
            lc, lc_clean,
            title=f"Curve: {filename}",
            save_path=OUTPUT_DIR / f'plot_{i+1}_{filename}.png'
        )
        
        if i == 0:
            print(f"  ✓ Plotted first curve: {filename}")
    except Exception as e:
        print(f"  ✗ Plot failed for {filename}: {e}")

if n_plots > 1:
    print(f"  ✓ Plotted {n_plots} curves total")

# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n" + "="*70)
print(f"✅ PROCESSING COMPLETE!")
print(f"="*70)

print(f"\n📊 Summary:")
print(f"  • Curves found: {len(all_curves)}")
print(f"  • Successfully processed: {processed_count}")
print(f"  • Failed: {failed_count}")
print(f"  • Features extracted: {len(dataset.columns) if all_features else 0}")

print(f"\n📁 Results saved to: {OUTPUT_DIR}/")
print(f"  • all_features.csv - Complete feature dataset")
print(f"  • feature_summary.csv - Statistical summary")
if 'label' in dataset.columns and 'n_dips' in dataset.columns:
    print(f"  • transit_candidates.csv - Potential exoplanet candidates")
print(f"  • plot_*.png - Sample visualizations")

print(f"\n🎯 Next steps:")
print(f"  1. Review {OUTPUT_DIR}/all_features.csv")
print(f"  2. Check sample plots in {OUTPUT_DIR}/")
if 'label' in dataset.columns and 'n_dips' in dataset.columns:
    print(f"  3. Investigate candidates in transit_candidates.csv")
print(f"  4. Use features for machine learning")

print(f"\n" + "="*70)
