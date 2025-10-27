"""
Update raw_metadata.csv with complete label information from exo_features_labeled.csv
"""
import pandas as pd
import os

print('='*70)
print('Updating raw_metadata.csv from exo_features_labeled.csv')
print('='*70)

# Check which labeled features file exists
if os.path.exists('outputs/exo_features_labeled_full.csv'):
    features_file = 'outputs/exo_features_labeled_full.csv'
elif os.path.exists('outputs/exo_features_labeled.csv'):
    features_file = 'outputs/exo_features_labeled.csv'
elif os.path.exists('outputs/exo_features.csv'):
    features_file = 'outputs/exo_features.csv'
    print('\n⚠ Using exo_features.csv (no labels yet, will extract from filename)')
else:
    print('❌ ERROR: No features CSV found in outputs/')
    exit(1)

print(f'\n📂 Loading: {features_file}')
feats = pd.read_csv(features_file, low_memory=False)
print(f'✓ Loaded: {feats.shape}')

# Extract filename from source column
feats['filename'] = feats['source'].apply(lambda x: os.path.basename(x))

# If label column doesn't exist, extract from filename
if 'label' not in feats.columns:
    print('\n⚠ No label column found. Extracting from filename prefix...')
    def get_label_from_filename(filename):
        if filename.startswith('positive_'):
            return 1
        elif filename.startswith('negative_'):
            return 0
        else:
            return None
    feats['label'] = feats['filename'].apply(get_label_from_filename)
    feats = feats[feats['label'].notna()]  # Keep only labeled files
    feats['label'] = feats['label'].astype(int)
    print(f'✓ Extracted labels from {len(feats)} filenames')

# If target column doesn't exist, try to extract from existing metadata or set as Unknown
if 'target' not in feats.columns:
    # Try to load existing metadata to get target names
    if os.path.exists('data/ExoplanetDataset/raw_metadata.csv'):
        old_meta = pd.read_csv('data/ExoplanetDataset/raw_metadata.csv')
        if 'target' in old_meta.columns:
            print('\n📋 Merging target names from existing metadata...')
            feats = feats.merge(old_meta[['filename', 'target']], on='filename', how='left')
        else:
            feats['target'] = 'Unknown'
    else:
        feats['target'] = 'Unknown'

# Fill missing targets
if feats['target'].isna().any():
    feats['target'] = feats['target'].fillna('Unknown')

# Get length from source if not present (estimate from file, or use placeholder)
if 'length' not in feats.columns:
    feats['length'] = 0  # Placeholder, actual length would need to load each npz

# Create complete metadata
metadata = feats[['filename', 'target', 'label', 'length']].copy()

# Ensure label is int
metadata['label'] = metadata['label'].astype(int)

# Sort by filename for readability
metadata = metadata.sort_values('filename').reset_index(drop=True)

print(f'\n✓ Created metadata with {len(metadata)} entries')
print(f'  Columns: {list(metadata.columns)}')
print(f'  Positives (label=1): {(metadata["label"]==1).sum()}')
print(f'  Negatives (label=0): {(metadata["label"]==0).sum()}')

# Save updated metadata
output_path = 'data/ExoplanetDataset/raw_metadata.csv'
metadata.to_csv(output_path, index=False)
print(f'\n✅ Updated: {output_path}')

# Show sample
print(f'\nSample (first 10 rows):')
print(metadata.head(10).to_string(index=False))

print(f'\nSample (last 10 rows):')
print(metadata.tail(10).to_string(index=False))

print('\n' + '='*70)
print('✅ DONE! raw_metadata.csv now has complete labels')
print('='*70)
