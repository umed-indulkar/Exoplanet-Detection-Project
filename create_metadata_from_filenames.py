"""
Create complete metadata CSV with labels extracted from filenames
positive_*.npz → label = 1
negative_*.npz → label = 0
"""
import pandas as pd
import os

print('='*60)
print('Creating complete metadata from filenames')
print('='*60)

# Load features
print('\nLoading features...')
feats = pd.read_csv('outputs/exo_features.csv', low_memory=False)
print(f'✓ Features loaded: {feats.shape}')

# Extract filename from source
feats['filename'] = feats['source'].apply(lambda x: os.path.basename(x))

# Extract label from filename prefix
def get_label_from_filename(filename):
    if filename.startswith('positive_'):
        return 1
    elif filename.startswith('negative_'):
        return 0
    else:
        return None  # Unknown

feats['label'] = feats['filename'].apply(get_label_from_filename)

# Check results
has_label = feats['label'].notna().sum()
no_label = feats['label'].isna().sum()

print(f'\n✓ Label extraction from filenames:')
print(f'  Files with label: {has_label}')
print(f'  Files without label: {no_label}')

if no_label > 0:
    print('\n  Files without positive/negative prefix:')
    unknown = feats[feats['label'].isna()]['filename'].tolist()
    for u in unknown[:10]:
        print(f'    - {u}')

# Filter to only labeled files
feats_labeled = feats[feats['label'].notna()].copy()
feats_labeled['label'] = feats_labeled['label'].astype(int)

print(f'\n✓ Kept only labeled files: {feats_labeled.shape}')
print(f'  Positives (label=1): {(feats_labeled["label"]==1).sum()}')
print(f'  Negatives (label=0): {(feats_labeled["label"]==0).sum()}')

# Create metadata CSV
metadata = feats_labeled[['filename', 'label']].copy()
metadata.to_csv('data/ExoplanetDataset/raw_metadata_complete.csv', index=False)
print(f'\n✓ Saved metadata: data/ExoplanetDataset/raw_metadata_complete.csv')
print(f'  Total files: {len(metadata)}')
print(f'  Columns: {list(metadata.columns)}')

# Save labeled features (drop filename column, keep label)
feats_labeled = feats_labeled.drop(columns=['filename'])
feats_labeled.to_csv('outputs/exo_features_labeled_full.csv', index=False)
print(f'\n✓ Saved labeled features: outputs/exo_features_labeled_full.csv')
print(f'  Shape: {feats_labeled.shape}')
print(f'  Positives: {(feats_labeled["label"]==1).sum()}')
print(f'  Negatives: {(feats_labeled["label"]==0).sum()}')

print('\n' + '='*60)
print('✅ DONE! All files now have labels (0 or 1)')
print('='*60)
