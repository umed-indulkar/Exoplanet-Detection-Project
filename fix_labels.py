import pandas as pd
import os

# Load features
print('Loading features...')
feats = pd.read_csv('outputs/exo_features.csv', low_memory=False)
print(f'Features: {feats.shape}')

# Load metadata
meta = pd.read_csv('data/ExoplanetDataset/raw_metadata.csv')
print(f'Metadata: {meta.shape}')

# Extract filename from source column (handle both / and \ path separators)
feats['filename'] = feats['source'].apply(lambda x: os.path.basename(x))
print(f'\nSample source paths:\n{feats["source"].head(3).tolist()}')
print(f'\nExtracted filenames:\n{feats["filename"].head(3).tolist()}')

# Merge with metadata on filename
print('\nMerging with metadata...')
merged = feats.merge(meta[['filename', 'label', 'target']], on='filename', how='left')
print(f'Merged shape: {merged.shape}')

# Check label coverage
missing_labels = merged['label'].isna().sum()
print(f'\nRows with labels: {merged["label"].notna().sum()}')
print(f'Rows missing labels: {missing_labels}')

if missing_labels > 0:
    print('\n⚠ Some files not found in metadata:')
    missing_files = merged[merged['label'].isna()]['filename'].head(10).tolist()
    print(missing_files)
    print('\nMetadata filenames (first 10):')
    print(meta['filename'].head(10).tolist())
    
# Drop rows without labels
if missing_labels > 0:
    print(f'\nDropping {missing_labels} rows without labels...')
    merged = merged.dropna(subset=['label'])

# Ensure label is int (0 or 1)
merged['label'] = merged['label'].astype(int)

# Drop filename column (no longer needed)
merged = merged.drop(columns=['filename'])

# Save
merged.to_csv('outputs/exo_features_labeled_fixed.csv', index=False)
print(f'\n✓ Saved outputs/exo_features_labeled_fixed.csv: {merged.shape}')
print(f'  Positives (label=1): {(merged["label"]==1).sum()}')
print(f'  Negatives (label=0): {(merged["label"]==0).sum()}')
