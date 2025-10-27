"""
Fix labels and split dataset into 70% train / 15% val / 15% test
"""
import pandas as pd
import os
from sklearn.model_selection import train_test_split

print('='*60)
print('STEP 1: Load and fix labels')
print('='*60)

# Load features
print('Loading features...')
feats = pd.read_csv('outputs/exo_features.csv', low_memory=False)
print(f'✓ Features loaded: {feats.shape}')

# Load metadata with labels
meta = pd.read_csv('data/ExoplanetDataset/raw_metadata.csv')
print(f'✓ Metadata loaded: {meta.shape}')

# Extract clean filename from source column
feats['filename'] = feats['source'].apply(lambda x: os.path.basename(x))

print(f'\nSample filenames from features:')
for f in feats['filename'].head(5):
    print(f'  - {f}')

print(f'\nSample filenames from metadata:')
for f in meta['filename'].head(5):
    print(f'  - {f}')

# Merge on filename
print('\nMerging features with metadata labels...')
merged = feats.merge(meta[['filename', 'label', 'target']], on='filename', how='left')
print(f'✓ Merged: {merged.shape}')

# Check label coverage
has_label = merged['label'].notna().sum()
no_label = merged['label'].isna().sum()
print(f'\n  Rows WITH labels: {has_label}')
print(f'  Rows WITHOUT labels: {no_label}')

if no_label > 0:
    print(f'\n⚠ WARNING: {no_label} rows have no labels (will be dropped)')
    print('  Sample files with no labels:')
    missing = merged[merged['label'].isna()]['filename'].head(5).tolist()
    for m in missing:
        print(f'    - {m}')
    
    # Drop rows without labels
    merged = merged.dropna(subset=['label'])
    print(f'\n✓ Kept only labeled rows: {merged.shape}')

# Ensure label is integer 0 or 1
merged['label'] = merged['label'].astype(int)

# Verify label values
unique_labels = sorted(merged['label'].unique())
print(f'\n✓ Label values: {unique_labels}')
if set(unique_labels) != {0, 1}:
    print(f'  ⚠ WARNING: Expected [0, 1], got {unique_labels}')

print(f'\nLabel distribution:')
print(f'  Positive (label=1): {(merged["label"]==1).sum()} ({(merged["label"]==1).sum()/len(merged)*100:.1f}%)')
print(f'  Negative (label=0): {(merged["label"]==0).sum()} ({(merged["label"]==0).sum()/len(merged)*100:.1f}%)')

# Drop filename column (no longer needed)
merged = merged.drop(columns=['filename'])

# Save full labeled dataset
merged.to_csv('outputs/exo_features_labeled_full.csv', index=False)
print(f'\n✓ Saved: outputs/exo_features_labeled_full.csv ({merged.shape})')

print('\n' + '='*60)
print('STEP 2: Split into Train (70%) / Val (15%) / Test (15%)')
print('='*60)

# Stratified split
train_df, temp_df = train_test_split(
    merged, 
    test_size=0.30,  # 30% for val+test
    stratify=merged['label'], 
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df, 
    test_size=0.50,  # 50% of 30% = 15% of total
    stratify=temp_df['label'], 
    random_state=42
)

# Save splits
train_df.to_csv('outputs/train.csv', index=False)
val_df.to_csv('outputs/val.csv', index=False)
test_df.to_csv('outputs/test.csv', index=False)

print(f'\n✓ Train set: {train_df.shape} ({train_df.shape[0]/len(merged)*100:.1f}%)')
print(f'  - Positives: {(train_df["label"]==1).sum()}')
print(f'  - Negatives: {(train_df["label"]==0).sum()}')
print(f'  - Saved: outputs/train.csv')

print(f'\n✓ Val set: {val_df.shape} ({val_df.shape[0]/len(merged)*100:.1f}%)')
print(f'  - Positives: {(val_df["label"]==1).sum()}')
print(f'  - Negatives: {(val_df["label"]==0).sum()}')
print(f'  - Saved: outputs/val.csv')

print(f'\n✓ Test set: {test_df.shape} ({test_df.shape[0]/len(merged)*100:.1f}%)')
print(f'  - Positives: {(test_df["label"]==1).sum()}')
print(f'  - Negatives: {(test_df["label"]==0).sum()}')
print(f'  - Saved: outputs/test.csv')

print('\n' + '='*60)
print('✅ DONE: Dataset ready for training!')
print('='*60)
print(f'Total samples: {len(merged)}')
print(f'Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}')
print(f'\nNext steps:')
print(f'1. Train baseline: python -m exodet.cli train --features outputs/train.csv --target label --model rf --output runs/rf.joblib')
print(f'2. Evaluate: python -m exodet.cli evaluate --model runs/rf.joblib --features outputs/val.csv --target label')
print(f'3. Train Siamese: python -m exodet.cli train-siamese --features outputs/train.csv --target label --epochs 10 --device cuda --output runs/siamese.pt')
