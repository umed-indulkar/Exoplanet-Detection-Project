"""
Complete pipeline:
1. Extract labels from filenames (positive_*.npz = 1, negative_*.npz = 0)
2. Create complete metadata CSV
3. Split into 70% train / 15% val / 15% test
"""
import pandas as pd
import os
from sklearn.model_selection import train_test_split

print('='*70)
print('STEP 1: Extract labels from filenames')
print('='*70)

# Load features
print('\nLoading features...')
feats = pd.read_csv('outputs/exo_features.csv', low_memory=False)
print(f'✓ Loaded: {feats.shape}')

# Extract filename from source
feats['filename'] = feats['source'].apply(lambda x: os.path.basename(x))

# Extract label from filename prefix
def get_label_from_filename(filename):
    """positive_*.npz → 1, negative_*.npz → 0"""
    if filename.startswith('positive_'):
        return 1
    elif filename.startswith('negative_'):
        return 0
    else:
        return None

feats['label'] = feats['filename'].apply(get_label_from_filename)

# Check results
has_label = feats['label'].notna().sum()
no_label = feats['label'].isna().sum()

print(f'\n✓ Label extraction:')
print(f'  WITH label: {has_label} files')
print(f'  WITHOUT label: {no_label} files')

if no_label > 0:
    print('\n  ⚠ Files without positive/negative prefix (will be dropped):')
    unknown = feats[feats['label'].isna()]['filename'].head(10).tolist()
    for u in unknown:
        print(f'    - {u}')
    feats = feats[feats['label'].notna()]
    print(f'\n  ✓ Kept only labeled files: {len(feats)}')

# Ensure label is integer
feats['label'] = feats['label'].astype(int)

print(f'\n✓ Label distribution:')
print(f'  Positive (label=1): {(feats["label"]==1).sum()} ({(feats["label"]==1).sum()/len(feats)*100:.1f}%)')
print(f'  Negative (label=0): {(feats["label"]==0).sum()} ({(feats["label"]==0).sum()/len(feats)*100:.1f}%)')

# Save complete metadata
metadata = feats[['filename', 'label']].copy()
metadata.to_csv('data/ExoplanetDataset/raw_metadata_complete.csv', index=False)
print(f'\n✓ Saved: data/ExoplanetDataset/raw_metadata_complete.csv ({len(metadata)} rows)')

# Drop filename column, keep label with features
feats = feats.drop(columns=['filename'])

# Save full labeled features
feats.to_csv('outputs/exo_features_labeled_full.csv', index=False)
print(f'✓ Saved: outputs/exo_features_labeled_full.csv ({feats.shape})')

print('\n' + '='*70)
print('STEP 2: Split into Train (70%) / Val (15%) / Test (15%)')
print('='*70)

# Stratified split to maintain label distribution
train_df, temp_df = train_test_split(
    feats, 
    test_size=0.30,  # 30% for val+test
    stratify=feats['label'], 
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df, 
    test_size=0.50,  # 50% of 30% = 15% total
    stratify=temp_df['label'], 
    random_state=42
)

# Save splits
train_df.to_csv('outputs/train.csv', index=False)
val_df.to_csv('outputs/val.csv', index=False)
test_df.to_csv('outputs/test.csv', index=False)

print(f'\n✓ TRAIN SET: {train_df.shape} ({len(train_df)/len(feats)*100:.1f}%)')
print(f'  - Positives (label=1): {(train_df["label"]==1).sum()}')
print(f'  - Negatives (label=0): {(train_df["label"]==0).sum()}')
print(f'  - File: outputs/train.csv')

print(f'\n✓ VAL SET: {val_df.shape} ({len(val_df)/len(feats)*100:.1f}%)')
print(f'  - Positives (label=1): {(val_df["label"]==1).sum()}')
print(f'  - Negatives (label=0): {(val_df["label"]==0).sum()}')
print(f'  - File: outputs/val.csv')

print(f'\n✓ TEST SET: {test_df.shape} ({len(test_df)/len(feats)*100:.1f}%)')
print(f'  - Positives (label=1): {(test_df["label"]==1).sum()}')
print(f'  - Negatives (label=0): {(test_df["label"]==0).sum()}')
print(f'  - File: outputs/test.csv')

print('\n' + '='*70)
print('✅ DATASET READY FOR TRAINING!')
print('='*70)
print(f'Total samples: {len(feats)}')
print(f'  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}')
print(f'  Split: {len(train_df)/len(feats)*100:.0f}% / {len(val_df)/len(feats)*100:.0f}% / {len(test_df)/len(feats)*100:.0f}%')

print(f'\n📋 NEXT STEPS:')
print(f'\n1️⃣ Train RandomForest (CPU):')
print(f'   python -m exodet.cli train --features outputs/train.csv --target label --model rf --output runs/exo_rf.joblib')

print(f'\n2️⃣ Evaluate on validation:')
print(f'   python -m exodet.cli evaluate --model runs/exo_rf.joblib --features outputs/val.csv --target label')

print(f'\n3️⃣ Train Siamese Network (GPU):')
print(f'   python -m exodet.cli train-siamese --features outputs/train.csv --target label --epochs 10 --embedding 32 --device cuda --output runs/exo_siamese.pt')

print(f'\n4️⃣ Evaluate Siamese on validation:')
print(f'   python -m exodet.cli evaluate-siamese --model runs/exo_siamese.pt --features outputs/val.csv --target label --device cuda')

print('\n' + '='*70)
