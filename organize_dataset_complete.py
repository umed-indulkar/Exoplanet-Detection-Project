"""
Complete dataset organization:
1. Load exo_features_labeled.csv and add label column (1=positive, 0=negative)
2. Split 70% train / 15% val / 15% test
3. Copy NPZ files to data/ExoplanetDataset/train, val, test folders
4. Create metadata CSV for each split
5. Update raw_metadata.csv with complete labels
"""
import pandas as pd
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

print('='*70)
print('COMPLETE DATASET ORGANIZATION')
print('='*70)

# ============================================================================
# STEP 1: Load features and extract labels from filenames
# ============================================================================
print('\n📂 STEP 1: Load features and extract labels from filenames')
print('-'*70)

features_file = 'outputs/exo_features_labeled.csv'
if not os.path.exists(features_file):
    features_file = 'outputs/exo_features.csv'
    print(f'⚠ exo_features_labeled.csv not found, using: {features_file}')

print(f'Loading: {features_file}')
df = pd.read_csv(features_file, low_memory=False)
print(f'✓ Loaded: {df.shape}')

# Extract filename from source
df['filename'] = df['source'].apply(lambda x: os.path.basename(x))

# Extract label from filename prefix
def get_label_from_filename(filename):
    """positive_*.npz → 1, negative_*.npz → 0"""
    if filename.startswith('positive_'):
        return 1
    elif filename.startswith('negative_'):
        return 0
    else:
        return None

# Create or update label column
df['label'] = df['filename'].apply(get_label_from_filename)

# Drop files without labels
unlabeled = df['label'].isna().sum()
if unlabeled > 0:
    print(f'⚠ Dropping {unlabeled} files without positive/negative prefix')
    df = df[df['label'].notna()]

df['label'] = df['label'].astype(int)

print(f'\n✓ Dataset after labeling: {df.shape}')
print(f'  Positives (label=1): {(df["label"]==1).sum()} ({(df["label"]==1).sum()/len(df)*100:.1f}%)')
print(f'  Negatives (label=0): {(df["label"]==0).sum()} ({(df["label"]==0).sum()/len(df)*100:.1f}%)')

# ============================================================================
# STEP 2: Split into train/val/test (70%/15%/15%)
# ============================================================================
print(f'\n📊 STEP 2: Split into train (70%) / val (15%) / test (15%)')
print('-'*70)

# Stratified split
train_df, temp_df = train_test_split(
    df, 
    test_size=0.30,  # 30% for val+test
    stratify=df['label'], 
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df, 
    test_size=0.50,  # 50% of 30% = 15% total
    stratify=temp_df['label'], 
    random_state=42
)

print(f'\n✓ TRAIN: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)')
print(f'  Positives: {(train_df["label"]==1).sum()}, Negatives: {(train_df["label"]==0).sum()}')

print(f'\n✓ VAL: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)')
print(f'  Positives: {(val_df["label"]==1).sum()}, Negatives: {(val_df["label"]==0).sum()}')

print(f'\n✓ TEST: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)')
print(f'  Positives: {(test_df["label"]==1).sum()}, Negatives: {(test_df["label"]==0).sum()}')

# ============================================================================
# STEP 3: Copy NPZ files to train/val/test folders
# ============================================================================
print(f'\n📁 STEP 3: Copy NPZ files to train/val/test folders')
print('-'*70)

base_path = Path('data/ExoplanetDataset')
train_dir = base_path / 'train'
val_dir = base_path / 'val'
test_dir = base_path / 'test'
raw_dir = base_path / 'raw'

# Create directories
train_dir.mkdir(exist_ok=True, parents=True)
val_dir.mkdir(exist_ok=True, parents=True)
test_dir.mkdir(exist_ok=True, parents=True)

def copy_files(df_split, target_dir, split_name):
    """Copy NPZ files to target directory"""
    print(f'\n  Copying {len(df_split)} files to {target_dir}...')
    copied = 0
    missing = 0
    
    for idx, row in df_split.iterrows():
        src_path = Path(row['source'])
        
        # If source path is not absolute, try raw folder
        if not src_path.exists():
            src_path = raw_dir / row['filename']
        
        if src_path.exists():
            dest_path = target_dir / row['filename']
            if not dest_path.exists():  # Don't overwrite
                shutil.copy2(src_path, dest_path)
            copied += 1
        else:
            missing += 1
            if missing <= 5:  # Show first 5 missing files
                print(f'    ⚠ Missing: {row["filename"]}')
    
    print(f'  ✓ {split_name}: Copied {copied} files, Missing {missing}')
    return copied, missing

train_copied, train_missing = copy_files(train_df, train_dir, 'TRAIN')
val_copied, val_missing = copy_files(val_df, val_dir, 'VAL')
test_copied, test_missing = copy_files(test_df, test_dir, 'TEST')

# ============================================================================
# STEP 4: Save feature CSVs for each split
# ============================================================================
print(f'\n💾 STEP 4: Save feature CSVs')
print('-'*70)

# Save to outputs folder (for training)
train_df.to_csv('outputs/train.csv', index=False)
val_df.to_csv('outputs/val.csv', index=False)
test_df.to_csv('outputs/test.csv', index=False)

print(f'✓ Saved: outputs/train.csv ({train_df.shape})')
print(f'✓ Saved: outputs/val.csv ({val_df.shape})')
print(f'✓ Saved: outputs/test.csv ({test_df.shape})')

# ============================================================================
# STEP 5: Create metadata CSVs for each split
# ============================================================================
print(f'\n📋 STEP 5: Create metadata CSVs')
print('-'*70)

def create_metadata(df_split, filename):
    """Create metadata CSV with filename, target, label, length"""
    meta = df_split[['filename', 'label']].copy()
    # Add target and length columns (placeholder values)
    meta['target'] = meta['filename'].apply(lambda x: 
        x.replace('positive_', 'Kepler-').replace('negative_', 'KIC ').split('.')[0]
    )
    meta['length'] = 0  # Placeholder
    meta = meta[['filename', 'target', 'label', 'length']]
    meta.to_csv(filename, index=False)
    return meta

train_meta = create_metadata(train_df, base_path / 'train_metadata.csv')
val_meta = create_metadata(val_df, base_path / 'val_metadata.csv')
test_meta = create_metadata(test_df, base_path / 'test_metadata.csv')

print(f'✓ Saved: {base_path}/train_metadata.csv ({len(train_meta)} rows)')
print(f'✓ Saved: {base_path}/val_metadata.csv ({len(val_meta)} rows)')
print(f'✓ Saved: {base_path}/test_metadata.csv ({len(test_meta)} rows)')

# ============================================================================
# STEP 6: Update raw_metadata.csv with complete labels
# ============================================================================
print(f'\n📝 STEP 6: Update raw_metadata.csv')
print('-'*70)

complete_meta = df[['filename', 'label']].copy()
complete_meta['target'] = complete_meta['filename'].apply(lambda x: 
    x.replace('positive_', 'Kepler-').replace('negative_', 'KIC ').split('.')[0]
)
complete_meta['length'] = 0  # Placeholder
complete_meta = complete_meta[['filename', 'target', 'label', 'length']]
complete_meta = complete_meta.sort_values('filename').reset_index(drop=True)

raw_meta_path = base_path / 'raw_metadata.csv'
complete_meta.to_csv(raw_meta_path, index=False)
print(f'✓ Updated: {raw_meta_path} ({len(complete_meta)} rows)')

# ============================================================================
# STEP 7: Save complete labeled features
# ============================================================================
print(f'\n💾 STEP 7: Save complete labeled features')
print('-'*70)

# Drop filename column, keep everything else
df_final = df.drop(columns=['filename'])
df_final.to_csv('outputs/exo_features_labeled_complete.csv', index=False)
print(f'✓ Saved: outputs/exo_features_labeled_complete.csv ({df_final.shape})')

# ============================================================================
# SUMMARY
# ============================================================================
print('\n' + '='*70)
print('✅ DATASET ORGANIZATION COMPLETE!')
print('='*70)

print(f'\n📊 DATASET SUMMARY:')
print(f'  Total samples: {len(df)}')
print(f'  Positives: {(df["label"]==1).sum()} | Negatives: {(df["label"]==0).sum()}')

print(f'\n📁 FILE ORGANIZATION:')
print(f'  data/ExoplanetDataset/train/ → {train_copied} files ({len(train_df)} features)')
print(f'  data/ExoplanetDataset/val/ → {val_copied} files ({len(val_df)} features)')
print(f'  data/ExoplanetDataset/test/ → {test_copied} files ({len(test_df)} features)')

print(f'\n📋 METADATA FILES:')
print(f'  data/ExoplanetDataset/raw_metadata.csv → Complete ({len(complete_meta)} rows)')
print(f'  data/ExoplanetDataset/train_metadata.csv → Train split')
print(f'  data/ExoplanetDataset/val_metadata.csv → Val split')
print(f'  data/ExoplanetDataset/test_metadata.csv → Test split')

print(f'\n💾 FEATURE CSVs (for training):')
print(f'  outputs/train.csv → {train_df.shape}')
print(f'  outputs/val.csv → {val_df.shape}')
print(f'  outputs/test.csv → {test_df.shape}')
print(f'  outputs/exo_features_labeled_complete.csv → {df_final.shape}')

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
