"""
Data Organization - Pipeline Team
==================================

Moved organize_dataset_complete.py to pipeline/data_organizer.py
for better modular structure.
"""

import pandas as pd
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def organize_dataset():
    """
    Complete dataset organization:
    1. Load exo_features_labeled.csv and add label column (1=positive, 0=negative)
    2. Split 70% train / 15% val / 15% test
    3. Copy NPZ files to data/ExoplanetDataset/train, val, test folders
    4. Create metadata CSV for each split
    5. Update raw_metadata.csv with complete labels
    """
    print('='*70)
    print('COMPLETE DATASET ORGANIZATION')
    print('='*70)

    # STEP 1: Load features and extract labels from filenames
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
    before = len(df)
    df = df.dropna(subset=['label'])
    after = len(df)
    print(f'✓ Labeled samples: {after} (dropped {before-after} without labels)')

    # STEP 2: Split into train/val/test
    print('\n📊 STEP 2: Split into train/val/test (70/15/15)')
    print('-'*70)

    # Remove rows with NaN in features
    feature_cols = [c for c in df.columns if c not in ['source', 'filename', 'target', 'label']]
    df_clean = df.dropna(subset=feature_cols)
    print(f'✓ Clean samples (no NaN features): {len(df_clean)}')

    # Stratified split
    train_df, temp_df = train_test_split(
        df_clean, 
        test_size=0.30,  # 30% for val+test
        stratify=df_clean['label'], 
        random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,  # 50% of remaining = 15% total
        stratify=temp_df['label'],
        random_state=42
    )

    print(f'✓ Train: {len(train_df)} samples')
    print(f'✓ Validation: {len(val_df)} samples') 
    print(f'✓ Test: {len(test_df)} samples')

    # STEP 3: Copy NPZ files to train/val/test folders
    print('\n📁 STEP 3: Copy NPZ files to train/val/test folders')
    print('-'*70)

    base_dir = Path('data/ExoplanetDataset')
    raw_dir = base_dir / 'raw'
    splits = ['train', 'val', 'test']
    split_dfs = [train_df, val_df, test_df]

    for split, split_df in zip(splits, split_dfs):
        split_dir = base_dir / split
        split_dir.mkdir(exist_ok=True)
        print(f'✓ {split_dir}/ - {len(split_df)} files')

        copied = 0
        for filename in split_df['filename']:
            src = raw_dir / filename
            dst = split_dir / filename
            if src.exists():
                shutil.copy2(src, dst)
                copied += 1
            else:
                print(f'⚠ Missing: {src}')

        print(f'  Copied {copied}/{len(split_df)} files')

    # STEP 4: Save feature CSVs for each split
    print('\n💾 STEP 4: Save feature CSVs for each split')
    print('-'*70)

    for split, split_df in zip(splits, split_dfs):
        output_path = f'outputs/{split}.csv'
        split_df.to_csv(output_path, index=False)
        print(f'✓ Saved: {output_path} ({len(split_df)} rows)')

    # STEP 5: Create metadata CSVs for each split
    print('\n📋 STEP 5: Create metadata CSVs for each split')
    print('-'*70)

    for split, split_df in zip(splits, split_dfs):
        # Create metadata with filename, target, label
        metadata = split_df[['filename', 'target', 'label']].copy()
        metadata_path = f'data/ExoplanetDataset/{split}/metadata.csv'
        metadata.to_csv(metadata_path, index=False)
        print(f'✓ Saved: {metadata_path}')

    # STEP 6: Update raw_metadata.csv with complete labels
    print('\n🔄 STEP 6: Update raw_metadata.csv with complete labels')
    print('-'*70)

    # Create complete metadata from our processed data
    complete_metadata = df_clean[['filename', 'target', 'label']].copy()
    complete_metadata_path = 'data/ExoplanetDataset/raw_metadata.csv'
    complete_metadata.to_csv(complete_metadata_path, index=False)
    print(f'✓ Updated: {complete_metadata_path} ({len(complete_metadata)} entries)')

    # STEP 7: Save complete labeled features
    print('\n✅ STEP 7: Save complete labeled features')
    print('-'*70)

    labeled_path = 'outputs/exo_features_labeled_full.csv'
    df_clean.to_csv(labeled_path, index=False)
    print(f'✓ Saved: {labeled_path} ({len(df_clean)} rows, {len(df_clean.columns)} columns)')

    print('\n' + '='*70)
    print('🎉 DATASET ORGANIZATION COMPLETE!')
    print('='*70)
    print(f'Total processed: {len(df_clean)} samples')
    print(f'Features per sample: {len(feature_cols)}')
    print(f'Files saved in: outputs/ and data/ExoplanetDataset/')
    print('='*70)

if __name__ == '__main__':
    organize_dataset()
