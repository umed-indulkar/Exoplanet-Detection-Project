import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the labeled features
df = pd.read_csv('outputs/exo_features_labeled.csv')
print('Shape:', df.shape)
print('\nColumns (last 10):', list(df.columns[-10:]))

# Check if label exists
if 'label' not in df.columns:
    print('\n❌ ERROR: No "label" column found!')
    print('Available columns:', df.columns.tolist())
    exit(1)

# Check label values
print('\nLabel column found ✓')
print('Label distribution:')
print(df['label'].value_counts())
print('\nLabel dtype:', df['label'].dtype)
print('Unique values:', sorted(df['label'].unique()))

# Ensure labels are 0/1 numeric
if df['label'].dtype not in ['int64', 'int32', 'float64']:
    print('\n⚠ Converting labels to numeric (0/1)...')
    df['label'] = df['label'].astype(int)

# Verify no NaN in labels
if df['label'].isna().any():
    print(f'\n⚠ WARNING: {df["label"].isna().sum()} rows have NaN labels. Dropping them.')
    df = df.dropna(subset=['label'])

print(f'\nFinal dataset: {df.shape[0]} rows, {df.shape[1]} columns')
print(f'Positives (label=1): {(df["label"]==1).sum()}')
print(f'Negatives (label=0): {(df["label"]==0).sum()}')

# Split: 70% train, 15% val, 15% test (stratified by label)
train_df, temp_df = train_test_split(
    df, 
    test_size=0.30,  # 30% for val+test
    stratify=df['label'], 
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

print('\n✓ Splits saved:')
print(f'  - outputs/train.csv: {train_df.shape[0]} rows ({train_df.shape[0]/df.shape[0]*100:.1f}%)')
print(f'    Positives: {(train_df["label"]==1).sum()}, Negatives: {(train_df["label"]==0).sum()}')
print(f'  - outputs/val.csv: {val_df.shape[0]} rows ({val_df.shape[0]/df.shape[0]*100:.1f}%)')
print(f'    Positives: {(val_df["label"]==1).sum()}, Negatives: {(val_df["label"]==0).sum()}')
print(f'  - outputs/test.csv: {test_df.shape[0]} rows ({test_df.shape[0]/df.shape[0]*100:.1f}%)')
print(f'    Positives: {(test_df["label"]==1).sum()}, Negatives: {(test_df["label"]==0).sum()}')
