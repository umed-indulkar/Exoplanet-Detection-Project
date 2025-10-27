import pandas as pd

df = pd.read_csv('outputs/features_combined.csv')
print('Shape:', df.shape)
cols = df.columns.tolist()
print('Total columns:', len(cols))
print('\nLast 10 columns:', cols[-10:])
print('\nHas "label" column?', 'label' in cols)
print('Has "source" column?', 'source' in cols)

# Check if there's an empty column
empty_cols = [c for c in cols if df[c].isna().all()]
if empty_cols:
    print('\nEmpty columns found:', empty_cols)

# Show first few values of last column
print('\nLast column name:', cols[-1])
print('Last column first 5 values:')
print(df[cols[-1]].head())
