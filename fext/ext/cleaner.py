import pandas as pd
import os
import numpy as np

# --- PATHS ---
INPUT_PATH  = r"D:\ppp\data\dataset_500\raw_curve_500_head.csv" 
OUTPUT_PATH = r"D:\ppp\data\dataset_500\training_curve_500_cleaned.csv"

def clean_data():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Input file {INPUT_PATH} not found.")
        return

    # Load data - assuming headers are already added from the previous step
    # If no headers yet, use: names=['kepid', 'label'] + [f'flux_{i}' for i in range(1, 501)]
    df = pd.read_csv(INPUT_PATH)
    initial_count = len(df)
    
    # 1. Identify flux columns (flux_1 to flux_500)
    flux_cols = [c for c in df.columns if 'flux' in c.lower()]
    
    if not flux_cols:
        print("Error: No flux columns detected. Check your CSV headers!")
        return

    # 2. Handle NaNs: Replace any remaining NaNs with 1.0 (the median baseline)
    # Instead of dropping them, we fill them so we don't lose potential data
    df[flux_cols] = df[flux_cols].fillna(1.0)

    # 3. Remove "Dead" Rows (True flat lines)
    # We check if the Max value minus the Min value is essentially zero.
    # This is more reliable than 'std' for normalized astronomical data.
    diff = df[flux_cols].max(axis=1) - df[flux_cols].min(axis=1)
    df_cleaned = df[diff > 0].copy() 

    # 4. Remove Extreme Outliers (Optional but recommended)
    # If a flux value is 0 or negative (impossible for normalized PDCSAP), it's bad data.
    df_cleaned = df_cleaned[(df_cleaned[flux_cols] > 0).all(axis=1)]

    print(f"--- CLEANING REPORT ---")
    print(f"Initial Rows: {initial_count}")
    print(f"Removed:      {initial_count - len(df_cleaned)} rows")
    print(f"Final Count:  {len(df_cleaned)} rows")
    
    # Save the result
    df_cleaned.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved cleaned dataset to: {OUTPUT_PATH}")

if __name__ == "__main__":
    clean_data()