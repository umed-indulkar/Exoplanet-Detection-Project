import pandas as pd
import os
import numpy as np

# --- PATHS ---
DATA_DIR = r"D:\ppp\data\dataset_500"
FILES_TO_CLEAN = ["raw_curve_500_head.csv"]

def clean_flat_rows(file_name):
    file_path = os.path.join(DATA_DIR, file_name)
    
    if not os.path.exists(file_path):
        print(f"Skipping: {file_name} not found.")
        return

    print(f"Processing {file_name}...")
    
    # 1. Load data
    df = pd.read_csv(file_path, header=None)
    initial_count = len(df)

    # 2. Extract flux columns (everything except the first column)
    # We force them to numeric. Any strings like 'CANDIDATE' in the flux 
    # area will become NaN, preventing the TypeError.
    flux_data = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

    # 3. Identify "Dead" rows
    # We check for Standard Deviation of 0 OR if the entire row is NaNs
    is_flat = (flux_data.std(axis=1) == 0) | (flux_data.isnull().all(axis=1))

    # 4. Filter out the flat rows
    df_clean = df[~is_flat]
    final_count = len(df_clean)
    removed = initial_count - final_count

    # 5. Save the cleaned file
    df_clean.to_csv(file_path, index=False, header=False)
    
    print(f"Done! Removed {removed} flat rows. New total: {final_count}")

if __name__ == "__main__":
    for f in FILES_TO_CLEAN:
        clean_flat_rows(f)