import pandas as pd
import os

INPUT_PATH = r"D:\ppp\data\dataset\training_with_headers.csv"
OUTPUT_PATH = r"D:\ppp\data\dataset\training_cleaned.csv"

def clean_data():
    if not os.path.exists(INPUT_PATH):
        print("Error: Input file not found.")
        return

    df = pd.read_csv(INPUT_PATH)
    initial_count = len(df)
    
    # Identify flux columns
    flux_cols = [c for c in df.columns if 'flux_' in c.lower()]
    
    # Remove rows with 0 variance (Flat lines)
    # We use a small epsilon (1e-9) to catch "nearly" flat lines
    df_cleaned = df[df[flux_cols].std(axis=1) > 1e-9].copy()
    
    # Remove rows that are entirely NaNs in the flux region
    df_cleaned = df_cleaned.dropna(subset=flux_cols, how='all')
    
    print(f"🧹 Cleaned Dataset: Removed {initial_count - len(df_cleaned)} dead rows.")
    df_cleaned.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    clean_data()