import pandas as pd
import numpy as np
import os

def audit_exoplanet_dataset(csv_path):
    if not os.path.exists(csv_path):
        print(f"❌ Error: File not found at {csv_path}")
        return

    print(f"🔬 Auditing Exoplanet Dataset: {csv_path}")
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Identify Columns
    label_col = 'Label'
    flux_cols = [f'flux_{i}' for i in range(1, 81)]
    
    if label_col not in df.columns:
        print(f"❌ Error: '{label_col}' column missing. Check your headers!")
        return

    print(f"✅ Detected {len(df)} rows, 80 Flux bins, and 1 Label column.")

    # --- 1. THE "ZERO-VALUE" LEAK ---
    # In transit data, 0.0 is usually an error (NaN fill). 
    # If one class has more 0s, the model learns 'Missing Data' instead of 'Planets'.
    print("\n--- 📏 STRUCTURAL AUDIT (Zero Check) ---")
    df['zero_count'] = (df[flux_cols] == 0).sum(axis=1)
    zero_leak = df.groupby(label_col)['zero_count'].mean()
    print(f"Avg 0.0 values per row:\n{zero_leak}")
    if abs(zero_leak.get(0, 0) - zero_leak.get(1, 0)) > 2:
        print("❌ LEAKAGE DETECTED: One class has significantly more missing data (0s).")
    else:
        print("✅ Data density is consistent across classes.")

    # --- 2. MAGNITUDE BIAS (The 1.0 Baseline) ---
    # Since we normalized, the mean should be very close to 1.0 for everyone.
    print("\n--- 📈 MAGNITUDE AUDIT (Mean Flux) ---")
    df['mean_flux'] = df[flux_cols].mean(axis=1)
    mean_bias = df.groupby(label_col)['mean_flux'].mean()
    print(f"Avg Flux Level (Should be ~1.0):\n{mean_bias}")
    if abs(mean_bias.get(0, 1) - mean_bias.get(1, 1)) > 0.05:
        print("❌ LEAKAGE: Significant brightness difference detected between classes.")

    # --- 3. SEQUENCE AUDIT (Shuffling Check) ---
    # If your file is [1,1,1... 0,0,0], a model might learn the row order.
    print("\n--- 🗂️ SEQUENCE AUDIT (Shuffle Check) ---")
    first_half_mean = df[label_col].iloc[:len(df)//2].mean()
    second_half_mean = df[label_col].iloc[len(df)//2:].mean()
    diff = abs(first_half_mean - second_half_mean)
    print(f"Label Mean - 1st Half: {first_half_mean:.2f} | 2nd Half: {second_half_mean:.2f}")
    if diff > 0.3:
        print("❌ LEAKAGE: Data is sorted by Label. SHUFFLE YOUR CSV!")
    else:
        print("✅ Data sequence appears randomized.")

    # --- 4. CORRELATION PEAK (The "Magic" Bin) ---
    # No single bin should be a perfect predictor. The model must look at the whole curve.
    print("\n--- 🎯 RAW STEP CORRELATION ---")
    # Convert Label to numeric for correlation calculation
    df[label_col] = pd.to_numeric(df[label_col], errors='coerce')
    correlations = df[flux_cols].corrwith(df[label_col]).abs()
    max_corr = correlations.max()
    max_bin = correlations.idxmax()
    
    if max_corr > 0.7:
        print(f"❌ EXTREME LEAKAGE: Flux bin '{max_bin}' has {max_corr:.2f} correlation.")
        print("   This usually means a non-physical artifact is present in that bin.")
    else:
        print(f"✅ Max raw correlation is {max_corr:.2f} in {max_bin} (Physical range).")

    # --- 5. VARIANCE CHECK ---
    # Dead rows (all 1.0) make the model lazy.
    print("\n--- ❄️ VARIANCE AUDIT ---")
    df['std_flux'] = df[flux_cols].std(axis=1)
    dead_rows = (df['std_flux'] == 0).sum()
    if dead_rows > 0:
        print(f"⚠️ WARNING: Found {dead_rows} 'Flat' rows (0 variance). Clean these out!")
    else:
        print("✅ All rows contain signal variance.")

if __name__ == '__main__':
    # Update this to your headered file path
    RAW_DATA = r"D:\ppp\data\dataset\training_with_headers.csv"
    audit_exoplanet_dataset(RAW_DATA)