import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# --- CONFIG ---
FILE_IN = r"D:\ppp\data\features\features_pruned.csv"
TRAIN_OUT = r"D:\ppp\data\features\train_balanced.csv"
TEST_OUT = r"D:\ppp\data\features\test_balanced.csv"

def balanced_50_50_split():
    print("🚀 Loading data for Balanced 50/50 Split...")
    df = pd.read_csv(FILE_IN)
    
    label_col = [c for c in df.columns if c.upper() == 'LABEL'][0]
    
    # 1. Separate the classes
    df_planets = df[df[label_col] == 1]
    df_noise = df[df[label_col] == 0]
    
    count_planets = len(df_planets)
    count_noise = len(df_noise)
    print(f"📉 Initial Distribution -> Planets: {count_planets} | Noise: {count_noise}")
    
    # 2. Find the minority class and downsample the majority class
    minority_count = min(count_planets, count_noise)
    
    if count_planets > count_noise:
        print("⚖️ Downsampling Planets to match Noise...")
        df_planets = resample(df_planets, replace=False, n_samples=minority_count, random_state=42)
    elif count_noise > count_planets:
        print("⚖️ Downsampling Noise to match Planets...")
        df_noise = resample(df_noise, replace=False, n_samples=minority_count, random_state=42)
    else:
        print("⚖️ Data is already perfectly balanced!")

    # 3. Recombine and Shuffle
    df_balanced = pd.concat([df_planets, df_noise]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"🎯 New Balanced Distribution -> Total: {len(df_balanced)} (50% Planets, 50% Noise)")
    
    # 4. Standard Stratified Split
    train_df, test_df = train_test_split(
        df_balanced, 
        test_size=0.2, 
        random_state=42, 
        stratify=df_balanced[label_col] # Ensures the 50/50 ratio holds in both Train and Test
    )
    
    train_df.to_csv(TRAIN_OUT, index=False)
    test_df.to_csv(TEST_OUT, index=False)
    
    print(f"✅ Success! Train and Test sets saved.")
    print(f"   Train Set: {len(train_df)} rows")
    print(f"   Test Set:  {len(test_df)} rows")

if __name__ == '__main__':
    balanced_50_50_split()