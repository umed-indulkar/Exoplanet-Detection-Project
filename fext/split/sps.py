import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# --- CONFIG ---
FILE_IN = r"D:\ppp\data\features\features_pruned.csv"
TRAIN_OUT = r"D:\ppp\data\features\train_1.csv"
TEST_OUT = r"D:\ppp\data\features\test_1.csv"
N_CLUSTERS = 5 # Groups the data into 5 unique "shapes"

def clustered_stratified_split():
    print("🚀 Loading data for Clustered Split...")
    df = pd.read_csv(FILE_IN)
    
    # Standardize label column name
    label_col = [c for c in df.columns if c.upper() == 'LABEL'][0]
    
    X = df.drop(columns=[label_col])
    y = df[label_col]

    print(f"📊 Running K-Means Clustering (K={N_CLUSTERS})...")
    # Standardize before clustering so high-variance features don't dominate
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Create a hybrid "Stratification Key" (e.g., Label_1_Cluster_3)
    # This forces the split to balance BOTH the Label AND the physical shape of the data
    df['Stratify_Key'] = df[label_col].astype(str) + "_" + clusters.astype(str)
    
    print("✂️ Splitting data based on Cluster and Label...")
    # Because some hybrid clusters might be too small to split perfectly, 
    # we filter out any group that has only 1 sample.
    valid_counts = df['Stratify_Key'].value_counts()
    valid_keys = valid_counts[valid_counts > 1].index
    df_valid = df[df['Stratify_Key'].isin(valid_keys)]
    
    train_df, test_df = train_test_split(
        df_valid, 
        test_size=0.2, 
        random_state=42, 
        stratify=df_valid['Stratify_Key']
    )
    
    # Clean up the temporary key before saving
    train_df = train_df.drop(columns=['Stratify_Key'])
    test_df = test_df.drop(columns=['Stratify_Key'])
    
    train_df.to_csv(TRAIN_OUT, index=False)
    test_df.to_csv(TEST_OUT, index=False)
    
    print(f"✅ Success! Data split perfectly across {N_CLUSTERS} structural groups.")
    print(f"   Train Size: {len(train_df)}")
    print(f"   Test Size:  {len(test_df)}")

if __name__ == '__main__':
    clustered_stratified_split()