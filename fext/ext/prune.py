import pandas as pd
import numpy as np
import gc
import psutil
from tsfresh import select_features
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# ---------- SETTINGS ----------
INPUT_PATH = r"D:\ppp\data\features\features_raw.csv"
OUTPUT_PATH = r"D:\ppp\data\features\features_pruned.csv"
CORR_THRESHOLD = 0.95  # If two features are 95% similar, keep only one

def print_ram_usage():
    mem = psutil.virtual_memory()
    print(f"--- [RAM Status] Used: {mem.percent}% ({mem.used / (1024**3):.2f}GB / {mem.total / (1024**3):.2f}GB) ---")

def remove_redundancy_advanced(df, corr_threshold=0.95):
    print(f"🔄 Starting Redundancy Pruning (Threshold: {corr_threshold})...")
    df = df.copy(deep=True) 
    
    # 1. Remove constant features (Variance = 0)
    variances = df.var()
    df = df.loc[:, variances > 0]
    print(f"✅ Removed constants. Features remaining: {df.shape[1]}")

    if df.shape[1] <= 2: return df

    # 2. Calculate Correlation Matrix
    # We use a smaller dtype to save RAM during the matrix calculation
    print(f"📊 Calculating correlation for {df.shape[1]} features...")
    corr_df = df.corr().abs().fillna(0)
    
    # 3. Hierarchical Clustering to find redundant groups
    # Force writable numpy array to avoid the 'read-only' error
    corr_matrix = np.array(corr_df.values, copy=True)
    dist_matrix = 1 - corr_matrix
    np.fill_diagonal(dist_matrix, 0)
    
    condensed_dist = squareform(dist_matrix, checks=False)
    linkage_matrix = linkage(condensed_dist, method='average')
    clusters = fcluster(linkage_matrix, t=1 - corr_threshold, criterion='distance')

    # 4. Pick one representative feature from each cluster
    selected_features = []
    for cluster_id in np.unique(clusters):
        members = df.columns[clusters == cluster_id]
        # Pick the feature with the highest variance (most signal)
        best_feat = df[members].var().idxmax()
        selected_features.append(best_feat)
        
    return df[selected_features]

def run_pruning():
    print_ram_usage()
    
    # 1. Load Data
    print(f"📂 Loading: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    
    # 2. Separate Label
    if 'Label' not in df.columns:
        print("❌ Error: 'Label' column not found in features file!")
        return
        
    y = df['Label']
    X = df.drop(columns=['Label'])
    
    # 3. TSFRSH Relevance Selection
    # This removes features that have no statistical relationship with the Label
    print("\n--- Phase 1: TSFresh Relevance Selection ---")
    X_selected = select_features(X, y)
    print(f"✅ Statistical selection complete. Features: {X_selected.shape[1]}")
    
    # Clear memory
    del X
    gc.collect()

    # 4. Redundancy Pruning
    # This removes features that are too similar to each other
    print("\n--- Phase 2: Redundancy Pruning ---")
    X_final = remove_redundancy_advanced(X_selected, corr_threshold=CORR_THRESHOLD)
    
    # 5. Final Save
    print(f"\n✨ SUCCESS! Features pruned from {df.shape[1]-1} down to {X_final.shape[1]}")
    X_final['Label'] = y.values
    X_final.to_csv(OUTPUT_PATH, index=False)
    print(f"📂 Saved pruned features to: {OUTPUT_PATH}")
    print_ram_usage()

if __name__ == "__main__":
    run_pruning()