import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import psutil
import os
from tsfresh.feature_selection.relevance import calculate_relevance_table
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------- CONFIGURATION ----------
FILE_IN = r"D:\ppp\data\features_500\features_curve_500_pruned.csv"  # Update path if necessary
FILE_OUT_RANKINGS = r"D:\ppp\data\features_500\pca_rankings_curve_500.csv"

def print_ram():
    mem = psutil.virtual_memory()
    print(f"--- [RAM Status] Used: {mem.percent}% ({mem.used / (1024**3):.2f}GB / {mem.total / (1024**3):.2f}GB) ---")

# ---------- PHASE 1: INDIVIDUAL SIGNIFICANCE (TSFRESH) ----------
def analyze_significance(X, y, top_n=20):
    print("\n--- Phase 1: Calculating Individual Feature Significance ---")
    print_ram()
    
    # Calculate p-values for all features against the Label
    relevance_table = calculate_relevance_table(X, y, n_jobs=8)
    relevance_table = relevance_table.sort_values("p_value")
    
    plot_data = relevance_table.head(top_n).copy()
    # Convert p-value to a visual score (-log10). Higher is better.
    plot_data["score"] = -np.log10(plot_data["p_value"] + 1e-300)

    plt.figure(figsize=(12, 8))
    sns.barplot(data=plot_data, x="score", y="feature", palette="viridis")
    plt.title(f"Top {top_n} Features (Statistical Significance)", fontsize=15)
    plt.xlabel("Significance Score [-log10(p-value)]")
    plt.tight_layout()
    plt.savefig("significance_ranking.png")
    plt.show()

# ---------- PHASE 2: GLOBAL PCA DIAGNOSTICS ----------
def analyze_pca(X, feature_names):
    print("\n--- Phase 2: Running Principal Component Analysis (PCA) ---")
    
    # Standardization is mandatory before PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA()
    pca.fit(X_scaled)
    
    # 1. SCREE PLOT (EXPLAINED VARIANCE)
    plt.figure(figsize=(14, 5))
    exp_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)
    
    plt.subplot(1, 2, 1)
    plt.bar(range(1, 21), exp_var[:20], alpha=0.7, align='center', label='Individual variance')
    plt.step(range(1, 21), cum_var[:20], where='mid', label='Cumulative variance', color='red')
    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('Principal Component Index (Top 20)')
    plt.title('Scree Plot: How much data is captured?')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cum_var) + 1), cum_var, color='red')
    plt.axhline(y=0.90, color='black', linestyle='--', label='90% Variance Threshold')
    plt.title('Total Variance Captured vs Components')
    plt.xlabel('Number of Components')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("pca_variance.png")
    plt.show()

    # 2. FEATURE RANKING (MEAN ABSOLUTE LOADING)
    n_top_pcs = 10
    loadings = pd.DataFrame(
        pca.components_[:n_top_pcs].T,
        columns=[f'PC{i+1}' for i in range(n_top_pcs)],
        index=feature_names
    )
    # Calculate global importance across the top 10 PCs
    loadings['Global_Importance'] = loadings.abs().mean(axis=1)
    loadings_ranked = loadings.sort_values(by='Global_Importance', ascending=False)
    
    loadings_ranked.to_csv(FILE_OUT_RANKINGS)
    print(f"✅ Full feature PCA rankings saved to: {FILE_OUT_RANKINGS}")

    # 3. TOP 30 FEATURES BAR PLOT
    plt.figure(figsize=(10, 8))
    top_30 = loadings_ranked.head(30)
    sns.barplot(x=top_30['Global_Importance'], y=top_30.index, palette="magma")
    plt.title("Top 30 Features (Global PCA Weight)")
    plt.xlabel("Average Absolute Loading across Top 10 PCs")
    plt.tight_layout()
    plt.savefig("pca_feature_weights.png")
    plt.show()

    # 4. LOADINGS HEATMAP
    plt.figure(figsize=(12, 10))
    heatmap_data = top_30.drop(columns=['Global_Importance'])
    sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, cbar_kws={'label': 'Loading Weight'})
    plt.title("Feature/Component Relationship Heatmap (Top 30)")
    plt.xlabel("Principal Components")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig("pca_heatmap.png")
    plt.show()

# ---------- MAIN EXECUTION ----------
if __name__ == '__main__':
    if not os.path.exists(FILE_IN):
        print(f"❌ Error: Could not find {FILE_IN}")
        exit()

    print(f"📂 Loading data...")
    df = pd.read_csv(FILE_IN)
    
    # Memory Management: Convert float64 to float32
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
        
    # Handle Label column generically (case-insensitive check)
    label_col = [c for c in df.columns if c.upper() == 'LABEL'][0]
    
    X = df.drop(columns=[label_col])
    y = df[label_col]
    feature_names = X.columns
    
    del df
    gc.collect()
    
    print(f"✅ Memory-Safe Data Ready: {X.shape[1]} features loaded.")
    
    try:
        analyze_significance(X, y, top_n=20)
        analyze_pca(X, feature_names)
        print("\n✨ All Diagnostics Complete! Images saved to your folder.")
    except Exception as e:
        print(f"❌ Error during diagnostics: {e}")