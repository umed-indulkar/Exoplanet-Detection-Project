import pandas as pd
import matplotlib.pyplot as plt
import os

def check_label_distribution(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        return

    print(f"\n{'='*40}")
    print(f"🔍 AUDITING: {os.path.basename(file_path)}")
    print(f"{'='*40}")
    
    try:
        # 1. Detect column name (handles 'Label', 'LABEL', or 'label')
        temp_df = pd.read_csv(file_path, nrows=1)
        label_col = [c for c in temp_df.columns if c.upper() == 'LABEL']
        
        if not label_col:
            print("❌ Error: Could not find a column named 'LABEL'.")
            return
        
        target = label_col[0]

        # 2. Load only the target column to save RAM
        labels = pd.read_csv(file_path, usecols=[target])[target]
        
        # 3. Calculate Stats
        counts = labels.value_counts().sort_index()
        percentages = labels.value_counts(normalize=True).sort_index() * 100
        
        print("\n[📊 DISTRIBUTION REPORT]")
        for label, count in counts.items():
            name = "Planet (Positive)" if label == 1 else "Star/Noise (Negative)"
            print(f"  • {name} [{label}]: {count:>5} rows ({percentages[label]:>6.2f}%)")
            
        print(f"\n  TOTAL ROWS: {len(labels)}")

        # 4. Health Warnings
        if 1 not in counts:
            print("\n🚨 ALARM: Zero Planets (Label 1) found! Your model cannot learn.")
        elif percentages[1] < 5:
            print("\n⚠️  WARNING: Extreme Imbalance (<5% planets). Your model may ignore the minority class.")
        elif 40 <= percentages[1] <= 60:
            print("\n✅ HEALTHY: Excellent balance for training.")

        # 5. Optional: Quick Visualization
        plt.figure(figsize=(6, 4))
        counts.plot(kind='bar', color=['skyblue', 'salmon'])
        plt.title(f"Class Balance: {os.path.basename(file_path)}")
        plt.xticks([0, 1], ['Noise (0)', 'Planet (1)'], rotation=0)
        plt.ylabel("Row Count")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == '__main__':
    # Update these paths to your generated splits
    FILES_TO_CHECK = [
        r"D:\ppp\data\features\train_1.csv",
        r"D:\ppp\data\features\test_1.csv",
        r"D:\ppp\data\features\train_balanced.csv",
        r"D:\ppp\data\features\test_balanced.csv"
    ]

    for file in FILES_TO_CHECK:
        if os.path.exists(file):
            check_label_distribution(file)
        else:
            print(f"Skipping: {file} (File not found yet)")