import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# =============================
# 1️⃣ Load Original Dataset
# =============================
data = pd.read_csv("lightcurve_features.csv")  # replace with your actual path

# =============================
# 2️⃣ Handle Missing Values & inf
# =============================

# Fill missing (empty) cells with 0 — or use median/mean if numeric stability matters
data = data.fillna(0)

# Replace infinite values with finite bounds
data = data.replace([np.inf, -np.inf], [1e6, -1e6])

# =============================
# 3️⃣ Drop Non-essential Columns
# =============================
drop_cols = ["kic_id", "filename"]
data = data.drop(columns=[c for c in drop_cols if c in data.columns], errors="ignore")



# =============================
# 5️⃣ Split into Train / Validation / Test
# =============================
train_temp_df, test_df = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)
train_df, val_df = train_test_split(train_temp_df, test_size=0.25, random_state=42, shuffle=True)
# → Train = 60%, Validation = 20%, Test = 20%

# =============================
# 6️⃣ Save Cleaned Datasets
# =============================
train_df.to_csv("train_dataset.csv", index=False)
val_df.to_csv("validation_dataset.csv", index=False)
test_df.to_csv("test_dataset.csv", index=False)

print("✅ Datasets created successfully:")
print(f"Train shape: {train_df.shape}")
print(f"Validation shape: {val_df.shape}")
print(f"Test shape: {test_df.shape}")

print("\n🔍 Label distribution:")
print(data["label"].value_counts())

print("\n⚙️ Data cleaning summary:")
print(f"Total inf values handled: {(np.isinf(data.values)).sum()}")
print(f"Total missing values (after fill): {data.isnull().sum().sum()}")
