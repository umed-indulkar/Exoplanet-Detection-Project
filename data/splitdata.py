import pandas as pd
from sklearn.model_selection import train_test_split

# Load your original CSV
data = pd.read_csv("lightcurve_features.csv")  # replace with your CSV path

# Split into train and validation (e.g., 80%-20%)
train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)

# Save validation CSV
val_df.to_csv("validation_dataset.csv", index=False)

# Optionally, overwrite train CSV if you want only remaining data in train
train_df.to_csv("train_dataset.csv", index=False)

print("Validation CSV created: validation_dataset.csv")
print("Updated train CSV: train_dataset.csv")
