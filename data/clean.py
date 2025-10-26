import pandas as pd

# Load original CSV
df = pd.read_csv("validation_dataset.csv")

# Drop ID columns
df_clean = df.drop(columns=["kic_id", "filename"])

# Save cleaned CSV
df_clean.to_csv("validation_data.csv", index=False)
print("Cleaned dataset saved to clean_exoplanet_dataset.csv")