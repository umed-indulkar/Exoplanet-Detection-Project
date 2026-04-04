import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. DATA PREPARATION (BALANCED + SPLIT) ---
def prepare_raw_cnn_data(file_path):
    print(f"📂 Loading Raw Flux: {file_path}")
    df = pd.read_csv(file_path)
    
    label_col = [c for c in df.columns if c.upper() == 'LABEL'][0]
    flux_cols = [c for c in df.columns if 'FLUX_' in c.upper()]

    # --- BALANCE DATA ---
    df_major = df[df[label_col] == 0]
    df_minor = df[df[label_col] == 1]

    min_size = min(len(df_major), len(df_minor))

    df_major_down = df_major.sample(min_size, random_state=42)
    df_minor_down = df_minor.sample(min_size, random_state=42)

    df_balanced = pd.concat([df_major_down, df_minor_down])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Balanced Dataset: {len(df_balanced)} samples ({min_size} per class)")

    # Stratified split
    train_df, test_df = train_test_split(
        df_balanced, test_size=0.2, random_state=42, stratify=df_balanced[label_col]
    )

    return train_df, test_df, flux_cols, label_col


# --- 2. DATASET CLASS ---
class RawCNN1DDataset(Dataset):
    def __init__(self, dataframe, flux_cols, label_col, scaler=None):
        self.X_raw = dataframe[flux_cols].values.astype(np.float32)
        self.y = dataframe[label_col].values.astype(np.float32)

        if scaler is None:
            self.scaler = StandardScaler()
            self.X_scaled = self.scaler.fit_transform(self.X_raw)
        else:
            self.scaler = scaler
            self.X_scaled = self.scaler.transform(self.X_raw)

        # Shape: (B, 1, 500)
        self.X = self.X_scaled[:, np.newaxis, :]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


# --- 3. CNN MODEL (FOR 500 FEATURES) ---
class ExoplanetCNN1D(nn.Module):
    def __init__(self):
        super(ExoplanetCNN1D, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)   # 500 → 250
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)   # 250 → 125
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 125, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc(x)
        return x.squeeze()


# --- 4. TRAINING + EVALUATION ---
if __name__ == '__main__':
    RAW_PATH = r"D:\ppp\data\dataset_500\raw_curve_500_cleaned.csv"

    train_df, test_df, flux_cols, label_col = prepare_raw_cnn_data(RAW_PATH)

    # --- sanity check ---
    print(f"Feature count: {len(flux_cols)} (expected 500)")

    train_ds = RawCNN1DDataset(train_df, flux_cols, label_col)
    test_ds = RawCNN1DDataset(test_df, flux_cols, label_col, scaler=train_ds.scaler)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # --- shape check ---
    for x, y in train_loader:
        print(f"Input shape: {x.shape}")  # expected: [B, 1, 500]
        break

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExoplanetCNN1D().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    print(f"\n🛰️ Training 1D-CNN on Raw Flux ({device})...")

    for epoch in range(50):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/50 | Loss: {loss.item():.4f}")

    # --- EVALUATION ---
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            outputs = model(x)
            preds = (torch.sigmoid(outputs) > 0.5).float().cpu()

            y_true.extend(y.tolist())
            y_pred.extend(preds.tolist())

    print("\n[RAW FLUX 1D-CNN REPORT]")
    print(classification_report(y_true, y_pred))

    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='YlOrRd')
    plt.title("1D-CNN Raw Flux Results")
    plt.show()