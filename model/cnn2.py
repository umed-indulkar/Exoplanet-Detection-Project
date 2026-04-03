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

# --- 1. DATA LOADING & SPLITTING ---
def prepare_raw_cnn_data(file_path):
    print(f"📂 Loading Raw Flux: {file_path}")
    df = pd.read_csv(file_path)
    
    label_col = [c for c in df.columns if c.upper() == 'LABEL'][0]
    flux_cols = [c for c in df.columns if 'FLUX_' in c.upper()]
    
    # Stratified Split
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[label_col]
    )
    return train_df, test_df, flux_cols, label_col

class RawCNN1DDataset(Dataset):
    def __init__(self, dataframe, flux_cols, label_col, scaler=None):
        self.X_raw = dataframe[flux_cols].values.astype(np.float32)
        self.y = dataframe[label_col].values.astype(np.float32)
        
        # Scaling is critical for CNN convergence
        if scaler is None:
            self.scaler = StandardScaler()
            self.X_scaled = self.scaler.fit_transform(self.X_raw)
        else:
            self.scaler = scaler
            self.X_scaled = self.scaler.transform(self.X_raw)

        # Reshape to (Batch, Channels, Length) -> (Batch, 1, 80)
        self.X = self.X_scaled[:, np.newaxis, :]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# --- 2. CNN ARCHITECTURE (Optimized for 80 Bins) ---
class ExoplanetCNN1D(nn.Module):
    def __init__(self):
        super(ExoplanetCNN1D, self).__init__()
        
        # Layer 1: Detects local edges/dips
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2) # 80 -> 40
        )
        
        # Layer 2: Detects complex shapes
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2) # 40 -> 20
        )
        
        # Fully Connected Classifier
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 20, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1) # Logits output
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc(x)
        return x.squeeze()

# --- 3. TRAINING ENGINE ---
if __name__ == '__main__':
    RAW_PATH = r"D:\ppp\data\dataset\raw_cleaned.csv"
    
    train_df, test_df, flux_cols, label_col = prepare_raw_cnn_data(RAW_PATH)
    
    train_ds = RawCNN1DDataset(train_df, flux_cols, label_col)
    test_ds = RawCNN1DDataset(test_df, flux_cols, label_col, scaler=train_ds.scaler)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

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
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/50 | Loss: {loss.item():.4f}")

    # --- 4. EVALUATION ---
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