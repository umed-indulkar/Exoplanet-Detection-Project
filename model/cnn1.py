import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. DATASET WITH 3D RESHAPING ---
class FeatureCNNDataset(Dataset):
    def __init__(self, csv_file, scaler=None, fit_scaler=False):
        df = pd.read_csv(csv_file)
        label_col = [c for c in df.columns if c.upper() == 'LABEL'][0]

        self.X_raw = df.drop(columns=[label_col]).values.astype(np.float32)
        self.y = df[label_col].values.astype(np.float32)

        # Standardize
        if scaler is None:
            self.scaler = StandardScaler()
            self.X_scaled = self.scaler.fit_transform(self.X_raw)
        else:
            self.scaler = scaler
            self.X_scaled = self.scaler.fit_transform(self.X_raw) if fit_scaler else self.scaler.transform(self.X_raw)

        # CNN 1D requires: (Batch, Channels, Length)
        # We treat the 777 features as a single channel sequence
        self.X = self.X_scaled[:, np.newaxis, :] 

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# --- 2. 1D-CNN ARCHITECTURE ---
class FeatureCNN1D(nn.Module):
    def __init__(self, input_size):
        super(FeatureCNN1D, self).__init__()
        
        # Conv Layer 1: Looks at local feature clusters
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Conv Layer 2: Higher level patterns
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        # Flatten and Dense
        # Calculate flattened size: (Input_size / 2 from first pool) * 64 channels
        self.flatten_size = (input_size // 2) * 64
        
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1) # Logits
        )

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)
        return x.squeeze()

# --- 3. TRAINING LOOP ---
def train_cnn():
    TRAIN_CSV = r"D:\ppp\data\features_500\train_balanced_curve_500.csv"
    TEST_CSV = r"D:\ppp\data\features_500\test_balanced_curve_500.csv"

    scaler = StandardScaler()
    train_ds = FeatureCNNDataset(TRAIN_CSV, scaler, fit_scaler=True)
    test_ds = FeatureCNNDataset(TEST_CSV, scaler, fit_scaler=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureCNN1D(input_size=train_ds.X_raw.shape[1]).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005)
    
    print(f"📡 Training 1D-CNN on {train_ds.X_raw.shape[1]} Features...")

    for epoch in range(50):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/50 | Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            outputs = model(x)
            preds = (torch.sigmoid(outputs) > 0.5).float().cpu()
            y_true.extend(y.tolist())
            y_pred.extend(preds.tolist())

    print("\n[1D-CNN FEATURE REPORT]")
    print(classification_report(y_true, y_pred))

    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Greens')
    plt.title("1D-CNN Feature Results")
    plt.show()

if __name__ == '__main__':
    train_cnn()