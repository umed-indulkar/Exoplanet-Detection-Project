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
import os

# --- 1. DATA SPLITTING & PREP ---
def prepare_raw_data(input_path):
    print(f"📂 Loading raw data: {input_path}")
    df = pd.read_csv(input_path)
    
    # Detect Label column and Flux columns
    label_col = [c for c in df.columns if c.upper() == 'LABEL'][0]
    flux_cols = [c for c in df.columns if 'flux_' in c.lower()]
    
    # Stratified Split (80% Train, 20% Test)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[label_col]
    )
    
    return train_df, test_df, flux_cols, label_col

class RawFluxDataset(Dataset):
    def __init__(self, dataframe, flux_cols, label_col, scaler=None):
        self.X = dataframe[flux_cols].values.astype(np.float32)
        self.y = dataframe[label_col].values.astype(np.float32)
        
        if scaler is None:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X)
        else:
            self.scaler = scaler
            self.X = self.scaler.transform(self.X)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# --- 2. DEEP ARCHITECTURE (80 -> 256 -> 128 -> 64 -> 32 -> 1) ---
class DeepRawFNN(nn.Module):
    def __init__(self, input_dim):
        super(DeepRawFNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1) # Output Logits
        )

    def forward(self, x):
        return self.net(x).squeeze()

# --- 3. MAIN TRAINING BLOCK ---
if __name__ == '__main__':
    RAW_FILE = r"D:\ppp\data\dataset_500\raw_curve_500_cleaned.csv"
    
    # Step 1: Split data in memory
    train_df, test_df, flux_cols, label_col = prepare_raw_data(RAW_FILE)

    # Step 2: Create Datasets (Sharing the scaler to prevent leakage)
    train_ds = RawFluxDataset(train_df, flux_cols, label_col)
    test_ds = RawFluxDataset(test_df, flux_cols, label_col, scaler=train_ds.scaler)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # Step 3: Setup Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepRawFNN(input_dim=len(flux_cols)).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # Step 4: Training Loop
    print(f"\n🧠 Training Deep Raw FNN on {len(flux_cols)} flux bins...")
    epochs = 100
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_raw_fnn.pth")

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Step 5: Evaluation
    print("\n🏁 Final Evaluation on Unseen Test Data...")
    model.load_state_dict(torch.load("best_raw_fnn.pth"))
    model.eval()
    
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            outputs = model(x)
            preds = (torch.sigmoid(outputs) > 0.5).float().cpu()
            y_true.extend(y.tolist())
            y_pred.extend(preds.tolist())

    print(classification_report(y_true, y_pred))

    # Confusion Matrix
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Oranges')
    plt.title("Deep Raw Flux FNN Results")
    plt.show()