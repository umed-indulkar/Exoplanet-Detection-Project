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

# --- 1. DATA PREPARATION (Leakage-Proof) ---
class ExoplanetDataset(Dataset):
    def __init__(self, csv_file, scaler=None, fit_scaler=False):
        df = pd.read_csv(csv_file)
        # Case-insensitive label detection
        label_col = [c for c in df.columns if c.upper() == 'LABEL'][0]

        self.X = df.drop(columns=[label_col]).values.astype(np.float32)
        self.y = df[label_col].values.astype(np.float32)

        if scaler is None:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X)
        else:
            self.scaler = scaler
            if fit_scaler:
                self.X = self.scaler.fit_transform(self.X)
            else:
                self.X = self.scaler.transform(self.X)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# --- 2. DEEP ARCHITECTURE (256 -> 128 -> 64 -> 32 -> 1) ---
class DeepFFNN(nn.Module):
    def __init__(self, input_size):
        super(DeepFFNN, self).__init__()

        self.net = nn.Sequential(
            # Block 1
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Block 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Block 3
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # Block 4
            nn.Linear(64, 32),
            nn.ReLU(),

            # Output (Logits)
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze()

# --- 3. TRAINING ENGINE ---
def train_model():
    # Setup Paths
    TRAIN_CSV = r"D:\ppp\data\features_500\train_balanced_curve_500.csv"
    TEST_CSV = r"D:\ppp\data\features_500\test_balanced_curve_500.csv"

    # Shared Scaler
    scaler = StandardScaler()
    train_data = ExoplanetDataset(TRAIN_CSV, scaler, fit_scaler=True)
    test_data = ExoplanetDataset(TEST_CSV, scaler, fit_scaler=False)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepFFNN(input_size=train_data.X.shape[1]).to(device)

    # Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    
    # Scheduler: Reduces Learning Rate if loss stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    print(f"🚀 Training Deep FFNN on {device}...")
    print(f"📊 Features: {train_data.X.shape[1]} | Samples: {len(train_data)}")

    epochs = 500
    best_loss = float('inf')
    early_stop_patience = 30
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        scheduler.step(avg_train_loss)

        # Early Stopping Logic
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), "best_exoplanet_model.pth")
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_train_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if patience_counter >= early_stop_patience:
            print(f"🛑 Early stopping at epoch {epoch+1}")
            break

    # --- 4. FINAL EVALUATION ---
    print("\n🏁 Evaluation Stage...")
    model.load_state_dict(torch.load("best_exoplanet_model.pth"))
    model.eval()
    
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float().cpu()

            all_preds.extend(preds.numpy())
            all_targets.extend(targets.numpy())

    # --- 5. VISUALIZATION ---
    print("\n[FINAL PERFORMANCE REPORT]")
    print(classification_report(all_targets, all_preds))

    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(all_targets, all_preds)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Purples', 
                xticklabels=['Noise', 'Planet'], yticklabels=['Noise', 'Planet'])
    plt.title("Deep FFNN: Final Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

if __name__ == '__main__':
    train_model()