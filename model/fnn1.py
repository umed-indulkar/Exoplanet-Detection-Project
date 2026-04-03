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

# --- 1. DATA PREPARATION ---
class ExoplanetDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        # Ensure Label column is removed from features
        label_col = [c for c in df.columns if c.upper() == 'LABEL'][0]
        
        self.X = df.drop(columns=[label_col]).values.astype(np.float32)
        self.y = df[label_col].values.astype(np.float32)
        
        # Scaling is mandatory for Neural Networks
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# --- 2. THE ARCHITECTURE ---
class SimpleFFNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleFFNN, self).__init__()
        # Layer 1: Input -> Hidden (128 neurons)
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2) # Prevents overfitting
        
        # Layer 2: Hidden -> Hidden (64 neurons)
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        
        # Layer 3: Output (1 neuron for Binary Classification)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.relu2(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# --- 3. TRAINING LOOP ---
def train_model():
    # Load Data
    train_data = ExoplanetDataset(r"D:\ppp\data\features\train_balanced.csv")
    test_data = ExoplanetDataset(r"D:\ppp\data\features\test_balanced.csv")
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    model = SimpleFFNN(input_size=train_data.X.shape[1])
    criterion = nn.BCELoss() # Binary Cross Entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"🧠 Training FFNN on {train_data.X.shape[1]} features...")
    
    epochs = 30
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

    # --- 4. EVALUATION ---
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs).squeeze()
            preds = (outputs > 0.5).float() # Threshold at 0.5
            all_preds.extend(preds.numpy())
            all_targets.extend(targets.numpy())

    print("\n[FEED-FORWARD NEURAL NETWORK REPORT]")
    print(classification_report(all_targets, all_preds))

    # Confusion Matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(confusion_matrix(all_targets, all_preds), annot=True, fmt='g', cmap='Purples')
    plt.title("FFNN Confusion Matrix")
    plt.show()

if __name__ == '__main__':
    train_model()