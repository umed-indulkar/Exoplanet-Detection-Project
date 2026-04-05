import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random

# --- 1. SIAMESE DATASET ---
class SiameseExoplanetDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        label_col = [c for c in df.columns if c.upper() == 'LABEL'][0]
        
        # Extract features and labels
        self.X_raw = df.drop(columns=[label_col]).values.astype(np.float32)
        self.y = df[label_col].values.astype(np.float32)
        
        # Standardize features
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X_raw)
        
        # Group indices by class for pairing
        self.positive_indices = np.where(self.y == 1)[0]
        self.negative_indices = np.where(self.y == 0)[0]
        
        print(f"📊 Data Info:")
        print(f"   - Input shape: {self.X_raw.shape}")
        print(f"   - Features: {self.X_raw.shape[1]}")
        print(f"   - Samples: {self.X_raw.shape[0]}")
        print(f"   - Positives: {len(self.positive_indices)}")
        print(f"   - Negatives: {len(self.negative_indices)}")
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # Create pairs: similar (same class) or dissimilar (different class)
        if random.random() > 0.5:
            # Similar pair
            target = 1.0  # Similar
            if self.y[idx] == 1:
                pair_idx = random.choice(self.positive_indices)
            else:
                pair_idx = random.choice(self.negative_indices)
        else:
            # Dissimilar pair
            target = 0.0  # Dissimilar
            if self.y[idx] == 1:
                pair_idx = random.choice(self.negative_indices)
            else:
                pair_idx = random.choice(self.positive_indices)
        
        return (torch.tensor(self.X[idx]), torch.tensor(self.X[pair_idx]), torch.tensor(target))

# --- 2. LIGHTWEIGHT ENCODER NETWORK ---
class LightweightEncoder(nn.Module):
    def __init__(self, input_size, embedding_dim=64):
        super(LightweightEncoder, self).__init__()
        
        # Feature encoder with bottleneck architecture
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)

# --- 3. SIAMESE NETWORK ---
class SiameseNetwork(nn.Module):
    def __init__(self, input_size, embedding_dim=64):
        super(SiameseNetwork, self).__init__()
        self.encoder = LightweightEncoder(input_size, embedding_dim)
        
    def forward(self, x1, x2):
        # Get embeddings for both inputs
        embedding1 = self.encoder(x1)
        embedding2 = self.encoder(x2)
        
        # L2 distance between embeddings
        distance = F.pairwise_distance(embedding1, embedding2, p=2)
        return distance
    
    def get_embedding(self, x):
        """Get single embedding for inference"""
        return self.encoder(x)

# --- 4. CONTRASTIVE LOSS ---
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, distance, target):
        # target = 1 for similar pairs, 0 for dissimilar pairs
        loss_similar = target * torch.pow(distance, 2)
        loss_dissimilar = (1 - target) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        loss = loss_similar + loss_dissimilar
        return torch.mean(loss)

# --- 5. TRAINING FUNCTION ---
def train_siamese_model():
    # Update paths for current structure
    TRAIN_CSV = "data/features/train_balanced.csv"
    TEST_CSV = "data/features/test_balanced.csv"
    
    print("🔄 Loading data for Siamese network...")
    train_dataset = SiameseExoplanetDataset(TRAIN_CSV)
    test_dataset = SiameseExoplanetDataset(TEST_CSV)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    input_size = train_dataset.X.shape[1]
    model = SiameseNetwork(input_size, embedding_dim=64)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n🎯 Model Architecture:")
    print(f"   - Input features: {input_size}")
    print(f"   - Embedding dimension: 64")
    print(f"   - Network: 151→128→64→64 (bottleneck)")
    
    # Training loop
    epochs = 30
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (x1, x2, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            distances = model(x1, x2)
            loss = criterion(distances, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    # --- 6. EVALUATION ---
    print("\n🔍 Evaluating Siamese Network...")
    model.eval()
    
    # Get embeddings for all test samples
    test_embeddings = []
    test_labels = []
    
    with torch.no_grad():
        # Use original test data (not pairs) for evaluation
        test_df = pd.read_csv(TEST_CSV)
        label_col = [c for c in test_df.columns if c.upper() == 'LABEL'][0]
        X_test = test_df.drop(columns=[label_col]).values.astype(np.float32)
        y_test = test_df[label_col].values.astype(np.float32)
        
        # Scale using training scaler
        X_test_scaled = train_dataset.scaler.transform(X_test)
        
        for i in range(len(X_test_scaled)):
            embedding = model.get_embedding(torch.tensor(X_test_scaled[i]).unsqueeze(0))
            test_embeddings.append(embedding.numpy().flatten())
            test_labels.append(y_test[i])
    
    test_embeddings = np.array(test_embeddings)
    test_labels = np.array(test_labels)
    
    # Simple classification using nearest neighbor in embedding space
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(test_embeddings, test_labels)
    predictions = knn.predict(test_embeddings)
    
    print("\n[SIAMESE NETWORK CLASSIFICATION REPORT]")
    print(classification_report(test_labels, predictions))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(test_labels, predictions), annot=True, fmt='d', cmap='Blues')
    plt.title("Siamese Network - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
    # Save the model
    torch.save(model.state_dict(), "model/siamese_exoplanet.pth")
    print(f"💾 Model saved as 'model/siamese_exoplanet.pth'")
    
    return model, train_dataset.scaler

# --- 7. INFERENCE FUNCTION ---
def predict_with_siamese(model, scaler, features, threshold=0.5):
    """
    Make prediction using Siamese network embeddings
    """
    model.eval()
    with torch.no_grad():
        # Scale and get embedding
        features_scaled = scaler.transform(features.reshape(1, -1))
        embedding = model.get_embedding(torch.tensor(features_scaled))
        
        # For binary classification, we can use embedding magnitude
        # or compare to a reference embedding
        embedding_norm = torch.norm(embedding).item()
        
        # Simple threshold-based prediction (can be improved)
        prediction = 1 if embedding_norm > threshold else 0
        confidence = min(embedding_norm, 2.0) / 2.0  # Normalize confidence
        
        return prediction, confidence

if __name__ == '__main__':
    model, scaler = train_siamese_model()
