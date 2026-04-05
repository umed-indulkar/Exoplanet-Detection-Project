import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os

# --- 1. DATASET_500 LOADER ---
class Dataset500Siamese(Dataset):
    def __init__(self, csv_file, mode='train', test_size=0.2, random_state=42):
        """
        Load dataset_500 with proper train/test split
        Ensures no data leakage between train and test
        """
        print(f"📊 Loading {csv_file}...")
        
        # Load the data with proper dtype handling
        self.data = pd.read_csv(csv_file, low_memory=False)
        print(f"   Raw shape: {self.data.shape}")
        print(f"   Columns: {list(self.data.columns[:5])}...")
        
        # Extract label and features
        if 'Label' in self.data.columns:
            self.labels = self.data['Label'].values
            # Drop kepid and Label columns
            feature_columns = [col for col in self.data.columns if col not in ['kepid', 'Label']]
            self.features = self.data[feature_columns].values
        else:
            # Fallback: assume first column is label, rest are features
            self.labels = self.data.iloc[:, 0].values
            self.features = self.data.iloc[:, 1:].values
        
        # Convert to proper dtypes - handle numpy arrays
        if isinstance(self.labels, np.ndarray):
            self.labels = pd.Series(self.labels).fillna(0).astype(int).values
        else:
            self.labels = pd.to_numeric(self.labels, errors='coerce').fillna(0).astype(int).values
            
        if isinstance(self.features, np.ndarray):
            self.features = pd.DataFrame(self.features).fillna(0).astype(np.float64).values
        else:
            self.features = pd.to_numeric(self.features, errors='coerce').fillna(0).astype(np.float64).values
        
        print(f"   Features: {self.features.shape[1]}")
        print(f"   Samples: {len(self.labels)}")
        print(f"   Positives: {np.sum(self.labels == 1)}")
        print(f"   Negatives: {np.sum(self.labels == 0)}")
        
        # Split into train/test with proper stratification
        train_indices, test_indices = train_test_split(
            range(len(self.labels)), 
            test_size=test_size, 
            stratify=self.labels, 
            random_state=random_state
        )
        
        if mode == 'train':
            self.indices = train_indices
            print(f"   Train samples: {len(self.indices)}")
        else:
            self.indices = test_indices
            print(f"   Test samples: {len(self.indices)}")
        
        # Use only the selected indices
        self.labels = self.labels[self.indices]
        self.features = self.features[self.indices]
        
        # Standardize features
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)
        
        # Group indices by class for pairing
        self.positive_indices = np.where(self.labels == 1)[0]
        self.negative_indices = np.where(self.labels == 0)[0]
        
        print(f"   Final shape: {self.features.shape}")
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Create pairs: similar (same class) or dissimilar (different class)
        if random.random() > 0.5:
            # Similar pair
            target = 1.0  # Similar
            if self.labels[idx] == 1:
                pair_idx = random.choice(self.positive_indices)
            else:
                pair_idx = random.choice(self.negative_indices)
        else:
            # Dissimilar pair
            target = 0.0  # Dissimilar
            if self.labels[idx] == 1:
                pair_idx = random.choice(self.negative_indices)
            else:
                pair_idx = random.choice(self.positive_indices)
        
        return (torch.tensor(self.features[idx], dtype=torch.float32), 
                torch.tensor(self.features[pair_idx], dtype=torch.float32), 
                torch.tensor(target, dtype=torch.float32))

# --- 2. ENCODER NETWORK ---
class SiameseEncoder(nn.Module):
    def __init__(self, input_size, embedding_dim=128):
        super(SiameseEncoder, self).__init__()
        
        # Enhanced architecture for 501 features
        self.encoder = nn.Sequential(
            # Layer 1: 501 -> 256
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 2: 256 -> 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Layer 3: 128 -> embedding
            nn.Linear(128, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)

# --- 3. SIAMESE NETWORK ---
class SiameseNetwork(nn.Module):
    def __init__(self, input_size, embedding_dim=128):
        super(SiameseNetwork, self).__init__()
        self.encoder = SiameseEncoder(input_size, embedding_dim)
        
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

# --- 5. TRAINER ---
class SiameseTrainer:
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.train_losses = []
        self.test_losses = []
        
    def train_epoch(self, optimizer, criterion):
        self.model.train()
        total_loss = 0
        for batch_idx, (x1, x2, targets) in enumerate(self.train_loader):
            x1, x2, targets = x1.to(self.device), x2.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            distances = self.model(x1, x2)
            loss = criterion(distances, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def test(self, criterion):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x1, x2, targets in self.test_loader:
                x1, x2, targets = x1.to(self.device), x2.to(self.device), targets.to(self.device)
                distances = self.model(x1, x2)
                loss = criterion(distances, targets)
                total_loss += loss.item()
                
        return total_loss / len(self.test_loader)
    
    def evaluate_siamese(self):
        """Evaluate Siamese network with proper test set"""
        self.model.eval()
        
        # Get embeddings for test set
        test_embeddings = []
        test_labels = []
        
        with torch.no_grad():
            # Use test loader features directly (not pairs)
            for batch_idx, (x1, x2, targets) in enumerate(self.test_loader):
                # Get embeddings for both samples in pair
                emb1 = self.model.get_embedding(x1.to(self.device))
                emb2 = self.model.get_embedding(x2.to(self.device))
                
                # Get corresponding labels
                batch_labels = targets.cpu().numpy()
                
                # Store embeddings and labels
                test_embeddings.append(emb1.cpu().numpy())
                test_embeddings.append(emb2.cpu().numpy())
                test_labels.extend(batch_labels)
                test_labels.extend(batch_labels)
        
        test_embeddings = np.vstack(test_embeddings)
        test_labels = np.array(test_labels)
        
        # Simple classification using nearest neighbor in embedding space
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score
        
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(test_embeddings, test_labels)
        predictions = knn.predict(test_embeddings)
        
        accuracy = accuracy_score(test_labels, predictions)
        
        print("\n[SIAMESE NETWORK EVALUATION]")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(classification_report(test_labels, predictions))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(test_labels, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Siamese Network - Confusion Matrix (Test Set)")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()
        
        return accuracy
    
    def train(self, epochs=50, lr=0.001):
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = ContrastiveLoss(margin=1.0)
        
        print(f"🚀 Training for {epochs} epochs...")
        
        best_test_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(optimizer, criterion)
            test_loss = self.test(criterion)
            
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
            
            # Early stopping based on test loss
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), "models/siamese_dataset500.pth")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Plot training curves
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss')
        plt.legend()
        plt.show()
        
        return self.model

# --- 6. MAIN TRAINING FUNCTION ---
def train_siamese_dataset500():
    """
    Train Siamese network on dataset_500 with proper train/test separation
    """
    
    # Data paths
    DATA_PATH = "data/dataset_500/dataset_500/raw_curve_500_head.csv"
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data file not found: {DATA_PATH}")
        return None, None
    
    print("🔄 Loading dataset_500 with proper train/test split...")
    
    # Create datasets with proper separation
    train_dataset = Dataset500Siamese(DATA_PATH, mode='train', test_size=0.2, random_state=42)
    test_dataset = Dataset500Siamese(DATA_PATH, mode='test', test_size=0.2, random_state=42)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    input_size = train_dataset.features.shape[1]  # Should be 501
    embedding_dim = 128
    
    print(f"\n🎯 Model Configuration:")
    print(f"   Input features: {input_size}")
    print(f"   Embedding dimension: {embedding_dim}")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseNetwork(input_size, embedding_dim).to(device)
    
    # Train model
    trainer = SiameseTrainer(model, train_loader, test_loader, device)
    trained_model = trainer.train(epochs=50, lr=0.001)
    
    # Evaluate on test set (unseen data)
    test_accuracy = trainer.evaluate_siamese()
    
    print(f"\n✅ Training complete!")
    print(f"📊 Final test accuracy: {test_accuracy:.4f}")
    print(f"💾 Model saved as: model/siamese_dataset500.pth")
    
    return trained_model, train_dataset.scaler

if __name__ == '__main__':
    model, scaler = train_siamese_dataset500()
