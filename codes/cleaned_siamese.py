import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import random0
import os

class CleanedSiameseDataset(Dataset):
    """Siamese dataset for cleaned data with train/test/candidates split"""
    
    def __init__(self, csv_file, mode='train', test_size=0.2, candidates_size=0.1, random_state=42):
        """
        Load cleaned dataset with train/test/candidates split
        """
        print(f"📊 Loading {csv_file} for {mode}...")
        
        # Load the cleaned data
        self.data = pd.read_csv(csv_file, low_memory=False)
        print(f"   Raw shape: {self.data.shape}")
        
        # Extract features and labels (assuming same structure as head file)
        if 'Label' in self.data.columns:
            self.labels = self.data['Label'].values
            feature_columns = [col for col in self.data.columns if col not in ['kepid', 'Label']]
            self.features = self.data[feature_columns].values
        else:
            # Fallback: assume first column is label, rest are features
            self.labels = self.data.iloc[:, 0].values
            self.features = self.data.iloc[:, 1:].values
        
        # Convert to proper dtypes
        self.labels = pd.Series(self.labels).fillna(0).astype(int).values
        self.features = pd.DataFrame(self.features).fillna(0).astype(np.float64).values
        
        print(f"   Features: {self.features.shape[1]}")
        print(f"   Samples: {len(self.labels)}")
        print(f"   Positives: {np.sum(self.labels == 1)}")
        print(f"   Negatives: {np.sum(self.labels == 0)}")
        
        # Create three-way split: train, test, candidates
        from sklearn.model_selection import train_test_split
        
        # First split: separate candidates
        remaining_indices, candidate_indices = train_test_split(
            range(len(self.labels)), 
            test_size=candidates_size, 
            stratify=self.labels, 
            random_state=random_state
        )
        
        # Second split: separate train and test from remaining
        train_indices, test_indices = train_test_split(
            remaining_indices, 
            test_size=test_size/(1-candidates_size), 
            stratify=self.labels[remaining_indices], 
            random_state=random_state
        )
        
        if mode == 'train':
            self.indices = train_indices
            print(f"   Train samples: {len(self.indices)}")
        elif mode == 'test':
            self.indices = test_indices
            print(f"   Test samples: {len(self.indices)}")
        elif mode == 'candidates':
            self.indices = candidate_indices
            print(f"   Candidate samples: {len(self.indices)}")
        
        # Use only the selected indices
        self.labels = self.labels[self.indices]
        self.features = self.features[self.indices]
        
        # Standardize features (fit on training data only)
        if mode == 'train':
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
            # Save scaler for other datasets
            import joblib
            joblib.dump(self.scaler, 'models/cleaned_scaler.pkl')
        else:
            # Load scaler fitted on training data
            import joblib
            self.scaler = joblib.load('models/cleaned_scaler.pkl')
            self.features = self.scaler.transform(self.features)
        
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

class CleanedSiameseNetwork(nn.Module):
    """Siamese network for cleaned dataset"""
    
    def __init__(self, input_size, embedding_dim=128):
        super(CleanedSiameseNetwork, self).__init__()
        
        # Enhanced encoder for cleaned data
        self.encoder = nn.Sequential(
            # Layer 1: input -> 256
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

class ContrastiveLoss(nn.Module):
    """Contrastive loss for Siamese network"""
    
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, distance, target):
        # target = 1 for similar pairs, 0 for dissimilar pairs
        loss_similar = target * torch.pow(distance, 2)
        loss_dissimilar = (1 - target) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        loss = loss_similar + loss_dissimilar
        return torch.mean(loss)

class CleanedSiameseTrainer:
    """Trainer for cleaned Siamese network"""
    
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
    
    def evaluate_on_dataset(self, dataset, dataset_name):
        """Evaluate Siamese network on a specific dataset"""
        self.model.eval()
        
        # Get embeddings for all samples in dataset
        embeddings = []
        labels = []
        
        with torch.no_grad():
            # Create a simple dataloader for evaluation
            eval_features = torch.FloatTensor(dataset.features).to(self.device)
            eval_labels = dataset.labels
            
            # Get embeddings in batches
            batch_size = 64
            for i in range(0, len(eval_features), batch_size):
                batch = eval_features[i:i+batch_size]
                batch_embeddings = self.model.get_embedding(batch)
                embeddings.append(batch_embeddings.cpu().numpy())
            
            embeddings = np.vstack(embeddings)
            labels = eval_labels
        
        # Simple classification using nearest neighbor in embedding space
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score, classification_report
        
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(embeddings, labels)
        predictions = knn.predict(embeddings)
        
        accuracy = accuracy_score(labels, predictions)
        
        print(f"\n[{dataset_name} EVALUATION]")
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(labels, predictions))
        
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
                torch.save(self.model.state_dict(), "models/cleaned_siamese.pth")
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
        plt.title('Training and Test Loss (Cleaned Data)')
        plt.legend()
        plt.savefig('output/cleaned_siamese_training.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return self.model

def train_cleaned_siamese():
    """Train Siamese network on cleaned dataset"""
    
    # Data paths
    DATA_PATH = "data/processed_curves/raw_curve_500_cleaned.csv"
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Cleaned data file not found: {DATA_PATH}")
        return None, None
    
    print("🔄 Training Siamese Network on Cleaned Dataset")
    print("=" * 60)
    
    # Create datasets with three-way split
    train_dataset = CleanedSiameseDataset(DATA_PATH, mode='train', test_size=0.2, candidates_size=0.1, random_state=42)
    test_dataset = CleanedSiameseDataset(DATA_PATH, mode='test', test_size=0.2, candidates_size=0.1, random_state=42)
    candidates_dataset = CleanedSiameseDataset(DATA_PATH, mode='candidates', test_size=0.2, candidates_size=0.1, random_state=42)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    input_size = train_dataset.features.shape[1]
    embedding_dim = 128
    
    print(f"\n🎯 Model Configuration:")
    print(f"   Input features: {input_size}")
    print(f"   Embedding dimension: {embedding_dim}")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    print(f"   Candidate samples: {len(candidates_dataset)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CleanedSiameseNetwork(input_size, embedding_dim).to(device)
    
    # Train model
    trainer = CleanedSiameseTrainer(model, train_loader, test_loader, device)
    trained_model = trainer.train(epochs=50, lr=0.001)
    
    # Evaluate on all three datasets
    print("\n" + "="*60)
    print("📊 FINAL EVALUATION ON ALL DATASETS")
    print("="*60)
    
    test_accuracy = trainer.evaluate_on_dataset(test_dataset, "TEST SET")
    candidates_accuracy = trainer.evaluate_on_dataset(candidates_dataset, "CANDIDATES SET")
    
    # Also evaluate on training set for reference
    train_accuracy = trainer.evaluate_on_dataset(train_dataset, "TRAIN SET")
    
    print(f"\n📈 SUMMARY:")
    print(f"   Train Accuracy: {train_accuracy:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.4f}")
    print(f"   Candidates Accuracy: {candidates_accuracy:.4f}")
    
    print(f"\n✅ Training complete!")
    print(f"💾 Model saved as: models/cleaned_siamese.pth")
    
    # Save results
    with open('output/cleaned_siamese_results.txt', 'w') as f:
        f.write("CLEANED SIAMESE NETWORK RESULTS\n")
        f.write("=" * 40 + "\n")
        f.write(f"Train Accuracy: {train_accuracy:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Candidates Accuracy: {candidates_accuracy:.4f}\n")
    
    print(f"📄 Results saved: output/cleaned_siamese_results.txt")
    
    return trained_model, train_dataset.scaler

if __name__ == "__main__":
    # Create output directory
    os.makedirs('models', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    model, scaler = train_cleaned_siamese()
