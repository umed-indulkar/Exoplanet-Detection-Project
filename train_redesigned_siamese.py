import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os
from datetime import datetime

class RedesignedSiameseNetwork(nn.Module):
    """Redesigned Siamese Network with Classification Head"""
    
    def __init__(self, input_size=500, embedding_dim=128, dropout_rate=0.3):
        super(RedesignedSiameseNetwork, self).__init__()
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),
            
            nn.Linear(128, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(32, 2)  # 2 classes
        )
        
    def get_embedding(self, x):
        return self.encoder(x)
    
    def forward_siamese(self, x1, x2):
        embedding1 = self.encoder(x1)
        embedding2 = self.encoder(x2)
        distance = F.pairwise_distance(embedding1, embedding2, p=2)
        return distance, embedding1, embedding2
    
    def forward_classification(self, x):
        embedding = self.encoder(x)
        logits = self.classification_head(embedding)
        probabilities = F.softmax(logits, dim=1)
        return logits, probabilities, embedding

class HybridLoss(nn.Module):
    """Hybrid loss combining contrastive and classification"""
    
    def __init__(self, margin=1.0, alpha=0.7, beta=0.3):
        super(HybridLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, distances, pair_labels, classification_logits, true_labels):
        # Contrastive loss
        contrastive_loss = torch.mean(
            (pair_labels.float() * torch.pow(distances, 2) + 
             (1 - pair_labels.float()) * torch.pow(F.relu(self.margin - distances), 2))
        )
        
        # Classification loss
        classification_loss = F.cross_entropy(classification_logits, true_labels)
        
        # Hybrid loss
        total_loss = self.alpha * contrastive_loss + self.beta * classification_loss
        
        return total_loss, contrastive_loss, classification_loss

class RawCurvesDataset(Dataset):
    """Dataset for raw 500-binned curves"""
    
    def __init__(self, data_path):
        print(f"Loading raw curves from: {data_path}")
        
        # Load data
        df = pd.read_csv(data_path, low_memory=False)
        
        # Extract features and labels
        self.labels = df['Label'].values
        feature_columns = [col for col in df.columns if col not in ['kepid', 'Label']]
        self.features = df[feature_columns].values
        
        # Convert to proper dtypes
        self.labels = pd.Series(self.labels).fillna(0).astype(int).values
        self.features = pd.DataFrame(self.features).fillna(0).astype(np.float64).values
        
        print(f"Loaded {len(self.features)} samples")
        print(f"Positive samples (exoplanets): {np.sum(self.labels == 1)}")
        print(f"Negative samples (non-exoplanets): {np.sum(self.labels == 0)}")
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]])[0]

def create_pairs(dataset, batch_size):
    """Create pairs for contrastive learning"""
    features, labels = dataset
    n = len(features)
    
    pairs = []
    pair_labels = []
    true_labels = []
    
    for _ in range(batch_size):
        # Random indices
        i, j = np.random.choice(n, 2, replace=False)
        
        # Get features and labels
        feat_i, label_i = features[i], labels[i]
        feat_j, label_j = features[j], labels[j]
        
        # Determine if similar (same class)
        is_similar = 1 if label_i == label_j else 0
        
        pairs.append((feat_i, feat_j))
        pair_labels.append(is_similar)
        true_labels.append(label_i)  # Use first sample's label for classification
    
    # Convert to tensors
    x1 = torch.stack([pair[0] for pair in pairs])
    x2 = torch.stack([pair[1] for pair in pairs])
    pair_labels = torch.LongTensor(pair_labels)
    true_labels = torch.LongTensor(true_labels)
    
    return x1, x2, pair_labels, true_labels

def train_redesigned_model():
    """Train the redesigned Siamese model on raw curves"""
    
    print("🚀 TRAINING REDESIGNED SIAMESE MODEL ON RAW CURVES")
    print("=" * 60)
    print("🎯 Hybrid Loss: Contrastive + Classification")
    print("🔬 Full 128D Vector Utilization")
    
    # Load data
    data_path = "data/processed_curves/raw_curve_500_cleaned.csv"
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return
    
    # Create dataset
    full_dataset = RawCurvesDataset(data_path)
    
    # Split data
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RedesignedSiameseNetwork().to(device)
    
    # Initialize scaler and fit on training data
    print("\n📊 Fitting scaler on training data...")
    train_features = []
    train_labels = []
    
    for i in range(len(train_dataset)):
        features, label = train_dataset[i]
        train_features.append(features.numpy())
        train_labels.append(label.numpy())
    
    train_features = np.array(train_features)
    scaler = StandardScaler()
    scaler.fit(train_features)
    
    # Save scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/redesigned_scaler.pkl')
    print("✅ Scaler saved: models/redesigned_scaler.pkl")
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    loss_fn = HybridLoss(margin=1.0, alpha=0.7, beta=0.3)
    
    # Training parameters
    epochs = 50
    batch_size = 64
    print(f"\n🎯 Training: {epochs} epochs, batch size {batch_size}")
    
    # Training loop
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        num_batches = 100  # Number of batches per epoch
        
        for batch_idx in range(num_batches):
            # Create pairs from training data
            train_features_batch = []
            train_labels_batch = []
            
            # Sample random training data
            indices = np.random.choice(len(train_dataset), batch_size * 2, replace=True)
            for idx in indices:
                features, label = train_dataset[idx]
                train_features_batch.append(scaler.transform(features.numpy().reshape(1, -1))[0])
                train_labels_batch.append(label.numpy())
            
            # Create pairs
            x1, x2, pair_labels, true_labels = create_pairs(
                (torch.FloatTensor(train_features_batch), train_labels_batch), 
                batch_size
            )
            
            # Move to device
            x1, x2, pair_labels, true_labels = (x1.to(device), x2.to(device), 
                                                  pair_labels.to(device), 
                                                  true_labels.to(device))
            
            # Forward pass
            distances, embeddings1, _ = model.forward_siamese(x1, x2)
            logits, _, _ = model.forward_classification(x1)
            
            # Loss
            total_loss, contrastive_loss, ce_loss = loss_fn(distances, pair_labels, logits, true_labels)
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_train_loss += total_loss.item()
        
        avg_train_loss = epoch_train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        if epoch % 5 == 0:
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            val_batches = 20
            
            with torch.no_grad():
                for _ in range(val_batches):
                    # Sample validation data
                    val_indices = np.random.choice(len(val_dataset), batch_size * 2, replace=True)
                    val_features_batch = []
                    val_labels_batch = []
                    
                    for idx in val_indices:
                        features, label = val_dataset[idx]
                        val_features_batch.append(scaler.transform(features.numpy().reshape(1, -1))[0])
                        val_labels_batch.append(label.numpy())
                    
                    # Create pairs
                    x1_val, x2_val, pair_labels_val, true_labels_val = create_pairs(
                        (torch.FloatTensor(val_features_batch), val_labels_batch), 
                        batch_size
                    )
                    
                    x1_val, x2_val, pair_labels_val, true_labels_val = (
                        x1_val.to(device), x2_val.to(device), 
                        pair_labels_val.to(device), 
                        true_labels_val.to(device)
                    )
                    
                    # Forward pass
                    distances_val, _, _ = model.forward_siamese(x1_val, x2_val)
                    logits_val, _, _ = model.forward_classification(x1_val)
                    
                    # Loss
                    total_loss_val, _, ce_loss_val = loss_fn(distances_val, pair_labels_val, logits_val, true_labels_val)
                    val_loss += total_loss_val.item()
                    
                    # Accuracy
                    predicted = torch.argmax(logits_val, dim=1)
                    correct += (predicted == true_labels_val).sum().item()
                    total += true_labels_val.size(0)
            
            avg_val_loss = val_loss / val_batches
            accuracy = correct / total
            
            val_losses.append(avg_val_loss)
            val_accuracies.append(accuracy)
            
            print(f"Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={accuracy:.4f}")
    
    # Save model
    torch.save(model.state_dict(), 'models/redesigned_siamese.pth')
    print("✅ Model saved: models/redesigned_siamese.pth")
    
    # Final evaluation
    print("\n🎯 FINAL EVALUATION")
    print("=" * 40)
    
    model.eval()
    all_predictions = []
    all_true_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for i in range(len(val_dataset)):
            features, true_label = val_dataset[i]
            
            # Process features
            features_scaled = scaler.transform(features.numpy().reshape(1, -1))
            features_tensor = torch.FloatTensor(features_scaled).to(device)
            
            # Get prediction
            logits, probabilities, _ = model.forward_classification(features_tensor)
            predicted = torch.argmax(logits, dim=1).item()
            exoplanet_prob = probabilities[0, 1].item()
            
            all_predictions.append(predicted)
            all_true_labels.append(true_label.numpy())
            all_probabilities.append(exoplanet_prob)
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    all_probabilities = np.array(all_probabilities)
    
    accuracy = np.mean(all_predictions == all_true_labels)
    
    print(f"Final Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"ROC-AUC Score: {roc_auc_score(all_true_labels, all_probabilities):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(all_true_labels, all_predictions, 
                              target_names=['Non-Exoplanet', 'Exoplanet']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_true_labels, all_predictions)
    print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
    
    return accuracy, model, scaler

def test_with_real_data(model, scaler):
    """Test the trained model with real star data"""
    
    print("\n🌍 TESTING WITH REAL STAR DATA")
    print("=" * 40)
    
    # Load real data
    tbl_path = "user_data/2018261.tbl"
    csv_path = "user_data/q1_q17_dr25_tce_2018261.csv"
    
    if not os.path.exists(tbl_path) or not os.path.exists(csv_path):
        print("❌ Real data files not found")
        return
    
    # Read light curve
    with open(tbl_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if '|set|' in line:
            data_start = i + 2
            break
    
    data = pd.read_csv(tbl_path, sep='\s+', skiprows=data_start, 
                     names=['QUARTER', 'TIME', 'PDCSAP_FLUX'], comment='#')
    
    time = data['TIME'].values
    flux = data['PDCSAP_FLUX'].values
    mask = ~np.isnan(time) & ~np.isnan(flux)
    time, flux = time[mask], flux[mask]
    
    # Read events
    events = pd.read_csv(csv_path, comment='#')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    print(f"Analyzing {len(events)} events with trained model...")
    
    for _, event in events.iterrows():
        period = event['tce_period']
        epoch = event['tce_time0bk']
        planet_num = event['tce_plnt_num']
        
        # Process light curve
        phase = ((time - epoch) / period) % 1.0
        sort_idx = np.argsort(phase)
        phase_sorted, flux_sorted = phase[sort_idx], flux[sort_idx]
        
        # Bin to 500 points
        bins = np.linspace(0, 1, 501)
        binned_flux = np.zeros(500)
        
        for i in range(500):
            mask = (phase_sorted >= bins[i]) & (phase_sorted < bins[i+1])
            if np.sum(mask) > 0:
                binned_flux[i] = np.median(flux_sorted[mask])
        
        # Handle empty bins
        empty_bins = binned_flux == 0
        if np.any(empty_bins):
            valid_indices = ~empty_bins
            if np.sum(valid_indices) > 1:
                bin_centers = (bins[:-1] + bins[1:]) / 2
                binned_flux[empty_bins] = np.interp(
                    bin_centers[empty_bins], 
                    bin_centers[valid_indices], 
                    binned_flux[valid_indices]
                )
        
        # Normalize and scale
        normalized_flux = binned_flux / np.median(binned_flux)
        processed_flux = scaler.transform(normalized_flux.reshape(1, -1))
        
        # Predict
        with torch.no_grad():
            logits, probabilities, embedding = model.forward_classification(
                torch.FloatTensor(processed_flux).to(device)
            )
            
            exoplanet_prob = probabilities[0, 1].item()
            prediction = 1 if exoplanet_prob > 0.5 else 0
            confidence = max(exoplanet_prob, 1 - exoplanet_prob)
            embedding_norm = torch.norm(embedding).item()
        
        pred_str = "🌍 EXOPLANET" if prediction == 1 else "⭐ NON-EXOPLANET"
        print(f"Planet {planet_num}: {pred_str} (P={exoplanet_prob:.3f}, Conf={confidence:.3f}, Norm={embedding_norm:.4f})")

def main():
    """Main training function"""
    
    # Train the model
    accuracy, model, scaler = train_redesigned_model()
    
    # Test with real data
    test_with_real_data(model, scaler)
    
    print(f"\n🎉 TRAINING COMPLETE!")
    print(f"📊 Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"📁 Model saved: models/redesigned_siamese.pth")
    print(f"📁 Scaler saved: models/redesigned_scaler.pkl")

if __name__ == "__main__":
    main()
