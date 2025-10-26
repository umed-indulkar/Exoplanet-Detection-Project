import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
import optuna
from torch.utils.data import DataLoader, TensorDataset
from itertools import combinations
import random

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class ContrastiveBackbone(nn.Module):
    """
    Shared backbone for Siamese network with advanced architecture
    This extracts discriminative embeddings from tsfresh features
    """
    def __init__(self, input_dim, embedding_dim=128, hidden_dims=[512, 256, 128], 
                 dropout_rate=0.3, use_batch_norm=True):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Projection head for embedding space
        self.projection = nn.Sequential(
            nn.Linear(hidden_dims[-1], embedding_dim),
            nn.BatchNorm1d(embedding_dim) if use_batch_norm else nn.Identity(),
            nn.GELU()
        )
        
        self._initialize_weights()
    
    def forward(self, x):
        features = self.feature_extractor(x)
        embeddings = self.projection(features)
        # L2 normalize embeddings for better contrastive learning
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class SiameseContrastiveNetwork(nn.Module):
    """
    Siamese network for contrastive learning during training
    Uses triplet loss + contrastive loss for better separation
    """
    def __init__(self, input_dim, embedding_dim=128, hidden_dims=[512, 256, 128],
                 dropout_rate=0.3, use_batch_norm=True):
        super().__init__()
        
        # Shared backbone
        self.backbone = ContrastiveBackbone(
            input_dim, embedding_dim, hidden_dims, dropout_rate, use_batch_norm
        )
        
        # Distance-based classifier for training
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, 1)
        )
        
    def forward(self, x1, x2=None, mode='train'):
        """
        mode='train': Returns embeddings for both inputs (contrastive learning)
        mode='inference': Returns classification logits for single input
        """
        if mode == 'train' and x2 is not None:
            # Siamese mode: process both inputs
            emb1 = self.backbone(x1)
            emb2 = self.backbone(x2)
            return emb1, emb2
        else:
            # Inference mode: single input classification
            emb = self.backbone(x1)
            logits = self.classifier(emb)
            return logits.squeeze()
    
    def get_embedding(self, x):
        """Extract embeddings for few-shot learning"""
        return self.backbone(x)


class HybridLoss(nn.Module):
    """
    Combines Contrastive Loss + Triplet Loss + Classification Loss
    for robust one-shot/few-shot learning capability
    """
    def __init__(self, margin=1.0, alpha=0.5, beta=0.3, gamma=0.2):
        super().__init__()
        self.margin = margin
        self.alpha = alpha  # Weight for contrastive loss
        self.beta = beta    # Weight for triplet loss
        self.gamma = gamma  # Weight for classification loss
        
    def contrastive_loss(self, emb1, emb2, labels):
        """
        Contrastive loss: pull similar pairs together, push different pairs apart
        labels: 1 for same class, 0 for different class
        """
        euclidean_distance = nn.functional.pairwise_distance(emb1, emb2)
        
        # Similar pairs: minimize distance
        loss_similar = labels * torch.pow(euclidean_distance, 2)
        
        # Different pairs: maximize distance up to margin
        loss_different = (1 - labels) * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0), 2
        )
        
        return torch.mean(loss_similar + loss_different)
    
    def triplet_loss(self, anchor, positive, negative):
        """
        Triplet loss: anchor-positive distance < anchor-negative distance
        """
        pos_dist = nn.functional.pairwise_distance(anchor, positive)
        neg_dist = nn.functional.pairwise_distance(anchor, negative)
        
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return torch.mean(loss)
    
    def forward(self, emb1, emb2, labels, triplet_data=None):
        """
        Combined loss function
        """
        # Contrastive loss
        contrastive = self.contrastive_loss(emb1, emb2, labels)
        
        total_loss = self.alpha * contrastive
        
        # Triplet loss (if triplet data provided)
        if triplet_data is not None:
            anchor, positive, negative = triplet_data
            triplet = self.triplet_loss(anchor, positive, negative)
            total_loss += self.beta * triplet
        
        return total_loss


class ContrastiveDataLoader:
    """
    Creates pairs and triplets for contrastive learning
    """
    @staticmethod
    def create_pairs(X, y, n_pairs=None):
        """
        Create positive and negative pairs for contrastive learning
        """
        if n_pairs is None:
            n_pairs = len(X)
        
        pairs_x1, pairs_x2, pair_labels = [], [], []
        
        # Group indices by class
        class_0_idx = np.where(y == 0)[0]
        class_1_idx = np.where(y == 1)[0]
        
        for _ in range(n_pairs):
            # Randomly decide positive or negative pair
            if random.random() < 0.5 and len(class_1_idx) > 1:
                # Positive pair (same class)
                if random.random() < 0.5 and len(class_0_idx) > 1:
                    # Both from class 0
                    idx = np.random.choice(class_0_idx, 2, replace=False)
                else:
                    # Both from class 1
                    idx = np.random.choice(class_1_idx, 2, replace=False)
                pairs_x1.append(X[idx[0]])
                pairs_x2.append(X[idx[1]])
                pair_labels.append(1.0)
            else:
                # Negative pair (different classes)
                idx1 = np.random.choice(class_0_idx)
                idx2 = np.random.choice(class_1_idx)
                pairs_x1.append(X[idx1])
                pairs_x2.append(X[idx2])
                pair_labels.append(0.0)
        
        return (torch.FloatTensor(pairs_x1), 
                torch.FloatTensor(pairs_x2), 
                torch.FloatTensor(pair_labels))
    
    @staticmethod
    def create_triplets(X, y, n_triplets=None):
        """
        Create triplets (anchor, positive, negative) for triplet loss
        """
        if n_triplets is None:
            n_triplets = len(X) // 2
        
        anchors, positives, negatives = [], [], []
        
        class_0_idx = np.where(y == 0)[0]
        class_1_idx = np.where(y == 1)[0]
        
        for _ in range(n_triplets):
            # Choose anchor class
            if random.random() < 0.5 and len(class_1_idx) > 1:
                # Anchor from class 1
                anchor_idx, positive_idx = np.random.choice(class_1_idx, 2, replace=False)
                negative_idx = np.random.choice(class_0_idx)
            elif len(class_0_idx) > 1:
                # Anchor from class 0
                anchor_idx, positive_idx = np.random.choice(class_0_idx, 2, replace=False)
                negative_idx = np.random.choice(class_1_idx)
            else:
                continue
                
            anchors.append(X[anchor_idx])
            positives.append(X[positive_idx])
            negatives.append(X[negative_idx])
        
        if len(anchors) == 0:
            return None
            
        return (torch.FloatTensor(anchors),
                torch.FloatTensor(positives),
                torch.FloatTensor(negatives))


def pretrain_siamese(model, X_train, y_train, X_val, y_val, epochs=100, 
                     batch_size=32, lr=1e-3, patience=15):
    """
    Phase 1: Pretrain using contrastive learning with Siamese architecture
    """
    print("\n" + "="*60)
    print("PHASE 1: CONTRASTIVE PRETRAINING")
    print("="*60)
    
    criterion = HybridLoss(margin=1.0, alpha=0.6, beta=0.4)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=lr/100
    )
    
    model.to(device)
    best_loss = float('inf')
    patience_counter = 0
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        # Create pairs for this epoch
        pairs_x1, pairs_x2, pair_labels = ContrastiveDataLoader.create_pairs(
            X_train, y_train, n_pairs=len(X_train) * 2
        )
        
        # Create triplets
        triplet_data = ContrastiveDataLoader.create_triplets(
            X_train, y_train, n_triplets=len(X_train)
        )
        
        # Training loop
        n_batches = len(pairs_x1) // batch_size
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            x1_batch = pairs_x1[start_idx:end_idx].to(device)
            x2_batch = pairs_x2[start_idx:end_idx].to(device)
            labels_batch = pair_labels[start_idx:end_idx].to(device)
            
            optimizer.zero_grad()
            
            # Get embeddings
            emb1, emb2 = model(x1_batch, x2_batch, mode='train')
            
            # Prepare triplet batch if available
            trip_batch = None
            if triplet_data is not None and i < len(triplet_data[0]) // batch_size:
                trip_start = i * batch_size
                trip_end = trip_start + batch_size
                anchor = model.backbone(triplet_data[0][trip_start:trip_end].to(device))
                positive = model.backbone(triplet_data[1][trip_start:trip_end].to(device))
                negative = model.backbone(triplet_data[2][trip_start:trip_end].to(device))
                trip_batch = (anchor, positive, negative)
            
            # Compute loss
            loss = criterion(emb1, emb2, labels_batch, trip_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            val_pairs_x1, val_pairs_x2, val_pair_labels = ContrastiveDataLoader.create_pairs(
                X_val, y_val, n_pairs=len(X_val)
            )
            
            n_val_batches = len(val_pairs_x1) // batch_size
            for i in range(n_val_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                x1_batch = val_pairs_x1[start_idx:end_idx].to(device)
                x2_batch = val_pairs_x2[start_idx:end_idx].to(device)
                labels_batch = val_pair_labels[start_idx:end_idx].to(device)
                
                emb1, emb2 = model(x1_batch, x2_batch, mode='train')
                loss = criterion(emb1, emb2, labels_batch)
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses) if val_losses else avg_train_loss
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        scheduler.step()
        
        # Early stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'siamese_pretrained.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
    
    print(f"\nPretraining Complete! Best Loss: {best_loss:.4f}")
    return history


def finetune_classifier(model, X_train, y_train, X_val, y_val, epochs=50,
                       batch_size=32, lr=1e-4, patience=10):
    """
    Phase 2: Fine-tune the classifier head for binary classification
    Backbone can be frozen or fine-tuned with lower learning rate
    """
    print("\n" + "="*60)
    print("PHASE 2: CLASSIFIER FINE-TUNING")
    print("="*60)
    
    # Option to freeze backbone (uncomment to use)
    # for param in model.backbone.parameters():
    #     param.requires_grad = False
    
    # Use different learning rates for backbone and classifier
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': lr * 0.1},
        {'params': model.classifier.parameters(), 'lr': lr}
    ], weight_decay=1e-4)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))  # Adjust based on class ratio
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    model.to(device)
    best_f1 = 0.0
    patience_counter = 0
    
    history = {
        'train_loss': [], 'val_loss': [], 'accuracy': [],
        'precision': [], 'recall': [], 'f1': [], 'auc': []
    }
    
    # Create datasets
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(x_batch, mode='inference')
            loss = criterion(logits, y_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation
        model.eval()
        val_losses, preds, targets = [], [], []
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                
                logits = model(x_batch, mode='inference')
                loss = criterion(logits, y_batch)
                val_losses.append(loss.item())
                
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.extend(probs)
                targets.extend(y_batch.cpu().numpy())
        
        avg_val_loss = np.mean(val_losses)
        
        # Calculate metrics
        targets_arr = np.array(targets)
        preds_arr = np.array(preds)
        
        # Find optimal threshold
        best_threshold = 0.5
        best_threshold_f1 = 0
        for thresh in np.arange(0.3, 0.7, 0.05):
            preds_bin = (preds_arr > thresh).astype(int)
            f1 = f1_score(targets_arr, preds_bin, zero_division=0)
            if f1 > best_threshold_f1:
                best_threshold_f1 = f1
                best_threshold = thresh
        
        preds_bin = (preds_arr > best_threshold).astype(int)
        
        acc = accuracy_score(targets_arr, preds_bin)
        prec = precision_score(targets_arr, preds_bin, zero_division=0)
        rec = recall_score(targets_arr, preds_bin, zero_division=0)
        f1 = f1_score(targets_arr, preds_bin, zero_division=0)
        
        try:
            auc = roc_auc_score(targets_arr, preds_arr)
        except:
            auc = 0.0
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['accuracy'].append(acc)
        history['precision'].append(prec)
        history['recall'].append(rec)
        history['f1'].append(f1)
        history['auc'].append(auc)
        
        scheduler.step(f1)
        
        # Early stopping on F1
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            torch.save(model.state_dict(), 'best_exoplanet_siamese.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss={avg_val_loss:.4f}, F1={f1:.4f}, "
                  f"AUC={auc:.4f}, Threshold={best_threshold:.2f}")
    
    print(f"\nFine-tuning Complete! Best F1: {best_f1:.4f}")
    return history, best_f1


def train_full_pipeline(X_train, y_train, X_val, y_val, input_dim,
                       pretrain_epochs=100, finetune_epochs=50,
                       batch_size=32, pretrain_lr=1e-3, finetune_lr=1e-4):
    """
    Complete two-phase training pipeline
    """
    print("\n" + "="*70)
    print(" SIAMESE NETWORK TRAINING PIPELINE FOR EXOPLANET CLASSIFICATION")
    print("="*70)
    
    # Initialize model
    model = SiameseContrastiveNetwork(
        input_dim=input_dim,
        embedding_dim=128,
        hidden_dims=[512, 256, 128],
        dropout_rate=0.3,
        use_batch_norm=True
    )
    
    # Phase 1: Contrastive pretraining
    pretrain_history = pretrain_siamese(
        model, X_train, y_train, X_val, y_val,
        epochs=pretrain_epochs, batch_size=batch_size, lr=pretrain_lr
    )
    
    # Load best pretrained weights
    model.load_state_dict(torch.load('siamese_pretrained.pth'))
    
    # Phase 2: Classifier fine-tuning
    finetune_history, best_f1 = finetune_classifier(
        model, X_train, y_train, X_val, y_val,
        epochs=finetune_epochs, batch_size=batch_size, lr=finetune_lr
    )
    
    return model, pretrain_history, finetune_history, best_f1


def inference_single_sample(model, x_sample):
    """
    Inference on a single sample using the trained Siamese network
    """
    model.eval()
    with torch.no_grad():
        x_tensor = torch.FloatTensor(x_sample).unsqueeze(0).to(device)
        logit = model(x_tensor, mode='inference')
        prob = torch.sigmoid(logit).item()
    return prob


# Example usage
if __name__ == "__main__":
    # Example with simulated data
    num_features = 794  # Typical tsfresh feature count
    n_samples = 2000
    
    # Simulated data (replace with your actual tsfresh features)
    X_train = np.random.randn(n_samples, num_features)
    y_train = np.random.binomial(1, 0.1, n_samples)  # 10% positive class
    
    X_val = np.random.randn(n_samples//5, num_features)
    y_val = np.random.binomial(1, 0.1, n_samples//5)
    
    # Train complete pipeline
    model, pretrain_hist, finetune_hist, best_f1 = train_full_pipeline(
        X_train, y_train, X_val, y_val, input_dim=num_features,
        pretrain_epochs=50, finetune_epochs=30
    )
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS: Best F1 Score = {best_f1:.4f}")
    print(f"{'='*70}")
    
    # Test inference on single sample
    test_sample = X_val[0]
    prob = inference_single_sample(model, test_sample)
    print(f"\nSingle sample prediction: {prob:.4f} ({'Exoplanet' if prob > 0.5 else 'Non-Exoplanet'})")