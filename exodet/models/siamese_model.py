"""
Siamese Neural Network - ML Team
=================================

Deep learning model with contrastive learning for exoplanet detection.
This is the ACTUAL Siamese code - no more duplication!
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader, random_split
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


@dataclass
class SiameseTrainResult:
    model_path: str
    metrics: Dict[str, float]


def _select_numeric_features(df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
    y = df[target_col].values
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).values
    return X.astype(np.float32), y.astype(np.int64)


class _PairDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, n_pairs: int = 50000, pos_ratio: float = 0.5, seed: int = 42):
        self.X = X
        self.y = y
        self.n = len(X)
        self.n_pairs = max(n_pairs, self.n)  # ensure enough steps
        self.pos_ratio = pos_ratio
        self.rng = random.Random(seed)
        # index by label for fast sampling
        self.by_label: Dict[int, list[int]] = {}
        for idx, lbl in enumerate(y):
            self.by_label.setdefault(int(lbl), []).append(idx)
        self.labels_sorted = list(self.by_label.keys())

    def __len__(self) -> int:
        return self.n_pairs

    def __getitem__(self, idx: int):
        # decide positive or negative
        is_pos = self.rng.random() < self.pos_ratio
        if is_pos and len(self.labels_sorted) > 0:
            lbl = self.rng.choice(self.labels_sorted)
            pool = self.by_label[lbl]
            if len(pool) < 2:
                # fallback to negative
                is_pos = False
            else:
                i, j = self.rng.sample(pool, 2)
                return self.X[i], self.X[j], 1
        # negative pair
        if len(self.labels_sorted) >= 2:
            a, b = self.rng.sample(self.labels_sorted, 2)
            i = self.rng.choice(self.by_label[a])
            j = self.rng.choice(self.by_label[b])
            return self.X[i], self.X[j], 0
        # degenerate fallback
        i, j = self.rng.sample(range(self.n), 2)
        return self.X[i], self.X[j], 0


class SiameseNet(nn.Module):
    def __init__(self, feature_dim: int, embedding_dim: int = 32):
        super().__init__()
        hidden = max(64, 2 * embedding_dim)
        self.backbone = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, embedding_dim),
        )
        self.l2 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        # unit-normalize embeddings
        z = torch.nn.functional.normalize(z, p=2, dim=1)
        return z


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # y=1: similar; y=0: dissimilar
        d = torch.nn.functional.pairwise_distance(z1, z2)
        pos = y * (d ** 2)
        neg = (1 - y) * torch.clamp(self.margin - d, min=0) ** 2
        return pos + neg


def train_siamese_from_csv(
    csv_path: str,
    target_col: str = 'label',
    embedding_dim: int = 32,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_split: float = 0.2,
    device: str = 'auto',
    output_path: str = None,
    seed: int = 42,
) -> SiameseTrainResult:
    """Train Siamese network from CSV features."""
    if not _HAS_TORCH:
        raise ImportError("PyTorch is required for Siamese training")

    # Load and prepare data
    df = pd.read_csv(csv_path)
    X, y = _select_numeric_features(df, target_col)

    # Feature standardization
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1  # Avoid division by zero
    X = (X - X_mean) / X_std

    # Handle NaN/Inf values
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

    # Device setup
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Split data
    n_val = int(len(X) * val_split)
    n_train = len(X) - n_val
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, len(X)))
    
    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]

    # Create datasets and loaders
    train_dataset = _PairDataset(X_train, y_train, n_pairs=epochs * (len(X_train) // batch_size), seed=seed)
    val_dataset = _PairDataset(X_val, y_val, n_pairs=5000, seed=seed)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model setup
    model = SiameseNet(X.shape[1], embedding_dim).to(device)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (x1, x2, y_batch) in enumerate(train_loader):
            x1, x2, y_batch = x1.to(device), x2.to(device), y_batch.to(device).float()
            
            optimizer.zero_grad()
            z1, z2 = model(x1), model(x2)
            loss = criterion(z1, z2, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x1, x2, y_batch in val_loader:
                x1, x2, y_batch = x1.to(device), x2.to(device), y_batch.to(device).float()
                z1, z2 = model(x1), model(x2)
                loss = criterion(z1, z2, y_batch)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if output_path:
                os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'embedding_dim': embedding_dim,
                    'feature_dim': X.shape[1],
                    'X_mean': X_mean,
                    'X_std': X_std,
                }, output_path)

    # Final metrics
    model.eval()
    with torch.no_grad():
        # Simple evaluation: compute embedding distances for validation pairs
        distances = []
        labels = []
        for x1, x2, y_batch in val_loader:
            x1, x2 = x1.to(device), x2.to(device)
            z1, z2 = model(x1), model(x2)
            d = torch.nn.functional.pairwise_distance(z1, z2).cpu().numpy()
            distances.extend(d)
            labels.extend(y_batch.numpy())
        
        distances = np.array(distances)
        labels = np.array(labels)
        
        # Simple metric: average distance for positive vs negative pairs
        pos_dist = distances[labels == 1].mean() if np.any(labels == 1) else 0
        neg_dist = distances[labels == 0].mean() if np.any(labels == 0) else 0
        
        metrics = {
            'val_loss': best_val_loss,
            'avg_positive_distance': pos_dist,
            'avg_negative_distance': neg_dist,
            'distance_separation': neg_dist - pos_dist,
        }

    return SiameseTrainResult(model_path=output_path, metrics=metrics)


def evaluate_siamese_from_csv(
    model_path: str,
    csv_path: str,
    target_col: str = 'label',
    device: str = 'auto',
) -> Dict[str, float]:
    """Evaluate Siamese model on CSV features."""
    if not _HAS_TORCH:
        raise ImportError("PyTorch is required for Siamese evaluation")

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = SiameseNet(checkpoint['feature_dim'], checkpoint['embedding_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    # Load and prepare data
    df = pd.read_csv(csv_path)
    X, y = _select_numeric_features(df, target_col)

    # Apply same standardization as training
    X = (X - checkpoint['X_mean']) / checkpoint['X_std']
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

    # Create evaluation dataset
    eval_dataset = _PairDataset(X, y, n_pairs=10000, seed=123)
    eval_loader = DataLoader(eval_dataset, batch_size=256, shuffle=False)

    # Evaluate
    distances = []
    labels = []
    with torch.no_grad():
        for x1, x2, y_batch in eval_loader:
            x1, x2 = x1.to(device), x2.to(device)
            z1, z2 = model(x1), model(x2)
            d = torch.nn.functional.pairwise_distance(z1, z2).cpu().numpy()
            distances.extend(d)
            labels.extend(y_batch.numpy())
    
    distances = np.array(distances)
    labels = np.array(labels)
    
    # Compute metrics
    pos_dist = distances[labels == 1].mean() if np.any(labels == 1) else 0
    neg_dist = distances[labels == 0].mean() if np.any(labels == 0) else 0
    
    # Simple classification using distance threshold
    threshold = (pos_dist + neg_dist) / 2
    pred = (distances < threshold).astype(int)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    try:
        from sklearn.metrics import roc_auc_score
        roc_auc = roc_auc_score(labels, -distances)  # negative distance = similarity
    except:
        roc_auc = 0.0
    
    metrics = {
        'accuracy': accuracy_score(labels, pred),
        'precision': precision_score(labels, pred, zero_division=0),
        'recall': recall_score(labels, pred, zero_division=0),
        'f1': f1_score(labels, pred, zero_division=0),
        'roc_auc': roc_auc,
        'avg_positive_distance': pos_dist,
        'avg_negative_distance': neg_dist,
        'distance_separation': neg_dist - pos_dist,
    }
    
    return metrics
