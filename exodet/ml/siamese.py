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
        return (pos + neg).mean()


def _device_from_arg(device: str) -> str:
    if device == 'auto':
        if _HAS_TORCH and torch.cuda.is_available():
            return 'cuda'
        return 'cpu'
    return device


def train_siamese_from_csv(
    csv_path: str,
    *,
    target_col: str = 'label',
    embedding_dim: int = 32,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_split: float = 0.2,
    device: str = 'auto',
    output_path: str = 'runs/siamese.pt',
    seed: int = 42,
) -> SiameseTrainResult:
    if not _HAS_TORCH:
        raise ImportError("PyTorch is not installed. Install: pip install torch --index-url https://download.pytorch.org/whl/cpu")

    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in CSV")

    X, y = _select_numeric_features(df, target_col)
    feature_dim = X.shape[1]

    # deterministic split
    n = len(X)
    n_val = int(math.ceil(n * val_split))
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    train_ds = _PairDataset(X_train, y_train, n_pairs=max(20000, len(X_train)))
    val_ds = _PairDataset(X_val, y_val, n_pairs=max(4000, len(X_val)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    dev = _device_from_arg(device)
    model = SiameseNet(feature_dim=feature_dim, embedding_dim=embedding_dim).to(dev)
    criterion = ContrastiveLoss(margin=1.0)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x1, x2, yy in train_loader:
            x1 = x1.to(dev)
            x2 = x2.to(dev)
            yy = yy.float().to(dev)
            optim.zero_grad()
            z1 = model(x1)
            z2 = model(x2)
            loss = criterion(z1, z2, yy)
            loss.backward()
            optim.step()
            total += loss.item() * x1.size(0)
        train_loss = total / len(train_loader.dataset)

        # validation
        model.eval()
        vtotal = 0.0
        with torch.no_grad():
            for x1, x2, yy in val_loader:
                x1 = x1.to(dev)
                x2 = x2.to(dev)
                yy = yy.float().to(dev)
                z1 = model(x1)
                z2 = model(x2)
                vloss = criterion(z1, z2, yy)
                vtotal += vloss.item() * x1.size(0)
        val_loss = vtotal / len(val_loader.dataset)
        if val_loss < best_val:
            best_val = val_loss
            _save_siamese(model, output_path, feature_dim=feature_dim, embedding_dim=embedding_dim)

    return SiameseTrainResult(model_path=output_path, metrics={"val_loss": best_val})


def evaluate_siamese_from_csv(model_path: str, csv_path: str, *, target_col: str = 'label', device: str = 'auto') -> Dict[str, float]:
    if not _HAS_TORCH:
        raise ImportError("PyTorch is not installed. Install: pip install torch --index-url https://download.pytorch.org/whl/cpu")
    model, meta = _load_siamese(model_path, map_location=_device_from_arg(device))
    X, y = _select_numeric_features(pd.read_csv(csv_path), target_col)

    # Build a small evaluation set of pairs
    ds = _PairDataset(X, y, n_pairs=min(10000, max(2000, len(X))))
    loader = DataLoader(ds, batch_size=512, shuffle=False)
    dev = _device_from_arg(device)
    model.to(dev).eval()
    dists = []
    labels = []
    with torch.no_grad():
        for x1, x2, yy in loader:
            x1 = x1.to(dev)
            x2 = x2.to(dev)
            z1 = model(x1)
            z2 = model(x2)
            d = torch.nn.functional.pairwise_distance(z1, z2).cpu().numpy()
            dists.append(d)
            labels.append(yy.numpy())
    import numpy as _np
    from sklearn.metrics import roc_auc_score
    dists = _np.concatenate(dists)
    labels = _np.concatenate(labels)
    # lower distance means positive; invert for AUC by using -d
    auc = float(roc_auc_score(labels, -dists))
    return {"roc_auc": auc}


def _save_siamese(model: 'SiameseNet', path: str, *, feature_dim: int, embedding_dim: int) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    payload = {
        'state_dict': model.state_dict(),
        'feature_dim': feature_dim,
        'embedding_dim': embedding_dim,
    }
    torch.save(payload, path)


def _load_siamese(path: str, map_location: str = 'cpu') -> Tuple['SiameseNet', Dict[str, Any]]:
    payload = torch.load(path, map_location=map_location)
    model = SiameseNet(feature_dim=payload['feature_dim'], embedding_dim=payload['embedding_dim'])
    model.load_state_dict(payload['state_dict'])
    return model, {'feature_dim': payload['feature_dim'], 'embedding_dim': payload['embedding_dim']}
