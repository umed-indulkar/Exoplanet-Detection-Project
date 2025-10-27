from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


@dataclass
class TrainResult:
    model: Any
    metrics: Dict[str, float]


def _split_features(df: pd.DataFrame, target_col: str):
    y = df[target_col].values
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_val, y_train, y_val


def train_baseline(df: pd.DataFrame, *, target_col: str = 'label', model_type: str = 'rf') -> tuple[Any, Dict[str, float]]:
    X_train, X_val, y_train, y_val = _split_features(df, target_col)

    if model_type == 'logreg':
        clf = LogisticRegression(max_iter=1000, n_jobs=None)
    elif model_type == 'rf':
        clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    metrics = {
        'accuracy': float(accuracy_score(y_val, y_pred)),
        'precision': float(precision_score(y_val, y_pred, zero_division=0)),
        'recall': float(recall_score(y_val, y_pred, zero_division=0)),
        'f1': float(f1_score(y_val, y_pred, zero_division=0)),
    }

    # ROC-AUC if probabilities available and both classes present
    try:
        y_prob = clf.predict_proba(X_val)[:, 1]
        metrics['roc_auc'] = float(roc_auc_score(y_val, y_prob))
    except Exception:
        pass

    return clf, metrics


def evaluate_baseline(model: Any, df: pd.DataFrame, *, target_col: str = 'label') -> Dict[str, float]:
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).values
    y = df[target_col].values
    y_pred = model.predict(X)
    metrics = {
        'accuracy': float(accuracy_score(y, y_pred)),
        'precision': float(precision_score(y, y_pred, zero_division=0)),
        'recall': float(recall_score(y, y_pred, zero_division=0)),
        'f1': float(f1_score(y, y_pred, zero_division=0)),
    }
    try:
        y_prob = model.predict_proba(X)[:, 1]
        metrics['roc_auc'] = float(roc_auc_score(y, y_prob))
    except Exception:
        pass
    return metrics


def predict_on_features(model: Any, df: pd.DataFrame) -> np.ndarray:
    X = df.select_dtypes(include=[np.number]).values
    return model.predict(X)


def save_model(model: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    dump(model, path)


def load_model(path: str) -> Any:
    return load(path)
