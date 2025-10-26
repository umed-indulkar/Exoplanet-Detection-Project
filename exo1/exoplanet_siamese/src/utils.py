"""
Utility functions for exoplanet detection project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
import torch
from typing import Dict, List, Tuple, Optional
import os
import json


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_light_curve(flux: np.ndarray, title: str = "Light Curve", 
                    save_path: Optional[str] = None):
    """
    Plot a light curve
    
    Args:
        flux: Array of flux measurements
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(12, 4))
    time = np.arange(len(flux))
    
    plt.plot(time, flux, 'b-', linewidth=0.5, alpha=0.8)
    plt.xlabel('Time Index')
    plt.ylabel('Normalized Flux')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Plot training history
    
    Args:
        history: Dictionary with training metrics
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    if 'train_acc' in history:
        axes[1].plot(history['train_acc'], label='Train Accuracy')
    if 'val_acc' in history:
        axes[1].plot(history['val_acc'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: List[str] = ['No Planet', 'Planet'],
                         save_path: Optional[str] = None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        save_path: Path to save figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, 
                  save_path: Optional[str] = None):
    """
    Plot ROC curve
    
    Args:
        y_true: True labels
        y_scores: Prediction scores
        save_path: Path to save figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'r--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_embedding_space(embeddings: np.ndarray, labels: np.ndarray, 
                        title: str = "Embedding Space",
                        save_path: Optional[str] = None):
    """
    Plot 2D embedding space using t-SNE or PCA
    
    Args:
        embeddings: Embedding vectors
        labels: Class labels
        title: Plot title
        save_path: Path to save figure
    """
    from sklearn.manifold import TSNE
    
    # Reduce to 2D if needed
    if embeddings.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
    else:
        embeddings_2d = embeddings
    
    plt.figure(figsize=(10, 8))
    
    # Plot each class
    unique_labels = np.unique(labels)
    colors = ['blue', 'red', 'green', 'orange']
    markers = ['o', 's', '^', 'D']
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=colors[i % len(colors)], 
                   marker=markers[i % len(markers)],
                   label=f'Class {label}', alpha=0.6, s=30)
    
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    # Convert labels to binary if needed (1 vs 2 -> 0 vs 1)
    y_true_binary = (y_true == 2).astype(int) if y_true.max() == 2 else y_true
    y_pred_binary = (y_pred == 2).astype(int) if y_pred.max() == 2 else y_pred
    
    metrics = {
        'accuracy': accuracy_score(y_true_binary, y_pred_binary),
        'precision': precision_score(y_true_binary, y_pred_binary, average='binary'),
        'recall': recall_score(y_true_binary, y_pred_binary, average='binary'),
        'f1': f1_score(y_true_binary, y_pred_binary, average='binary')
    }
    
    return metrics


def print_metrics(metrics: Dict[str, float]):
    """Print metrics in a formatted way"""
    print("\n" + "="*50)
    print("Performance Metrics:")
    print("="*50)
    for name, value in metrics.items():
        print(f"{name.capitalize():15s}: {value:.4f}")
    print("="*50)


def save_model_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                         epoch: int, loss: float, metrics: Dict[str, float],
                         filepath: str):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        metrics: Current metrics
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_model_checkpoint(model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer],
                         filepath: str, device: torch.device = torch.device('cpu')):
    """
    Load model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer (optional)
        filepath: Path to checkpoint
        device: Device to load to
        
    Returns:
        Loaded epoch and metrics
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    print(f"Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
    
    return checkpoint['epoch'], checkpoint.get('metrics', {})


def create_results_summary(test_metrics: Dict[str, float], 
                          model_config: Dict, 
                          training_time: float,
                          save_path: str):
    """
    Create and save results summary
    
    Args:
        test_metrics: Test set metrics
        model_config: Model configuration
        training_time: Training duration
        save_path: Path to save summary
    """
    summary = {
        'test_metrics': test_metrics,
        'model_config': model_config,
        'training_time_seconds': training_time,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"Results summary saved to {save_path}")


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'min'):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
    
    def __call__(self, value: float) -> bool:
        """
        Check if should stop training
        
        Args:
            value: Current metric value
            
        Returns:
            True if should stop
        """
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'min':
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


if __name__ == "__main__":
    # Example usage
    print("Utility functions loaded successfully")
