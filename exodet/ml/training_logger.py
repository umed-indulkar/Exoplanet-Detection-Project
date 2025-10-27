"""
Training logger with comprehensive visualization
Generates plots for loss, accuracy, precision, recall, confusion matrix, ROC curve, etc.
"""
import os
from typing import Dict, List
import json

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
    import numpy as np
    _HAS_VIZ = True
except ImportError:
    _HAS_VIZ = False


class TrainingLogger:
    """Logs training metrics and generates visualization plots"""
    
    def __init__(self, output_dir: str = 'runs/logs'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_precision': [],
            'val_precision': [],
            'train_recall': [],
            'val_recall': [],
            'train_f1': [],
            'val_f1': [],
            'train_auc': [],
            'val_auc': [],
            'epoch': [],
        }
    
    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Log metrics for one epoch"""
        self.history['epoch'].append(epoch)
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def save_history(self):
        """Save training history to JSON"""
        path = os.path.join(self.output_dir, 'training_history.json')
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved: {path}")
    
    def plot_all(self):
        """Generate all visualization plots"""
        if not _HAS_VIZ:
            print("Matplotlib not available. Skipping plots.")
            return
        
        print(f"\nGenerating training visualizations in: {self.output_dir}")
        
        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # Create multi-panel figure
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Loss plot
        ax1 = plt.subplot(2, 3, 1)
        self._plot_loss(ax1)
        
        # 2. Accuracy plot
        ax2 = plt.subplot(2, 3, 2)
        self._plot_accuracy(ax2)
        
        # 3. Precision & Recall
        ax3 = plt.subplot(2, 3, 3)
        self._plot_precision_recall(ax3)
        
        # 4. F1 Score
        ax4 = plt.subplot(2, 3, 4)
        self._plot_f1(ax4)
        
        # 5. AUC
        ax5 = plt.subplot(2, 3, 5)
        self._plot_auc(ax5)
        
        # 6. Summary text
        ax6 = plt.subplot(2, 3, 6)
        self._plot_summary(ax6)
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'training_curves.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {path}")
    
    def _plot_loss(self, ax):
        """Plot training and validation loss"""
        epochs = self.history['epoch']
        if self.history['train_loss']:
            ax.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        if self.history['val_loss']:
            ax.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_accuracy(self, ax):
        """Plot training and validation accuracy"""
        epochs = self.history['epoch']
        if self.history['train_acc']:
            ax.plot(epochs, self.history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
        if self.history['val_acc']:
            ax.plot(epochs, self.history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
        ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='80% baseline')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_precision_recall(self, ax):
        """Plot precision and recall"""
        epochs = self.history['epoch']
        if self.history['train_precision']:
            ax.plot(epochs, self.history['train_precision'], 'b-', label='Train Precision', linewidth=2)
        if self.history['val_precision']:
            ax.plot(epochs, self.history['val_precision'], 'g-', label='Val Precision', linewidth=2)
        if self.history['train_recall']:
            ax.plot(epochs, self.history['train_recall'], 'c-', label='Train Recall', linewidth=2, alpha=0.7)
        if self.history['val_recall']:
            ax.plot(epochs, self.history['val_recall'], 'r-', label='Val Recall', linewidth=2, alpha=0.7)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Precision & Recall', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    def _plot_f1(self, ax):
        """Plot F1 score"""
        epochs = self.history['epoch']
        if self.history['train_f1']:
            ax.plot(epochs, self.history['train_f1'], 'b-', label='Train F1', linewidth=2)
        if self.history['val_f1']:
            ax.plot(epochs, self.history['val_f1'], 'r-', label='Val F1', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title('F1 Score', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_auc(self, ax):
        """Plot AUC score"""
        epochs = self.history['epoch']
        if self.history['train_auc']:
            ax.plot(epochs, self.history['train_auc'], 'b-', label='Train AUC', linewidth=2)
        if self.history['val_auc']:
            ax.plot(epochs, self.history['val_auc'], 'r-', label='Val AUC', linewidth=2)
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect AUC')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('AUC', fontsize=12)
        ax.set_title('Model AUC', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
    
    def _plot_summary(self, ax):
        """Plot summary statistics as text"""
        ax.axis('off')
        
        # Get best metrics
        best_val_loss = min(self.history['val_loss']) if self.history['val_loss'] else 0
        best_val_acc = max(self.history['val_acc']) if self.history['val_acc'] else 0
        best_val_f1 = max(self.history['val_f1']) if self.history['val_f1'] else 0
        best_val_auc = max(self.history['val_auc']) if self.history['val_auc'] else 0
        
        summary_text = f"""
Training Summary
{'='*30}

Best Validation Metrics:
  Loss:      {best_val_loss:.4f}
  Accuracy:  {best_val_acc:.4f} ({best_val_acc*100:.2f}%)
  F1 Score:  {best_val_f1:.4f} ({best_val_f1*100:.2f}%)
  ROC-AUC:   {best_val_auc:.4f} ({best_val_auc*100:.2f}%)

Total Epochs: {len(self.history['epoch'])}
        """
        
        ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=ax.transAxes)


def plot_confusion_matrix(y_true, y_pred, output_path: str = 'runs/logs/confusion_matrix.png'):
    """Plot confusion matrix heatmap"""
    if not _HAS_VIZ:
        return
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['No Exoplanet', 'Exoplanet'],
                yticklabels=['No Exoplanet', 'Exoplanet'])
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_roc_curve(y_true, y_scores, output_path: str = 'runs/logs/roc_curve.png'):
    """Plot ROC curve"""
    if not _HAS_VIZ:
        return
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_precision_recall_curve(y_true, y_scores, output_path: str = 'runs/logs/precision_recall_curve.png'):
    """Plot Precision-Recall curve"""
    if not _HAS_VIZ:
        return
    
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = np.mean(precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'g-', linewidth=2, label=f'AP = {avg_precision:.4f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_metrics_bar(metrics: Dict[str, float], output_path: str = 'runs/logs/metrics_bar.png'):
    """Plot metrics as bar chart"""
    if not _HAS_VIZ:
        return
    
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metric_keys = ['accuracy', 'precision', 'recall', 'f1_score']
    values = [metrics.get(k, 0) for k in metric_keys]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metric_names, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.ylim([0, 1.1])
    plt.ylabel('Score', fontsize=12)
    plt.title('Performance Metrics', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")
