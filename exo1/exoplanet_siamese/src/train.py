"""
Training script for Siamese network
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import os
import time
from tqdm import tqdm
import argparse

from siamese_model import SiameseNetwork, ContrastiveLoss, create_siamese_model
from pair_generation import SiamesePairGenerator
from utils import (
    set_seed, plot_training_history, save_model_checkpoint,
    EarlyStopping, calculate_metrics, print_metrics
)


class SiameseTrainer:
    """Trainer for Siamese network"""
    
    def __init__(self, model: SiameseNetwork, device: torch.device = None):
        """
        Initialize trainer
        
        Args:
            model: Siamese network model
            device: Computing device
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
    def create_data_loader(self, X1: np.ndarray, X2: np.ndarray, labels: np.ndarray,
                          batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """
        Create PyTorch DataLoader from numpy arrays
        
        Args:
            X1: First set of samples
            X2: Second set of samples
            labels: Pair labels
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Returns:
            DataLoader object
        """
        X1_tensor = torch.FloatTensor(X1)
        X2_tensor = torch.FloatTensor(X2)
        labels_tensor = torch.FloatTensor(labels)
        
        dataset = TensorDataset(X1_tensor, X2_tensor, labels_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
        return loader
    
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer,
                   criterion: nn.Module) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            
        Returns:
            Average loss and accuracy
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (x1, x2, labels) in enumerate(tqdm(train_loader, desc="Training")):
            x1, x2, labels = x1.to(self.device), x2.to(self.device), labels.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            embedding1, embedding2 = self.model(x1, x2)
            
            # Calculate loss
            loss = criterion(embedding1, embedding2, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            
            # Calculate accuracy
            with torch.no_grad():
                distance = torch.sqrt(torch.sum((embedding1 - embedding2) ** 2, dim=1))
                predictions = (distance < 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """
        Validate model
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Average loss and accuracy
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x1, x2, labels in tqdm(val_loader, desc="Validation"):
                x1, x2, labels = x1.to(self.device), x2.to(self.device), labels.to(self.device)
                
                # Forward pass
                embedding1, embedding2 = self.model(x1, x2)
                
                # Calculate loss
                loss = criterion(embedding1, embedding2, labels)
                total_loss += loss.item()
                
                # Calculate accuracy
                distance = torch.sqrt(torch.sum((embedding1 - embedding2) ** 2, dim=1))
                predictions = (distance < 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
             epochs: int = 100, learning_rate: float = 0.001,
             weight_decay: float = 1e-5, margin: float = 1.0,
             early_stopping_patience: int = 10,
             checkpoint_dir: str = '../outputs/models/'):
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            margin: Margin for contrastive loss
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory to save checkpoints
        """
        # Setup optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        criterion = ContrastiveLoss(margin=margin)
        
        # Early stopping
        early_stopping = EarlyStopping(patience=early_stopping_patience, mode='min')
        
        # Training loop
        best_val_loss = float('inf')
        
        print(f"Training on {self.device}")
        print("="*50)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            # Validate
            if val_loader:
                val_loss, val_acc = self.validate(val_loader, criterion)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
                    save_model_checkpoint(
                        self.model, optimizer, epoch, val_loss,
                        {'accuracy': val_acc}, checkpoint_path
                    )
                
                # Early stopping
                if early_stopping(val_loss):
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break
        
        # Save final model
        final_path = os.path.join(checkpoint_dir, 'final_model.pth')
        save_model_checkpoint(
            self.model, optimizer, epoch, 
            self.history['train_loss'][-1],
            {'accuracy': self.history['train_acc'][-1]}, 
            final_path
        )
        
        return self.history


def train_siamese_network(config: dict):
    """
    Main training function
    
    Args:
        config: Training configuration
    """
    # Set seed
    set_seed(config['seed'])
    
    # Load pairs
    print("Loading training pairs...")
    generator = SiamesePairGenerator()
    
    train_X1 = np.load(os.path.join(config['pairs_dir'], 'train_X1.npy'))
    train_X2 = np.load(os.path.join(config['pairs_dir'], 'train_X2.npy'))
    train_labels = np.load(os.path.join(config['pairs_dir'], 'train_labels.npy'))
    
    if os.path.exists(os.path.join(config['pairs_dir'], 'val_X1.npy')):
        val_X1 = np.load(os.path.join(config['pairs_dir'], 'val_X1.npy'))
        val_X2 = np.load(os.path.join(config['pairs_dir'], 'val_X2.npy'))
        val_labels = np.load(os.path.join(config['pairs_dir'], 'val_labels.npy'))
    else:
        val_X1, val_X2, val_labels = None, None, None
    
    print(f"Training pairs: {len(train_labels)}")
    if val_X1 is not None:
        print(f"Validation pairs: {len(val_labels)}")
    
    # Create model
    input_dim = train_X1.shape[1]
    model = create_siamese_model(input_dim, config['model'])
    
    # Create trainer
    trainer = SiameseTrainer(model)
    
    # Create data loaders
    train_loader = trainer.create_data_loader(
        train_X1, train_X2, train_labels,
        batch_size=config['batch_size'], shuffle=True
    )
    
    val_loader = None
    if val_X1 is not None:
        val_loader = trainer.create_data_loader(
            val_X1, val_X2, val_labels,
            batch_size=config['batch_size'], shuffle=False
        )
    
    # Train
    print("\nStarting training...")
    start_time = time.time()
    
    history = trainer.train(
        train_loader, val_loader,
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        margin=config['margin'],
        early_stopping_patience=config['early_stopping_patience'],
        checkpoint_dir=config['checkpoint_dir']
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Plot history
    plot_path = os.path.join(config['output_dir'], 'training_history.png')
    plot_training_history(history, plot_path)
    
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Siamese Network')
    parser.add_argument('--pairs_dir', type=str, default='../data/pairs/',
                       help='Directory containing training pairs')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--margin', type=float, default=1.0,
                       help='Margin for contrastive loss')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'seed': 42,
        'pairs_dir': args.pairs_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': 1e-5,
        'margin': args.margin,
        'early_stopping_patience': 10,
        'checkpoint_dir': '../outputs/models/',
        'output_dir': '../outputs/',
        'model': {
            'hidden_dims': [256, 128, 64],
            'embedding_dim': 32,
            'dropout_rate': 0.3
        }
    }
    
    # Train
    model, history = train_siamese_network(config)
