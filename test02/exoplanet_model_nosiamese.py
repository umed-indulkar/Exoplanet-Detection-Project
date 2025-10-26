import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
import optuna
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from collections import Counter

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ImprovedExoplanetClassifier(nn.Module):
    """
    Optimized architecture for tsfresh features classification
    Uses residual connections, batch normalization, and dropout
    """
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64], 
                 dropout_rate=0.3, use_batch_norm=True):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(nn.GELU())  # GELU often works better than SiLU for tabular data
            
            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(32, 1)
        )
        
        self._initialize_weights()
    
    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits.squeeze()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Better than BCEWithLogitsLoss for imbalanced datasets
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def create_balanced_dataloader(X, y, batch_size=32, is_train=True):
    """
    Creates a dataloader with weighted sampling for class balance
    """
    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    
    if is_train:
        # Calculate class weights
        class_counts = Counter(y.astype(int))
        total = len(y)
        weights = [total / class_counts[int(label)] for label in y]
        sampler = WeightedRandomSampler(weights, len(weights))
        
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def setup_training(model, lr, weight_decay=1e-4, optimizer_name='AdamW'):
    """
    Setup with Focal Loss and advanced optimizers
    """
    criterion = FocalLoss(alpha=0.75, gamma=2.0)  # Adjust alpha based on your class ratio
    
    if optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, 
                             weight_decay=weight_decay, nesterov=True)
    else:
        raise ValueError("Optimizer not recognized")
    
    # Cosine annealing with warm restarts
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=lr/100
    )
    
    return criterion, optimizer, scheduler


def train_model(model, train_loader, val_loader, epochs=100, patience=15, 
                lr=1e-3, weight_decay=1e-4, optimizer_name='AdamW'):
    """
    Enhanced training with better metrics tracking
    """
    criterion, optimizer, scheduler = setup_training(model, lr, weight_decay, optimizer_name)
    model.to(device)
    
    best_f1 = 0.0  # Track F1 instead of just loss
    best_val_loss = float('inf')
    patience_counter = 0
    
    history = {
        'train_loss': [], 'val_loss': [], 'accuracy': [], 
        'precision': [], 'recall': [], 'f1': [], 'auc': []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation phase
        model.eval()
        val_losses, preds, targets = [], [], []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_losses.append(loss.item())
                
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.extend(probs)
                targets.extend(y.cpu().numpy())
        
        avg_val_loss = np.mean(val_losses)
        
        # Calculate metrics with optimal threshold
        targets_arr = np.array(targets)
        preds_arr = np.array(preds)
        
        # Find optimal threshold based on F1 score
        thresholds = np.arange(0.3, 0.7, 0.05)
        best_threshold = 0.5
        best_threshold_f1 = 0
        
        for thresh in thresholds:
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
        
        scheduler.step()
        
        # Early stopping based on F1 score
        if f1 > best_f1:
            best_f1 = f1
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_exoplanet_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Val Loss={avg_val_loss:.4f}, "
                  f"F1={f1:.4f}, AUC={auc:.4f}")
    
    return history, best_val_loss, best_f1


def objective(trial, train_loader, val_loader, input_dim):
    """
    Enhanced Optuna objective with more hyperparameters
    """
    # Hyperparameter search space
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    optimizer_name = trial.suggest_categorical('optimizer', ['AdamW', 'Adam'])
    
    # Architecture parameters
    n_layers = trial.suggest_int('n_layers', 3, 5)
    hidden_dim_base = trial.suggest_int('hidden_dim_base', 256, 768, step=128)
    
    # Create layer dimensions with geometric decay
    hidden_dims = [hidden_dim_base // (2**i) for i in range(n_layers)]
    
    use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
    
    model = ImprovedExoplanetClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm
    )
    
    history, val_loss, val_f1 = train_model(
        model, train_loader, val_loader, 
        epochs=50, patience=10,
        lr=lr, weight_decay=weight_decay,
        optimizer_name=optimizer_name
    )
    
    # Optimize for F1 score (maximize) by returning negative
    return -val_f1


def hyperparameter_tuning(train_loader, val_loader, input_dim, n_trials=50):
    """
    Run comprehensive hyperparameter search
    """
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader, input_dim),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print("\n" + "="*50)
    print("Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    print(f"\nBest F1 Score: {-study.best_trial.value:.4f}")
    print("="*50)
    
    return study


# Example usage
if __name__ == "__main__":
    # Replace with your actual data
    num_features = 794  # Typical tsfresh feature count
    n_samples = 1000
    
    # Simulated data (replace with your actual tsfresh features)
    X_train = np.random.randn(n_samples, num_features)
    y_train = np.random.binomial(1, 0.1, n_samples)  # 10% positive class
    
    X_val = np.random.randn(n_samples//5, num_features)
    y_val = np.random.binomial(1, 0.1, n_samples//5)
    
    # Create balanced dataloaders
    train_loader = create_balanced_dataloader(X_train, y_train, batch_size=32, is_train=True)
    val_loader = create_balanced_dataloader(X_val, y_val, batch_size=64, is_train=False)
    
    # Train with default hyperparameters
    model = ImprovedExoplanetClassifier(input_dim=num_features)
    history, val_loss, val_f1 = train_model(
        model, train_loader, val_loader, 
        epochs=100, lr=1e-3
    )
    
    print(f"\nFinal Results: Val Loss={val_loss:.4f}, F1={val_f1:.4f}")
    
    # Uncomment to run hyperparameter tuning
    # study = hyperparameter_tuning(train_loader, val_loader, num_features, n_trials=30)
