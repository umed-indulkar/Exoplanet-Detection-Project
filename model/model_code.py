import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import optuna
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from lock_layer_parameters import LockedHyperparams

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize locked hyperparameters 


#important put the number of features here or else it will be locked

##############################################################
##############################################################

num_features =   # Replace later when features are ready here
HYPER = LockedHyperparams(num_features)

###############################################################
###################################################################



class FCNNBackbone(nn.Module):
    def __init__(self, input_dim=HYPER.INPUT_DIM, hidden1=HYPER.HIDDEN1, hidden2=HYPER.HIDDEN2, embedding_dim=HYPER.EMBEDDING_DIM):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.SiLU(),
            nn.Linear(hidden1, hidden2),
            nn.SiLU(),
            nn.Linear(hidden2, embedding_dim),
            nn.SiLU()
        )
        self._initialize_weights()

    def forward(self, x):
        return self.layers(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='silu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class SiameseFCNN(nn.Module):
    def __init__(self, input_dim=HYPER.INPUT_DIM, hidden1=HYPER.HIDDEN1, hidden2=HYPER.HIDDEN2, embedding_dim=HYPER.EMBEDDING_DIM, classifier_hidden=HYPER.CLASSIFIER_HIDDEN):
        super().__init__()
        self.backbone = FCNNBackbone(input_dim, hidden1, hidden2, embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, classifier_hidden),
            nn.SiLU(),
            nn.Linear(classifier_hidden, 1)
        )
        self._initialize_weights()

    def forward(self, x1, x2):
        emb1 = self.backbone(x1)
        emb2 = self.backbone(x2)
        diff = torch.abs(emb1 - emb2)
        logits = self.classifier(diff)
        return logits

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='silu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

# Training helpers
def setup_training(model, lr, optimizer_name='Adam'):
    criterion = nn.BCEWithLogitsLoss()
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == 'RMSProp':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError("Optimizer not recognized")
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    return criterion, optimizer, scheduler

def train_model(model, train_loader, val_loader, epochs=50, patience=5, lr=1e-3, optimizer_name='Adam'):
    criterion, optimizer, scheduler = setup_training(model, lr, optimizer_name)
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0

    history = {'train_loss': [], 'val_loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'auc': []}

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for x1, x2, y in train_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x1, x2).squeeze()
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        avg_train_loss = np.mean(train_losses)

        model.eval()
        val_losses, preds, targets = [], [], []
        with torch.no_grad():
            for x1, x2, y in val_loader:
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                logits = model(x1, x2).squeeze()
                loss = criterion(logits, y)
                val_losses.append(loss.item())
                preds.extend(torch.sigmoid(logits).cpu().numpy())
                targets.extend(y.cpu().numpy())
        avg_val_loss = np.mean(val_losses)

        preds_bin = [1 if p > 0.5 else 0 for p in preds]
        acc = accuracy_score(targets, preds_bin)
        prec = precision_score(targets, preds_bin, zero_division=0)
        rec = recall_score(targets, preds_bin, zero_division=0)
        try:
            auc = roc_auc_score(targets, preds)
        except:
            auc = 0.0

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['accuracy'].append(acc)
        history['precision'].append(prec)
        history['recall'].append(rec)
        history['auc'].append(auc)

        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return history, best_val_loss

# Hyperparameter tuning
def objective(trial, train_loader, val_loader):
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSProp'])
    model = SiameseFCNN()
    _, val_loss = train_model(model, train_loader, val_loader, epochs=20, lr=lr, optimizer_name=optimizer_name)
    return val_loss

def hyperparameter_tuning(train_loader, val_loader, n_trials=10):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, train_loader, val_loader), n_trials=n_trials)
    print("Best trial:", study.best_trial.params)
    return study

def plot_study_analytics(study):
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.show()
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.show()

# # Example DataLoader
# X = torch.randn(200, HYPER.INPUT_DIM)
# Y = torch.randint(0,2,(200,), dtype=torch.float32)
# dataset = TensorDataset(X, X, Y)
# train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
# val_loader = DataLoader(dataset, batch_size=16)