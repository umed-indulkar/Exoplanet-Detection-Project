import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import os

class FeedforwardNN(nn.Module):
    """Feedforward Neural Network for exoplanet detection"""
    
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], dropout_rate=0.3):
        super(FeedforwardNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class ConvolutionalNN(nn.Module):
    """1D Convolutional Neural Network for time series data"""
    
    def __init__(self, input_size, num_filters=[64, 128, 256], kernel_sizes=[3, 5, 7], dropout_rate=0.3):
        super(ConvolutionalNN, self).__init__()
        
        # Reshape input for 1D CNN: (batch, 1, sequence_length)
        self.conv_layers = nn.ModuleList()
        
        in_channels = 1
        for i, (num_filter, kernel_size) in enumerate(zip(num_filters, kernel_sizes)):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, num_filter, kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(num_filter),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout_rate)
                )
            )
            in_channels = num_filter
        
        # Calculate the size after convolutions
        self._calculate_conv_output_size(input_size)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.conv_output_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def _calculate_conv_output_size(self, input_size):
        """Calculate output size after convolutional layers"""
        size = input_size
        for conv_layer in self.conv_layers:
            # Each conv layer has Conv1d + MaxPool1d(2)
            size = size // 2  # MaxPool1d with kernel_size=2
        self.conv_output_size = size * self.conv_layers[-1][0].out_channels
    
    def forward(self, x):
        # Reshape for 1D CNN: (batch, sequence_length) -> (batch, 1, sequence_length)
        x = x.unsqueeze(1)
        
        # Apply convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = self.fc_layers(x)
        
        return x

class NeuralNetworkTrainer:
    """Trainer for neural network models"""
    
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
    
    def train_epoch(self, optimizer, criterion):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in self.train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def test(self, criterion):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                
                total_loss += loss.item()
                predicted = (outputs.squeeze() > 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def train(self, epochs=50, lr=0.001):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        
        print(f"🚀 Training for {epochs} epochs...")
        
        best_test_accuracy = 0
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(optimizer, criterion)
            test_loss, test_acc = self.test(criterion)
            
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            self.train_accuracies.append(train_acc)
            self.test_accuracies.append(test_acc)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
            
            # Early stopping based on test accuracy
            if test_acc > best_test_accuracy:
                best_test_accuracy = test_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), f'models/{self.model.__class__.__name__.lower()}_best.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Plot training curves
        self.plot_training_curves()
        
        return best_test_accuracy
    
    def plot_training_curves(self):
        """Plot training and test curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.test_losses, label='Test Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Test Loss')
        ax1.legend()
        
        # Accuracy curves
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.test_accuracies, label='Test Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Test Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('output/training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

class NeuralNetworkModels:
    """Neural network models for exoplanet detection"""
    
    def __init__(self, data_path="data/extracted_features/features_curve_500_pruned.csv"):
        self.data_path = data_path
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and preprocess the dataset"""
        print("📊 Loading dataset_500...")
        
        # Load data
        data = pd.read_csv(self.data_path, low_memory=False)
        print(f"   Raw shape: {data.shape}")
        
        # Extract features and labels
        if 'Label' in data.columns:
            labels = data['Label'].values
            feature_columns = [col for col in data.columns if col not in ['kepid', 'Label']]
            features = data[feature_columns].values
        else:
            labels = data.iloc[:, 0].values
            features = data.iloc[:, 1:].values
        
        # Convert to proper dtypes
        labels = pd.Series(labels).fillna(0).astype(int).values
        features = pd.DataFrame(features).fillna(0).astype(np.float64).values
        
        print(f"   Features: {features.shape[1]}")
        print(f"   Samples: {len(labels)}")
        print(f"   Positives: {np.sum(labels == 1)}")
        print(f"   Negatives: {np.sum(labels == 0)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, stratify=labels, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_train_tensor = torch.FloatTensor(y_train)
        y_test_tensor = torch.FloatTensor(y_test)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        return train_loader, test_loader, X_train_scaled.shape[1]
    
    def train_feedforward_nn(self):
        """Train Feedforward Neural Network"""
        print("\n🧠 Training Feedforward Neural Network...")
        
        train_loader, test_loader, input_size = self.load_data()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = FeedforwardNN(input_size).to(device)
        
        trainer = NeuralNetworkTrainer(model, train_loader, test_loader, device)
        accuracy = trainer.train(epochs=50, lr=0.001)
        
        print(f"   Best Test Accuracy: {accuracy:.4f}")
        
        # Save final model
        torch.save(model.state_dict(), 'models/feedforward_nn.pth')
        print(f"   Model saved: models/feedforward_nn.pth")
        
        return accuracy
    
    def train_cnn(self):
        """Train Convolutional Neural Network"""
        print("\n🌊 Training Convolutional Neural Network...")
        
        train_loader, test_loader, input_size = self.load_data()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ConvolutionalNN(input_size).to(device)
        
        trainer = NeuralNetworkTrainer(model, train_loader, test_loader, device)
        accuracy = trainer.train(epochs=50, lr=0.001)
        
        print(f"   Best Test Accuracy: {accuracy:.4f}")
        
        # Save final model
        torch.save(model.state_dict(), 'models/convolutional_nn.pth')
        print(f"   Model saved: models/convolutional_nn.pth")
        
        return accuracy
    
    def train_all_models(self):
        """Train all neural network models"""
        print("🚀 TRAINING NEURAL NETWORK MODELS")
        print("=" * 50)
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('output', exist_ok=True)
        
        # Train models
        results = {}
        results['feedforward_nn'] = self.train_feedforward_nn()
        results['convolutional_nn'] = self.train_cnn()
        
        # Save scaler
        import joblib
        joblib.dump(self.scaler, 'models/nn_scaler.pkl')
        print(f"\n💾 Scaler saved: models/nn_scaler.pkl")
        
        # Summary
        print("\n📊 NEURAL NETWORK RESULTS")
        print("=" * 50)
        for model, accuracy in results.items():
            print(f"{model:20s}: {accuracy:.4f}")
        
        return results

if __name__ == "__main__":
    # Train neural network models
    nn_models = NeuralNetworkModels()
    results = nn_models.train_all_models()
