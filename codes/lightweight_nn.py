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

class LightweightFFNN(nn.Module):
    """Lightweight Feedforward Neural Network"""
    
    def __init__(self, input_size):
        super(LightweightFFNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)

class LightweightCNN(nn.Module):
    """Lightweight 1D CNN for faster training"""
    
    def __init__(self, input_size):
        super(LightweightCNN, self).__init__()
        
        # Simple 1D CNN architecture
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        
        # Calculate flattened size
        conv_output_size = (input_size // 4) * 64
        
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Reshape for CNN: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

def quick_train(model, train_loader, test_loader, model_name, epochs=20):
    """Quick training function"""
    
    print(f"\n🧠 Training {model_name}...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Testing
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()
                
                predicted = (outputs > 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        accuracy = correct / total
        
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Acc: {accuracy:.4f}")
    
    # Final evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x).squeeze()
            predicted = (outputs > 0.5).float()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"   ✅ {model_name} Accuracy: {accuracy:.4f}")
    
    # Save model
    torch.save(model.state_dict(), f'models/{model_name.lower()}.pth')
    print(f"   💾 Model saved: models/{model_name.lower()}.pth")
    
    return accuracy

def main():
    """Main function to train neural networks"""
    
    print("🚀 LIGHTWEIGHT NEURAL NETWORK TRAINING")
    print("=" * 50)
    
    # Load data
    print("📊 Loading dataset_500...")
    data = pd.read_csv("data/dataset_500/dataset_500/raw_curve_500_head.csv", low_memory=False)
    
    # Extract features and labels
    labels = data['Label'].values
    feature_columns = [col for col in data.columns if col not in ['kepid', 'Label']]
    features = data[feature_columns].values
    
    # Convert to proper dtypes
    labels = pd.Series(labels).fillna(0).astype(int).values
    features = pd.DataFrame(features).fillna(0).astype(np.float64).values
    
    print(f"   Features: {features.shape[1]}")
    print(f"   Samples: {len(labels)}")
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_train_tensor = torch.FloatTensor(y_train)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    # Train models
    results = {}
    
    # Feedforward Neural Network
    ffnn = LightweightFFNN(X_train_scaled.shape[1])
    results['FeedforwardNN'] = quick_train(ffnn, train_loader, test_loader, 'FeedforwardNN', epochs=20)
    
    # Convolutional Neural Network
    cnn = LightweightCNN(X_train_scaled.shape[1])
    results['CNN'] = quick_train(cnn, train_loader, test_loader, 'CNN', epochs=20)
    
    # Save scaler
    import joblib
    joblib.dump(scaler, 'models/nn_scaler.pkl')
    
    # Summary
    print("\n📊 NEURAL NETWORK RESULTS")
    print("=" * 50)
    for model, accuracy in results.items():
        print(f"{model:15s}: {accuracy:.4f}")
    
    # Save results
    with open('output/nn_results.txt', 'w') as f:
        f.write("NEURAL NETWORK RESULTS\n")
        f.write("=" * 30 + "\n")
        for model, accuracy in results.items():
            f.write(f"{model}: {accuracy:.4f}\n")
    
    print(f"\n📄 Results saved: output/nn_results.txt")
    
    return results

if __name__ == "__main__":
    results = main()
