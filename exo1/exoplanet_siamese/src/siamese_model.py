"""
Siamese Neural Network Model for Exoplanet Detection
Implements a fully connected neural network with contrastive loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class FeatureExtractorNetwork(nn.Module):
    """Base network for feature extraction in Siamese architecture"""
    
    def __init__(self, input_dim: int, hidden_dims: list = [256, 128, 64], 
                 dropout_rate: float = 0.3, activation: str = 'relu'):
        """
        Initialize feature extractor network
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            activation: Activation function ('relu', 'leaky_relu', 'elu')
        """
        super(FeatureExtractorNetwork, self).__init__()
        
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Build layers
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Set activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation == 'elu':
            self.activation = F.elu
        else:
            self.activation = F.relu
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x: Input tensor
            
        Returns:
            Embedded features
        """
        for i, (layer, bn, dropout) in enumerate(zip(self.layers, self.batch_norms, self.dropouts)):
            x = layer(x)
            x = bn(x)
            x = self.activation(x)
            x = dropout(x)
        
        return x


class SiameseNetwork(nn.Module):
    """Siamese network for similarity learning"""
    
    def __init__(self, input_dim: int, hidden_dims: list = [256, 128, 64],
                 embedding_dim: int = 32, dropout_rate: float = 0.3):
        """
        Initialize Siamese network
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            embedding_dim: Final embedding dimension
            dropout_rate: Dropout probability
        """
        super(SiameseNetwork, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = FeatureExtractorNetwork(
            input_dim, hidden_dims, dropout_rate
        )
        
        # Final embedding layer
        last_hidden_dim = hidden_dims[-1] if hidden_dims else input_dim
        self.embedding_layer = nn.Linear(last_hidden_dim, embedding_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for one input
        
        Args:
            x: Input tensor
            
        Returns:
            Embedding vector
        """
        features = self.feature_extractor(x)
        embedding = self.embedding_layer(features)
        # L2 normalize embeddings
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for pair of inputs
        
        Args:
            x1: First input tensor
            x2: Second input tensor
            
        Returns:
            Tuple of embeddings for both inputs
        """
        embedding1 = self.forward_one(x1)
        embedding2 = self.forward_one(x2)
        return embedding1, embedding2
    
    def get_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Calculate Euclidean distance between embeddings
        
        Args:
            x1: First input tensor
            x2: Second input tensor
            
        Returns:
            Distance tensor
        """
        embedding1, embedding2 = self.forward(x1, x2)
        distance = torch.sqrt(torch.sum((embedding1 - embedding2) ** 2, dim=1))
        return distance
    
    def predict_similarity(self, x1: torch.Tensor, x2: torch.Tensor, 
                          threshold: float = 0.5) -> torch.Tensor:
        """
        Predict if two inputs are similar
        
        Args:
            x1: First input tensor
            x2: Second input tensor
            threshold: Distance threshold for similarity
            
        Returns:
            Binary predictions (1 for similar, 0 for dissimilar)
        """
        distance = self.get_distance(x1, x2)
        predictions = (distance < threshold).float()
        return predictions


class ContrastiveLoss(nn.Module):
    """Contrastive loss for Siamese network training"""
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize contrastive loss
        
        Args:
            margin: Margin for dissimilar pairs
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, embedding1: torch.Tensor, embedding2: torch.Tensor, 
                label: torch.Tensor) -> torch.Tensor:
        """
        Calculate contrastive loss
        
        Args:
            embedding1: First embedding tensor
            embedding2: Second embedding tensor
            label: Label tensor (1 for similar, 0 for dissimilar)
            
        Returns:
            Loss value
        """
        # Calculate Euclidean distance
        distance = torch.sqrt(torch.sum((embedding1 - embedding2) ** 2, dim=1) + 1e-6)
        
        # Contrastive loss
        loss = label * torch.pow(distance, 2) + \
               (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        
        return torch.mean(loss)


class TripletLoss(nn.Module):
    """Triplet loss for Siamese network training"""
    
    def __init__(self, margin: float = 0.5):
        """
        Initialize triplet loss
        
        Args:
            margin: Margin between positive and negative distances
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        """
        Calculate triplet loss
        
        Args:
            anchor: Anchor embedding
            positive: Positive embedding (same class)
            negative: Negative embedding (different class)
            
        Returns:
            Loss value
        """
        # Calculate distances
        pos_distance = torch.sqrt(torch.sum((anchor - positive) ** 2, dim=1) + 1e-6)
        neg_distance = torch.sqrt(torch.sum((anchor - negative) ** 2, dim=1) + 1e-6)
        
        # Triplet loss
        loss = torch.clamp(pos_distance - neg_distance + self.margin, min=0.0)
        
        return torch.mean(loss)


class SiameseClassifier(nn.Module):
    """Siamese network with classification head"""
    
    def __init__(self, siamese_network: SiameseNetwork, num_classes: int = 2):
        """
        Initialize Siamese classifier
        
        Args:
            siamese_network: Pre-trained Siamese network
            num_classes: Number of output classes
        """
        super(SiameseClassifier, self).__init__()
        
        self.siamese = siamese_network
        embedding_dim = siamese_network.embedding_layer.out_features
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification
        
        Args:
            x: Input tensor
            
        Returns:
            Class logits
        """
        # Get embedding from Siamese network
        embedding = self.siamese.forward_one(x)
        
        # Classify
        logits = self.classifier(embedding)
        
        return logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted class labels
        """
        logits = self.forward(x)
        predictions = torch.argmax(logits, dim=1)
        return predictions


def create_siamese_model(input_dim: int, config: dict = None) -> SiameseNetwork:
    """
    Create Siamese network with specified configuration
    
    Args:
        input_dim: Input feature dimension
        config: Model configuration dictionary
        
    Returns:
        Siamese network model
    """
    if config is None:
        config = {
            'hidden_dims': [256, 128, 64],
            'embedding_dim': 32,
            'dropout_rate': 0.3
        }
    
    model = SiameseNetwork(
        input_dim=input_dim,
        hidden_dims=config.get('hidden_dims', [256, 128, 64]),
        embedding_dim=config.get('embedding_dim', 32),
        dropout_rate=config.get('dropout_rate', 0.3)
    )
    
    return model


if __name__ == "__main__":
    # Example usage
    input_dim = 50  # Number of features
    batch_size = 32
    
    # Create model
    model = create_siamese_model(input_dim)
    
    # Create dummy data
    x1 = torch.randn(batch_size, input_dim)
    x2 = torch.randn(batch_size, input_dim)
    labels = torch.randint(0, 2, (batch_size,)).float()
    
    # Forward pass
    embedding1, embedding2 = model(x1, x2)
    
    # Calculate loss
    criterion = ContrastiveLoss(margin=1.0)
    loss = criterion(embedding1, embedding2, labels)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Loss: {loss.item():.4f}")
