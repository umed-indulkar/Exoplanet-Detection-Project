"""
Evaluation script for Siamese network
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import os
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import argparse

from siamese_model import SiameseNetwork, SiameseClassifier, create_siamese_model
from utils import (
    set_seed, load_model_checkpoint, calculate_metrics, print_metrics,
    plot_confusion_matrix, plot_roc_curve, plot_embedding_space,
    create_results_summary
)


class SiameseEvaluator:
    """Evaluator for Siamese network"""
    
    def __init__(self, model: SiameseNetwork, device: torch.device = None):
        """
        Initialize evaluator
        
        Args:
            model: Trained Siamese network
            device: Computing device
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def extract_embeddings(self, features: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Extract embeddings for all samples
        
        Args:
            features: Feature array
            batch_size: Batch size for processing
            
        Returns:
            Embedding array
        """
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(features), batch_size), desc="Extracting embeddings"):
                batch = features[i:i+batch_size]
                batch_tensor = torch.FloatTensor(batch).to(self.device)
                batch_embeddings = self.model.forward_one(batch_tensor)
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def find_optimal_threshold(self, X1: np.ndarray, X2: np.ndarray, 
                             labels: np.ndarray) -> float:
        """
        Find optimal distance threshold for classification
        
        Args:
            X1: First set of samples
            X2: Second set of samples
            labels: True labels (1 for similar, 0 for dissimilar)
            
        Returns:
            Optimal threshold
        """
        distances = []
        
        with torch.no_grad():
            for i in range(0, len(X1), 32):
                batch_x1 = torch.FloatTensor(X1[i:i+32]).to(self.device)
                batch_x2 = torch.FloatTensor(X2[i:i+32]).to(self.device)
                batch_distances = self.model.get_distance(batch_x1, batch_x2)
                distances.extend(batch_distances.cpu().numpy())
        
        distances = np.array(distances)
        
        # Try different thresholds
        best_threshold = 0.5
        best_accuracy = 0
        
        for threshold in np.linspace(distances.min(), distances.max(), 100):
            predictions = (distances < threshold).astype(int)
            accuracy = np.mean(predictions == labels)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        print(f"Optimal threshold: {best_threshold:.4f} (Accuracy: {best_accuracy:.4f})")
        return best_threshold
    
    def evaluate_pairs(self, X1: np.ndarray, X2: np.ndarray, labels: np.ndarray,
                      threshold: float = 0.5) -> Dict[str, float]:
        """
        Evaluate model on pairs
        
        Args:
            X1: First set of samples
            X2: Second set of samples
            labels: True labels
            threshold: Distance threshold
            
        Returns:
            Dictionary of metrics
        """
        predictions = []
        distances = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(X1), 32), desc="Evaluating"):
                batch_x1 = torch.FloatTensor(X1[i:i+32]).to(self.device)
                batch_x2 = torch.FloatTensor(X2[i:i+32]).to(self.device)
                
                batch_distances = self.model.get_distance(batch_x1, batch_x2)
                batch_predictions = (batch_distances < threshold).float()
                
                distances.extend(batch_distances.cpu().numpy())
                predictions.extend(batch_predictions.cpu().numpy())
        
        predictions = np.array(predictions)
        distances = np.array(distances)
        
        # Calculate metrics
        metrics = calculate_metrics(labels, predictions)
        metrics['mean_distance_positive'] = np.mean(distances[labels == 1])
        metrics['mean_distance_negative'] = np.mean(distances[labels == 0])
        metrics['threshold'] = threshold
        
        return metrics
    
    def classify_samples(self, features: np.ndarray, reference_features: Dict[int, np.ndarray],
                        threshold: float = 0.5) -> np.ndarray:
        """
        Classify samples using nearest neighbor in embedding space
        
        Args:
            features: Features to classify
            reference_features: Reference features for each class
            threshold: Distance threshold
            
        Returns:
            Predicted labels
        """
        embeddings = self.extract_embeddings(features)
        predictions = []
        
        for embedding in tqdm(embeddings, desc="Classifying"):
            min_distances = {}
            
            for class_label, ref_features in reference_features.items():
                ref_embeddings = self.extract_embeddings(ref_features)
                
                # Calculate distances to all reference samples
                embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(self.device)
                ref_tensor = torch.FloatTensor(ref_embeddings).to(self.device)
                
                distances = torch.cdist(embedding_tensor, ref_tensor).squeeze()
                min_distances[class_label] = distances.min().item()
            
            # Predict class with minimum distance
            predicted_class = min(min_distances, key=min_distances.get)
            
            # Check if distance is below threshold
            if min_distances[predicted_class] > threshold:
                predicted_class = -1  # Unknown class
            
            predictions.append(predicted_class)
        
        return np.array(predictions)
    
    def evaluate_classification(self, test_features: pd.DataFrame,
                              train_features: pd.DataFrame,
                              threshold: float = 0.5) -> Dict[str, float]:
        """
        Evaluate classification performance
        
        Args:
            test_features: Test set features with labels
            train_features: Training set features for reference
            threshold: Distance threshold
            
        Returns:
            Dictionary of metrics
        """
        # Prepare test data
        feature_cols = [col for col in test_features.columns if col != 'LABEL']
        X_test = test_features[feature_cols].values
        y_test = test_features['LABEL'].values
        
        # Prepare reference data
        reference_features = {}
        for label in train_features['LABEL'].unique():
            mask = train_features['LABEL'] == label
            reference_features[label] = train_features[mask][feature_cols].values[:100]  # Use subset
        
        # Classify
        predictions = self.classify_samples(X_test, reference_features, threshold)
        
        # Filter out unknown predictions
        known_mask = predictions != -1
        y_test_known = y_test[known_mask]
        predictions_known = predictions[known_mask]
        
        # Calculate metrics
        metrics = calculate_metrics(y_test_known, predictions_known)
        metrics['coverage'] = np.mean(known_mask)  # Percentage of samples classified
        
        return metrics, predictions


def evaluate_model(config: dict):
    """
    Main evaluation function
    
    Args:
        config: Evaluation configuration
    """
    # Set seed
    set_seed(config['seed'])
    
    # Load test data
    print("Loading test data...")
    test_features = pd.read_csv(config['test_features_path'])
    
    if config['test_pairs_path']:
        test_X1 = np.load(os.path.join(config['test_pairs_path'], 'test_X1.npy'))
        test_X2 = np.load(os.path.join(config['test_pairs_path'], 'test_X2.npy'))
        test_labels = np.load(os.path.join(config['test_pairs_path'], 'test_labels.npy'))
        print(f"Test pairs: {len(test_labels)}")
    
    # Create model
    input_dim = len([col for col in test_features.columns if col != 'LABEL'])
    model = create_siamese_model(input_dim, config['model'])
    
    # Load checkpoint
    checkpoint_path = os.path.join(config['checkpoint_dir'], config['checkpoint_name'])
    load_model_checkpoint(model, None, checkpoint_path)
    
    # Create evaluator
    evaluator = SiameseEvaluator(model)
    
    # Evaluate on pairs
    if config['test_pairs_path']:
        print("\nEvaluating on test pairs...")
        
        # Find optimal threshold on validation set if available
        if config['val_pairs_path']:
            val_X1 = np.load(os.path.join(config['val_pairs_path'], 'val_X1.npy'))
            val_X2 = np.load(os.path.join(config['val_pairs_path'], 'val_X2.npy'))
            val_labels = np.load(os.path.join(config['val_pairs_path'], 'val_labels.npy'))
            
            optimal_threshold = evaluator.find_optimal_threshold(val_X1, val_X2, val_labels)
        else:
            optimal_threshold = 0.5
        
        # Evaluate
        pair_metrics = evaluator.evaluate_pairs(test_X1, test_X2, test_labels, optimal_threshold)
        
        print("\nPair Evaluation Results:")
        print_metrics(pair_metrics)
    
    # Evaluate classification
    if config['train_features_path']:
        print("\nEvaluating classification...")
        train_features = pd.read_csv(config['train_features_path'])
        
        classification_metrics, predictions = evaluator.evaluate_classification(
            test_features, train_features, threshold=optimal_threshold if 'optimal_threshold' in locals() else 0.5
        )
        
        print("\nClassification Results:")
        print_metrics(classification_metrics)
        
        # Save predictions
        test_features['predictions'] = predictions
        predictions_path = os.path.join(config['output_dir'], 'test_predictions.csv')
        test_features.to_csv(predictions_path, index=False)
        print(f"Predictions saved to {predictions_path}")
    
    # Extract and visualize embeddings
    if config['visualize']:
        print("\nVisualizing embeddings...")
        feature_cols = [col for col in test_features.columns if col not in ['LABEL', 'predictions']]
        X_test = test_features[feature_cols].values[:1000]  # Use subset for visualization
        y_test = test_features['LABEL'].values[:1000]
        
        embeddings = evaluator.extract_embeddings(X_test)
        
        # Plot embedding space
        embed_plot_path = os.path.join(config['output_dir'], 'embedding_space.png')
        plot_embedding_space(embeddings, y_test, save_path=embed_plot_path)
        
        # Plot confusion matrix if predictions available
        if 'predictions' in test_features.columns:
            y_pred = test_features['predictions'].values[:1000]
            valid_mask = y_pred != -1
            
            cm_plot_path = os.path.join(config['output_dir'], 'confusion_matrix.png')
            plot_confusion_matrix(y_test[valid_mask], y_pred[valid_mask], save_path=cm_plot_path)
    
    # Save results summary
    all_metrics = {}
    if 'pair_metrics' in locals():
        all_metrics['pair_evaluation'] = pair_metrics
    if 'classification_metrics' in locals():
        all_metrics['classification'] = classification_metrics
    
    summary_path = os.path.join(config['output_dir'], 'evaluation_results.json')
    create_results_summary(all_metrics, config['model'], 0, summary_path)
    
    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Siamese Network')
    parser.add_argument('--test_features', type=str, required=True,
                       help='Path to test features CSV')
    parser.add_argument('--checkpoint', type=str, default='best_model.pth',
                       help='Model checkpoint name')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'seed': 42,
        'test_features_path': args.test_features,
        'test_pairs_path': '../data/pairs/',
        'val_pairs_path': '../data/pairs/',
        'train_features_path': '../data/features/train_features.csv',
        'checkpoint_dir': '../outputs/models/',
        'checkpoint_name': args.checkpoint,
        'output_dir': '../outputs/results/',
        'visualize': args.visualize,
        'model': {
            'hidden_dims': [256, 128, 64],
            'embedding_dim': 32,
            'dropout_rate': 0.3
        }
    }
    
    # Evaluate
    metrics = evaluate_model(config)
