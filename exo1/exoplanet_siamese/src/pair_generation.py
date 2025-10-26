"""
Pair Generation Module for Siamese Network Training
Creates positive and negative pairs from feature vectors
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.utils import shuffle
import os
from tqdm import tqdm


class SiamesePairGenerator:
    """Generate training pairs for Siamese network"""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize pair generator
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
    def create_pairs(self, features_df: pd.DataFrame, 
                    num_pairs: Optional[int] = None,
                    positive_ratio: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create positive and negative pairs from features
        
        Args:
            features_df: DataFrame with features and labels
            num_pairs: Total number of pairs to generate (None for all possible)
            positive_ratio: Ratio of positive pairs (same class)
            
        Returns:
            Tuple of (X1, X2, labels) where labels=1 for positive pairs
        """
        # Separate features and labels
        feature_cols = [col for col in features_df.columns if col != 'LABEL']
        X = features_df[feature_cols].values
        y = features_df['LABEL'].values
        
        # Get indices for each class
        planet_indices = np.where(y == 2)[0]  # Label 2 for planets
        no_planet_indices = np.where(y == 1)[0]  # Label 1 for no planets
        
        print(f"Found {len(planet_indices)} planet samples and {len(no_planet_indices)} non-planet samples")
        
        # Calculate number of positive and negative pairs
        if num_pairs is None:
            # Generate balanced dataset
            max_positive = min(len(planet_indices) * (len(planet_indices) - 1) // 2,
                             len(no_planet_indices) * (len(no_planet_indices) - 1) // 2)
            max_negative = len(planet_indices) * len(no_planet_indices)
            num_pairs = min(max_positive * 2, max_negative * 2)
        
        num_positive = int(num_pairs * positive_ratio)
        num_negative = num_pairs - num_positive
        
        print(f"Generating {num_positive} positive and {num_negative} negative pairs")
        
        # Generate positive pairs (same class)
        positive_pairs = []
        
        # Positive pairs from planet class
        planet_positive = num_positive // 2
        for _ in range(planet_positive):
            idx1, idx2 = np.random.choice(planet_indices, 2, replace=False)
            positive_pairs.append((X[idx1], X[idx2], 1))
        
        # Positive pairs from no-planet class
        no_planet_positive = num_positive - planet_positive
        for _ in range(no_planet_positive):
            idx1, idx2 = np.random.choice(no_planet_indices, 2, replace=False)
            positive_pairs.append((X[idx1], X[idx2], 1))
        
        # Generate negative pairs (different classes)
        negative_pairs = []
        for _ in range(num_negative):
            idx1 = np.random.choice(planet_indices)
            idx2 = np.random.choice(no_planet_indices)
            # Randomly swap to avoid bias
            if np.random.random() > 0.5:
                idx1, idx2 = idx2, idx1
            negative_pairs.append((X[idx1], X[idx2], 0))
        
        # Combine all pairs
        all_pairs = positive_pairs + negative_pairs
        
        # Shuffle pairs
        np.random.shuffle(all_pairs)
        
        # Separate into arrays
        X1 = np.array([pair[0] for pair in all_pairs])
        X2 = np.array([pair[1] for pair in all_pairs])
        labels = np.array([pair[2] for pair in all_pairs])
        
        return X1, X2, labels
    
    def create_balanced_pairs(self, features_df: pd.DataFrame,
                            pairs_per_sample: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create balanced pairs ensuring each sample appears equally often
        
        Args:
            features_df: DataFrame with features and labels
            pairs_per_sample: Number of pairs to generate per sample
            
        Returns:
            Tuple of (X1, X2, labels)
        """
        feature_cols = [col for col in features_df.columns if col != 'LABEL']
        X = features_df[feature_cols].values
        y = features_df['LABEL'].values
        
        pairs = []
        
        for i in tqdm(range(len(X)), desc="Creating balanced pairs"):
            current_label = y[i]
            
            # Generate positive pairs
            same_class_indices = np.where(y == current_label)[0]
            same_class_indices = same_class_indices[same_class_indices != i]
            
            if len(same_class_indices) > 0:
                positive_samples = min(pairs_per_sample // 2, len(same_class_indices))
                selected_positive = np.random.choice(same_class_indices, positive_samples, replace=False)
                
                for j in selected_positive:
                    pairs.append((X[i], X[j], 1))
            
            # Generate negative pairs
            different_class_indices = np.where(y != current_label)[0]
            
            if len(different_class_indices) > 0:
                negative_samples = pairs_per_sample - (pairs_per_sample // 2)
                selected_negative = np.random.choice(different_class_indices, negative_samples, replace=True)
                
                for j in selected_negative:
                    pairs.append((X[i], X[j], 0))
        
        # Shuffle pairs
        np.random.shuffle(pairs)
        
        # Separate into arrays
        X1 = np.array([pair[0] for pair in pairs])
        X2 = np.array([pair[1] for pair in pairs])
        labels = np.array([pair[2] for pair in pairs])
        
        print(f"Generated {len(pairs)} total pairs")
        print(f"Positive pairs: {np.sum(labels)}, Negative pairs: {len(labels) - np.sum(labels)}")
        
        return X1, X2, labels
    
    def create_hard_pairs(self, features_df: pd.DataFrame, 
                         model=None, 
                         num_pairs: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create hard positive and negative pairs using model predictions
        
        Args:
            features_df: DataFrame with features and labels
            model: Trained model to find hard examples (optional)
            num_pairs: Number of pairs to generate
            
        Returns:
            Tuple of (X1, X2, labels)
        """
        feature_cols = [col for col in features_df.columns if col != 'LABEL']
        X = features_df[feature_cols].values
        y = features_df['LABEL'].values
        
        if model is None:
            # Without model, use distance-based hard mining
            return self._create_distance_based_hard_pairs(X, y, num_pairs)
        else:
            # With model, use prediction-based hard mining
            return self._create_model_based_hard_pairs(X, y, model, num_pairs)
    
    def _create_distance_based_hard_pairs(self, X: np.ndarray, y: np.ndarray, 
                                         num_pairs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create hard pairs based on feature distances
        
        Hard positives: Far apart samples from same class
        Hard negatives: Close samples from different classes
        """
        from sklearn.metrics.pairwise import euclidean_distances
        
        # Compute pairwise distances
        distances = euclidean_distances(X)
        
        pairs = []
        
        # Generate hard positive pairs
        for _ in range(num_pairs // 2):
            # Random class
            class_label = np.random.choice([1, 2])
            class_indices = np.where(y == class_label)[0]
            
            if len(class_indices) > 1:
                # Find farthest pair within class
                class_distances = distances[np.ix_(class_indices, class_indices)]
                np.fill_diagonal(class_distances, -np.inf)
                max_idx = np.unravel_index(np.argmax(class_distances), class_distances.shape)
                idx1, idx2 = class_indices[max_idx[0]], class_indices[max_idx[1]]
                pairs.append((X[idx1], X[idx2], 1))
        
        # Generate hard negative pairs
        planet_indices = np.where(y == 2)[0]
        no_planet_indices = np.where(y == 1)[0]
        
        for _ in range(num_pairs - num_pairs // 2):
            # Find closest pair from different classes
            inter_class_distances = distances[np.ix_(planet_indices, no_planet_indices)]
            min_idx = np.unravel_index(np.argmin(inter_class_distances), inter_class_distances.shape)
            idx1, idx2 = planet_indices[min_idx[0]], no_planet_indices[min_idx[1]]
            pairs.append((X[idx1], X[idx2], 0))
        
        # Shuffle and return
        np.random.shuffle(pairs)
        
        X1 = np.array([pair[0] for pair in pairs])
        X2 = np.array([pair[1] for pair in pairs])
        labels = np.array([pair[2] for pair in pairs])
        
        return X1, X2, labels
    
    def _create_model_based_hard_pairs(self, X: np.ndarray, y: np.ndarray, 
                                      model, num_pairs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create hard pairs based on model predictions
        """
        # This would use the model to find hard examples
        # Implementation depends on the model interface
        # For now, fall back to distance-based
        return self._create_distance_based_hard_pairs(X, y, num_pairs)
    
    def save_pairs(self, X1: np.ndarray, X2: np.ndarray, labels: np.ndarray, 
                  output_dir: str, prefix: str = "pairs"):
        """
        Save generated pairs to files
        
        Args:
            X1: First set of samples
            X2: Second set of samples
            labels: Pair labels
            output_dir: Directory to save pairs
            prefix: Filename prefix
        """
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, f"{prefix}_X1.npy"), X1)
        np.save(os.path.join(output_dir, f"{prefix}_X2.npy"), X2)
        np.save(os.path.join(output_dir, f"{prefix}_labels.npy"), labels)
        
        print(f"Saved pairs to {output_dir}")
    
    def load_pairs(self, input_dir: str, prefix: str = "pairs") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load saved pairs from files
        
        Args:
            input_dir: Directory containing saved pairs
            prefix: Filename prefix
            
        Returns:
            Tuple of (X1, X2, labels)
        """
        X1 = np.load(os.path.join(input_dir, f"{prefix}_X1.npy"))
        X2 = np.load(os.path.join(input_dir, f"{prefix}_X2.npy"))
        labels = np.load(os.path.join(input_dir, f"{prefix}_labels.npy"))
        
        print(f"Loaded {len(labels)} pairs from {input_dir}")
        
        return X1, X2, labels


def generate_training_pairs(features_path: str, output_dir: str, 
                           method: str = "balanced",
                           pairs_per_sample: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate and save training pairs
    
    Args:
        features_path: Path to features CSV
        output_dir: Directory to save pairs
        method: Pair generation method ('balanced', 'random', 'hard')
        pairs_per_sample: Number of pairs per sample for balanced method
        
    Returns:
        Generated pairs
    """
    # Load features
    features_df = pd.read_csv(features_path)
    
    # Initialize generator
    generator = SiamesePairGenerator()
    
    # Generate pairs based on method
    if method == "balanced":
        X1, X2, labels = generator.create_balanced_pairs(features_df, pairs_per_sample)
    elif method == "random":
        X1, X2, labels = generator.create_pairs(features_df)
    elif method == "hard":
        X1, X2, labels = generator.create_hard_pairs(features_df)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Save pairs
    generator.save_pairs(X1, X2, labels, output_dir, prefix=method)
    
    return X1, X2, labels


if __name__ == "__main__":
    # Example usage
    features_file = "../data/features/train_features.csv"
    pairs_dir = "../data/pairs/"
    
    pairs = generate_training_pairs(features_file, pairs_dir, method="balanced")
