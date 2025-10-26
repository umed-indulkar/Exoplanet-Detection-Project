"""
Main entry point for Exoplanet Detection using Siamese Networks
Complete pipeline from data preprocessing to evaluation
"""

import os
import sys
import yaml
import argparse
import time
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import preprocess_pipeline
from src.feature_extraction import extract_and_save_features
from src.pair_generation import generate_training_pairs
from src.train import train_siamese_network
from src.evaluate import evaluate_model
from src.utils import set_seed, print_metrics


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_preprocessing(config: dict):
    """Run data preprocessing pipeline"""
    print("\n" + "="*60)
    print("STEP 1: DATA PREPROCESSING")
    print("="*60)
    
    # Process data
    data_splits = preprocess_pipeline(
        input_path=config['data']['raw_data_path'],
        output_dir=config['data']['processed_dir'],
        normalization=config['preprocessing']['normalization_method'],
        detrend=config['preprocessing']['detrend']
    )
    
    print("✓ Data preprocessing completed")
    return data_splits


def run_feature_extraction(config: dict):
    """Run feature extraction pipeline"""
    print("\n" + "="*60)
    print("STEP 2: FEATURE EXTRACTION")
    print("="*60)
    
    # Extract features for each split
    for split in ['train', 'val', 'test']:
        input_path = os.path.join(config['data']['processed_dir'], f'{split}_processed.csv')
        output_path = os.path.join(config['data']['features_dir'], f'{split}_features.csv')
        
        if os.path.exists(input_path):
            print(f"\nExtracting features for {split} set...")
            extract_and_save_features(input_path, output_path)
    
    print("✓ Feature extraction completed")


def run_pair_generation(config: dict):
    """Generate training pairs"""
    print("\n" + "="*60)
    print("STEP 3: PAIR GENERATION")
    print("="*60)
    
    # Generate pairs for each split
    for split in ['train', 'val', 'test']:
        features_path = os.path.join(config['data']['features_dir'], f'{split}_features.csv')
        
        if os.path.exists(features_path):
            print(f"\nGenerating pairs for {split} set...")
            generate_training_pairs(
                features_path=features_path,
                output_dir=config['data']['pairs_dir'],
                method=config['pairs']['method'],
                pairs_per_sample=config['pairs']['pairs_per_sample']
            )
            
            # Rename files to include split name
            for suffix in ['X1', 'X2', 'labels']:
                old_path = os.path.join(config['data']['pairs_dir'], f"{config['pairs']['method']}_{suffix}.npy")
                new_path = os.path.join(config['data']['pairs_dir'], f"{split}_{suffix}.npy")
                if os.path.exists(old_path):
                    os.rename(old_path, new_path)
    
    print("✓ Pair generation completed")


def run_training(config: dict):
    """Run model training"""
    print("\n" + "="*60)
    print("STEP 4: MODEL TRAINING")
    print("="*60)
    
    # Prepare training configuration
    train_config = {
        'seed': config['system']['seed'],
        'pairs_dir': config['data']['pairs_dir'],
        'epochs': config['training']['epochs'],
        'batch_size': config['training']['batch_size'],
        'learning_rate': config['training']['learning_rate'],
        'weight_decay': config['training']['weight_decay'],
        'margin': config['training']['margin'],
        'early_stopping_patience': config['training']['early_stopping_patience'],
        'checkpoint_dir': config['output']['models_dir'],
        'output_dir': config['output']['results_dir'],
        'model': config['model']
    }
    
    # Train model
    model, history = train_siamese_network(train_config)
    
    print("✓ Model training completed")
    return model, history


def run_evaluation(config: dict):
    """Run model evaluation"""
    print("\n" + "="*60)
    print("STEP 5: MODEL EVALUATION")
    print("="*60)
    
    # Prepare evaluation configuration
    eval_config = {
        'seed': config['system']['seed'],
        'test_features_path': os.path.join(config['data']['features_dir'], 'test_features.csv'),
        'test_pairs_path': config['data']['pairs_dir'],
        'val_pairs_path': config['data']['pairs_dir'],
        'train_features_path': os.path.join(config['data']['features_dir'], 'train_features.csv'),
        'checkpoint_dir': config['output']['models_dir'],
        'checkpoint_name': 'best_model.pth',
        'output_dir': config['output']['results_dir'],
        'visualize': config['evaluation']['visualize'],
        'model': config['model']
    }
    
    # Evaluate model
    metrics = evaluate_model(eval_config)
    
    print("✓ Model evaluation completed")
    return metrics


def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description='Exoplanet Detection Pipeline')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--steps', nargs='+', 
                       default=['preprocess', 'features', 'pairs', 'train', 'evaluate'],
                       choices=['preprocess', 'features', 'pairs', 'train', 'evaluate'],
                       help='Pipeline steps to run')
    parser.add_argument('--force', action='store_true',
                       help='Force re-run even if outputs exist')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set seed for reproducibility
    set_seed(config['system']['seed'])
    
    # Start pipeline
    print("\n" + "="*60)
    print("EXOPLANET DETECTION PIPELINE")
    print("="*60)
    print(f"Configuration: {args.config}")
    print(f"Steps to run: {args.steps}")
    
    start_time = time.time()
    
    try:
        # Run selected steps
        if 'preprocess' in args.steps:
            # Check if processed data already exists
            processed_path = os.path.join(config['data']['processed_dir'], 'train_processed.csv')
            if not os.path.exists(processed_path) or args.force:
                run_preprocessing(config)
            else:
                print("\n✓ Processed data already exists, skipping preprocessing")
        
        if 'features' in args.steps:
            # Check if features already exist
            features_path = os.path.join(config['data']['features_dir'], 'train_features.csv')
            if not os.path.exists(features_path) or args.force:
                run_feature_extraction(config)
            else:
                print("\n✓ Features already exist, skipping feature extraction")
        
        if 'pairs' in args.steps:
            # Check if pairs already exist
            pairs_path = os.path.join(config['data']['pairs_dir'], 'train_X1.npy')
            if not os.path.exists(pairs_path) or args.force:
                run_pair_generation(config)
            else:
                print("\n✓ Pairs already exist, skipping pair generation")
        
        if 'train' in args.steps:
            # Check if model already exists
            model_path = os.path.join(config['output']['models_dir'], 'best_model.pth')
            if not os.path.exists(model_path) or args.force:
                model, history = run_training(config)
            else:
                print("\n✓ Trained model already exists, skipping training")
        
        if 'evaluate' in args.steps:
            metrics = run_evaluation(config)
            
            # Print final results
            print("\n" + "="*60)
            print("FINAL RESULTS")
            print("="*60)
            
            if 'classification' in metrics:
                print("\nClassification Metrics:")
                print_metrics(metrics['classification'])
            
            if 'pair_evaluation' in metrics:
                print("\nPair Evaluation Metrics:")
                print_metrics(metrics['pair_evaluation'])
        
        # Calculate total time
        total_time = time.time() - start_time
        print(f"\n✓ Pipeline completed successfully in {total_time:.2f} seconds")
        
    except Exception as e:
        print(f"\n✗ Pipeline failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
