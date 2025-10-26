"""
Data Preprocessing Module for Exoplanet Detection
Handles loading, cleaning, and normalizing light curve data
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, Any
import os
import joblib
from tqdm import tqdm


class LightCurvePreprocessor:
    """Preprocessor for light curve time series data"""
    
    def __init__(self, normalization_method: str = 'standard'):
        """
        Initialize preprocessor
        
        Args:
            normalization_method: 'standard', 'minmax', or 'robust'
        """
        self.normalization_method = normalization_method
        self.scaler = None
        self.flux_columns = []
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load light curve data from CSV file
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with light curve data
        """
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        # Identify flux columns
        self.flux_columns = [col for col in df.columns if col.startswith('FLUX')]
        print(f"Found {len(self.flux_columns)} flux measurements")
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the light curve data
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        print("Cleaning data...")
        
        # Remove any rows with all NaN flux values
        df_clean = df.dropna(how='all', subset=self.flux_columns)
        
        # Fill remaining NaN values with interpolation
        for col in self.flux_columns:
            if df_clean[col].isna().any():
                df_clean[col] = df_clean[col].interpolate(method='linear', limit_direction='both')
        
        # Remove outliers using IQR method for each light curve
        for idx in tqdm(range(len(df_clean)), desc="Removing outliers"):
            flux_values = df_clean.iloc[idx][self.flux_columns].values
            Q1 = np.percentile(flux_values, 25)
            Q3 = np.percentile(flux_values, 75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Clip outliers
            flux_values = np.clip(flux_values, lower_bound, upper_bound)
            df_clean.loc[df_clean.index[idx], self.flux_columns] = flux_values
        
        print(f"Data cleaned. Shape: {df_clean.shape}")
        return df_clean
    
    def normalize_flux(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Normalize flux measurements
        
        Args:
            df: DataFrame with flux data
            fit: Whether to fit the scaler (True for training data)
            
        Returns:
            DataFrame with normalized flux
        """
        print(f"Normalizing flux using {self.normalization_method} method...")
        
        df_norm = df.copy()
        
        if self.normalization_method == 'standard':
            if fit:
                self.scaler = StandardScaler()
                df_norm[self.flux_columns] = self.scaler.fit_transform(df[self.flux_columns])
            else:
                df_norm[self.flux_columns] = self.scaler.transform(df[self.flux_columns])
                
        elif self.normalization_method == 'minmax':
            if fit:
                self.scaler = MinMaxScaler()
                df_norm[self.flux_columns] = self.scaler.fit_transform(df[self.flux_columns])
            else:
                df_norm[self.flux_columns] = self.scaler.transform(df[self.flux_columns])
                
        elif self.normalization_method == 'local':
            # Local normalization (per light curve)
            for idx in range(len(df_norm)):
                flux_values = df_norm.iloc[idx][self.flux_columns].values
                mean_flux = np.mean(flux_values)
                std_flux = np.std(flux_values)
                if std_flux > 0:
                    normalized_flux = (flux_values - mean_flux) / std_flux
                else:
                    normalized_flux = flux_values - mean_flux
                df_norm.loc[df_norm.index[idx], self.flux_columns] = normalized_flux
        
        return df_norm
    
    def detrend_flux(self, df: pd.DataFrame, window_size: int = 10) -> pd.DataFrame:
        """
        Remove trends from flux measurements using moving average
        
        Args:
            df: DataFrame with flux data
            window_size: Window size for moving average
            
        Returns:
            DataFrame with detrended flux
        """
        print(f"Detrending flux with window size {window_size}...")
        
        df_detrended = df.copy()
        
        for idx in tqdm(range(len(df)), desc="Detrending"):
            flux_values = df.iloc[idx][self.flux_columns].values
            
            # Calculate moving average
            moving_avg = np.convolve(flux_values, np.ones(window_size)/window_size, mode='same')
            
            # Subtract trend
            detrended = flux_values - moving_avg
            df_detrended.loc[df_detrended.index[idx], self.flux_columns] = detrended
        
        return df_detrended
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                   val_size: float = 0.1, random_state: int = 42) -> Dict[str, pd.DataFrame]:
        """
        Split data into train, validation, and test sets
        
        Args:
            df: DataFrame to split
            test_size: Proportion for test set
            val_size: Proportion for validation set
            random_state: Random seed
            
        Returns:
            Dictionary with train, val, and test DataFrames
        """
        print("Splitting data...")
        
        # First split: train+val and test
        train_val, test = train_test_split(
            df, test_size=test_size, random_state=random_state, 
            stratify=df['LABEL'] if 'LABEL' in df.columns else None
        )
        
        # Second split: train and val
        val_size_adjusted = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val, test_size=val_size_adjusted, random_state=random_state,
            stratify=train_val['LABEL'] if 'LABEL' in train_val.columns else None
        )
        
        print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        
        return {
            'train': train.reset_index(drop=True),
            'val': val.reset_index(drop=True),
            'test': test.reset_index(drop=True)
        }
    
    def save_processed_data(self, df: pd.DataFrame, filepath: str):
        """Save processed data to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Saved processed data to {filepath}")
    
    def save_scaler(self, filepath: str):
        """Save scaler for later use"""
        if self.scaler:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(self.scaler, filepath)
            print(f"Saved scaler to {filepath}")
    
    def load_scaler(self, filepath: str):
        """Load saved scaler"""
        self.scaler = joblib.load(filepath)
        print(f"Loaded scaler from {filepath}")


def preprocess_pipeline(input_path: str, output_dir: str, 
                        normalization: str = 'local',
                        detrend: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Complete preprocessing pipeline
    
    Args:
        input_path: Path to raw data
        output_dir: Directory to save processed data
        normalization: Normalization method
        detrend: Whether to detrend the data
        
    Returns:
        Dictionary with processed train, val, and test sets
    """
    # Initialize preprocessor
    preprocessor = LightCurvePreprocessor(normalization_method=normalization)
    
    # Load and clean data
    df = preprocessor.load_data(input_path)
    df_clean = preprocessor.clean_data(df)
    
    # Detrend if requested
    if detrend:
        df_clean = preprocessor.detrend_flux(df_clean)
    
    # Normalize
    df_normalized = preprocessor.normalize_flux(df_clean, fit=True)
    
    # Split data
    data_splits = preprocessor.split_data(df_normalized)
    
    # Save processed data
    for split_name, split_df in data_splits.items():
        output_path = os.path.join(output_dir, f'{split_name}_processed.csv')
        preprocessor.save_processed_data(split_df, output_path)
    
    # Save scaler
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    preprocessor.save_scaler(scaler_path)
    
    return data_splits


if __name__ == "__main__":
    # Example usage
    input_file = "../data/raw/exoTest.csv"
    output_directory = "../data/processed/"
    
    processed_data = preprocess_pipeline(
        input_file, 
        output_directory,
        normalization='local',
        detrend=True
    )
