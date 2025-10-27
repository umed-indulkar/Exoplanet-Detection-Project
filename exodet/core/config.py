"""
Configuration Management Module
================================

Unified configuration system supporting:
- YAML configuration files
- Dictionary-based configuration
- Environment variable overrides
- Default configurations
- Validation and schema checking
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from copy import deepcopy

from .exceptions import ConfigError


class Config:
    """
    Configuration manager for the Exoplanet Detection System.
    
    Provides a unified interface for loading, validating, and accessing
    configuration parameters from multiple sources.
    """
    
    DEFAULT_CONFIG = {
        # System settings
        'system': {
            'seed': 42,
            'device': 'auto',  # 'auto', 'cpu', 'cuda'
            'num_workers': 4,
            'verbose': True,
            'log_level': 'INFO',
        },
        
        # Data settings
        'data': {
            'input_dir': 'data/raw/',
            'output_dir': 'data/processed/',
            'file_format': 'auto',  # 'auto', 'npz', 'csv', 'fits'
        },
        
        # Preprocessing settings
        'preprocessing': {
            'remove_nans': True,
            'detrend': {
                'enabled': True,
                'method': 'polynomial',
                'order': 3,
                'window_length': 51,
            },
            'sigma_clip': {
                'enabled': True,
                'sigma': 3.0,
                'iterations': 3,
                'method': 'iterative',
            },
            'normalize': {
                'enabled': True,
                'method': 'zscore',
            },
            'fold': {
                'enabled': False,
                'period': None,
                'epoch': 0.0,
            },
            'bin': {
                'enabled': False,
                'bin_size': 0.01,
                'method': 'weighted',
            },
            'quality_mask': {
                'enabled': True,
                'mad_threshold': 10.0,
            },
            'min_points': 10,
        },
        
        # Feature extraction settings
        'features': {
            'tier': 'standard',  # 'fast', 'standard', 'comprehensive'
            'custom_features': [],
            'parallel': True,
            'cache_enabled': True,
            'fast': {
                'timeout': 1.0,
            },
            'standard': {
                'timeout': 5.0,
            },
            'comprehensive': {
                'timeout': 30.0,
                'n_jobs': -1,
            },
        },
        
        # Model settings
        'model': {
            'type': 'siamese',  # 'siamese', 'fcnn', 'ensemble'
            'architecture': {
                'hidden_dims': [256, 128, 64],
                'embedding_dim': 32,
                'dropout_rate': 0.3,
                'activation': 'relu',
            },
        },
        
        # Training settings
        'training': {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 0.00001,
            'optimizer': 'adam',
            'margin': 1.0,
            'scheduler': {
                'enabled': True,
                'patience': 5,
                'factor': 0.5,
            },
            'early_stopping': {
                'enabled': True,
                'patience': 10,
                'min_delta': 0.0001,
            },
            'pairs': {
                'method': 'balanced',
                'pairs_per_sample': 5,
                'positive_ratio': 0.5,
            },
        },
        
        # Evaluation settings
        'evaluation': {
            'metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc'],
            'visualizations': ['confusion_matrix', 'roc_curve', 'embedding'],
            'save_predictions': True,
            'distance_threshold': 0.5,
        },
        
        # Output settings
        'output': {
            'save_plots': True,
            'save_features': True,
            'save_models': True,
            'format': 'csv',  # 'csv', 'excel', 'hdf5'
            'compression': None,  # None, 'gzip', 'bz2'
            'models_dir': 'outputs/models/',
            'logs_dir': 'outputs/logs/',
            'results_dir': 'outputs/results/',
            'plots_dir': 'outputs/plots/',
        },
    }
    
    def __init__(self, config: Optional[Union[Dict, str, Path]] = None):
        """
        Initialize configuration.
        
        Args:
            config: Configuration source. Can be:
                - None: Use default configuration
                - Dict: Use provided dictionary
                - str/Path: Load from YAML file
        """
        # Start with default config
        self._config = deepcopy(self.DEFAULT_CONFIG)
        
        # Load and merge additional config
        if config is not None:
            if isinstance(config, (str, Path)):
                loaded_config = self.load_from_yaml(config)
                self._merge_config(loaded_config)
            elif isinstance(config, dict):
                self._merge_config(config)
            else:
                raise ConfigError(f"Invalid config type: {type(config)}")
        
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        # Validate configuration
        self.validate()
    
    def load_from_yaml(self, path: Union[str, Path]) -> Dict:
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to YAML file
            
        Returns:
            Configuration dictionary
            
        Raises:
            ConfigError: If file cannot be loaded
        """
        path = Path(path)
        
        if not path.exists():
            raise ConfigError(f"Configuration file not found: {path}")
        
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            return config or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in {path}: {str(e)}")
        except Exception as e:
            raise ConfigError(f"Failed to load config from {path}: {str(e)}")
    
    def save_to_yaml(self, path: Union[str, Path]):
        """
        Save configuration to YAML file.
        
        Args:
            path: Path to save YAML file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            raise ConfigError(f"Failed to save config to {path}: {str(e)}")
    
    def _merge_config(self, new_config: Dict):
        """
        Recursively merge new configuration into existing config.
        
        Args:
            new_config: Configuration dictionary to merge
        """
        self._config = self._deep_merge(self._config, new_config)
    
    @staticmethod
    def _deep_merge(base: Dict, update: Dict) -> Dict:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            update: Dictionary with updates
            
        Returns:
            Merged dictionary
        """
        result = deepcopy(base)
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        
        return result
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides to configuration."""
        # Define environment variable mappings
        env_mappings = {
            'EXODET_SEED': ['system', 'seed'],
            'EXODET_DEVICE': ['system', 'device'],
            'EXODET_NUM_WORKERS': ['system', 'num_workers'],
            'EXODET_VERBOSE': ['system', 'verbose'],
            'EXODET_DATA_DIR': ['data', 'input_dir'],
            'EXODET_OUTPUT_DIR': ['data', 'output_dir'],
            'EXODET_FEATURE_TIER': ['features', 'tier'],
            'EXODET_MODEL_TYPE': ['model', 'type'],
            'EXODET_BATCH_SIZE': ['training', 'batch_size'],
            'EXODET_LEARNING_RATE': ['training', 'learning_rate'],
            'EXODET_EPOCHS': ['training', 'epochs'],
        }
        
        for env_var, config_path in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # Type conversion
                if config_path[-1] in ['seed', 'num_workers', 'batch_size', 'epochs']:
                    value = int(value)
                elif config_path[-1] in ['learning_rate']:
                    value = float(value)
                elif config_path[-1] in ['verbose']:
                    value = value.lower() in ['true', '1', 'yes']
                
                # Set value in config
                current = self._config
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[config_path[-1]] = value
    
    def validate(self):
        """
        Validate configuration.
        
        Raises:
            ConfigError: If configuration is invalid
        """
        # Validate system settings
        if self._config['system']['device'] not in ['auto', 'cpu', 'cuda']:
            raise ConfigError(f"Invalid device: {self._config['system']['device']}")
        
        if self._config['system']['num_workers'] < 1:
            raise ConfigError("num_workers must be >= 1")
        
        # Validate preprocessing settings
        if self._config['preprocessing']['detrend']['method'] not in ['polynomial', 'savgol', 'median', 'none']:
            raise ConfigError(f"Invalid detrend method: {self._config['preprocessing']['detrend']['method']}")
        
        if self._config['preprocessing']['normalize']['method'] not in ['zscore', 'minmax', 'robust', 'median']:
            raise ConfigError(f"Invalid normalize method: {self._config['preprocessing']['normalize']['method']}")
        
        # Validate feature settings
        if self._config['features']['tier'] not in ['fast', 'standard', 'comprehensive']:
            raise ConfigError(f"Invalid feature tier: {self._config['features']['tier']}")
        
        # Validate model settings
        if self._config['model']['type'] not in ['siamese', 'fcnn', 'ensemble']:
            raise ConfigError(f"Invalid model type: {self._config['model']['type']}")
        
        # Validate training settings
        if self._config['training']['batch_size'] < 1:
            raise ConfigError("batch_size must be >= 1")
        
        if self._config['training']['learning_rate'] <= 0:
            raise ConfigError("learning_rate must be > 0")
        
        if self._config['training']['epochs'] < 1:
            raise ConfigError("epochs must be >= 1")
        
        if self._config['training']['optimizer'] not in ['adam', 'sgd', 'rmsprop']:
            raise ConfigError(f"Invalid optimizer: {self._config['training']['optimizer']}")
        
        # Validate output settings
        if self._config['output']['format'] not in ['csv', 'excel', 'hdf5']:
            raise ConfigError(f"Invalid output format: {self._config['output']['format']}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'system.seed' or 'model.architecture.hidden_dims')
            default: Default value if key not found
            
        Returns:
            Configuration value
            
        Example:
            >>> config.get('system.seed')
            42
            >>> config.get('model.architecture.hidden_dims')
            [256, 128, 64]
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'system.seed')
            value: Value to set
            
        Example:
            >>> config.set('system.seed', 123)
            >>> config.set('model.architecture.hidden_dims', [512, 256, 128])
        """
        keys = key.split('.')
        current = self._config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def to_dict(self) -> Dict:
        """
        Get configuration as dictionary.
        
        Returns:
            Configuration dictionary
        """
        return deepcopy(self._config)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access to config."""
        return self._config[key]
    
    def __setitem__(self, key: str, value: Any):
        """Allow dict-style setting of config."""
        self._config[key] = value
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return key in self._config
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Config({len(self._config)} sections)"
    
    def print_config(self, section: Optional[str] = None):
        """
        Print configuration in human-readable format.
        
        Args:
            section: Optional section name to print (None for all)
        """
        if section:
            if section not in self._config:
                print(f"Section '{section}' not found")
                return
            config_to_print = {section: self._config[section]}
        else:
            config_to_print = self._config
        
        print("\n" + "="*60)
        print("CONFIGURATION")
        print("="*60)
        print(yaml.dump(config_to_print, default_flow_style=False, sort_keys=False))
        print("="*60 + "\n")


# Convenience functions
def load_config(path: Union[str, Path]) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        path: Path to YAML file
        
    Returns:
        Config object
    """
    return Config(path)


def get_default_config() -> Config:
    """
    Get default configuration.
    
    Returns:
        Config object with default settings
    """
    return Config()
