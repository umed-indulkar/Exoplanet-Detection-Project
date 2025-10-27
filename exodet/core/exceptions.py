"""
Custom Exceptions for Exoplanet Detection System
=================================================

Defines custom exceptions for better error handling and debugging.
"""


class ExoDetError(Exception):
    """Base exception for all exodet errors."""
    pass


class DataLoadError(ExoDetError):
    """Raised when data loading fails."""
    pass


class PreprocessingError(ExoDetError):
    """Raised when preprocessing fails."""
    pass


class ConfigError(ExoDetError):
    """Raised when configuration is invalid."""
    pass


class FeatureExtractionError(ExoDetError):
    """Raised when feature extraction fails."""
    pass


class ModelError(ExoDetError):
    """Raised when model operations fail."""
    pass


class TrainingError(ExoDetError):
    """Raised when model training fails."""
    pass


class ValidationError(ExoDetError):
    """Raised when data validation fails."""
    pass


class FileFormatError(DataLoadError):
    """Raised when file format is not supported or invalid."""
    pass


class InsufficientDataError(DataLoadError):
    """Raised when data has insufficient points for processing."""
    pass
