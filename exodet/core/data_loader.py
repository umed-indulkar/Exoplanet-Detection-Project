"""
Universal Data Loader Module
============================

Unified data loading system supporting multiple formats:
- NPZ files (NumPy archives)
- CSV files (tabular data)
- FITS files (astronomical data)

Combines best practices from all branches with automatic format detection.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
from dataclasses import dataclass, field
import warnings

from .exceptions import DataLoadError, FileFormatError, InsufficientDataError


@dataclass
class LightCurve:
    """
    Data class representing a light curve.
    
    Attributes:
        time (np.ndarray): Time values
        flux (np.ndarray): Flux measurements
        flux_err (np.ndarray): Flux uncertainties
        metadata (dict): Additional metadata (period, epoch, label, etc.)
        source_file (str): Original filename
        format (str): File format (npz, csv, fits)
    """
    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    metadata: Dict = field(default_factory=dict)
    source_file: str = ""
    format: str = ""
    
    def __post_init__(self):
        """Validate light curve data after initialization."""
        self._validate()
    
    def _validate(self):
        """Ensure data integrity."""
        if len(self.time) != len(self.flux):
            raise ValueError(f"Time and flux arrays must have same length: "
                           f"{len(self.time)} != {len(self.flux)}")
        
        if len(self.flux_err) > 0 and len(self.flux_err) != len(self.flux):
            raise ValueError(f"Flux error array length mismatch: "
                           f"{len(self.flux_err)} != {len(self.flux)}")
        
        if len(self.time) == 0:
            raise InsufficientDataError("Light curve has no data points")
    
    def __len__(self) -> int:
        """Return number of data points."""
        return len(self.time)
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"LightCurve(points={len(self)}, "
                f"time_span={self.time[-1]-self.time[0]:.2f}, "
                f"source={self.source_file})")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            'time': self.time,
            'flux': self.flux,
            'flux_err': self.flux_err,
            'metadata': self.metadata,
            'source_file': self.source_file,
            'format': self.format
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LightCurve':
        """Create LightCurve from dictionary."""
        return cls(
            time=data['time'],
            flux=data['flux'],
            flux_err=data.get('flux_err', np.array([])),
            metadata=data.get('metadata', {}),
            source_file=data.get('source_file', ''),
            format=data.get('format', '')
        )


class UniversalDataLoader:
    """
    Universal data loader supporting multiple file formats.
    
    Automatically detects format and applies appropriate loading strategy.
    Combines best practices from all branches.
    """
    
    # Possible key names for time, flux, and errors in files
    TIME_KEYS = ['time', 't', 'TIME', 'T', 'bjd', 'BJD', 'hjd', 'HJD']
    FLUX_KEYS = ['flux', 'FLUX', 'f', 'F', 'magnitude', 'mag', 'SAP_FLUX', 'PDCSAP_FLUX']
    ERROR_KEYS = ['flux_err', 'flux_error', 'error', 'ERROR', 'err', 'ERR', 
                  'flux_err', 'sigma', 'SIGMA', 'SAP_FLUX_ERR', 'PDCSAP_FLUX_ERR']
    LABEL_KEYS = ['label', 'LABEL', 'class', 'CLASS', 'target', 'TARGET']
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the data loader.
        
        Args:
            verbose: Whether to print loading information
        """
        self.verbose = verbose
    
    def load(self, file_path: Union[str, Path]) -> LightCurve:
        """
        Load a single light curve file with automatic format detection.
        
        Args:
            file_path: Path to the file
            
        Returns:
            LightCurve object
            
        Raises:
            DataLoadError: If loading fails
            FileFormatError: If format is not supported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise DataLoadError(f"File not found: {file_path}")
        
        # Detect format
        format_type = self._detect_format(file_path)
        
        # Load based on format
        if format_type == 'npz':
            return self.load_npz(file_path)
        elif format_type == 'csv':
            return self.load_csv(file_path)
        elif format_type == 'fits':
            return self.load_fits(file_path)
        else:
            raise FileFormatError(f"Unsupported file format: {file_path.suffix}")
    
    def load_npz(self, file_path: Union[str, Path]) -> LightCurve:
        """
        Load light curve from NPZ file (NumPy archive).
        
        Implements flexible key detection from main branch.
        
        Args:
            file_path: Path to NPZ file
            
        Returns:
            LightCurve object
            
        Raises:
            DataLoadError: If required keys not found
        """
        file_path = Path(file_path)
        
        try:
            data = np.load(file_path, allow_pickle=True)
            
            # Find time array
            time_key = self._find_key(data.files, self.TIME_KEYS)
            if time_key is None:
                raise DataLoadError(f"No time array found in {file_path}. "
                                  f"Available keys: {list(data.files)}")
            
            # Find flux array
            flux_key = self._find_key(data.files, self.FLUX_KEYS)
            if flux_key is None:
                raise DataLoadError(f"No flux array found in {file_path}. "
                                  f"Available keys: {list(data.files)}")
            
            # Find error array (optional)
            error_key = self._find_key(data.files, self.ERROR_KEYS)
            
            # Extract arrays
            time = np.asarray(data[time_key])
            flux = np.asarray(data[flux_key])
            flux_err = np.asarray(data[error_key]) if error_key else np.ones_like(flux) * 0.001
            
            # Extract metadata
            metadata = {}
            label_key = self._find_key(data.files, self.LABEL_KEYS)
            if label_key:
                metadata['label'] = int(data[label_key])
            
            # Store all other keys as metadata
            for key in data.files:
                if key not in [time_key, flux_key, error_key, label_key]:
                    try:
                        metadata[key] = data[key]
                    except:
                        pass
            
            if self.verbose:
                print(f"✓ Loaded NPZ: {file_path.name} ({len(time)} points)")
            
            return LightCurve(
                time=time,
                flux=flux,
                flux_err=flux_err,
                metadata=metadata,
                source_file=str(file_path),
                format='npz'
            )
            
        except Exception as e:
            raise DataLoadError(f"Failed to load NPZ file {file_path}: {str(e)}")
    
    def load_csv(self, file_path: Union[str, Path], 
                 time_col: Optional[str] = None,
                 flux_cols: Optional[List[str]] = None,
                 label_col: Optional[str] = None) -> LightCurve:
        """
        Load light curve from CSV file.
        
        Supports two CSV formats:
        1. Standard format: time, flux, flux_err columns
        2. Kepler format: LABEL, FLUX.1, FLUX.2, ... (from newcode)
        
        Args:
            file_path: Path to CSV file
            time_col: Name of time column (auto-detected if None)
            flux_cols: List of flux column names (auto-detected if None)
            label_col: Name of label column (auto-detected if None)
            
        Returns:
            LightCurve object
        """
        file_path = Path(file_path)
        
        try:
            df = pd.read_csv(file_path)
            
            # Detect label column
            if label_col is None:
                label_col = self._find_key(df.columns, self.LABEL_KEYS)
            
            # Extract label if present
            metadata = {}
            if label_col and label_col in df.columns:
                metadata['label'] = int(df[label_col].iloc[0])
                df = df.drop(columns=[label_col])
            
            # Check if this is a Kepler-style CSV (FLUX.1, FLUX.2, ...)
            flux_pattern_cols = [col for col in df.columns if col.startswith('FLUX.')]
            
            if flux_pattern_cols:
                # Kepler format: each row is a light curve
                if len(df) > 1:
                    warnings.warn(f"CSV has {len(df)} rows, using first row only")
                
                flux = df[flux_pattern_cols].iloc[0].values
                time = np.arange(len(flux))  # Generate time indices
                flux_err = np.ones_like(flux) * 0.001  # Default errors
                
            else:
                # Standard format: time and flux columns
                if time_col is None:
                    time_col = self._find_key(df.columns, self.TIME_KEYS)
                if time_col is None:
                    # Use index as time
                    time = np.arange(len(df))
                else:
                    time = df[time_col].values
                
                flux_col = self._find_key(df.columns, self.FLUX_KEYS)
                if flux_col is None:
                    # Assume first numeric column is flux
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    flux_col = numeric_cols[0] if len(numeric_cols) > 0 else None
                
                if flux_col is None:
                    raise DataLoadError(f"Could not find flux column in {file_path}")
                
                flux = df[flux_col].values
                
                # Try to find error column
                error_col = self._find_key(df.columns, self.ERROR_KEYS)
                flux_err = df[error_col].values if error_col else np.ones_like(flux) * 0.001
            
            if self.verbose:
                print(f"✓ Loaded CSV: {file_path.name} ({len(time)} points)")
            
            return LightCurve(
                time=time,
                flux=flux,
                flux_err=flux_err,
                metadata=metadata,
                source_file=str(file_path),
                format='csv'
            )
            
        except Exception as e:
            raise DataLoadError(f"Failed to load CSV file {file_path}: {str(e)}")
    
    def load_fits(self, file_path: Union[str, Path]) -> LightCurve:
        """
        Load light curve from FITS file.
        
        Args:
            file_path: Path to FITS file
            
        Returns:
            LightCurve object
        """
        file_path = Path(file_path)
        
        try:
            from astropy.io import fits
            
            with fits.open(file_path) as hdul:
                # Usually data is in first extension
                data = hdul[1].data if len(hdul) > 1 else hdul[0].data
                
                # Find columns
                time_col = self._find_key(data.columns.names, self.TIME_KEYS)
                flux_col = self._find_key(data.columns.names, self.FLUX_KEYS)
                error_col = self._find_key(data.columns.names, self.ERROR_KEYS)
                
                if time_col is None or flux_col is None:
                    raise DataLoadError(f"Required columns not found in {file_path}")
                
                time = data[time_col]
                flux = data[flux_col]
                flux_err = data[error_col] if error_col else np.ones_like(flux) * 0.001
                
                # Extract header metadata
                metadata = dict(hdul[0].header)
                
                if self.verbose:
                    print(f"✓ Loaded FITS: {file_path.name} ({len(time)} points)")
                
                return LightCurve(
                    time=time,
                    flux=flux,
                    flux_err=flux_err,
                    metadata=metadata,
                    source_file=str(file_path),
                    format='fits'
                )
                
        except ImportError:
            raise DataLoadError("astropy is required to load FITS files. "
                              "Install with: pip install astropy")
        except Exception as e:
            raise DataLoadError(f"Failed to load FITS file {file_path}: {str(e)}")
    
    def load_batch(self, directory: Union[str, Path], 
                   pattern: str = "*", 
                   recursive: bool = False,
                   max_files: Optional[int] = None) -> List[LightCurve]:
        """
        Load multiple light curve files from a directory.
        
        Args:
            directory: Directory containing light curve files
            pattern: Glob pattern for file matching (e.g., "*.npz", "*.csv")
            recursive: Whether to search recursively
            max_files: Maximum number of files to load (None for all)
            
        Returns:
            List of LightCurve objects
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise DataLoadError(f"Directory not found: {directory}")
        
        if not directory.is_dir():
            raise DataLoadError(f"Not a directory: {directory}")
        
        # Find files
        if recursive:
            files = list(directory.rglob(pattern))
        else:
            files = list(directory.glob(pattern))
        
        # Filter out directories
        files = [f for f in files if f.is_file()]
        
        if not files:
            warnings.warn(f"No files found in {directory} matching pattern '{pattern}'")
            return []
        
        if max_files:
            files = files[:max_files]
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Loading {len(files)} files from {directory}")
            print(f"{'='*60}")
        
        # Load all files
        light_curves = []
        failed = []
        
        for file_path in files:
            try:
                lc = self.load(file_path)
                light_curves.append(lc)
            except Exception as e:
                failed.append((file_path, str(e)))
                if self.verbose:
                    print(f"✗ Failed: {file_path.name} - {str(e)}")
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"✓ Successfully loaded: {len(light_curves)}/{len(files)} files")
            if failed:
                print(f"✗ Failed: {len(failed)} files")
            print(f"{'='*60}\n")
        
        return light_curves
    
    def _detect_format(self, file_path: Path) -> str:
        """
        Detect file format from extension.
        
        Args:
            file_path: Path to file
            
        Returns:
            Format string ('npz', 'csv', or 'fits')
        """
        suffix = file_path.suffix.lower()
        
        if suffix == '.npz':
            return 'npz'
        elif suffix == '.csv':
            return 'csv'
        elif suffix in ['.fits', '.fit', '.fts']:
            return 'fits'
        else:
            raise FileFormatError(f"Unknown file format: {suffix}")
    
    @staticmethod
    def _find_key(available_keys: List[str], possible_keys: List[str]) -> Optional[str]:
        """
        Find first matching key from a list of possibilities.
        
        Args:
            available_keys: List of available keys
            possible_keys: List of possible key names to search for
            
        Returns:
            First matching key or None if no match
        """
        available_set = set(available_keys)
        for key in possible_keys:
            if key in available_set:
                return key
        return None


# Convenience functions
def load_lightcurve(file_path: Union[str, Path], verbose: bool = True) -> LightCurve:
    """
    Load a single light curve file.
    
    Args:
        file_path: Path to the light curve file
        verbose: Whether to print loading information
        
    Returns:
        LightCurve object
    """
    loader = UniversalDataLoader(verbose=verbose)
    return loader.load(file_path)


def load_batch_lightcurves(directory: Union[str, Path], 
                           pattern: str = "*",
                           recursive: bool = False,
                           max_files: Optional[int] = None,
                           verbose: bool = True) -> List[LightCurve]:
    """
    Load multiple light curve files from a directory.
    
    Args:
        directory: Directory containing light curve files
        pattern: Glob pattern for file matching
        recursive: Whether to search recursively
        max_files: Maximum number of files to load
        verbose: Whether to print progress information
        
    Returns:
        List of LightCurve objects
    """
    loader = UniversalDataLoader(verbose=verbose)
    return loader.load_batch(directory, pattern, recursive, max_files)
