"""
Data Loader Module - Preprocessing Team
=====================================

Universal data loading system supporting multiple formats:
- NPZ files (NumPy archives)
- CSV files (tabular data)  
- FITS files (astronomical data)

This is the ACTUAL data loader code - no more duplication!
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
from dataclasses import dataclass, field
import warnings

# Define exceptions locally to avoid core dependency
class DataLoadError(Exception):
    """Data loading failed."""
    pass

class FileFormatError(Exception):
    """Unsupported file format."""
    pass

class InsufficientDataError(Exception):
    """Not enough data points."""
    pass


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
    """
    
    def __init__(self):
        """Initialize data loader."""
        self.supported_formats = {'.npz', '.csv', '.fits', '.fit'}
    
    def load(self, file_path: Union[str, Path]) -> LightCurve:
        """
        Load light curve from file, auto-detecting format.
        
        Args:
            file_path: Path to the file
            
        Returns:
            LightCurve object
            
        Raises:
            DataLoadError: If loading fails
            FileFormatError: If format is unsupported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise DataLoadError(f"File not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        if suffix not in self.supported_formats:
            raise FileFormatError(f"Unsupported format: {suffix}")
        
        try:
            if suffix == '.npz':
                return self._load_npz(file_path)
            elif suffix == '.csv':
                return self._load_csv(file_path)
            elif suffix in {'.fits', '.fit'}:
                return self._load_fits(file_path)
        except Exception as e:
            raise DataLoadError(f"Failed to load {file_path}: {str(e)}")
    
    def _load_npz(self, file_path: Path) -> LightCurve:
        """Load NPZ file."""
        data = np.load(file_path)
        
        # Extract arrays
        time = data['time']
        flux = data['flux']
        flux_err = data.get('flux_err', np.array([]))
        
        # Extract metadata
        metadata = {}
        for key in data.files:
            if key not in ['time', 'flux', 'flux_err']:
                metadata[key] = data[key]
        
        return LightCurve(
            time=time,
            flux=flux,
            flux_err=flux_err,
            metadata=metadata,
            source_file=str(file_path),
            format='npz'
        )
    
    def _load_csv(self, file_path: Path) -> LightCurve:
        """Load CSV file."""
        df = pd.read_csv(file_path)
        
        # Find time and flux columns
        time_col = self._find_column(df, ['time', 'timestamp', 'jd', 'bjd'])
        flux_col = self._find_column(df, ['flux', 'brightness', 'magnitude'])
        flux_err_col = self._find_column(df, ['flux_err', 'error', 'uncertainty'], optional=True)
        
        time = df[time_col].values
        flux = df[flux_col].values
        flux_err = df[flux_err_col].values if flux_err_col else np.array([])
        
        # Extract other columns as metadata
        metadata = {}
        for col in df.columns:
            if col not in [time_col, flux_col, flux_err_col]:
                metadata[col] = df[col].iloc[0] if len(df) == 1 else df[col].values
        
        return LightCurve(
            time=time,
            flux=flux,
            flux_err=flux_err,
            metadata=metadata,
            source_file=str(file_path),
            format='csv'
        )
    
    def _load_fits(self, file_path: Path) -> LightCurve:
        """Load FITS file."""
        try:
            from astropy.io import fits
        except ImportError:
            raise DataLoadError("astropy required for FITS support. Install: pip install astropy")
        
        with fits.open(file_path) as hdul:
            # Use first extension with data
            data = None
            for hdu in hdul:
                if hdu.data is not None and len(hdu.data.shape) >= 2:
                    data = hdu.data
                    break
            
            if data is None:
                raise DataLoadError("No data found in FITS file")
            
            # Assume columns: time, flux, flux_err
            time = data[:, 0]
            flux = data[:, 1]
            flux_err = data[:, 2] if data.shape[1] > 2 else np.array([])
            
            # Extract header metadata
            metadata = {}
            if hasattr(hdul[0], 'header'):
                for key, value in hdul[0].header.items():
                    if not key.startswith('COMMENT') and not key.startswith('HISTORY'):
                        metadata[key] = value
        
        return LightCurve(
            time=time,
            flux=flux,
            flux_err=flux_err,
            metadata=metadata,
            source_file=str(file_path),
            format='fits'
        )
    
    def _find_column(self, df: pd.DataFrame, names: List[str], optional: bool = False) -> Optional[str]:
        """Find column by possible names."""
        for name in names:
            for col in df.columns:
                if col.lower() == name.lower():
                    return col
        
        if not optional:
            raise DataLoadError(f"Could not find column with names: {names}")
        return None


# Global loader instance
_loader = UniversalDataLoader()

def load_lightcurve(file_path: Union[str, Path]) -> LightCurve:
    """
    Load a single light curve file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        LightCurve object
    """
    return _loader.load(file_path)

def load_batch_lightcurves(file_paths: List[Union[str, Path]]) -> List[LightCurve]:
    """
    Load multiple light curve files.
    
    Args:
        file_paths: List of file paths
        
    Returns:
        List of LightCurve objects
    """
    lightcurves = []
    for path in file_paths:
        try:
            lc = load_lightcurve(path)
            lightcurves.append(lc)
        except Exception as e:
            warnings.warn(f"Failed to load {path}: {str(e)}")
    
    return lightcurves
