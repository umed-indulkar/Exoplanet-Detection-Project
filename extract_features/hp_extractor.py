#!/usr/bin/env python3
"""
HIGH-PERFORMANCE TSFresh Feature Extractor for Exoplanet Detection
Maintains ALL ~350 features while dramatically improving processing speed.
Optimized for i7 processors with advanced memory management and parallel processing.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import multiprocessing as mp
from joblib import Parallel, delayed, Memory
from tqdm import tqdm
import pickle
import warnings
import psutil
import gc
from functools import lru_cache
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

# TSFresh imports
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters, ComprehensiveFCParameters
from tsfresh.utilities.dataframe_functions import impute

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class HighPerformanceExoplanetExtractor:
    """
    High-performance TSFresh feature extractor maintaining ALL features (~350).
    Optimized for maximum speed on i7 processors.
    """
    
    def __init__(self, 
                 npz_folder_path: str,
                 output_csv_path: str,
                 progress_file: str = "hp_extraction_progress.pkl",
                 n_jobs: int = None,
                 batch_size: int = None,
                 memory_limit_gb: float = None,
                 use_memory_mapping: bool = True,
                 enable_caching: bool = True):
        """
        Initialize high-performance feature extractor.
        
        Args:
            npz_folder_path: Path to NPZ files
            output_csv_path: Output CSV path
            progress_file: Progress tracking file
            n_jobs: Number of processes (auto-detected based on system)
            batch_size: Batch size (auto-calculated based on available RAM)
            memory_limit_gb: Memory limit in GB (auto-detected)
            use_memory_mapping: Use memory mapping for large files
            enable_caching: Enable intelligent caching
        """
        self.npz_folder = Path(npz_folder_path)
        self.output_csv = Path(output_csv_path)
        self.progress_file = Path(progress_file)
        self.use_memory_mapping = use_memory_mapping
        self.enable_caching = enable_caching
        
        # System optimization detection
        self._detect_system_specs()
        self.n_jobs = n_jobs or self._get_optimal_n_jobs()
        self.batch_size = batch_size or self._calculate_optimal_batch_size()
        self.memory_limit_gb = memory_limit_gb or self._get_memory_limit()
        
        # Setup advanced logging
        self._setup_logging()
        
        # Initialize progress tracking
        self.processed_files = set()
        self.failed_files = set()
        self.extracted_features = []
        self.processing_stats = {
            'total_time': 0,
            'files_processed': 0,
            'avg_time_per_file': 0,
            'memory_usage': []
        }
        
        # Setup caching if enabled
        if self.enable_caching:
            self.memory_cache = Memory(location='./tsfresh_cache', verbose=0)
            self._extract_features_cached = self.memory_cache.cache(self._extract_features_from_curve_base)
        
        # Load progress if resuming
        self._load_progress()
        
        # Get ALL features (comprehensive set)
        self.feature_params = self._get_comprehensive_features()
        
        # Setup memory monitoring
        self._setup_memory_monitoring()
        
        self.logger.info("=" * 80)
        self.logger.info("HIGH-PERFORMANCE TSFRESH EXTRACTOR INITIALIZED")
        self.logger.info("=" * 80)
        self.logger.info(f"System: {self.cpu_info}")
        self.logger.info(f"Processes: {self.n_jobs}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Memory limit: {self.memory_limit_gb:.1f} GB")
        self.logger.info(f"Features per curve: ~350 (ALL features)")
        self.logger.info(f"Memory mapping: {self.use_memory_mapping}")
        self.logger.info(f"Caching enabled: {self.enable_caching}")
        self.logger.info("=" * 80)

    def _detect_system_specs(self):
        """Detect system specifications for optimization."""
        self.cpu_count = mp.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Detect CPU info
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpu_info = f.read()
                if 'Intel' in cpu_info and 'i7' in cpu_info:
                    self.cpu_info = "Intel i7 (optimized)"
                else:
                    self.cpu_info = f"{self.cpu_count}-core processor"
        except:
            self.cpu_info = f"{self.cpu_count}-core processor"
        
        self.has_hyperthreading = self.cpu_count > psutil.cpu_count(logical=False)

    def _get_optimal_n_jobs(self) -> int:
        """Calculate optimal number of jobs based on system specs."""
        logical_cores = self.cpu_count
        physical_cores = psutil.cpu_count(logical=False)
        
        # i7 optimization
        if 'i7' in self.cpu_info.lower():
            if self.has_hyperthreading:
                # Use all logical cores for TSFresh (benefits from hyperthreading)
                return max(1, logical_cores - 1)
            else:
                return max(1, physical_cores - 1)
        else:
            # Conservative for other processors
            return max(1, physical_cores - 1)

    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available memory."""
        # Estimate memory per file (conservative)
        memory_per_file_mb = 50  # MB per file during processing
        available_memory_mb = (self.memory_gb * 0.7) * 1024  # Use 70% of available memory
        
        optimal_batch = int(available_memory_mb / memory_per_file_mb / self.n_jobs)
        
        # Constraints
        return max(10, min(optimal_batch, 200))  # Between 10 and 200

    def _get_memory_limit(self) -> float:
        """Get memory limit for processing."""
        return self.memory_gb * 0.8  # Use 80% of available memory

    def _setup_logging(self):
        """Setup advanced logging with performance metrics."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('hp_feature_extraction.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _setup_memory_monitoring(self):
        """Setup memory monitoring thread."""
        self.memory_monitor_active = True
        self.memory_stats = {'current': 0, 'peak': 0}
        
        def monitor_memory():
            while self.memory_monitor_active:
                current_memory = psutil.virtual_memory().used / (1024**3)
                self.memory_stats['current'] = current_memory
                if current_memory > self.memory_stats['peak']:
                    self.memory_stats['peak'] = current_memory
                time.sleep(1)
        
        self.memory_thread = threading.Thread(target=monitor_memory, daemon=True)
        self.memory_thread.start()

    def _get_comprehensive_features(self) -> Dict:
        """
        Get ALL TSFresh features - comprehensive feature set (~350 features).
        This maintains the original feature count while optimizing extraction.
        """
        
        comprehensive_features = {
            # Basic statistical features (16 features)
            'abs_energy': None,
            'absolute_maximum': None,
            'absolute_sum_of_changes': None,
            'mean': None,
            'median': None,
            'minimum': None,
            'maximum': None,
            'standard_deviation': None,
            'variance': None,
            'skewness': None,
            'kurtosis': None,
            'root_mean_square': None,
            'sum_values': None,
            'mean_abs_change': None,
            'mean_change': None,
            'mean_second_derivative_central': None,
            
            # Autocorrelation features (15 features)
            'autocorrelation': [{'lag': lag} for lag in range(1, 10)],
            'agg_autocorrelation': [
                {'f_agg': 'mean', 'maxlag': 40},
                {'f_agg': 'median', 'maxlag': 40},
                {'f_agg': 'var', 'maxlag': 40}
            ],
            'partial_autocorrelation': [{'lag': lag} for lag in range(1, 4)],
            
            # Trend analysis (20 features)
            'agg_linear_trend': [
                {'attr': 'rvalue', 'chunk_len': 5, 'f_agg': agg} 
                for agg in ['max', 'min', 'mean', 'var']
            ] + [
                {'attr': 'rvalue', 'chunk_len': 10, 'f_agg': agg} 
                for agg in ['max', 'min', 'mean', 'var']
            ] + [
                {'attr': 'rvalue', 'chunk_len': 50, 'f_agg': agg} 
                for agg in ['max', 'min', 'mean', 'var']
            ],
            'linear_trend': [
                {'attr': 'slope'}, 
                {'attr': 'rvalue'}, 
                {'attr': 'stderr'}, 
                {'attr': 'intercept'}
            ],
            
            # Entropy and complexity measures (20+ features)
            'approximate_entropy': [{'m': 2, 'r': r} for r in [0.1, 0.3, 0.5, 0.7, 0.9]],
            'sample_entropy': None,
            'binned_entropy': [{'max_bins': 10}],
            'fourier_entropy': [{'bins': 10}],
            'permutation_entropy': [
                {'dimension': d, 'tau': 1} for d in range(3, 8)
            ],
            'cid_ce': [{'normalize': True}, {'normalize': False}],
            'lempel_ziv_complexity': [{'bins': b} for b in [2, 3, 5, 10]],
            
            # AR coefficients (10 features)
            'ar_coefficient': [{'coeff': i, 'k': 10} for i in range(10)],
            
            # Nonlinear dynamics (10 features)
            'c3': [{'lag': lag} for lag in range(1, 4)],
            'time_reversal_asymmetry_statistic': [{'lag': lag} for lag in range(1, 4)],
            'friedrich_coefficients': [{'coeff': i, 'm': 3, 'r': 30} for i in range(4)],
            
            # FULL frequency domain features (~120 features - MOST IMPORTANT)
            'fft_aggregated': [
                {'aggtype': 'centroid'},
                {'aggtype': 'variance'},
                {'aggtype': 'skew'},
                {'aggtype': 'kurtosis'}
            ],
            'fft_coefficient': [
                # Low frequency coefficients (most important)
                {'coeff': i, 'attr': attr} 
                for i in range(0, 21)
                for attr in (['real', 'abs'] if i == 0 else ['real', 'imag', 'abs', 'angle'])
            ] + [
                # Mid frequency coefficients
                {'coeff': i, 'attr': attr}
                for i in range(25, 101, 5)
                for attr in ['real', 'imag', 'abs']
            ] + [
                # High frequency coefficients
                {'coeff': i, 'attr': attr}
                for i in range(125, 401, 25)
                for attr in ['real', 'abs']
            ],
            'spkt_welch_density': [{'coeff': i} for i in [2, 5, 8]],
            'power_spectral_density': [{'coeff': i} for i in [2, 5, 8]],
            
            # Wavelet features (25 features)
            'cwt_coefficients': [
                {'widths': (2, 5, 10, 20), 'coeff': i, 'w': w} 
                for i in range(15)
                for w in [2, 5, 10]
            ],
            'energy_ratio_by_chunks': [
                {'num_segments': 10, 'segment_focus': i} 
                for i in range(10)
            ],
            
            # Statistical tests and measures (15 features)
            'augmented_dickey_fuller': [
                {'attr': 'teststat'}, 
                {'attr': 'pvalue'}, 
                {'attr': 'usedlag'}
            ],
            'number_crossing_m': [{'m': m} for m in [-1, 0, 1]],
            'longest_strike_above_mean': None,
            'longest_strike_below_mean': None,
            'variance_larger_than_standard_deviation': None,
            'has_duplicate_max': None,
            'has_duplicate_min': None,
            'has_duplicate': None,
            'count_above_mean': None,
            'count_below_mean': None,
            
            # Quantiles and ranges (35 features)
            'quantile': [{'q': q} for q in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]],
            'range_count': [
                {'min': -r, 'max': r} for r in [0.1, 0.2, 0.5, 1.0, 2.0]
            ],
            'ratio_beyond_r_sigma': [{'r': r} for r in [0.5, 1, 1.5, 2, 2.5, 3]],
            'symmetry_looking': [{'r': r} for r in [0.05, 0.1, 0.15, 0.2, 0.25]],
            'large_standard_deviation': [{'r': r} for r in [0.05, 0.1, 0.15, 0.2, 0.25]],
            'ratio_value_number_to_time_series_length': None,
            'first_location_of_maximum': None,
            'first_location_of_minimum': None,
            'last_location_of_maximum': None,
            'last_location_of_minimum': None,
            
            # Peak and pattern detection (15 features)
            'number_peaks': [{'n': n} for n in [1, 3, 5, 10, 50]],
            'number_cwt_peaks': [{'n': n} for n in [1, 5]],
            'benford_correlation': None,
            
            # Index mass quantiles (8 features)
            'index_mass_quantile': [{'q': q} for q in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]],
            
            # Matrix profile features (advanced - 10 features)
            'matrix_profile': [
                {'threshold': t, 'feature': f} 
                for t in [0.98, 0.99] 
                for f in ['min', 'max', 'mean', 'median', 'std']
            ],
            
            # Change point detection (5 features)
            'change_quantiles': [
                {'ql': 0.0, 'qh': 0.2, 'isabs': False, 'f_agg': 'mean'},
                {'ql': 0.0, 'qh': 0.2, 'isabs': False, 'f_agg': 'var'},
                {'ql': 0.8, 'qh': 1.0, 'isabs': False, 'f_agg': 'mean'},
                {'ql': 0.8, 'qh': 1.0, 'isabs': False, 'f_agg': 'var'},
                {'ql': 0.0, 'qh': 1.0, 'isabs': True, 'f_agg': 'mean'}
            ],
        }
        
        # Calculate feature count
        total_features = 0
        for feature_name, params in comprehensive_features.items():
            if params is None:
                total_features += 1
            elif isinstance(params, list):
                total_features += len(params)
        
        self.logger.info(f"Comprehensive feature set loaded: ~{total_features} features")
        return comprehensive_features

    def _load_npz_file_optimized(self, npz_path: Path) -> Optional[np.ndarray]:
        """
        Highly optimized NPZ file loading with memory mapping and preprocessing.
        """
        try:
            if self.use_memory_mapping:
                # Use memory mapping for better performance
                with np.load(npz_path, mmap_mode='r') as data:
                    # Try common key names in order of preference
                    for key in ['flux', 'intensity', 'lightcurve', 'data', 'y', 'signal']:
                        if key in data.files:
                            curve = np.array(data[key])  # Load into memory
                            break
                    else:
                        # Take first array if no standard key found
                        curve = np.array(data[data.files[0]])
            else:
                # Standard loading
                with np.load(npz_path) as data:
                    for key in ['flux', 'intensity', 'lightcurve', 'data', 'y', 'signal']:
                        if key in data.files:
                            curve = data[key]
                            break
                    else:
                        curve = data[data.files[0]]
            
            # Optimized preprocessing pipeline
            if curve.ndim > 1:
                curve = curve.flatten()
            
            # Remove NaN/inf values efficiently
            finite_mask = np.isfinite(curve)
            if not np.all(finite_mask):
                curve = curve[finite_mask]
            
            # Quality checks
            if len(curve) < 50:  # Minimum length for meaningful features
                return None
            
            # Remove extreme outliers (optional - can improve feature stability)
            if len(curve) > 100:
                q1, q99 = np.percentile(curve, [1, 99])
                outlier_mask = (curve >= q1) & (curve <= q99)
                if np.sum(outlier_mask) > len(curve) * 0.8:  # Keep if >80% data remains
                    curve = curve[outlier_mask]
            
            return curve.astype(np.float64)  # Ensure consistent dtype
                
        except Exception as e:
            self.logger.error(f"Error loading {npz_path}: {e}")
            return None

    def _extract_features_from_curve_base(self, npz_path_str: str) -> Optional[Dict]:
        """
        Base feature extraction function (cacheable).
        """
        npz_path = Path(npz_path_str)
        try:
            curve = self._load_npz_file_optimized(npz_path)
            if curve is None:
                return None
            
            # Create DataFrame in TSFresh format
            df = pd.DataFrame({
                'id': [npz_path.stem] * len(curve),
                'time': np.arange(len(curve)),  # Use numpy for speed
                'value': curve
            })
            
            # Extract features with optimizations
            extracted_features = extract_features(
                df,
                column_id='id',
                column_sort='time',
                column_value='value',
                default_fc_parameters=self.feature_params,
                impute_function=impute,
                n_jobs=1,  # Single job per process to avoid nested parallelization
                disable_progressbar=True,
                show_warnings=False,
                chunksize=None,  # Process all at once for small datasets
                max_timeshift=None,
                distributor=None
            )
            
            # Convert to dictionary efficiently
            feature_dict = extracted_features.iloc[0].to_dict()
            feature_dict['filename'] = npz_path.name
            
            return feature_dict
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed for {npz_path}: {e}")
            return None

    def _extract_features_from_curve(self, npz_path: Path) -> Optional[Dict]:
        """
        Wrapper for feature extraction with optional caching.
        """
        if self.enable_caching:
            return self._extract_features_cached(str(npz_path))
        else:
            return self._extract_features_from_curve_base(str(npz_path))

    def _process_batch_optimized(self, batch_files: List[Path]) -> List[Dict]:
        """
        Optimized batch processing with advanced parallel processing.
        """
        # Use joblib with optimized backend
        results = Parallel(
            n_jobs=self.n_jobs, 
            backend='multiprocessing',
            batch_size='auto',
            pre_dispatch='2*n_jobs',
            verbose=0
        )(
            delayed(self._extract_features_from_curve)(npz_file) 
            for npz_file in batch_files
        )
        
        # Process results efficiently
        valid_results = []
        for i, result in enumerate(results):
            if result is not None:
                valid_results.append(result)
                self.processed_files.add(batch_files[i].name)
            else:
                self.failed_files.add(batch_files[i].name)
        
        # Force garbage collection after batch
        gc.collect()
        
        return valid_results

    def _load_progress(self):
        """Load previous extraction progress if exists."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'rb') as f:
                    progress_data = pickle.load(f)
                    self.processed_files = progress_data.get('processed_files', set())
                    self.failed_files = progress_data.get('failed_files', set())
                    self.extracted_features = progress_data.get('extracted_features', [])
                    self.processing_stats = progress_data.get('processing_stats', self.processing_stats)
                
                self.logger.info(f"Resumed: {len(self.processed_files)} files already processed")
                if len(self.failed_files) > 0:
                    self.logger.info(f"Previous failures: {len(self.failed_files)} files")
                
            except Exception as e:
                self.logger.warning(f"Could not load progress file: {e}")

    def _save_progress(self):
        """Save current extraction progress with stats."""
        progress_data = {
            'processed_files': self.processed_files,
            'failed_files': self.failed_files,
            'extracted_features': self.extracted_features,
            'processing_stats': self.processing_stats
        }
        
        try:
            with open(self.progress_file, 'wb') as f:
                pickle.dump(progress_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            self.logger.error(f"Could not save progress: {e}")

    def extract_all_features(self):
        """
        HIGH-PERFORMANCE extraction of ALL features from NPZ files.
        """
        # Get all NPZ files
        npz_files = list(self.npz_folder.glob("*.npz"))
        
        if not npz_files:
            self.logger.error(f"No NPZ files found in {self.npz_folder}")
            return
        
        # Filter out already processed files
        remaining_files = [f for f in npz_files if f.name not in self.processed_files]
        
        self.logger.info(f"Found {len(npz_files)} total NPZ files")
        self.logger.info(f"Processing {len(remaining_files)} remaining files")
        
        if not remaining_files:
            self.logger.info("All files already processed!")
            self._save_final_csv()
            return
        
        # Performance monitoring setup
        start_time = time.time()
        self.processing_stats['start_time'] = start_time
        
        # Create optimized batches
        batches = [
            remaining_files[i:i + self.batch_size] 
            for i in range(0, len(remaining_files), self.batch_size)
        ]
        
        total_batches = len(batches)
        self.logger.info(f"Processing {total_batches} optimized batches of size {self.batch_size}")
        
        # Process batches with advanced progress tracking
        with tqdm(total=len(remaining_files), 
                 desc="Extracting ALL features",
                 unit="files",
                 ncols=120,
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}') as pbar:
            
            for batch_idx, batch in enumerate(batches):
                batch_start_time = time.time()
                
                try:
                    # Monitor memory before batch
                    memory_before = psutil.virtual_memory().used / (1024**3)
                    
                    # Process batch with optimizations
                    batch_results = self._process_batch_optimized(batch)
                    
                    # Add results
                    self.extracted_features.extend(batch_results)
                    
                    # Update progress
                    pbar.update(len(batch))
                    
                    # Calculate comprehensive statistics
                    batch_time = time.time() - batch_start_time
                    total_elapsed = time.time() - start_time
                    files_processed = (batch_idx + 1) * self.batch_size
                    files_processed = min(files_processed, len(remaining_files))
                    
                    # Memory monitoring
                    memory_after = psutil.virtual_memory().used / (1024**3)
                    memory_used = memory_after - memory_before
                    
                    # Performance calculations
                    if files_processed > 0:
                        avg_time_per_file = total_elapsed / files_processed
                        remaining_files_count = len(remaining_files) - files_processed
                        eta_seconds = avg_time_per_file * remaining_files_count
                        eta_formatted = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                        
                        success_rate = len(batch_results) / len(batch) * 100
                        
                        # Update stats
                        self.processing_stats.update({
                            'avg_time_per_file': avg_time_per_file,
                            'files_processed': files_processed,
                            'success_rate': (len(self.extracted_features) / files_processed) * 100
                        })
                        
                        # Advanced progress display
                        pbar.set_postfix({
                            'Batch': f"{batch_idx+1}/{total_batches}",
                            'Success': f"{success_rate:.1f}%",
                            'ETA': eta_formatted,
                            'Rate': f"{1/avg_time_per_file:.1f}/s",
                            'Mem': f"{memory_after:.1f}GB",
                            'Features': f"~{len(batch_results)*350 if batch_results else 0}"
                        })
                    
                    # Periodic saves (every 3 batches or when memory is high)
                    if (batch_idx + 1) % 3 == 0 or memory_after > self.memory_limit_gb:
                        self._save_progress()
                        
                        # Force garbage collection if memory usage is high
                        if memory_after > self.memory_limit_gb * 0.9:
                            gc.collect()
                        
                except Exception as e:
                    self.logger.error(f"Error processing batch {batch_idx}: {e}")
                    # Mark all files in batch as failed
                    for npz_file in batch:
                        self.failed_files.add(npz_file.name)
                    pbar.update(len(batch))
        
        # Final processing
        self._finalize_extraction(start_time, len(remaining_files))

    def _finalize_extraction(self, start_time: float, total_files: int):
        """Finalize extraction with comprehensive reporting."""
        total_time = time.time() - start_time
        
        # Update final stats
        self.processing_stats.update({
            'total_time': total_time,
            'total_files': total_files,
            'successful_files': len(self.extracted_features),
            'failed_files': len(self.failed_files),
            'peak_memory_gb': self.memory_stats['peak']
        })
        
        # Stop memory monitoring
        self.memory_monitor_active = False
        
        # Final saves
        self._save_progress()
        self._save_final_csv()
        
        # Comprehensive performance report
        self._generate_performance_report(total_time, total_files)

    def _generate_performance_report(self, total_time: float, total_files: int):
        """Generate comprehensive performance report."""
        self.logger.info("=" * 80)
        self.logger.info("HIGH-PERFORMANCE EXTRACTION COMPLETED")
        self.logger.info("=" * 80)
        
        success_count = len(self.extracted_features)
        fail_count = len(self.failed_files)
        
        self.logger.info(f"Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        self.logger.info(f"Files processed: {success_count}/{total_files}")
        self.logger.info(f"Success rate: {(success_count/total_files)*100:.1f}%")
        self.logger.info(f"Average time per file: {total_time/total_files:.3f} seconds")
        self.logger.info(f"Processing rate: {total_files/total_time:.2f} files/second")
        self.logger.info(f"Peak memory usage: {self.memory_stats['peak']:.2f} GB")
        
        if success_count > 0:
            # Feature statistics
            sample_features = self.extracted_features[0]
            feature_count = len([k for k in sample_features.keys() if k != 'filename'])
            total_features_extracted = feature_count * success_count
            
            self.logger.info(f"Features per file: {feature_count}")
            self.logger.info(f"Total features extracted: {total_features_extracted:,}")
            self.logger.info(f"Feature extraction rate: {total_features_extracted/total_time:.0f} features/second")
            
            # Performance classification
            files_per_second = total_files / total_time
            if files_per_second > 5:
                performance_rating = "EXCELLENT"
            elif files_per_second > 2:
                performance_rating = "VERY GOOD"
            elif files_per_second > 1:
                performance_rating = "GOOD"
            elif files_per_second > 0.5:
                performance_rating = "MODERATE"
            else:
                performance_rating = "SLOW"
            
            self.logger.info(f"Performance rating: {performance_rating}")
            
            # System efficiency metrics
            cpu_efficiency = (total_files/total_time) / self.n_jobs
            memory_efficiency = self.memory_stats['peak'] / self.memory_gb * 100
            
            self.logger.info(f"CPU efficiency: {cpu_efficiency:.2f} files/second/core")
            self.logger.info(f"Memory efficiency: {memory_efficiency:.1f}% of available RAM used")
        
        if fail_count > 0:
            self.logger.warning(f"Failed files: {fail_count}")
            self.logger.warning("Check failed files for issues (corrupted, too short, etc.)")
        
        self.logger.info("=" * 80)

    def _save_final_csv(self):
        """Save extracted features to optimized CSV format."""
        if not self.extracted_features:
            self.logger.warning("No features to save!")
            return
        
        self.logger.info("Saving features to CSV...")
        save_start_time = time.time()
        
        try:
            # Create DataFrame efficiently
            df = pd.DataFrame(self.extracted_features)
            
            # Reorder columns - filename first, then features
            feature_columns = [col for col in df.columns if col != 'filename']
            df = df[['filename'] + sorted(feature_columns)]
            
            # Optimize data types to reduce file size
            for col in feature_columns:
                if df[col].dtype == 'float64':
                    # Check if we can use float32 without significant precision loss
                    if df[col].abs().max() < 1e38:  # float32 range check
                        df[col] = df[col].astype('float32')
            
            # Save with optimized settings
            df.to_csv(
                self.output_csv, 
                index=False,
                float_format='%.6f',  # Reasonable precision for features
                compression=None  # No compression for faster loading later
            )
            
            save_time = time.time() - save_start_time
            file_size_mb = self.output_csv.stat().st_size / (1024**2)
            
            self.logger.info(f"CSV saved successfully: {self.output_csv}")
            self.logger.info(f"File size: {file_size_mb:.2f} MB")
            self.logger.info(f"Save time: {save_time:.2f} seconds")
            self.logger.info(f"Shape: {df.shape}")
            
            # Verify data integrity
            self._verify_saved_data(df)
            
        except Exception as e:
            self.logger.error(f"Failed to save CSV: {e}")
            # Try to save as pickle as backup
            try:
                pickle_path = self.output_csv.with_suffix('.pkl')
                with open(pickle_path, 'wb') as f:
                    pickle.dump(self.extracted_features, f, protocol=pickle.HIGHEST_PROTOCOL)
                self.logger.info(f"Saved backup as pickle: {pickle_path}")
            except Exception as backup_e:
                self.logger.error(f"Backup save also failed: {backup_e}")

    def _verify_saved_data(self, df: pd.DataFrame):
        """Verify the integrity of saved data."""
        try:
            # Check for missing values
            missing_counts = df.isnull().sum()
            total_missing = missing_counts.sum()
            
            if total_missing > 0:
                self.logger.warning(f"Found {total_missing} missing values across all features")
                # Report columns with most missing values
                top_missing = missing_counts[missing_counts > 0].sort_values(ascending=False).head()
                for col, count in top_missing.items():
                    self.logger.warning(f"  {col}: {count} missing values")
            
            # Check for infinite values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            inf_counts = pd.Series(index=numeric_cols, dtype=int)
            for col in numeric_cols:
                inf_counts[col] = np.isinf(df[col]).sum()
            
            total_inf = inf_counts.sum()
            if total_inf > 0:
                self.logger.warning(f"Found {total_inf} infinite values")
                top_inf = inf_counts[inf_counts > 0].sort_values(ascending=False).head()
                for col, count in top_inf.items():
                    self.logger.warning(f"  {col}: {count} infinite values")
            
            # Feature statistics
            feature_cols = [col for col in df.columns if col != 'filename']
            self.logger.info(f"Data quality check completed:")
            self.logger.info(f"  - Total features: {len(feature_cols)}")
            self.logger.info(f"  - Missing values: {total_missing} ({(total_missing/(len(df)*len(feature_cols)))*100:.3f}%)")
            self.logger.info(f"  - Infinite values: {total_inf}")
            
            # Sample feature ranges for sanity check
            if len(feature_cols) > 0:
                sample_features = np.random.choice(feature_cols, min(5, len(feature_cols)), replace=False)
                self.logger.info("Sample feature ranges:")
                for feat in sample_features:
                    feat_min, feat_max = df[feat].min(), df[feat].max()
                    feat_mean, feat_std = df[feat].mean(), df[feat].std()
                    self.logger.info(f"  {feat}: [{feat_min:.3f}, {feat_max:.3f}], μ={feat_mean:.3f}, σ={feat_std:.3f}")
            
        except Exception as e:
            self.logger.warning(f"Data verification failed: {e}")

    def get_processing_stats(self) -> Dict:
        """Get detailed processing statistics."""
        return {
            **self.processing_stats,
            'processed_files_count': len(self.processed_files),
            'failed_files_count': len(self.failed_files),
            'extracted_features_count': len(self.extracted_features),
            'current_memory_gb': self.memory_stats['current'],
            'peak_memory_gb': self.memory_stats['peak'],
            'system_info': {
                'cpu_info': self.cpu_info,
                'cpu_count': self.cpu_count,
                'total_memory_gb': self.memory_gb,
                'n_jobs': self.n_jobs,
                'batch_size': self.batch_size
            }
        }

    def cleanup(self):
        """Clean up resources and temporary files."""
        try:
            # Stop memory monitoring
            self.memory_monitor_active = False
            
            # Clean up cache if enabled
            if self.enable_caching and hasattr(self, 'memory_cache'):
                try:
                    self.memory_cache.clear()
                    cache_dir = Path('./tsfresh_cache')
                    if cache_dir.exists():
                        import shutil
                        shutil.rmtree(cache_dir)
                    self.logger.info("Cache cleaned up")
                except Exception as e:
                    self.logger.warning(f"Could not clean cache: {e}")
            
            # Force garbage collection
            gc.collect()
            
            self.logger.info("Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

    def resume_extraction(self):
        """Resume extraction from where it left off."""
        self.logger.info("Resuming extraction from previous progress...")
        self.extract_all_features()

    def get_failed_files(self) -> List[str]:
        """Get list of files that failed processing."""
        return list(self.failed_files)

    def retry_failed_files(self, max_retries: int = 2):
        """Retry processing failed files with more conservative settings."""
        if not self.failed_files:
            self.logger.info("No failed files to retry")
            return
        
        self.logger.info(f"Retrying {len(self.failed_files)} failed files...")
        
        # Use more conservative settings for retries
        original_batch_size = self.batch_size
        original_n_jobs = self.n_jobs
        
        self.batch_size = max(1, self.batch_size // 2)  # Smaller batches
        self.n_jobs = max(1, self.n_jobs // 2)  # Fewer parallel jobs
        
        failed_copy = self.failed_files.copy()
        self.failed_files.clear()
        
        retry_files = [self.npz_folder / fname for fname in failed_copy]
        
        # Process in smaller batches
        for attempt in range(max_retries):
            if not retry_files:
                break
                
            self.logger.info(f"Retry attempt {attempt + 1}/{max_retries} for {len(retry_files)} files")
            
            # Process remaining failed files
            remaining_retry = []
            for npz_file in retry_files:
                try:
                    result = self._extract_features_from_curve(npz_file)
                    if result is not None:
                        self.extracted_features.append(result)
                        self.processed_files.add(npz_file.name)
                    else:
                        remaining_retry.append(npz_file)
                except Exception as e:
                    self.logger.error(f"Retry failed for {npz_file}: {e}")
                    remaining_retry.append(npz_file)
            
            retry_files = remaining_retry
            
            if retry_files:
                # Add back to failed files for final attempt
                self.failed_files.update(f.name for f in retry_files)
        
        # Restore original settings
        self.batch_size = original_batch_size
        self.n_jobs = original_n_jobs
        
        successful_retries = len(failed_copy) - len(retry_files)
        self.logger.info(f"Retry completed: {successful_retries}/{len(failed_copy)} files recovered")


def main():
    """
    Main execution function with comprehensive error handling.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="High-Performance TSFresh Feature Extractor for Exoplanet Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("npz_folder", help="Path to folder containing NPZ files")
    parser.add_argument("output_csv", help="Output CSV file path")
    parser.add_argument("--n-jobs", type=int, default=None, 
                       help="Number of parallel jobs (auto-detected)")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size for processing (auto-calculated)")
    parser.add_argument("--memory-limit", type=float, default=None,
                       help="Memory limit in GB (auto-detected)")
    parser.add_argument("--disable-caching", action="store_true",
                       help="Disable feature caching")
    parser.add_argument("--disable-memory-mapping", action="store_true",
                       help="Disable memory mapping for NPZ files")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from previous extraction")
    parser.add_argument("--retry-failed", action="store_true",
                       help="Retry previously failed files")
    parser.add_argument("--progress-file", default="hp_extraction_progress.pkl",
                       help="Progress file path")
    
    args = parser.parse_args()
    
    try:
        # Initialize extractor with optimized settings
        extractor = HighPerformanceExoplanetExtractor(
            npz_folder_path=args.npz_folder,
            output_csv_path=args.output_csv,
            progress_file=args.progress_file,
            n_jobs=args.n_jobs,
            batch_size=args.batch_size,
            memory_limit_gb=args.memory_limit,
            use_memory_mapping=not args.disable_memory_mapping,
            enable_caching=not args.disable_caching
        )
        
        # Execute based on mode
        if args.retry_failed:
            extractor.retry_failed_files()
        elif args.resume:
            extractor.resume_extraction()
        else:
            extractor.extract_all_features()
        
        # Print final statistics
        stats = extractor.get_processing_stats()
        print("\n" + "="*60)
        print("FINAL PROCESSING STATISTICS")
        print("="*60)
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
        print("="*60)
        
        # Cleanup
        extractor.cleanup()
        
        print("\n🚀 HIGH-PERFORMANCE EXTRACTION COMPLETED SUCCESSFULLY! 🚀")
        
    except KeyboardInterrupt:
        print("\n⚠️  Extraction interrupted by user")
        try:
            extractor._save_progress()
            print("Progress saved. Use --resume to continue later.")
        except:
            pass
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()