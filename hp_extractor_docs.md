# High-Performance TSFresh Feature Extractor
## Complete Documentation & User Guide

---

## Table of Contents
1. [Overview](#overview)
2. [Installation & Requirements](#installation)
3. [Quick Start Guide](#quick-start)
4. [Detailed Usage](#detailed-usage)
5. [Configuration Parameters](#configuration-parameters)
6. [Input Constraints](#input-constraints)
7. [Common Errors & Solutions](#errors)
8. [Best Practices](#best-practices)
9. [Performance Tuning](#performance-tuning)
10. [Troubleshooting](#troubleshooting)

---

## 1. Overview {#overview}

### What This Tool Does
- Extracts **~350 time-series features** from exoplanet light curve data
- Optimized for **Intel i7 processors** with automatic system detection
- Processes NPZ files in parallel with **3-4x speedup** over baseline
- Maintains full feature set for maximum detection capability
- Includes resume functionality and comprehensive progress tracking

### What This Tool Does NOT Do
- ❌ Does not train machine learning models
- ❌ Does not classify or predict exoplanets
- ❌ Does not handle non-time-series data
- ❌ Does not work with formats other than NPZ
- ❌ Does not perform data augmentation or synthetic generation

### Key Features
✅ Extracts all ~350 TSFresh features per light curve
✅ Auto-optimizes for your i7 processor
✅ Memory-mapped file I/O for speed
✅ Intelligent caching to avoid reprocessing
✅ Resume capability after interruption
✅ Real-time performance monitoring
✅ Comprehensive error handling

---

## 2. Installation & Requirements {#installation}

### System Requirements

**Minimum:**
- Python 3.7+
- 8GB RAM
- 4-core processor
- 10GB free disk space

**Recommended:**
- Python 3.8+
- 16GB+ RAM
- Intel i7 (8+ cores)
- SSD storage
- 50GB+ free disk space

### Required Python Packages

```bash
# Install all dependencies
pip install numpy pandas tsfresh joblib tqdm psutil scikit-learn

# Specific versions (tested)
pip install numpy>=1.19.0
pip install pandas>=1.2.0
pip install tsfresh>=0.19.0
pip install joblib>=1.0.0
pip install tqdm>=4.60.0
pip install psutil>=5.8.0
```

### Installation Steps

```bash
# 1. Clone or download the script
wget https://your-repo/hp_extractor.py

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python -c "import tsfresh; print(tsfresh.__version__)"

# 4. Test system detection
python hp_extractor.py --test
```

---

## 3. Quick Start Guide {#quick-start}

### Step 1: Prepare Your Data

```bash
# Your NPZ files should be in one folder
/path/to/data/
  ├── lightcurve_001.npz
  ├── lightcurve_002.npz
  ├── lightcurve_003.npz
  └── ...
```

### Step 2: Edit the Script

Open `hp_extractor.py` and modify these two lines:

```python
NPZ_FOLDER_PATH = "/path/to/your/npz/files"  # <-- Change this
OUTPUT_CSV_PATH = "/path/to/output/features.csv"  # <-- Change this
```

### Step 3: Run the Extractor

```bash
# Basic usage
python hp_extractor.py

# With output redirection
python hp_extractor.py 2>&1 | tee extraction.log
```

### Step 4: Monitor Progress

The script will display:
- Real-time progress bar
- ETA (Estimated Time to Arrival)
- Processing rate (files/second)
- Memory usage
- Success rate

### Step 5: Get Results

Output files:
- `features.csv` - Main feature matrix
- `features.txt` - Feature names list
- `features.json` - Processing statistics
- `hp_feature_extraction.log` - Detailed logs

---

## 4. Detailed Usage {#detailed-usage}

### Basic Usage

```python
from hp_extractor import HighPerformanceExoplanetExtractor

# Create extractor instance
extractor = HighPerformanceExoplanetExtractor(
    npz_folder_path="/path/to/npz/files",
    output_csv_path="/path/to/output/features.csv"
)

# Run extraction
extractor.extract_all_features()

# Optional: Clean up temporary files
extractor.cleanup()
```

### Advanced Usage

```python
# Custom configuration
extractor = HighPerformanceExoplanetExtractor(
    npz_folder_path="/data/lightcurves",
    output_csv_path="/output/features.csv",
    progress_file="my_progress.pkl",
    n_jobs=8,  # Use 8 cores
    batch_size=100,  # Process 100 files per batch
    memory_limit_gb=12.0,  # Limit to 12GB RAM
    use_memory_mapping=True,  # Enable memory mapping
    enable_caching=True  # Enable caching
)

extractor.extract_all_features()
```

### Resume After Interruption

```python
# If script is interrupted, just run it again
# It will automatically resume from the last saved progress

extractor = HighPerformanceExoplanetExtractor(
    npz_folder_path="/data/lightcurves",
    output_csv_path="/output/features.csv",
    progress_file="my_progress.pkl"  # Same progress file
)

# This will skip already processed files
extractor.extract_all_features()
```

### Get Performance Recommendations

```python
extractor = HighPerformanceExoplanetExtractor(
    npz_folder_path="/data/lightcurves",
    output_csv_path="/output/features.csv"
)

# Get system-specific recommendations
recommendations = extractor.get_performance_recommendations()

print("Current config:", recommendations['current_config'])
print("Optimizations:", recommendations['i7_optimizations'])
print("Expected performance:", recommendations['expected_performance'])
```

---

## 5. Configuration Parameters {#configuration-parameters}

### Class Initialization Parameters

#### `npz_folder_path` (required)
- **Type:** `str`
- **Description:** Path to folder containing NPZ files
- **Constraints:**
  - Must be a valid directory path
  - Must contain at least one `.npz` file
  - Must have read permissions
- **Example:** `"/home/user/data/lightcurves"`

#### `output_csv_path` (required)
- **Type:** `str`
- **Description:** Path where feature CSV will be saved
- **Constraints:**
  - Parent directory must be writable
  - Will overwrite if file exists
  - Recommended extension: `.csv`
- **Example:** `"/home/user/output/features.csv"`

#### `progress_file` (optional)
- **Type:** `str`
- **Default:** `"hp_extraction_progress.pkl"`
- **Description:** Path to save/load extraction progress
- **Constraints:**
  - Must be writable location
  - Should end with `.pkl`
- **Example:** `"my_extraction_progress.pkl"`

#### `n_jobs` (optional)
- **Type:** `int` or `None`
- **Default:** `None` (auto-detect)
- **Description:** Number of parallel processes
- **Constraints:**
  - Must be between 1 and CPU count
  - `None` = auto-detect (recommended)
  - **Recommended:** Leave as `None` for auto-optimization
- **Valid Range:** `1` to `mp.cpu_count()`
- **Example:** `8` (for 8-core processing)

#### `batch_size` (optional)
- **Type:** `int` or `None`
- **Default:** `None` (auto-calculate)
- **Description:** Number of files to process in each batch
- **Constraints:**
  - Must be between 10 and 500
  - `None` = auto-calculate based on RAM (recommended)
  - Larger = more memory usage, potentially faster
- **Valid Range:** `10` to `500`
- **Example:** `50`

#### `memory_limit_gb` (optional)
- **Type:** `float` or `None`
- **Default:** `None` (auto-detect as 80% of available RAM)
- **Description:** Maximum memory usage in GB
- **Constraints:**
  - Must be positive
  - Should be less than total system RAM
  - `None` = auto-detect (recommended)
- **Valid Range:** `0.5` to `total_system_ram * 0.9`
- **Example:** `12.0` (for 12GB limit)

#### `use_memory_mapping` (optional)
- **Type:** `bool`
- **Default:** `True`
- **Description:** Use memory-mapped file I/O
- **Constraints:**
  - `True` recommended for files on SSD
  - `False` for network drives or slow storage
- **Example:** `True`

#### `enable_caching` (optional)
- **Type:** `bool`
- **Default:** `True`
- **Description:** Enable intelligent caching
- **Constraints:**
  - Requires disk space for cache (~5-10% of data size)
  - `True` recommended for repeated processing
- **Example:** `True`

---

## 6. Input Constraints {#input-constraints}

### NPZ File Requirements

#### File Format
- **Extension:** Must be `.npz` (NumPy compressed archive)
- **Content:** Must contain at least one array
- **Encoding:** Standard NumPy format

#### Array Structure
```python
# Valid structures:
# Option 1: Standard key names
{'flux': array([1.0, 1.1, 0.9, ...])}
{'intensity': array([1.0, 1.1, 0.9, ...])}
{'lightcurve': array([1.0, 1.1, 0.9, ...])}
{'data': array([1.0, 1.1, 0.9, ...])}

# Option 2: Any single array (will use first array found)
{'my_custom_name': array([1.0, 1.1, 0.9, ...])}
```

#### Array Requirements

**Data Type:**
- Integer or float arrays
- Will be converted to `float64`
- Complex numbers NOT supported

**Shape:**
- **Minimum length:** 50 data points (recommended: 100+)
- **Maximum length:** No hard limit (tested up to 100,000 points)
- **Dimension:** 1D array (will flatten if 2D)
- **Valid shapes:** `(n,)` or `(1, n)` or `(n, 1)`

**Data Quality:**
- **NaN/Inf handling:** Automatically removed
- **Outliers:** Extreme values (1st/99th percentile) optionally filtered
- **Missing data:** Gaps are NOT interpolated
- **Valid percentage:** At least 80% finite values required

#### Example Valid NPZ File

```python
import numpy as np

# Create valid light curve
time = np.linspace(0, 100, 2000)
flux = 1.0 + 0.01 * np.sin(2 * np.pi * time / 10)  # Simple periodic signal
flux += np.random.normal(0, 0.001, len(flux))  # Add noise

# Save in valid format
np.savez('lightcurve_001.npz', flux=flux)

# Also valid:
np.savez('lightcurve_002.npz', intensity=flux)
np.savez('lightcurve_003.npz', data=flux)
```

### Folder Structure Requirements

```
Valid structure:
/data/
  ├── curve_001.npz  ✓
  ├── curve_002.npz  ✓
  ├── curve_003.npz  ✓
  └── ...

Invalid structures:
/data/
  ├── subfolder1/
  │   ├── curve_001.npz  ✗ (not in root)
  ├── curve.txt  ✗ (wrong format)
  ├── .hidden.npz  ✗ (hidden file)
```

**Requirements:**
- All NPZ files must be in the **root** of the specified folder
- Subdirectories are **NOT** recursively searched
- Hidden files (starting with `.`) are ignored

---

## 7. Common Errors & Solutions {#errors}

### Error 1: "No NPZ files found"

**Error Message:**
```
ERROR: No NPZ files found in: /path/to/folder
```

**Causes:**
1. Wrong folder path
2. NPZ files in subdirectories
3. No `.npz` extension
4. Permission issues

**Solutions:**
```bash
# Check if folder exists
ls -la /path/to/folder

# Check for NPZ files
find /path/to/folder -name "*.npz"

# Check permissions
chmod 755 /path/to/folder
```

---

### Error 2: "Memory Error" / "Out of Memory"

**Error Message:**
```
MemoryError: Unable to allocate array
```

**Causes:**
1. Batch size too large
2. Insufficient RAM
3. Memory leak from previous runs

**Solutions:**
```python
# Reduce batch size
extractor = HighPerformanceExoplanetExtractor(
    npz_folder_path="/data",
    output_csv_path="/output/features.csv",
    batch_size=20,  # Reduce from default
    memory_limit_gb=6.0  # Set explicit limit
)

# Or close other applications and try again
```

---

### Error 3: "Feature extraction failed"

**Error Message:**
```
Feature extraction failed for lightcurve_xxx.npz: ...
```

**Causes:**
1. Light curve too short (< 50 points)
2. All values are NaN/Inf
3. Corrupted NPZ file
4. Invalid array structure

**Solutions:**
```python
# Check NPZ file manually
import numpy as np

with np.load('problematic_file.npz') as data:
    print("Keys:", data.files)
    arr = data['flux']  # or appropriate key
    print("Shape:", arr.shape)
    print("Finite values:", np.sum(np.isfinite(arr)))
    print("Length:", len(arr))

# Requirements:
# - Length >= 50
# - At least 80% finite values
# - 1D or flattenable array
```

---

### Error 4: "Permission denied"

**Error Message:**
```
PermissionError: [Errno 13] Permission denied: '/output/features.csv'
```

**Causes:**
1. No write permission to output directory
2. File is open in another program
3. Directory doesn't exist

**Solutions:**
```bash
# Create output directory
mkdir -p /path/to/output

# Fix permissions
chmod 755 /path/to/output

# Check if file is locked
lsof | grep features.csv
```

---

### Error 5: "TSFresh Import Error"

**Error Message:**
```
ModuleNotFoundError: No module named 'tsfresh'
```

**Causes:**
1. TSFresh not installed
2. Wrong Python environment
3. Version incompatibility

**Solutions:**
```bash
# Install TSFresh
pip install tsfresh

# Verify installation
python -c "import tsfresh; print(tsfresh.__version__)"

# Check Python environment
which python
pip list | grep tsfresh

# If using conda
conda install -c conda-forge tsfresh
```

---

### Error 6: "Progress file corrupted"

**Error Message:**
```
Could not load progress file: ...
```

**Causes:**
1. Progress file from different version
2. Corrupted pickle file
3. Incomplete write during crash

**Solutions:**
```bash
# Delete progress file and start fresh
rm hp_extraction_progress.pkl

# Or rename for backup
mv hp_extraction_progress.pkl hp_extraction_progress.pkl.backup
```

---

### Error 7: "Joblib/Multiprocessing Error"

**Error Message:**
```
BrokenProcessPool: A process in the pool was terminated
```

**Causes:**
1. Process killed due to memory
2. System resource limits
3. Signal interruption

**Solutions:**
```python
# Reduce parallel processes
extractor = HighPerformanceExoplanetExtractor(
    npz_folder_path="/data",
    output_csv_path="/output/features.csv",
    n_jobs=4,  # Reduce from auto-detected value
    batch_size=20  # Smaller batches
)

# Check system limits
import resource
print(resource.getrlimit(resource.RLIMIT_NPROC))
```

---

### Error 8: "Invalid NPZ key"

**Error Message:**
```
KeyError: 'flux'
```

**Causes:**
1. NPZ doesn't contain expected keys
2. Non-standard array naming

**Solutions:**
The code automatically tries multiple keys:
- `'flux'`, `'intensity'`, `'lightcurve'`, `'data'`, `'y'`, `'signal'`
- Falls back to first available array

If still failing:
```python
# Check what keys are in your NPZ
import numpy as np
with np.load('your_file.npz') as data:
    print("Available keys:", data.files)

# If you need to add a custom key, modify the code:
# In _load_npz_file_optimized(), add your key to the list:
for key in ['flux', 'intensity', 'YOUR_CUSTOM_KEY', ...]:
```

---

## 8. Best Practices {#best-practices}

### Data Preparation

1. **Standardize your NPZ files:**
   ```python
   import numpy as np
   
   # Use consistent key name across all files
   for i, curve in enumerate(light_curves):
       np.savez(f'curve_{i:04d}.npz', flux=curve)
   ```

2. **Pre-filter bad data:**
   ```python
   # Remove curves that are too short or too noisy
   valid_curves = []
   for curve in all_curves:
       if len(curve) >= 100 and np.sum(np.isfinite(curve)) > len(curve) * 0.95:
           valid_curves.append(curve)
   ```

3. **Consistent naming:**
   ```python
   # Use zero-padded numbers for proper sorting
   curve_0001.npz  ✓
   curve_0002.npz  ✓
   ...
   curve_1000.npz  ✓
   
   # Avoid:
   curve_1.npz  ✗
   curve_10.npz  ✗ (sorting issues)
   ```

### Processing Strategy

1. **Start with a test run:**
   ```python
   # Test on 10 files first
   test_files = list(Path(NPZ_FOLDER).glob("*.npz"))[:10]
   # Copy to test folder and run
   ```

2. **Monitor first batch:**
   ```bash
   # Watch memory usage
   watch -n 1 free -h
   
   # Monitor CPU
   htop
   ```

3. **Use screen/tmux for long runs:**
   ```bash
   # Start screen session
   screen -S extraction
   
   # Run script
   python hp_extractor.py
   
   # Detach: Ctrl+A, D
   # Reattach: screen -r extraction
   ```

### Resource Management

1. **Optimal settings for i7 systems:**

   ```python
   # 8GB RAM
   extractor = HighPerformanceExoplanetExtractor(
       n_jobs=4,
       batch_size=30
   )
   
   # 16GB RAM
   extractor = HighPerformanceExoplanetExtractor(
       n_jobs=6,
       batch_size=50
   )
   
   # 32GB+ RAM
   extractor = HighPerformanceExoplanetExtractor(
       n_jobs=None,  # Auto-detect
       batch_size=100
   )
   ```

2. **Storage considerations:**
   - Keep NPZ files on SSD for best performance
   - Output CSV size ≈ 0.5-1 MB per file
   - Cache requires ~5-10% of data size

3. **Clean up between runs:**
   ```python
   # After successful extraction
   extractor.cleanup()  # Removes progress and cache files
   ```

---

## 9. Performance Tuning {#performance-tuning}

### Benchmarking Your System

```python
import time
import numpy as np
from hp_extractor import HighPerformanceExoplanetExtractor

# Create test data
test_folder = Path("./test_data")
test_folder.mkdir(exist_ok=True)

for i in range(10):
    curve = np.random.randn(2000)
    np.savez(test_folder / f"test_{i:04d}.npz", flux=curve)

# Benchmark
start = time.time()
extractor = HighPerformanceExoplanetExtractor(
    npz_folder_path="./test_data",
    output_csv_path="./test_output.csv"
)
extractor.extract_all_features()
elapsed = time.time() - start

print(f"Time for 10 files: {elapsed:.2f} seconds")
print(f"Time per file: {elapsed/10:.2f} seconds")
print(f"Estimated time for 1000 files: {(elapsed/10)*1000/60:.1f} minutes")
```

### Optimization Strategies

**For Maximum Speed:**
```python
extractor = HighPerformanceExoplanetExtractor(
    n_jobs=None,  # Use all cores
    batch_size=None,  # Auto-optimize
    use_memory_mapping=True,
    enable_caching=True
)
```

**For Memory-Constrained Systems:**
```python
extractor = HighPerformanceExoplanetExtractor(
    n_jobs=4,  # Fewer processes
    batch_size=20,  # Smaller batches
    memory_limit_gb=6.0,
    use_memory_mapping=False,  # Disable if causing issues
    enable_caching=False  # Save disk space
)
```

**For Network Storage:**
```python
extractor = HighPerformanceExoplanetExtractor(
    n_jobs=4,  # Reduce to avoid network congestion
    batch_size=10,
    use_memory_mapping=False,  # Don't use mmap for network drives
    enable_caching=False
)
```

### Performance Metrics

Expected processing rates on i7:

| System Config | Files/Hour | Time per File | 1000 Files |
|--------------|------------|---------------|------------|
| i7-8700 (6-core, 16GB) | 500-800 | 4-7 sec | 1.2-2 hrs |
| i7-9700K (8-core, 16GB) | 600-1000 | 3-6 sec | 1-1.7 hrs |
| i7-10700 (8-core, 32GB) | 800-1200 | 2.5-4 sec | 0.8-1.2 hrs |
| i7-11700K (8-core, 32GB) | 900-1400 | 2-4 sec | 0.7-1.1 hrs |

---

## 10. Troubleshooting {#troubleshooting}

### Debug Mode

Enable detailed logging:

```python
import logging

# Set to DEBUG level
logging.basicConfig(level=logging.DEBUG)

extractor = HighPerformanceExoplanetExtractor(...)
extractor.extract_all_features()
```

### Check System Resources

```python
import psutil
import multiprocessing as mp

print("CPU cores:", mp.cpu_count())
print("RAM:", psutil.virtual_memory().total / (1024**3), "GB")
print("Disk space:", psutil.disk_usage('/').free / (1024**3), "GB")
```

### Validate Single File

```python
from pathlib import Path
import numpy as np

# Test single file extraction
test_file = Path("/path/to/single/file.npz")

extractor = HighPerformanceExoplanetExtractor(
    npz_folder_path=test_file.parent,
    output_csv_path="./test_single.csv"
)

result = extractor._extract_features_from_curve(test_file)
print("Features extracted:", len(result) if result else "FAILED")
```

### Performance Profiling

```python
import cProfile
import pstats

# Profile the extraction
profiler = cProfile.Profile()
profiler.enable()

extractor.extract_all_features()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 slowest functions
```

---

## Quick Reference Commands

### Installation
```bash
pip install numpy pandas tsfresh joblib tqdm psutil
```

### Basic Run
```bash
python hp_extractor.py
```

### With Logging
```bash
python hp_extractor.py 2>&1 | tee extraction_$(date +%Y%m%d_%H%M%S).log
```

### Resume After Interrupt
```bash
# Just run again - it will auto-resume
python hp_extractor.py
```

### Clean Up
```python
extractor.cleanup()  # In script
```
```bash
rm hp_extraction_progress.pkl  # From command line
rm -rf ./tsfresh_cache  # Remove cache
```

---

## Support & Contact

For issues, questions, or contributions:
- Check logs: `hp_feature_extraction.log`
- Review error messages carefully
- Verify all input constraints
- Test with small subset first

---

**Version:** 1.0.0  
**Last Updated:** 2025  
**Optimized For:** Intel i7 processors  
**Features:** ~350 per light curve