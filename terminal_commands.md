# High-Performance TSFresh Extractor - Terminal Commands Guide

## Complete Command Reference for Terminal Usage

---

## Installation Commands

```bash
# 1. Install required packages
pip install numpy pandas tsfresh joblib tqdm psutil scikit-learn

# 2. Or install from requirements file (if available)
pip install -r requirements.txt

# 3. Verify installation
python -c "import tsfresh; print(f'TSFresh version: {tsfresh.__version__}')"
```

---

## Basic Usage Commands

### 1. **Most Basic Command (Default Settings)**
```bash
python hp_extractor.py /path/to/npz_folder /path/to/output.csv
```

**Example with your KIC data:**
```bash
python hp_extractor.py ./lightcurve_ultraclean_5000 ./output/kic_features.csv
```

---

### 2. **With Custom Number of Cores**
```bash
python hp_extractor.py /path/to/npz_folder /path/to/output.csv --n-jobs 8
```

**Parameters:**
- `--n-jobs` : Number of parallel processes (1 to max cores)
- **Default:** Auto-detected (CPU cores - 1)

**Examples:**
```bash
# Use 4 cores
python hp_extractor.py ./lightcurve_ultraclean_5000 ./output/features.csv --n-jobs 4

# Use 8 cores
python hp_extractor.py ./lightcurve_ultraclean_5000 ./output/features.csv --n-jobs 8

# Use 12 cores
python hp_extractor.py ./lightcurve_ultraclean_5000 ./output/features.csv --n-jobs 12
```

---

### 3. **With Custom Batch Size**
```bash
python hp_extractor.py /path/to/npz_folder /path/to/output.csv --batch-size 50
```

**Parameters:**
- `--batch-size` : Number of files per batch (10-500)
- **Default:** Auto-calculated based on RAM

**Examples:**
```bash
# Small batches (low RAM systems)
python hp_extractor.py ./lightcurve_ultraclean_5000 ./output/features.csv --batch-size 20

# Medium batches
python hp_extractor.py ./lightcurve_ultraclean_5000 ./output/features.csv --batch-size 50

# Large batches (high RAM systems)
python hp_extractor.py ./lightcurve_ultraclean_5000 ./output/features.csv --batch-size 100
```

---

### 4. **With Memory Limit**
```bash
python hp_extractor.py /path/to/npz_folder /path/to/output.csv --memory-limit 12.0
```

**Parameters:**
- `--memory-limit` : Maximum RAM usage in GB
- **Default:** 80% of available RAM

**Examples:**
```bash
# Limit to 8GB RAM
python hp_extractor.py ./lightcurve_ultraclean_5000 ./output/features.csv --memory-limit 8.0

# Limit to 16GB RAM
python hp_extractor.py ./lightcurve_ultraclean_5000 ./output/features.csv --memory-limit 16.0
```

---

### 5. **Resume Previous Extraction**
```bash
python hp_extractor.py /path/to/npz_folder /path/to/output.csv --resume
```

**When to use:**
- After interruption (Ctrl+C)
- After system crash
- After power failure

**Example:**
```bash
python hp_extractor.py ./lightcurve_ultraclean_5000 ./output/features.csv --resume
```

---

### 6. **Retry Failed Files**
```bash
python hp_extractor.py /path/to/npz_folder /path/to/output.csv --retry-failed
```

**When to use:**
- After initial extraction completes with failures
- Uses more conservative settings for retry

**Example:**
```bash
python hp_extractor.py ./lightcurve_ultraclean_5000 ./output/features.csv --retry-failed
```

---

### 7. **Disable Caching**
```bash
python hp_extractor.py /path/to/npz_folder /path/to/output.csv --disable-caching
```

**When to use:**
- Limited disk space
- One-time processing
- Testing different feature sets

**Example:**
```bash
python hp_extractor.py ./lightcurve_ultraclean_5000 ./output/features.csv --disable-caching
```

---

### 8. **Disable Memory Mapping**
```bash
python hp_extractor.py /path/to/npz_folder /path/to/output.csv --disable-memory-mapping
```

**When to use:**
- Network drives
- Slow storage devices
- Memory mapping causing issues

**Example:**
```bash
python hp_extractor.py ./lightcurve_ultraclean_5000 ./output/features.csv --disable-memory-mapping
```

---

### 9. **Custom Progress File**
```bash
python hp_extractor.py /path/to/npz_folder /path/to/output.csv --progress-file my_progress.pkl
```

**When to use:**
- Running multiple extractions simultaneously
- Custom file organization

**Example:**
```bash
python hp_extractor.py ./lightcurve_ultraclean_5000 ./output/features.csv --progress-file kic_progress.pkl
```

---

## Combined Commands (Most Common)

### **Optimized for 8-Core i7 with 16GB RAM**
```bash
python hp_extractor.py ./lightcurve_ultraclean_5000 ./output/kic_features.csv \
  --n-jobs 7 \
  --batch-size 50 \
  --memory-limit 12.0
```

### **Conservative Settings (Low RAM)**
```bash
python hp_extractor.py ./lightcurve_ultraclean_5000 ./output/kic_features.csv \
  --n-jobs 4 \
  --batch-size 20 \
  --memory-limit 6.0
```

### **Maximum Performance (High-End System)**
```bash
python hp_extractor.py ./lightcurve_ultraclean_5000 ./output/kic_features.csv \
  --n-jobs 11 \
  --batch-size 100 \
  --memory-limit 24.0
```

### **Network Storage / Slow Drive**
```bash
python hp_extractor.py ./lightcurve_ultraclean_5000 ./output/kic_features.csv \
  --n-jobs 4 \
  --batch-size 10 \
  --disable-memory-mapping \
  --disable-caching
```

---

## Background Processing Commands

### **Run in Background with Logging**
```bash
nohup python hp_extractor.py ./lightcurve_ultraclean_5000 ./output/features.csv > extraction.log 2>&1 &
```

### **Run in Screen Session**
```bash
# Start screen session
screen -S extraction

# Run extractor
python hp_extractor.py ./lightcurve_ultraclean_5000 ./output/features.csv

# Detach: Ctrl+A then D
# Reattach later: screen -r extraction
```

### **Run in tmux Session**
```bash
# Start tmux session
tmux new -s extraction

# Run extractor
python hp_extractor.py ./lightcurve_ultraclean_5000 ./output/features.csv

# Detach: Ctrl+B then D
# Reattach later: tmux attach -t extraction
```

---

## Monitoring Commands

### **Monitor Progress in Real-Time**
```bash
# In another terminal, watch the log file
tail -f hp_feature_extraction.log
```

### **Monitor System Resources**
```bash
# CPU and Memory usage
htop

# Memory only
watch -n 1 free -h

# Disk I/O
iostat -x 2

# Process-specific monitoring
top -p $(pgrep -f hp_extractor.py)
```

---

## File Management Commands

### **Check NPZ Files**
```bash
# Count NPZ files
ls -1 ./lightcurve_ultraclean_5000/*.npz | wc -l

# List first 10 files
ls ./lightcurve_ultraclean_5000/*.npz | head -10

# Check file sizes
du -sh ./lightcurve_ultraclean_5000/*.npz | sort -h | tail -10
```

### **Verify Output**
```bash
# Check if CSV was created
ls -lh ./output/kic_features.csv

# Count rows in CSV (including header)
wc -l ./output/kic_features.csv

# View first few lines
head ./output/kic_features.csv

# Check file size
du -h ./output/kic_features.csv
```

### **Clean Up Progress Files**
```bash
# Remove progress file (start fresh)
rm hp_extraction_progress.pkl

# Remove cache directory
rm -rf ./tsfresh_cache

# Remove log file
rm hp_feature_extraction.log
```

---

## Testing Commands

### **Test with 10 Files**
```bash
# Create test directory
mkdir -p ./test_kic
cp ./lightcurve_ultraclean_5000/KIC_*.npz ./test_kic/ | head -10

# Run on test directory
python hp_extractor.py ./test_kic ./test_output.csv --n-jobs 4
```

### **Dry Run (Check System)**
```bash
# Test imports and system detection
python -c "
from hp_extractor import HighPerformanceExoplanetExtractor
import psutil
print('CPU cores:', psutil.cpu_count())
print('RAM (GB):', psutil.virtual_memory().total / (1024**3))
"
```

---

## Troubleshooting Commands

### **Check for Errors in Log**
```bash
grep -i error hp_feature_extraction.log
grep -i failed hp_feature_extraction.log
grep -i warning hp_feature_extraction.log
```

### **Kill Stuck Process**
```bash
# Find process ID
ps aux | grep hp_extractor

# Kill process
kill -9 <PID>

# Or kill all Python processes (careful!)
pkill -f hp_extractor.py
```

### **Check Disk Space**
```bash
df -h
```

### **Verify Single NPZ File**
```bash
python -c "
import numpy as np
with np.load('./lightcurve_ultraclean_5000/KIC_1234244.npz') as data:
    print('Keys:', data.files)
    print('Shape:', data[data.files[0]].shape)
"
```

---

## Complete Command Reference Table

| Command | Purpose | Default | Example |
|---------|---------|---------|---------|
| `npz_folder` | Input folder (required) | - | `./lightcurve_ultraclean_5000` |
| `output_csv` | Output file (required) | - | `./output/features.csv` |
| `--n-jobs` | Parallel processes | Auto | `--n-jobs 8` |
| `--batch-size` | Batch size | Auto | `--batch-size 50` |
| `--memory-limit` | Max RAM (GB) | 80% | `--memory-limit 12.0` |
| `--progress-file` | Progress file path | `hp_extraction_progress.pkl` | `--progress-file my.pkl` |
| `--disable-caching` | Disable cache | False | `--disable-caching` |
| `--disable-memory-mapping` | Disable mmap | False | `--disable-memory-mapping` |
| `--resume` | Resume extraction | False | `--resume` |
| `--retry-failed` | Retry failed files | False | `--retry-failed` |

---

## Quick Reference - Common Scenarios

### **First Time Running**
```bash
python hp_extractor.py ./lightcurve_ultraclean_5000 ./output/kic_features.csv
```

### **After Interruption**
```bash
python hp_extractor.py ./lightcurve_ultraclean_5000 ./output/kic_features.csv --resume
```

### **Low Memory System**
```bash
python hp_extractor.py ./lightcurve_ultraclean_5000 ./output/kic_features.csv --n-jobs 4 --batch-size 20
```

### **Maximum Speed**
```bash
python hp_extractor.py ./lightcurve_ultraclean_5000 ./output/kic_features.csv --n-jobs 12 --batch-size 100
```

### **With Logging to File**
```bash
python hp_extractor.py ./lightcurve_ultraclean_5000 ./output/kic_features.csv 2>&1 | tee extraction_$(date +%Y%m%d_%H%M%S).log
```

---

## Help Command

```bash
# View all available options
python hp_extractor.py --help
```

**Output:**
```
usage: hp_extractor.py [-h] [--n-jobs N_JOBS] [--batch-size BATCH_SIZE]
                       [--memory-limit MEMORY_LIMIT] [--disable-caching]
                       [--disable-memory-mapping] [--resume] [--retry-failed]
                       [--progress-file PROGRESS_FILE]
                       npz_folder output_csv

High-Performance TSFresh Feature Extractor for Exoplanet Detection

positional arguments:
  npz_folder            Path to folder containing NPZ files
  output_csv            Output CSV file path

optional arguments:
  -h, --help            show this help message and exit
  --n-jobs N_JOBS       Number of parallel jobs (auto-detected) (default: None)
  --batch-size BATCH_SIZE
                        Batch size for processing (auto-calculated) (default: None)
  --memory-limit MEMORY_LIMIT
                        Memory limit in GB (auto-detected) (default: None)
  --disable-caching     Disable feature caching (default: False)
  --disable-memory-mapping
                        Disable memory mapping for NPZ files (default: False)
  --resume              Resume from previous extraction (default: False)
  --retry-failed        Retry previously failed files (default: False)
  --progress-file PROGRESS_FILE
                        Progress file path (default: hp_extraction_progress.pkl)
```

---

## Exit Codes

- **0** : Success
- **1** : Error or user interruption
- **Non-zero** : Fatal error (check logs)

---

## Pro Tips

1. **Always test with small subset first**
2. **Use `--resume` if interrupted**
3. **Monitor memory usage during first batch**
4. **Keep progress file until completion**
5. **Use screen/tmux for long runs**
6. **Save output to different location than input**
7. **Check logs for warnings/errors**