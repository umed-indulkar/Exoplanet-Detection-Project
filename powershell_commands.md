# High-Performance TSFresh Extractor - PowerShell Commands Guide

## Complete Command Reference for Windows PowerShell

---

## Installation Commands (PowerShell)

```powershell
# 1. Install required packages
pip install numpy pandas tsfresh joblib tqdm psutil scikit-learn

# 2. Or install from requirements file
pip install -r requirements.txt

# 3. Verify installation
python -c "import tsfresh; print(f'TSFresh version: {tsfresh.__version__}')"
```

---

## Basic Usage Commands (PowerShell)

### 1. **Most Basic Command (Default Settings)**
```powershell
python hp_extractor.py .\lightcurve_ultraclean_5000 .\output\kic_features.csv
```

**Note:** Use backslashes `\` instead of forward slashes `/` in Windows paths

**Examples:**
```powershell
# Current directory
python hp_extractor.py .\lightcurve_ultraclean_5000 .\output\features.csv

# Full path
python hp_extractor.py C:\Users\YourName\data\lightcurve_ultraclean_5000 C:\Users\YourName\output\features.csv

# Relative path
python hp_extractor.py ..\data\lightcurve_ultraclean_5000 ..\output\features.csv
```

---

### 2. **With Custom Number of Cores**
```powershell
python hp_extractor.py .\lightcurve_ultraclean_5000 .\output\features.csv --n-jobs 8
```

**Examples:**
```powershell
# Use 4 cores
python hp_extractor.py .\lightcurve_ultraclean_5000 .\output\features.csv --n-jobs 4

# Use 8 cores
python hp_extractor.py .\lightcurve_ultraclean_5000 .\output\features.csv --n-jobs 8
```

---

### 3. **With Custom Batch Size**
```powershell
python hp_extractor.py .\lightcurve_ultraclean_5000 .\output\features.csv --batch-size 50
```

**Examples:**
```powershell
# Small batches (8GB RAM)
python hp_extractor.py .\lightcurve_ultraclean_5000 .\output\features.csv --batch-size 20

# Large batches (32GB RAM)
python hp_extractor.py .\lightcurve_ultraclean_5000 .\output\features.csv --batch-size 100
```

---

### 4. **With Memory Limit**
```powershell
python hp_extractor.py .\lightcurve_ultraclean_5000 .\output\features.csv --memory-limit 12.0
```

**Examples:**
```powershell
# Limit to 8GB
python hp_extractor.py .\lightcurve_ultraclean_5000 .\output\features.csv --memory-limit 8.0

# Limit to 16GB
python hp_extractor.py .\lightcurve_ultraclean_5000 .\output\features.csv --memory-limit 16.0
```

---

### 5. **Resume Previous Extraction**
```powershell
python hp_extractor.py .\lightcurve_ultraclean_5000 .\output\features.csv --resume
```

---

### 6. **Retry Failed Files**
```powershell
python hp_extractor.py .\lightcurve_ultraclean_5000 .\output\features.csv --retry-failed
```

---

### 7. **Disable Caching**
```powershell
python hp_extractor.py .\lightcurve_ultraclean_5000 .\output\features.csv --disable-caching
```

---

### 8. **Disable Memory Mapping**
```powershell
python hp_extractor.py .\lightcurve_ultraclean_5000 .\output\features.csv --disable-memory-mapping
```

---

### 9. **Custom Progress File**
```powershell
python hp_extractor.py .\lightcurve_ultraclean_5000 .\output\features.csv --progress-file kic_progress.pkl
```

---

## Combined Commands (PowerShell)

### **Optimized for 8-Core i7 with 16GB RAM**
```powershell
python hp_extractor.py .\lightcurve_ultraclean_5000 .\output\kic_features.csv `
  --n-jobs 7 `
  --batch-size 50 `
  --memory-limit 12.0
```

**Note:** Use backtick `` ` `` for line continuation in PowerShell

### **Conservative Settings (Low RAM)**
```powershell
python hp_extractor.py .\lightcurve_ultraclean_5000 .\output\kic_features.csv `
  --n-jobs 4 `
  --batch-size 20 `
  --memory-limit 6.0
```

### **Maximum Performance (High-End System)**
```powershell
python hp_extractor.py .\lightcurve_ultraclean_5000 .\output\kic_features.csv `
  --n-jobs 11 `
  --batch-size 100 `
  --memory-limit 24.0
```

### **Network Storage / Slow Drive**
```powershell
python hp_extractor.py .\lightcurve_ultraclean_5000 .\output\kic_features.csv `
  --n-jobs 4 `
  --batch-size 10 `
  --disable-memory-mapping `
  --disable-caching
```

---

## Background Processing (PowerShell)

### **Run in Background with Logging**
```powershell
# Method 1: Redirect output to file
python hp_extractor.py .\lightcurve_ultraclean_5000 .\output\features.csv > extraction.log 2>&1
```

```powershell
# Method 2: Start as background job
Start-Job -ScriptBlock {
    python hp_extractor.py .\lightcurve_ultraclean_5000 .\output\features.csv
} -Name "FeatureExtraction"

# Check job status
Get-Job -Name "FeatureExtraction"

# Get job output
Receive-Job -Name "FeatureExtraction"
```

```powershell
# Method 3: Start in new window
Start-Process python -ArgumentList "hp_extractor.py", ".\lightcurve_ultraclean_5000", ".\output\features.csv" -NoNewWindow
```

### **Run with Timestamp Logging**
```powershell
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
python hp_extractor.py .\lightcurve_ultraclean_5000 .\output\features.csv > "extraction_$timestamp.log" 2>&1
```

---

## Monitoring Commands (PowerShell)

### **Monitor Progress in Real-Time**
```powershell
# Watch log file (updates every 2 seconds)
Get-Content .\hp_feature_extraction.log -Wait -Tail 20
```

### **Monitor System Resources**
```powershell
# CPU and Memory usage
Get-Process python | Select-Object CPU, PM, WS, Name

# Continuous monitoring (every 2 seconds)
while ($true) {
    Clear-Host
    Get-Process python | Select-Object CPU, PM, WS, Name
    Start-Sleep 2
}

# Memory usage in GB
Get-Process python | Select-Object Name, @{Name="Memory(GB)";Expression={$_.WS / 1GB}}
```

### **Check Disk Space**
```powershell
Get-PSDrive C | Select-Object Used, Free
```

---

## File Management Commands (PowerShell)

### **Check NPZ Files**
```powershell
# Count NPZ files
(Get-ChildItem .\lightcurve_ultraclean_5000\*.npz).Count

# List first 10 files
Get-ChildItem .\lightcurve_ultraclean_5000\*.npz | Select-Object -First 10

# Check total size of NPZ files
(Get-ChildItem .\lightcurve_ultraclean_5000\*.npz | Measure-Object -Property Length -Sum).Sum / 1GB

# List KIC files specifically
Get-ChildItem .\lightcurve_ultraclean_5000\KIC_*.npz | Select-Object Name, Length
```

### **Verify Output**
```powershell
# Check if CSV was created
Get-ChildItem .\output\kic_features.csv

# Check file size
(Get-Item .\output\kic_features.csv).Length / 1MB

# Count lines in CSV
(Get-Content .\output\kic_features.csv).Count

# View first 5 lines
Get-Content .\output\kic_features.csv -Head 5

# View last 5 lines
Get-Content .\output\kic_features.csv -Tail 5
```

### **Clean Up Progress Files**
```powershell
# Remove progress file
Remove-Item hp_extraction_progress.pkl -ErrorAction SilentlyContinue

# Remove cache directory
Remove-Item .\tsfresh_cache -Recurse -Force -ErrorAction SilentlyContinue

# Remove log file
Remove-Item hp_feature_extraction.log -ErrorAction SilentlyContinue

# Remove all temporary files
Remove-Item hp_extraction_progress.pkl, hp_feature_extraction.log -ErrorAction SilentlyContinue
Remove-Item .\tsfresh_cache -Recurse -Force -ErrorAction SilentlyContinue
```

---

## Testing Commands (PowerShell)

### **Test with 10 Files**
```powershell
# Create test directory
New-Item -Path .\test_kic -ItemType Directory -Force

# Copy first 10 files
Get-ChildItem .\lightcurve_ultraclean_5000\*.npz | Select-Object -First 10 | Copy-Item -Destination .\test_kic

# Run on test directory
python hp_extractor.py .\test_kic .\test_output.csv --n-jobs 4
```

### **Dry Run (Check System)**
```powershell
python -c "from hp_extractor import HighPerformanceExoplanetExtractor; import psutil; print('CPU cores:', psutil.cpu_count()); print('RAM (GB):', psutil.virtual_memory().total / (1024**3))"
```

### **Check Python and Package Versions**
```powershell
# Python version
python --version

# Package versions
pip list | Select-String "numpy|pandas|tsfresh|joblib|tqdm|psutil"

# Detailed package info
pip show tsfresh
```

---

## Troubleshooting Commands (PowerShell)

### **Check for Errors in Log**
```powershell
# Find errors
Select-String -Path .\hp_feature_extraction.log -Pattern "error" -CaseSensitive:$false

# Find failures
Select-String -Path .\hp_feature_extraction.log -Pattern "failed" -CaseSensitive:$false

# Find warnings
Select-String -Path .\hp_feature_extraction.log -Pattern "warning" -CaseSensitive:$false

# Count errors
(Select-String -Path .\hp_feature_extraction.log -Pattern "error" -CaseSensitive:$false).Count
```

### **Kill Stuck Process**
```powershell
# Find Python processes
Get-Process python

# Kill specific process by ID
Stop-Process -Id <PID> -Force

# Kill all Python processes (careful!)
Get-Process python | Stop-Process -Force

# Kill processes running hp_extractor
Get-Process | Where-Object {$_.CommandLine -like "*hp_extractor*"} | Stop-Process -Force
```

### **Verify Single NPZ File**
```powershell
python -c "import numpy as np; data = np.load('.\lightcurve_ultraclean_5000\KIC_1234244.npz'); print('Keys:', data.files); print('Shape:', data[data.files[0]].shape)"
```

### **Check Available Memory**
```powershell
# Total RAM
[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 2)

# Available RAM
[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB / 1024, 2)

# Memory usage percentage
$os = Get-CimInstance Win32_OperatingSystem
$total = $os.TotalVisibleMemorySize / 1MB
$free = $os.FreePhysicalMemory / 1MB
$used = $total - $free
Write-Host "Memory Usage: $([math]::Round(($used/$total)*100, 2))%"
```

---

## PowerShell-Specific Tips

### **Setting Execution Policy (if needed)**
```powershell
# Check current policy
Get-ExecutionPolicy

# Set to allow scripts (run as Administrator)
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

# Or bypass for single session
powershell -ExecutionPolicy Bypass
```

### **Creating Output Directory**
```powershell
# Create directory if it doesn't exist
New-Item -Path .\output -ItemType Directory -Force
```

### **Check if Script is Running**
```powershell
# Check if hp_extractor is running
Get-Process python | Where-Object {$_.CommandLine -like "*hp_extractor*"}

# Or simpler
Get-Process python -ErrorAction SilentlyContinue
```

### **Measure Execution Time**
```powershell
# Measure how long extraction takes
Measure-Command {
    python hp_extractor.py .\lightcurve_ultraclean_5000 .\output\features.csv
}
```

---

## PowerShell Aliases for Convenience

```powershell
# Create aliases in your PowerShell profile
function Run-Extractor {
    param(
        [string]$InputFolder = ".\lightcurve_ultraclean_5000",
        [string]$OutputFile = ".\output\kic_features.csv",
        [int]$Jobs = 7,
        [int]$BatchSize = 50
    )
    python hp_extractor.py $InputFolder $OutputFile --n-jobs $Jobs --batch-size $BatchSize
}

# Usage:
# Run-Extractor
# Run-Extractor -Jobs 8 -BatchSize 100
```

---

## Complete PowerShell Script Example

```powershell
# complete_extraction.ps1
# Complete PowerShell script for feature extraction

param(
    [string]$NPZFolder = ".\lightcurve_ultraclean_5000",
    [string]$OutputCSV = ".\output\kic_features.csv",
    [int]$NJobs = 7,
    [int]$BatchSize = 50
)

# Create output directory
New-Item -Path (Split-Path $OutputCSV) -ItemType Directory -Force | Out-Null

# Check if NPZ folder exists
if (!(Test-Path $NPZFolder)) {
    Write-Error "NPZ folder not found: $NPZFolder"
    exit 1
}

# Count NPZ files
$fileCount = (Get-ChildItem "$NPZFolder\*.npz").Count
Write-Host "Found $fileCount NPZ files"

if ($fileCount -eq 0) {
    Write-Error "No NPZ files found in $NPZFolder"
    exit 1
}

# Create log filename with timestamp
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = "extraction_$timestamp.log"

# Run extraction with logging
Write-Host "Starting extraction..."
Write-Host "Input: $NPZFolder"
Write-Host "Output: $OutputCSV"
Write-Host "Cores: $NJobs, Batch: $BatchSize"
Write-Host "Log: $logFile"
Write-Host ""

python hp_extractor.py $NPZFolder $OutputCSV `
    --n-jobs $NJobs `
    --batch-size $BatchSize `
    *> $logFile

# Check result
if ($LASTEXITCODE -eq 0) {
    Write-Host "Extraction completed successfully!" -ForegroundColor Green
    
    # Show output file info
    $outputInfo = Get-Item $OutputCSV
    Write-Host "Output file: $($outputInfo.FullName)"
    Write-Host "File size: $([math]::Round($outputInfo.Length / 1MB, 2)) MB"
    Write-Host "Lines: $((Get-Content $OutputCSV).Count)"
} else {
    Write-Host "Extraction failed! Check log: $logFile" -ForegroundColor Red
    exit 1
}
```

**Save as `complete_extraction.ps1` and run:**
```powershell
.\complete_extraction.ps1
# Or with custom parameters:
.\complete_extraction.ps1 -NPZFolder "C:\data\lightcurves" -NJobs 8
```

---

## Quick Reference - PowerShell vs Bash

| Task | PowerShell | Bash (Linux) |
|------|------------|--------------|
| Path separator | `\` (backslash) | `/` (forward slash) |
| Line continuation | `` ` `` (backtick) | `\` (backslash) |
| Redirect output | `>` or `Out-File` | `>` or `>>` |
| Current directory | `.\` | `./` |
| List files | `Get-ChildItem` or `ls` | `ls` |
| Remove file | `Remove-Item` or `rm` | `rm` |
| Count files | `(Get-ChildItem).Count` | `ls | wc -l` |
| Watch file | `Get-Content -Wait` | `tail -f` |

---

## Common PowerShell Scenarios

### **First Time Running**
```powershell
python hp_extractor.py .\lightcurve_ultraclean_5000 .\output\kic_features.csv
```

### **After Interruption (Ctrl+C)**
```powershell
python hp_extractor.py .\lightcurve_ultraclean_5000 .\output\kic_features.csv --resume
```

### **With Full Logging**
```powershell
python hp_extractor.py .\lightcurve_ultraclean_5000 .\output\kic_features.csv > extraction.log 2>&1
```

### **Maximum Performance**
```powershell
python hp_extractor.py .\lightcurve_ultraclean_5000 .\output\kic_features.csv `
  --n-jobs 12 `
  --batch-size 100
```

---

## Help Command (PowerShell)

```powershell
python hp_extractor.py --help
```

---

## Environment Variables (PowerShell)

```powershell
# Set Python to use all CPU cores (if needed)
$env:OMP_NUM_THREADS = "8"
$env:MKL_NUM_THREADS = "8"

# Disable warnings
$env:PYTHONWARNINGS = "ignore"
```

---

## Notes for Windows Users

1. **Use PowerShell (not CMD)** - Better features and compatibility
2. **Paths use backslashes** - Windows style: `.\folder\file`
3. **Line continuation uses backtick** - `` ` `` instead of `\`
4. **May need Admin rights** - For high-priority processing
5. **Antivirus can slow down** - Consider temporary exclusion for data folder
6. **Use SSD if possible** - Much faster I/O performance