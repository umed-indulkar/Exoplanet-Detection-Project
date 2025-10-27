# How to Run

This guide describes how to install dependencies, verify the environment, and run the core workflows (single-file and batch) using the Python API. It focuses on what is implemented in this repository today.

Last updated: October 27, 2025

---

## 1) Prerequisites

- Python 3.10+ (tested with Python 3.13)
- pip available on PATH

Optional but recommended: a virtual environment.

---

## 2) Installation

```bash
pip install -r requirements.txt
```

Verify the setup:
```bash
python simple_test.py
```
All checks should pass.

---

## 3) Single-File Workflow

Load a light curve (NPZ/CSV/FITS), preprocess, extract features, and save them.

```python
from exodet import load_lightcurve, preprocess_lightcurve
from exodet.features import extract_basic_features

# Load
lc = load_lightcurve('path/to/lightcurve.npz')  # also supports .csv, .fits

# Preprocess (defaults are sensible; pass overrides as needed)
lc_clean = preprocess_lightcurve(
    lc,
    detrend={'enabled': True, 'method': 'polynomial', 'order': 2},
    sigma_clip={'enabled': True, 'sigma': 3.0},
    normalize={'enabled': True, 'method': 'zscore'}
)

# Extract features (~49 columns)
features = extract_basic_features(lc_clean, verbose=False)
features.to_csv('features.csv', index=False)
```

---

## 4) Batch Workflow

Process all files in a directory and build a combined dataset.

```python
import pandas as pd
from exodet import load_batch_lightcurves, preprocess_lightcurve
from exodet.features import extract_basic_features

curves = load_batch_lightcurves('data/', pattern='*.npz')

rows = []
for lc in curves:
    clean = preprocess_lightcurve(lc)
    feats = extract_basic_features(clean, verbose=False)
    feats['source'] = lc.source_file
    rows.append(feats)

dataset = pd.concat(rows, ignore_index=True)
dataset.to_csv('dataset.csv', index=False)
```

Notes:
- The loader supports NPZ/CSV/FITS.
- Adjust `pattern` (e.g., `*.csv`, `*.fits`) as needed.

---

## 5) Configuration

Use `config.yaml` to centralize preprocessing options. You can also modify and save custom settings.

```python
from exodet import Config, preprocess_lightcurve, load_lightcurve

config = Config('config.yaml')
lc = load_lightcurve('path/to/lightcurve.npz')

lc_clean = preprocess_lightcurve(lc, config=config['preprocessing'])

# Example: modify and save custom config
config.set('system.num_workers', 8)
config.save_to_yaml('production_config.yaml')
```

---

## 6) Visualization

Generate plots for inspection and reporting.

```python
from exodet import load_lightcurve, preprocess_lightcurve
from exodet.visualization import plot_lightcurve, plot_folded_lightcurve, plot_comparison

lc = load_lightcurve('path/to/lightcurve.npz')
lc_clean = preprocess_lightcurve(lc)

# Raw/processed
plot_lightcurve(lc, title='Raw', save_path='raw.png')
plot_lightcurve(lc_clean, title='Processed', save_path='processed.png')
plot_comparison(lc, lc_clean, title='Comparison', save_path='comparison.png')

# Phase-folded (provide period)
plot_folded_lightcurve(lc_clean, period=5.0, save_path='folded.png')
```

---

## 7) Outputs

- `features.csv` or `dataset.csv`: tabular features (~49 columns)
- `raw.png`, `processed.png`, `comparison.png`, `folded.png`: diagnostic plots

---

## 8) Troubleshooting

- Import/module errors: ensure `pip install -r requirements.txt` completed successfully.
- File not found: verify paths and file extensions.
- Large datasets: process in batches or reduce memory footprint by saving intermediate results.

Minimal environment check:
```bash
python -c "import numpy, pandas, scipy, yaml; print('OK')"
```

---

## 9) Notes on Feature Tiers

Older branches mention “fast/standard/comprehensive” modes and 900+ features. This repository ships the basic, curated feature set (~49 features) implemented in `exodet/features/basic_extractor.py`. TSFresh-based comprehensive features and ML model training are not included here and can be added on top if needed.

