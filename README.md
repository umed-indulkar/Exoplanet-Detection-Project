# Unified Exoplanet Detection System

Production-ready toolkit for working with astronomical light curves: loading, preprocessing, feature extraction, and visualization.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents

- Overview
- Features
- Installation
- Quick Start
- Usage Examples
- Project Structure
- Notes on Feature Tiers
- Contributing
- Citation
- License

---

## Overview

This repository consolidates clean, maintainable utilities to work with light curve data. The focus is on:
- Universal data loading (NPZ/CSV/FITS)
- A configurable preprocessing pipeline (detrending, sigma clipping, normalization, folding, binning)
- A curated basic feature set (~49 features) suitable for downstream analysis
- Practical plotting utilities for inspection and reporting

Scope of this repository is the “core” workflow. Model training, dashboards, and TSFresh-based large feature sets are out-of-scope here and can be layered on later.

---

## Features

- Universal data loader (NPZ, CSV, FITS)
- Configurable preprocessing: detrending (polynomial/Savitzky–Golay/median), sigma clipping, normalization (z-score/min–max/robust/median), period folding, time binning
- Basic feature extraction (~49 features): statistics, variability, frequency-domain summaries, simple transit indicators
- Batch processing helpers
- Plotting: raw, processed, folded curves; feature distribution summaries

---

## Installation

### Prerequisites
- Python 3.10 or higher (tested with Python 3.13)

### Step 1: Clone repository
```bash
git clone <this-repo-url>
cd exocode
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify
```bash
python simple_test.py
```
Expected: all checks pass.

---

## Quick Start

### Example 1: Load and visualize a light curve
```python
from exodet import load_lightcurve, preprocess_lightcurve
from exodet.visualization import plot_lightcurve

lc = load_lightcurve('path/to/lightcurve.npz')  # also supports .csv, .fits
lc_clean = preprocess_lightcurve(lc)
plot_lightcurve(lc_clean, title="Processed Light Curve", save_path="plot.png")
```

### Example 2: Extract features
```python
from exodet.features import extract_basic_features

features = extract_basic_features(lc_clean)
print(f"Features: {features.shape}")
features.to_csv('features.csv', index=False)
```

### Example 3: Batch processing
```python
from exodet import load_batch_lightcurves, preprocess_lightcurve
from exodet.features import extract_basic_features
import pandas as pd

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

## Usage Examples
Additional, end-to-end examples are in `HOW_TO_RUN.md`.

---

## Project Structure

```
exocode/
├── exodet/
│   ├── core/
│   │   ├── data_loader.py       # NPZ/CSV/FITS loaders, LightCurve type
│   │   ├── preprocessing.py     # Preprocessing pipeline
│   │   └── config.py            # YAML-backed configuration
│   ├── features/
│   │   └── basic_extractor.py   # ~49 basic features
│   └── visualization/
│       └── plots.py             # Light curve and feature plots
├── config.yaml                  # Default settings
├── requirements.txt             # Dependencies
├── test_unified_system.py       # Integration test (reference)
└── simple_test.py               # Environment sanity check
```

---

## Notes on Feature Tiers

You may see references to “fast/standard/comprehensive” tiers or “900+ features” in older branches. In this repository:
- Implemented: a curated basic feature set (~49 features) in `exodet/features/basic_extractor.py`.
- Not included here: TSFresh-based comprehensive features and ML model training. These can be integrated later if needed.

### Python API

#### Complete Pipeline
```python
from exodet import Config, load_lightcurve, preprocess_lightcurve
from exodet.features import get_feature_extractor
from exodet.models import SiameseNetwork
from exodet.training import Trainer

# Load configuration
config = Config('config.yaml')

# Load and preprocess data
lc = load_lightcurve('lightcurve.npz')
lc_clean = preprocess_lightcurve(lc, config=config['preprocessing'])

# Extract features
extractor = get_feature_extractor(tier=config.get('features.tier'))
features = extractor.extract(lc_clean)

# Initialize and train model
model = SiameseNetwork(
    input_dim=len(features.columns),
    **config['model']['architecture']
)

trainer = Trainer(config=config)
trainer.train(model, train_features, val_features)

# Make predictions
predictions = model.predict(test_features)
```

---

## Contributing
Contributions are welcome for documentation improvements, new preprocessing options, and additional vetted features.

---

## Citation
If you use this software in your research, please cite:

---

```

```bibtex
@software{exoplanet_detection_2025,
  title = {Unified Exoplanet Detection System},
  author = {Exoplanet Detection Team},
  year = {2025},
  version = {2.0.0},
  url = {https://github.com/yourusername/exoplanet-detection}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).

---

## 🙏 Acknowledgments

- NASA Kepler Mission for providing exoplanet data
- TSFresh team for time-series feature extraction
- PyTorch and scikit-learn communities
- All contributors to the original branches

---

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/exoplanet-detection/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/exoplanet-detection/discussions)
- **Email**: contact@exoplanet-detection.org

---

## 🗺️ Roadmap

### Version 2.1 (Planned)
- [ ] Additional ML models (Random Forest, XGBoost)
- [ ] GPU acceleration for feature extraction
- [ ] Real-time data streaming support
- [ ] Advanced data augmentation

### Version 2.2 (Planned)
- [ ] Transfer learning capabilities
- [ ] Multi-planet detection
- [ ] Uncertainty quantification
- [ ] Cloud deployment support

### Version 3.0 (Future)
- [ ] Deep learning architectures (CNNs, Transformers)
- [ ] Automated pipeline optimization
- [ ] Integration with astronomical databases
- [ ] Production deployment framework

---

**Happy Exoplanet Hunting! 🌟🪐**

Made with ❤️ by the Exoplanet Detection Team
