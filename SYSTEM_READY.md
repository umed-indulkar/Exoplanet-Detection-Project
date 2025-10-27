# System Ready Summary

Date: October 27, 2025

Status: Production-ready core for light-curve processing (data loading, preprocessing, feature extraction, visualization, configuration).

---

## What’s Included

- Package: `exodet/`
  - `core/data_loader.py` — NPZ/CSV/FITS loaders, `LightCurve` type
  - `core/preprocessing.py` — configurable pipeline (detrend, sigma clip, normalize, fold, bin)
  - `core/config.py` — YAML-backed configuration
  - `features/basic_extractor.py` — ~49 curated features
  - `visualization/plots.py` — light-curve and feature plots
- Reference scripts (optional): `test_unified_system.py`, `demo_complete_system.py`, `process_your_data.py`
- Documentation: `README.md`, `HOW_TO_RUN.md`, `RUN_THIS_FIRST.md`

---

## How to Run (Essentials)

Install:
```bash
pip install -r requirements.txt
```

Verify environment:
```bash
python simple_test.py
```

Minimal usage:
```python
from exodet import load_lightcurve, preprocess_lightcurve
from exodet.features import extract_basic_features

lc = load_lightcurve('path/to/lightcurve.npz')  # also supports .csv, .fits
lc_clean = preprocess_lightcurve(lc)
features = extract_basic_features(lc_clean)
features.to_csv('features.csv', index=False)
```

Visualization (optional):
```python
from exodet.visualization import plot_lightcurve
plot_lightcurve(lc_clean, title='Processed', save_path='processed.png')
```

For full workflows (batch, config usage), see `HOW_TO_RUN.md`.

---

## Capabilities (Current)

- Universal data loading: NPZ, CSV, FITS; batch loading
- Preprocessing: polynomial/Savitzky–Golay/median detrending, sigma clipping, normalization (z-score/min–max/robust/median), folding, binning
- Feature extraction: ~49 curated features (statistics, variability, frequency summaries, simple transit indicators)
- Visualization: raw/processed/folded plots, feature distributions
- Configuration: YAML-driven with sensible defaults

---

## Notes on Feature Tiers

Older branches referenced “fast/standard/comprehensive” modes and 900+ features. This repository intentionally includes the basic, curated feature set (~49 features) implemented in `exodet/features/basic_extractor.py`. TSFresh-based comprehensive features and ML model training are not part of this core and can be added separately if needed.

---

## References

- Quick start: `RUN_THIS_FIRST.md`
- Detailed runbook: `HOW_TO_RUN.md`
- API overview and examples: `README.md`
