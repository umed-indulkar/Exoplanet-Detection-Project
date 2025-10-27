# Start Here

This is a short quick-start for the core light-curve workflow. For detailed usage, see `HOW_TO_RUN.md` and `README.md`.

---

## 1) Install

```bash
pip install -r requirements.txt
```

Verify:
```bash
python simple_test.py
```
All checks should pass.

---

## 2) Minimal Usage

Load a light curve, preprocess it, and extract features.

```python
from exodet import load_lightcurve, preprocess_lightcurve
from exodet.features import extract_basic_features

lc = load_lightcurve('path/to/lightcurve.npz')  # also supports .csv, .fits
lc_clean = preprocess_lightcurve(lc)
features = extract_basic_features(lc_clean)
features.to_csv('features.csv', index=False)
```

Optional plotting (saved to PNG):
```python
from exodet.visualization import plot_lightcurve
plot_lightcurve(lc_clean, title='Processed', save_path='processed.png')
```

---

## 3) Next Steps

- Batch processing example: see `HOW_TO_RUN.md` (Batch Workflow)
- Configuration via YAML: see `HOW_TO_RUN.md` (Configuration)
- Additional examples: see `README.md`

Notes on features: this repository provides a curated basic feature set (~49 features). Larger TSFresh-based sets and ML training are intentionally out of scope here and can be integrated later.
