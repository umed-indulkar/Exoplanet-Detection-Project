# Colab: End-to-End Guide (Real Data → 900+ Features → Baseline ML → Siamese → Outputs)

Follow these cells exactly in a fresh Google Colab notebook. This pipeline assumes you have real light-curve files (NPZ/CSV/FITS). If you also have labels, include them as a CSV with columns: `source` (matching your file names/IDs) and `label` (0/1).

Note: Colab’s runtime resets when idle; re-run setup if needed.

---

## 0) Runtime and prerequisites

- Runtime → Change runtime type → Hardware accelerator:
  - None or GPU (optional). CPU is fine for this pipeline. GPU helps Siamese training.

---

## 1) Install dependencies

```python
!pip -q install numpy pandas scipy pyyaml matplotlib seaborn tsfresh statsmodels scikit-learn optuna streamlit
# Torch CPU wheel (works everywhere). If GPU runtime already includes a compatible torch, you can skip or adjust.
!pip -q install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 2) Get your code

Replace YOUR_REPO_URL with your repo URL.

```python
!rm -rf exocode
!git clone YOUR_REPO_URL exocode
%cd exocode
```

Verify imports quickly:
```python
!python simple_test.py
```

---

## 3) Add your real data

Place files in `exocode/data/`. Choose one approach:

- Option A: Upload from your computer
```python
from google.colab import files
from pathlib import Path
import shutil

Path('data').mkdir(exist_ok=True)
uploaded = files.upload()  # Choose many: .npz, .csv, .fits
for name in uploaded:
    shutil.move(name, f"data/{name}")
len(list(Path('data').glob('*')))
```

- Option B: Mount Drive and copy
```python
from google.colab import drive
from pathlib import Path
drive.mount('/content/drive')
Path('data').mkdir(exist_ok=True)
# Example: copy all NPZ from Drive folder
!cp /content/drive/MyDrive/your_lightcurves/*.npz data/
```

Your `data/` folder should now contain your real files.

---

## 4) Extract TSFresh comprehensive features (~900+)

Use the CLI with the comprehensive preset.

```python
!python -m exodet.cli extract \
  --input "data/*" \
  --output outputs/features_tsfresh_comprehensive.csv \
  --tier tsfresh \
  --tsfresh-params comprehensive \
  --workers 4
```

Check shape:
```python
import pandas as pd
df = pd.read_csv('outputs/features_tsfresh_comprehensive.csv')
df.shape, df.columns[:10].tolist()
```

Notes:
- For very large datasets, reduce `--workers` or process in chunks.

---

## 5) (Optional) Merge labels for supervised training

If you have labels, provide a CSV with columns: `source,label`.
- `source` must match the filename or identifier stored in the `features` CSV `source` column.

```python
from pathlib import Path
import pandas as pd

# Option A: Upload a labels CSV
from google.colab import files
Path('labels').mkdir(exist_ok=True)
uploaded = files.upload()  # choose labels.csv
# move uploaded file to labels/labels.csv if needed

features = pd.read_csv('outputs/features_tsfresh_comprehensive.csv')
labels = pd.read_csv('labels/labels.csv')  # must have columns: source,label
merged = features.merge(labels, on='source', how='inner')  # keep only labeled rows
Path('outputs').mkdir(exist_ok=True)
merged.to_csv('outputs/features_labeled.csv', index=False)
merged.shape
```

If you don’t have labels, you can still inspect features and skip supervised training.

---

## 6) Baseline ML training/evaluation (RandomForest)

Train:
```python
!python -m exodet.cli train \
  --features outputs/features_labeled.csv \
  --target label \
  --model rf \
  --output runs/rf.joblib
```

Evaluate (same file or a separate validation CSV):
```python
!python -m exodet.cli evaluate \
  --model runs/rf.joblib \
  --features outputs/features_labeled.csv \
  --target label
```

Predict on raw curves (auto basic features):
```python
!python -m exodet.cli predict \
  --model runs/rf.joblib \
  --input "data/*" \
  --output outputs/predictions_baseline.csv
```

---

## 7) Siamese model training/evaluation (deep)

Train:
```python
!python -m exodet.cli train-siamese \
  --features outputs/features_labeled.csv \
  --target label \
  --epochs 10 \
  --embedding 32 \
  --device auto \
  --output runs/siamese.pt
```

Evaluate:
```python
!python -m exodet.cli evaluate-siamese \
  --model runs/siamese.pt \
  --features outputs/features_labeled.csv \
  --target label \
  --device auto
```

- Metric reported: ROC-AUC over pairwise similarity.
- `--device auto` uses CUDA if available, otherwise CPU.

---

## 8) Download results

```python
from google.colab import files

# Features (~900+)
files.download('outputs/features_tsfresh_comprehensive.csv')
# Labeled features (if created)
# files.download('outputs/features_labeled.csv')

# Models
files.download('runs/rf.joblib')
files.download('runs/siamese.pt')

# Predictions (if generated)
# files.download('outputs/predictions_baseline.csv')
```

---

## 9) Quick visual inspection (optional)

```python
import pandas as pd
from exodet import load_lightcurve, preprocess_lightcurve
import matplotlib.pyplot as plt

one = 'data/your_file.npz'  # or .csv/.fits
lc = load_lightcurve(one)
lc_clean = preprocess_lightcurve(lc)

plt.figure(figsize=(10,4))
plt.plot(lc.time, lc.flux, '.', alpha=0.6, label='raw')
plt.plot(lc_clean.time, lc_clean.flux, '.', alpha=0.6, label='processed')
plt.legend(); plt.title('Light curve (raw vs processed)'); plt.show()
```

---

## 10) Tips for large datasets

- Split input patterns and run extraction in chunks (e.g., `data/part1/*`, `data/part2/*`).
- Reduce `--workers` if memory is constrained.
- Save outputs to Drive to persist across runtime restarts.

---

## Summary

- Install deps → clone repo → add data to `data/`
- Extract comprehensive TSFresh features (`--tsfresh-params comprehensive`) → `outputs/features_tsfresh_comprehensive.csv` (~900+)
- If supervised: merge labels → train baseline RF → train Siamese → evaluate
- Download outputs/models

If you want a prebuilt Colab notebook in the repo, create an issue or request and it can be added as `colab_end_to_end.ipynb` with these cells prefilled.
