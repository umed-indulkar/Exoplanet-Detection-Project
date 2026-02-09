# Exoplanet Detection System

A modular toolkit for detecting exoplanets from light curve data.

## 🏗️ Structure

```
exodet/
├── preprocessing/         # Data loading and cleaning
├── feature_extraction/    # Extract features from light curves
├── models/               # ML models (RF, Logistic, Siamese)
├── visualization/        # Visualization and plotting tools
├── pipeline/             # Integration and workflows
└── cli/                  # Command line interface
```

## 👥 Team Independence

Each team works in their own module:
- **Preprocessing Team**: `preprocessing/` only
- **Feature Team**: `feature_extraction/` only  
- **ML Team**: `models/` only
- **Visualization Team**: `visualization/` only
- **Pipeline Team**: Coordinates everything

## 🚀 Usage

```bash
# Organize dataset
python -m exodet.cli organize

# Extract features
python -m exodet.cli batch --input data/ExoplanetDataset/raw --output outputs/features.csv

# Train models
python -m exodet.cli train --features outputs/train.csv --model rf --output runs/rf.joblib
python -m exodet.cli train-siamese --features outputs/train.csv --output runs/siamese.pt

# Evaluate models
python -m exodet.cli evaluate --model runs/rf.joblib --features outputs/test.csv
```

## 📦 Dependencies

See `requirements.txt` for minimal dependencies (9 packages only).
