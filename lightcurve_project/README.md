# Light Curve Analysis Project

A comprehensive Python toolkit for processing, visualizing, and extracting features from astronomical light curves stored in `.npz` format.

## 🚀 Features

- **Data Loading**: Flexible `.npz` file loading with automatic key detection
- **Preprocessing**: Ultra-clean processing pipeline with detrending, outlier removal, and normalization
- **Visualization**: Comprehensive plotting capabilities for light curves, folded curves, and features
- **Feature Extraction**: 100+ astronomical and time-series features
- **Interactive Pruning**: Manual feature selection with intuitive CLI interface
- **Modular Design**: Clean, well-documented codebase for easy extension

## 📦 Installation

1. Clone or download this repository
2. Install dependencies:

```bash
cd lightcurve_project
pip install -r requirements.txt
```

## 🏃 Quick Start

### Basic Usage

```bash
# Run example with synthetic data
python main.py

# Load and visualize a light curve
python main.py --load your_data.npz --visualize

# Extract features and save to CSV
python main.py --load your_data.npz --extract --save features.csv

# Full pipeline with preprocessing and folding
python main.py --load your_data.npz --preprocess --period 2.5 --visualize --extract --save results.csv
```

### Interactive Feature Pruning

```bash
# Extract features and prune interactively
python main.py --load your_data.npz --extract --prune --save pruned_features.csv
```

## 📁 Project Structure

```
lightcurve_project/
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── main.py                  # CLI entry point
├── src/                     # Source code
│   ├── __init__.py
│   ├── data_loader.py       # Data loading & preprocessing
│   ├── visualization.py     # Plotting functions
│   ├── feature_extraction.py # Feature extraction (100+ features)
│   └── feature_pruning.py   # Manual pruning utilities
├── notebooks/               # Jupyter notebooks
│   └── exploration.ipynb    # Interactive analysis
└── data/                    # Data directory
    └── sample_curves/       # Sample .npz files
```

## 🔧 Core Functionality

### 1. Data Loading & Preprocessing

The `data_loader.py` module provides:

- **Flexible loading**: Automatically detects time, flux, and flux_err arrays in .npz files
- **Comprehensive preprocessing**:
  1. Remove NaNs and normalize
  2. Flatten stellar variability (polynomial detrending)
  3. Mask good quality points
  4. Sigma-clip outliers (iterative)
  5. Fold if period & epoch provided
  6. Bin data (5-minute bins)
  7. Normalize flux (zero mean, unit variance)

```python
from src.data_loader import load_npz_curve, preprocess_lightcurve

# Load raw light curve
lc_data = load_npz_curve('your_data.npz')

# Apply preprocessing
clean_lc = preprocess_lightcurve(lc_data, period=2.5, epoch_time=0)
```

### 2. Visualization

The `visualization.py` module provides:

- `plot_lightcurve()`: Basic light curve plotting with error bars
- `plot_folded_curve()`: Phase-folded light curves with binning
- `plot_feature_distribution()`: Feature histograms and boxplots
- `plot_comprehensive_analysis()`: Multi-panel analysis plots

```python
from src.visualization import plot_lightcurve, plot_folded_curve

# Plot light curve
plot_lightcurve(time, flux, flux_err, save_path='lightcurve.png')

# Plot folded curve
plot_folded_curve(time, flux, period=2.5, save_path='folded.png')
```

### 3. Feature Extraction

The `feature_extraction.py` module extracts 100+ features including:

**Statistical Features**:
- Mean, median, standard deviation, variance
- Skewness, kurtosis, percentiles
- Robust statistics (trimmed means, MAD)

**Time-Domain Features**:
- Autocorrelation functions
- Linear trends and slopes
- Difference statistics

**Frequency-Domain Features**:
- FFT coefficients and spectral properties
- Lomb-Scargle periodogram analysis
- Spectral entropy and centroids

**Transit-Specific Features**:
- Dip detection and characterization
- Transit depth, duration, and slopes
- Ingress/egress asymmetry

**Variability Features**:
- Amplitude measures
- Stetson variability indices
- Von Neumann ratio

```python
from src.feature_extraction import extract_features

# Extract all features
features_df = extract_features(time, flux, flux_err)
print(f"Extracted {len(features_df.columns)} features")
```

### 4. Feature Pruning

The `feature_pruning.py` module provides:

- Interactive CLI for feature selection
- Category-based feature grouping
- Pattern-based selection/removal
- Feature importance analysis
- Save/load feature selections

```python
from src.feature_pruning import interactive_feature_selection, manual_prune

# Interactive selection
selected_features = interactive_feature_selection(features_df)

# Create pruned dataset
pruned_df = manual_prune(features_df, selected_features)
```

## 📊 Example Workflow

### 1. Load and Preprocess

```python
# Load your .npz file
lc_data = load_npz_curve('transit_data.npz')

# Clean the data
clean_lc = preprocess_lightcurve(lc_data, period=3.2, epoch_time=0)
```

### 2. Visualize

```python
# Basic light curve
plot_lightcurve(clean_lc['time'], clean_lc['flux'], clean_lc['flux_err'])

# Folded curve
plot_folded_curve(clean_lc['time'], clean_lc['flux'], period=3.2)
```

### 3. Extract Features

```python
# Get comprehensive feature set
features = extract_features(clean_lc['time'], clean_lc['flux'], clean_lc['flux_err'])

# Visualize feature distributions
plot_feature_distribution(features)
```

### 4. Prune and Save

```python
# Interactive pruning
selected = interactive_feature_selection(features)

# Save results
pruned_features = manual_prune(features, selected)
pruned_features.to_csv('final_features.csv', index=False)
```

## 🎯 Command Line Interface

The main CLI supports various workflows:

### Basic Commands

- `--load FILE`: Load light curve from .npz file
- `--visualize`: Create visualization plots
- `--extract`: Extract features
- `--prune`: Interactive feature pruning
- `--save FILE`: Save results to CSV

### Processing Options

- `--preprocess`: Apply full preprocessing pipeline
- `--period P`: Period for folding
- `--epoch T0`: Epoch time for folding

### Advanced Options

- `--features-file FILE`: Load/save feature selection
- `--report`: Generate feature analysis report
- `--output-dir DIR`: Specify output directory

### Examples

```bash
# Full analysis pipeline
python main.py --load data.npz --preprocess --period 2.1 --visualize --extract --prune --save results.csv --report

# Quick visualization
python main.py --load data.npz --visualize --output-dir plots/

# Feature extraction only
python main.py --load data.npz --extract --save features.csv
```

## 📝 Input Data Format

Your `.npz` files should contain arrays with keys like:
- Time: `time`, `t`, `TIME`, etc.
- Flux: `flux`, `FLUX`, `magnitude`, etc.
- Errors: `flux_err`, `error`, `ERROR`, `flux_error`, etc.

Example:
```python
import numpy as np

# Save your data
np.savez('my_lightcurve.npz', 
         time=time_array,
         flux=flux_array, 
         flux_err=error_array)
```

## 🔬 Extending the Code

### Adding New Features

1. Add your feature extraction function to `feature_extraction.py`
2. Update the `extract_features()` function to call your new function
3. Follow the naming convention for feature categories

### Custom Preprocessing

1. Modify `preprocess_lightcurve()` in `data_loader.py`
2. Add new preprocessing steps as needed
3. Make steps optional with boolean parameters

### New Visualizations

1. Add plotting functions to `visualization.py`
2. Follow the existing pattern for save paths and figure management
3. Update the CLI to support new plot types

## 🤝 Contributing

This is a modular codebase designed for easy extension. Key principles:

- **Clean interfaces**: Each module has clear input/output specifications
- **Comprehensive documentation**: All functions are documented with examples
- **Flexible design**: Features can be easily added or removed
- **Error handling**: Robust error handling throughout

## 📋 Requirements

See `requirements.txt` for the complete list. Key dependencies:

- `numpy`, `scipy`, `pandas`: Core scientific computing
- `matplotlib`, `seaborn`: Visualization
- `astropy`, `lightkurve`: Astronomical analysis
- `tsfresh`: Time series feature extraction
- `scikit-learn`: Machine learning utilities

## 🎓 Getting Started

1. **Try the example**: Run `python main.py` to see the tool in action with synthetic data
2. **Use your data**: Replace with your own `.npz` files
3. **Explore features**: Use the interactive pruning to understand what features are available
4. **Extend**: Add your own feature extraction or visualization functions

Happy analyzing! 🌟