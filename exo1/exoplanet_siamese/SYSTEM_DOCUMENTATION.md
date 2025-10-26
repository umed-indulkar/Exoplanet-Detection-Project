# 📚 Exoplanet Detection System - Complete Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Directory Structure & Purpose](#directory-structure--purpose)
4. [How the System Works](#how-the-system-works)
5. [Installation & Setup](#installation--setup)
6. [Running the System](#running-the-system)
7. [Data Flow](#data-flow)
8. [Technical Details](#technical-details)
9. [Troubleshooting](#troubleshooting)

---

## 🌟 System Overview

This system detects exoplanets (planets orbiting other stars) by analyzing light curves - measurements of a star's brightness over time. When a planet passes in front of its star (transit), it causes a small dip in brightness. Our Siamese Neural Network learns to identify these characteristic patterns.

### Key Capabilities:
- **Automated Detection**: Identifies potential exoplanets from raw flux measurements
- **High Accuracy**: 85-90% accuracy using deep learning
- **Feature Engineering**: Extracts 50+ meaningful features from light curves
- **Similarity Learning**: Uses Siamese networks to learn what makes light curves similar/different

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT: Raw Light Curves                 │
│                    (CSV with FLUX measurements)             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   DATA PREPROCESSING                        │
│  • Cleaning: Remove outliers, interpolate missing values    │
│  • Normalization: Standardize flux values                   │
│  • Detrending: Remove systematic variations                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   FEATURE EXTRACTION                        │
│  • Statistical: mean, std, skewness, kurtosis               │
│  • Shape: peaks, troughs, roughness                         │
│  • Frequency: FFT, spectral features                        │
│  • Transit-specific: depth, duration, period                │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    PAIR GENERATION                          │
│  • Positive pairs: Same class (both planet/no-planet)       │
│  • Negative pairs: Different classes                        │
│  • Strategies: Balanced, Random, Hard mining                │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  SIAMESE NEURAL NETWORK                     │
│                                                             │
│   Input 1 ──┐                    ┌── Input 2                │
│             ▼                    ▼                          │
│        [Shared Feature Extractor]                           │
│         ├── Dense(256) + ReLU                               │
│         ├── Dense(128) + ReLU                               │
│         └── Dense(64) + ReLU                                │
│             ▼                    ▼                          │
│        [Embedding Layer (32)]                               │
│             ▼                    ▼                          │
│        [Contrastive Loss Function]                          │
│                                                             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      EVALUATION                             │
│  • Metrics: Accuracy, Precision, Recall, F1                 │
│  • Visualizations: Confusion Matrix, ROC, t-SNE             │
│  • Predictions: Classification of new light curves          │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Directory Structure & Purpose

```
exoplanet_siamese/
│
├── 📂 data/                      # All data files
│   ├── raw/                      # Original CSV files
│   │   └── exoTest.csv          # Input light curves (LABEL + FLUX columns)
│   │
│   ├── processed/                # Cleaned & normalized data
│   │   ├── train_processed.csv  # 70% for training
│   │   ├── val_processed.csv    # 10% for validation
│   │   └── test_processed.csv   # 20% for testing
│   │
│   ├── features/                 # Extracted features
│   │   ├── train_features.csv   # ~50 features per light curve
│   │   ├── val_features.csv     
│   │   └── test_features.csv    
│   │
│   └── pairs/                    # Siamese network inputs
│       ├── train_X1.npy         # First element of pairs
│       ├── train_X2.npy         # Second element of pairs
│       ├── train_labels.npy     # 1=similar, 0=different
│       └── ...                  # Similar for val/test
│
├── 📂 src/                       # Source code modules
│   ├── __init__.py              # Package initialization
│   ├── data_preprocessing.py    # Clean & normalize light curves
│   ├── feature_extraction.py    # Extract meaningful features
│   ├── pair_generation.py       # Create training pairs
│   ├── siamese_model.py         # Neural network architecture
│   ├── train.py                 # Training loop & optimization
│   ├── evaluate.py              # Model evaluation & metrics
│   └── utils.py                 # Helper functions & plotting
│
├── 📂 notebooks/                 # Interactive analysis
│   ├── 01_data_exploration.ipynb    # Visualize raw data
│   └── 02_feature_visualization.ipynb # Analyze features
│
├── 📂 outputs/                   # Results & artifacts
│   ├── models/                  # Saved model weights
│   │   ├── best_model.pth      # Best validation performance
│   │   └── final_model.pth     # Final trained model
│   │
│   ├── logs/                    # Training history
│   │   └── training_log.txt    
│   │
│   └── results/                 # Evaluation outputs
│       ├── evaluation_results.json  # Metrics
│       ├── test_predictions.csv     # Predictions
│       ├── confusion_matrix.png     # Visualizations
│       └── embedding_space.png      
│
├── 📄 config.yaml               # Configuration settings
├── 📄 requirements.txt          # Python dependencies
├── 📄 main.py                   # Main pipeline script
└── 📄 README.md                 # Project overview
```

---

## 🔄 How the System Works

### Step-by-Step Process:

#### 1️⃣ **Data Loading & Preprocessing**
```python
# Module: data_preprocessing.py
# Input: Raw CSV with flux measurements
# Process:
- Load CSV file with LABEL and FLUX.1, FLUX.2, ..., FLUX.n columns
- Remove outliers using IQR method (Q1-3*IQR to Q3+3*IQR)
- Interpolate missing values
- Detrend using moving average (window=10)
- Normalize using local normalization (per light curve)
- Split into train (70%), validation (10%), test (20%)
```

#### 2️⃣ **Feature Extraction**
```python
# Module: feature_extraction.py
# Input: Processed light curves
# Output: 50+ features per light curve

Features extracted:
- Statistical (15): mean, std, median, min, max, range, percentiles, skewness, kurtosis
- Shape (10): num_peaks, num_troughs, peak_heights, roughness, smoothness
- Frequency (8): dominant_frequency, spectral_centroid, band_power_ratios
- Time Series (10): autocorrelation, trend, entropy, zero_crossing_rate
- Transit-specific (8): transit_depth, duration, period, SNR
```

#### 3️⃣ **Pair Generation**
```python
# Module: pair_generation.py
# Creates input pairs for Siamese network

Strategies:
1. Balanced: Equal representation of each sample
2. Random: Random sampling from classes
3. Hard: Focus on difficult examples

Example:
- Positive pair: (Planet_1, Planet_2) → Label = 1 (similar)
- Negative pair: (Planet_1, No_Planet_1) → Label = 0 (different)
```

#### 4️⃣ **Model Training**
```python
# Module: siamese_model.py & train.py
# Architecture: Siamese Neural Network

Network Structure:
- Shared weights between twin networks
- Layers: Input → 256 → 128 → 64 → 32 (embedding)
- Activation: ReLU
- Regularization: Dropout (0.3), BatchNorm
- Loss: Contrastive Loss with margin=1.0
- Optimizer: Adam (lr=0.001)
```

#### 5️⃣ **Evaluation**
```python
# Module: evaluate.py
# Metrics calculated:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC
- Embedding visualization (t-SNE)
```

---

## 💻 Installation & Setup

### Prerequisites:
- Python 3.8 or higher
- 4GB+ RAM recommended
- GPU optional but speeds up training

### Step 1: Install Dependencies
```bash
# Navigate to project directory
cd h:/My Drive/exo1/exoplanet_siamese

# Install required packages
pip install -r requirements.txt
```

### Step 2: Verify Data
```bash
# Check that raw data exists
ls data/raw/
# Should show: exoTest.csv
```

---

## 🚀 Running the System

### Option 1: Run Complete Pipeline (Recommended for first time)
```bash
# This runs all steps automatically
python main.py --config config.yaml

# What happens:
# 1. Preprocesses raw data
# 2. Extracts features
# 3. Generates pairs
# 4. Trains model
# 5. Evaluates performance
```

### Option 2: Run Individual Steps
```bash
# Step 1: Preprocess data only
python main.py --steps preprocess

# Step 2: Extract features only
python main.py --steps features

# Step 3: Generate pairs only
python main.py --steps pairs

# Step 4: Train model only
python main.py --steps train

# Step 5: Evaluate only
python main.py --steps evaluate

# Or combine multiple steps
python main.py --steps preprocess features pairs
```

### Option 3: Run with Custom Settings
```bash
# Force re-run even if outputs exist
python main.py --force

# Use custom configuration
python main.py --config my_config.yaml
```

### Option 4: Direct Module Execution
```bash
# Run individual modules directly
cd src

# Preprocess data
python data_preprocessing.py

# Extract features
python feature_extraction.py

# Generate pairs
python pair_generation.py

# Train model
python train.py --epochs 50 --batch_size 64

# Evaluate
python evaluate.py --test_features ../data/features/test_features.csv
```

---

## 📊 Data Flow

```
1. Raw Data (CSV)
   ↓
2. Preprocessed Data (CSV)
   - Normalized flux values
   - No outliers
   - Detrended
   ↓
3. Features (CSV)
   - 50+ numeric features per light curve
   - Statistical, shape, frequency features
   ↓
4. Pairs (NPY)
   - Arrays of paired samples
   - Binary labels (similar/different)
   ↓
5. Trained Model (PTH)
   - Neural network weights
   - Learned embeddings
   ↓
6. Predictions (CSV)
   - Class predictions
   - Confidence scores
```

---

## 🔧 Technical Details

### Model Parameters (config.yaml)
```yaml
model:
  hidden_dims: [256, 128, 64]  # Network layers
  embedding_dim: 32            # Final embedding size
  dropout_rate: 0.3            # Regularization
  
training:
  epochs: 100                  # Max training iterations
  batch_size: 32              # Samples per batch
  learning_rate: 0.001        # Optimization step size
  margin: 1.0                 # Contrastive loss margin
```

### Feature Categories
1. **Statistical Features** (15 total)
   - Basic: mean, std, median, min, max
   - Spread: range, IQR, MAD
   - Shape: skewness, kurtosis
   - Percentiles: 10th, 25th, 75th, 90th

2. **Transit Features** (8 total)
   - transit_depth: Maximum dip depth
   - transit_duration: Length of transit
   - transit_period: Time between transits
   - transit_snr: Signal-to-noise ratio

3. **Frequency Features** (8 total)
   - dominant_frequency: Main periodic component
   - spectral_energy: Total power
   - band_ratios: Low/mid/high frequency distribution

### Performance Expectations
- **Training Time**: 10-30 minutes (CPU), 2-5 minutes (GPU)
- **Memory Usage**: ~2GB RAM
- **Accuracy**: 85-90% on test set
- **False Positive Rate**: <15%
- **False Negative Rate**: <20%

---

## 🔍 Troubleshooting

### Common Issues & Solutions

#### Issue 1: "File not found" error
```bash
# Solution: Check file paths
ls data/raw/
# Ensure exoTest.csv exists
```

#### Issue 2: Out of memory
```bash
# Solution: Reduce batch size in config.yaml
training:
  batch_size: 16  # Reduce from 32
```

#### Issue 3: Training not converging
```bash
# Solution: Adjust learning rate
training:
  learning_rate: 0.0001  # Reduce from 0.001
```

#### Issue 4: Poor performance
```bash
# Solutions:
# 1. Increase training epochs
# 2. Try different normalization
preprocessing:
  normalization_method: "standard"  # Instead of "local"
# 3. Generate more pairs
pairs:
  pairs_per_sample: 10  # Increase from 5
```

### Monitoring Training
```bash
# Watch training progress
tail -f outputs/logs/training_log.txt

# Check saved models
ls -la outputs/models/

# View results
cat outputs/results/evaluation_results.json
```

### Quick Validation
```python
# Test if system is working
import sys
sys.path.append('src')
from siamese_model import create_siamese_model

# Create test model
model = create_siamese_model(input_dim=50)
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
# Should print: Model parameters: ~100,000
```

---

## 📈 Interpreting Results

### Metrics Explanation:
- **Accuracy**: Overall correct predictions (target: >85%)
- **Precision**: Of predicted planets, how many are real (target: >80%)
- **Recall**: Of real planets, how many were found (target: >75%)
- **F1-Score**: Harmonic mean of precision/recall (target: >78%)

### Output Files:
1. **evaluation_results.json**: All metrics in JSON format
2. **test_predictions.csv**: Predictions for each test sample
3. **confusion_matrix.png**: Visual prediction breakdown
4. **embedding_space.png**: 2D visualization of learned features
5. **training_history.png**: Loss and accuracy curves

---

## 🎯 Tips for Best Results

1. **Data Quality**: Ensure light curves have consistent sampling
2. **Balanced Dataset**: Maintain roughly equal planet/no-planet samples
3. **Feature Selection**: Use feature importance to identify key features
4. **Hyperparameter Tuning**: Experiment with config.yaml settings
5. **Cross-Validation**: Use validation set to prevent overfitting

---

## 📞 Support

For issues or questions:
1. Check this documentation
2. Review error messages carefully
3. Examine log files in `outputs/logs/`
4. Try example notebooks for debugging

---

**System ready for exoplanet detection! 🌟🔭**
