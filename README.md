# Exoplanet Detection Project

A comprehensive machine learning project for detecting exoplanets from Kepler light curve data using various algorithms including Siamese neural networks, traditional ML models, and deep learning approaches.

## 📁 Project Structure

```
Exoplanet-Detection-Project/
├── data/                   # Dataset storage
│   ├── raw_nasa/                    # Original NASA Kepler data
│   │   ├── 005607631-temp.fits      # Raw FITS files from NASA
│   │   ├── 007671950-temp.fits      # 3 example light curves
│   │   ├── 007918172-temp.fits      
│   │   ├── all_tbl_links.txt        # Download links
│   │   ├── master_koi_catalog.csv   # KOI (Kepler Object of Interest) catalog
│   │   ├── candidates_only/         # Candidate data (important)
│   │   └── processed_log.txt        # Processing log (important)
│   ├── processed_curves/            # 500-binned light curves
│   │   ├── raw_curve_500.csv        # All processed curves (6,349 stars)
│   │   ├── raw_curve_500_cleaned.csv # Cleaned version (removed outliers)
│   │   └── raw_curve_500_head.csv   # Sample/head version for testing
│   ├── extracted_features/          # Engineered features from curves
│   │   ├── features_curve_500.csv           # All 958 extracted features
│   │   ├── features_curve_500_pruned.csv    # Selected/reduced feature set
│   │   ├── pca_rankings_curve_500.csv       # PCA feature importance
│   │   ├── pca_variance.png                  # PCA variance explained
│   │   ├── pca_heatmap.png                  # PCA feature correlations
│   │   ├── significance_ranking.png         # Feature significance
│   │   └── checkpoints/                     # Training checkpoints
│   ├── splits/                      # Train/test splits for ML
│   │   ├── train_1_curve_500.csv           # Training set (unbalanced)
│   │   ├── train_balanced_curve_500.csv     # Training set (balanced)
│   │   ├── test_1_curve_500.csv            # Test set (unbalanced)
│   │   └── test_balanced_curve_500.csv      # Test set (balanced)
│   └── old_exoplanet_dataset/       # Legacy dataset (not used)
├── codes/                  # Source code
│   ├── siamese.py                 # Original Siamese architecture
│   ├── siamese_dataset500.py      # Siamese model for dataset_500
│   ├── cleaned_siamese.py         # Siamese model for cleaned data
│   ├── baseline_models.py         # RF, Logistic Regression, XGBoost
│   ├── neural_networks.py         # FFNN and CNN models
│   └── lightweight_nn.py          # Lightweight NN versions
├── models/                 # Trained models
│   ├── siamese_dataset500.pth     # Original Siamese model (75.06% accuracy)
│   ├── cleaned_siamese.pth        # Cleaned Siamese model (81.73% accuracy)
│   ├── random_forest.pkl          # Random Forest model (73.33% accuracy)
│   ├── logistic_regression.pkl    # Logistic Regression model (70.42% accuracy)
│   ├── xgboost.pkl                # XGBoost model (75.53% accuracy)
│   ├── feedforwardnn.pth          # Feedforward Neural Network (70.97% accuracy)
│   ├── cnn.pth                    # Convolutional Neural Network (73.17% accuracy)
│   └── *.pkl                      # Scalers and utilities
├── features/              # Extracted features (empty initially)
├── output/                # Results and outputs
│   ├── baseline_results.txt       # Model comparison results
│   ├── nn_results.txt             # Neural network results
│   ├── cleaned_siamese_results.txt# Cleaned Siamese results
│   └── *.png                      # Training curves and visualizations
├── visualization/          # Visualizations (empty - for future work)
├── README.md              # This file
├── requirements.txt       # All dependencies
├── train_all.py           # Main training script
└── pyproject.toml         # Project configuration
```

## 🎯 Project Overview

This project implements multiple machine learning approaches for exoplanet detection from Kepler space telescope data:

### 📊 Dataset
- **Source**: Kepler light curve data from NASA
- **Processing Pipeline**: NASA FITS files → 500-binned curves → Extracted features
- **Samples**: 6,349 stars with confirmed/candidate exoplanets
- **Labels**: Binary classification (0 = non-exoplanet, 1 = exoplanet)
- **Split**: 70% training, 20% testing, 10% candidates (stratified, no leakage)

### 🤖 Models Implemented

#### 1. Siamese Neural Networks
- **siamese.py**: Original Siamese architecture
- **siamese_dataset500.py**: Optimized for dataset_500 (75.06% accuracy)
- **cleaned_siamese.py**: Best performing model on cleaned data (81.73% accuracy)
- **Architecture**: 500 → 256 → 128 → 128 embedding with contrastive loss

#### 2. Baseline Traditional Models
- **Random Forest**: Ensemble decision trees (73.33% accuracy)
- **Logistic Regression**: Linear classification (70.42% accuracy)
- **XGBoost**: Gradient boosting (75.53% accuracy)
- **Implementation**: `baseline_models.py`

#### 3. Deep Learning Models
- **Feedforward Neural Network**: Multi-layer perceptron (70.97% accuracy)
- **Convolutional Neural Network**: 1D CNN for time series (73.17% accuracy)
- **Implementation**: `neural_networks.py` and `lightweight_nn.py`

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt
# or with uv
uv sync
```

### 2. Train All Models
```bash
python train_all.py
```

### 3. Individual Model Training
```bash
# Baseline models (RF, Logistic Regression, XGBoost)
cd codes && python baseline_models.py

# Neural networks (FFNN, CNN)
cd codes && python neural_networks.py

# Siamese model (best performance)
cd codes && python cleaned_siamese.py
```

## 📈 Model Performance

### Current Best Results
| Model | Test Accuracy | Type | Data Used |
|-------|---------------|------|-----------|
| Cleaned Siamese | 81.73% | Deep Learning | Cleaned curves |
| XGBoost | 75.53% | Ensemble | Extracted features |
| Original Siamese | 75.06% | Deep Learning | Raw curves |
| CNN | 73.17% | Deep Learning | Extracted features |
| Random Forest | 73.33% | Ensemble | Extracted features |
| Feedforward NN | 70.97% | Deep Learning | Extracted features |
| Logistic Regression | 70.42% | Linear | Extracted features |

## 📊 Data Organization

### **Data Processing Pipeline:**

1. **Raw NASA Data** (`data/raw_nasa/`)
   - Original FITS files from NASA Kepler archive
   - 3 example light curves + KOI catalog
   - Source: NASA Exoplanet Archive

2. **Processed Curves** (`data/processed_curves/`)
   - 500-binned light curves after phase folding
   - `raw_curve_500_cleaned.csv`: 6,349 stars × 500 flux measurements
   - Used for: Siamese networks, CNNs

3. **Extracted Features** (`data/extracted_features/`)
   - 958 statistical features extracted using tsfresh
   - `features_curve_500_pruned.csv`: Selected best features (~300)
   - Used for: Traditional ML, Feedforward NN

4. **Train/Test Splits** (`data/splits/`)
   - Pre-split datasets for convenience
   - Balanced and unbalanced versions available

### **File Usage Guide:**

```python
# For Siamese Networks (time series data)
data = pd.read_csv('data/processed_curves/raw_curve_500_cleaned.csv')

# For Traditional ML (tabular features)
data = pd.read_csv('data/extracted_features/features_curve_500_pruned.csv')

# For quick training with pre-made splits
train = pd.read_csv('data/splits/train_balanced_curve_500.csv')
test = pd.read_csv('data/splits/test_balanced_curve_500.csv')
```

## 🔧 Technical Details

### Data Preprocessing
- **Standardization**: Features scaled using StandardScaler
- **Missing Values**: Filled with 0
- **Train/Test Split**: Stratified sampling to maintain class balance
- **Data Leakage Prevention**: Strict separation of training and testing data

### Siamese Network Architecture
```
Input (500 features)
    ↓
Dense Layer (256 units) + BatchNorm + ReLU + Dropout
    ↓
Dense Layer (128 units) + BatchNorm + ReLU + Dropout
    ↓
Dense Layer (128 units) + BatchNorm
    ↓
L2 Normalization (Embedding)
```

### Loss Functions
- **Siamese**: Contrastive Loss with margin = 1.0
- **Classification**: Binary Cross-Entropy
- **Traditional Models**: Default sklearn loss functions

### Optimizers
- **All Neural Networks**: Adam optimizer (lr=0.001, weight_decay=1e-5)
- **Early Stopping**: Patience = 10 epochs
- **Batch Size**: 32 (Siamese), 64 (Neural Networks)

## 📊 Evaluation Metrics

All models are evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification results

## 🛠️ Dependencies

### Core Libraries
- `torch`: PyTorch for deep learning
- `sklearn`: Traditional machine learning
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `xgboost`: Gradient boosting

### Visualization
- `matplotlib`: Plotting and visualization
- `seaborn`: Statistical visualization

### Utilities
- `joblib`: Model serialization
- `scipy`: Scientific computing

## 📝 Usage Examples

### Loading and Using Trained Models

```python
import joblib
import torch
from codes.cleaned_siamese import CleanedSiameseNetwork

# Load Random Forest model
rf_model = joblib.load('models/random_forest.pkl')
scaler = joblib.load('models/scaler.pkl')

# Load best Siamese model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
siamese_model = CleanedSiameseNetwork(500, 128).to(device)
siamese_model.load_state_dict(torch.load('models/cleaned_siamese.pth', map_location=device))
```

### Making Predictions

```python
# Prepare your data (same preprocessing as training)
data_scaled = scaler.transform(your_data)

# Random Forest prediction
rf_pred = rf_model.predict(data_scaled)

# Siamese prediction (requires embedding extraction)
with torch.no_grad():
    embedding = siamese_model.get_embedding(torch.FloatTensor(data_scaled))
```

## 🔬 Research Notes

### Key Findings
1. **Cleaned Siamese network** achieves best performance (81.73%)
2. **Data preprocessing** is crucial for model performance
3. **Class imbalance** requires careful handling
4. **Feature engineering** from light curves significantly improves traditional ML

### Data Processing Insights
- **Phase folding** and **binning** preserves transit signals
- **Outlier removal** improves Siamese network performance
- **Feature extraction** creates interpretable patterns for traditional ML
- **Proper train/test separation** prevents data leakage

### Future Improvements
- [ ] Hyperparameter optimization for all models
- [ ] Ensemble methods combining multiple approaches
- [ ] Advanced architectures (Transformers, GNNs)
- [ ] Cross-validation with multiple folds
- [ ] Feature importance analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- **NASA Kepler Mission**: For providing the light curve data
- **Scikit-learn**: For traditional ML implementations
- **PyTorch**: For deep learning frameworks
- **Research Community**: For exoplanet detection research
- **Your Friend**: For processing the raw NASA data and creating the dataset

## 📞 Contact

For questions or collaborations, please refer to the project repository.

---

**Note**: The `visualization/` folder is intentionally empty and reserved for future visualization work.

**Best Model**: Cleaned Siamese Network (81.73% accuracy) on `data/processed_curves/raw_curve_500_cleaned.csv`

Pre-trained models are available in the `model/` directory.