# Exoplanet Detection Project

A comprehensive machine learning project for detecting exoplanets from Kepler light curve data using various algorithms including Siamese neural networks, traditional ML models, and deep learning approaches.

## 📁 Project Structure

```
Exoplanet-Detection-Project/
├── data/                   # Dataset storage
│   └── dataset_500/       # Main dataset (500 flux features)
│       └── dataset_500/
│           ├── raw_curve_500_head.csv    # Main dataset file
│           └── processed_log.txt         # Processing log
├── codes/                  # Source code
│   ├── siamese.py                 # Original Siamese model
│   ├── siamese_dataset500.py      # Siamese model for dataset_500
│   ├── baseline_models.py         # RF, Logistic Regression, XGBoost
│   └── neural_networks.py         # FFNN and CNN models
├── models/                 # Trained models
│   ├── siamese_dataset500.pth     # Best Siamese model
│   ├── random_forest.pkl          # Random Forest model
│   ├── logistic_regression.pkl    # Logistic Regression model
│   ├── xgboost.pkl                # XGBoost model
│   ├── feedforward_nn.pth         # Feedforward Neural Network
│   ├── convolutional_nn.pth       # Convolutional Neural Network
│   └── *.pkl                      # Scalers and utilities
├── features/              # Extracted features (empty initially)
├── output/                # Results and outputs
│   ├── baseline_results.txt       # Model comparison results
│   └── training_curves.png        # Training visualization
├── visualization/          # Visualizations (empty - for future work)
├── README.md              # This file
└── pyproject.toml         # Project dependencies
```

## 🎯 Project Overview

This project implements multiple machine learning approaches for exoplanet detection from Kepler space telescope data:

### 📊 Dataset
- **Source**: Kepler light curve data
- **Features**: 500 flux measurements per star
- **Samples**: 6,352 stars
- **Labels**: Binary classification (0 = non-exoplanet, 1 = exoplanet)
- **Split**: 80% training, 20% testing (stratified, no leakage)

### 🤖 Models Implemented

#### 1. Siamese Neural Networks
- **siamese.py**: Original Siamese architecture
- **siamese_dataset500.py**: Optimized for dataset_500
- **Performance**: 74.67% test accuracy
- **Architecture**: 500 → 256 → 128 → 128 embedding

#### 2. Baseline Traditional Models
- **Random Forest**: Ensemble decision trees
- **Logistic Regression**: Linear classification
- **XGBoost**: Gradient boosting
- **Implementation**: `baseline_models.py`

#### 3. Deep Learning Models
- **Feedforward Neural Network**: Multi-layer perceptron
- **Convolutional Neural Network**: 1D CNN for time series
- **Implementation**: `neural_networks.py`

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt
# or with uv
uv sync
```

### 2. Train Baseline Models
```bash
cd codes
python baseline_models.py
```

### 3. Train Neural Networks
```bash
cd codes
python neural_networks.py
```

### 4. Train Siamese Model
```bash
cd codes
python siamese_dataset500.py
```

## 📈 Model Performance

### Current Best Results
| Model | Test Accuracy | Type |
|-------|---------------|------|
| Siamese Network | 74.67% | Deep Learning |
| Random Forest | TBD | Ensemble |
| XGBoost | TBD | Gradient Boosting |
| Feedforward NN | TBD | Deep Learning |
| CNN | TBD | Deep Learning |
| Logistic Regression | TBD | Linear |

*Run the training scripts to get updated performance metrics*

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
from codes.baseline_models import BaselineModels

# Load Random Forest model
rf_model = joblib.load('models/random_forest.pkl')
scaler = joblib.load('models/scaler.pkl')

# Load Siamese model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
siamese_model = torch.load('models/siamese_dataset500.pth', map_location=device)
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

## � Research Notes

### Key Findings
1. **Siamese networks** show promising results for exoplanet detection
2. **Data preprocessing** is crucial for model performance
3. **Class imbalance** requires careful handling
4. **Feature engineering** from light curves is important

### Future Improvements
- [ ] Hyperparameter optimization
- [ ] Ensemble methods
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

## 📞 Contact

For questions or collaborations, please refer to the project repository.

---

**Note**: The `visualization/` folder is intentionally empty and reserved for future visualization work.

Pre-trained models are available in the `model/` directory.