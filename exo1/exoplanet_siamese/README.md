# Exoplanet Detection using Siamese Neural Networks

## 🎯 Project Overview

This project implements a Siamese neural network approach for detecting exoplanets from light curve data. The system learns to distinguish between planetary transits and other stellar variations by training on pairs of light curves and learning a similarity metric.

## 🚀 Features

- **Data Preprocessing**: Normalization, detrending, and outlier removal for light curves
- **Feature Extraction**: Statistical, shape-based, frequency, and transit-specific features
- **Siamese Network**: Deep learning model for similarity learning
- **Pair Generation**: Balanced, random, and hard negative mining strategies
- **Comprehensive Evaluation**: Multiple metrics and visualizations

## 📁 Project Structure

```
exoplanet_siamese/
│
├── data/
│   ├── raw/                     # Raw light curve data (CSV)
│   ├── processed/               # Processed and normalized data
│   ├── features/                # Extracted features
│   └── pairs/                   # Generated training pairs
│
├── src/
│   ├── data_preprocessing.py    # Data cleaning and normalization
│   ├── feature_extraction.py    # Feature engineering
│   ├── pair_generation.py       # Siamese pair creation
│   ├── siamese_model.py        # Model architecture
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   └── utils.py                # Utility functions
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_feature_visualization.ipynb
│
├── outputs/
│   ├── models/                 # Saved models
│   ├── logs/                   # Training logs
│   └── results/                # Evaluation results
│
├── config.yaml                 # Configuration file
├── requirements.txt           # Dependencies
├── main.py                    # Main pipeline script
└── README.md                  # This file
```

## 🛠️ Installation

1. Clone the repository:
```bash
cd exoplanet_siamese
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Data Preparation

Place your raw light curve data (CSV format) in the `data/raw/` directory. The expected format:
- Column `LABEL`: 1 for no planet, 2 for planet
- Columns `FLUX.1`, `FLUX.2`, ...: Flux measurements over time

## 🏃 Quick Start

### Run Complete Pipeline

```bash
python main.py --config config.yaml
```

This will run all steps:
1. Data preprocessing
2. Feature extraction
3. Pair generation
4. Model training
5. Evaluation

### Run Individual Steps

```bash
# Only preprocessing
python main.py --steps preprocess

# Only training (assuming data is prepared)
python main.py --steps train

# Multiple steps
python main.py --steps features pairs train
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

- **Data paths**: Input/output directories
- **Preprocessing**: Normalization method, detrending
- **Model architecture**: Hidden layers, embedding dimension
- **Training**: Learning rate, batch size, epochs
- **Evaluation**: Thresholds, visualization options

## 🧠 Model Architecture

The Siamese network consists of:
- **Feature Extractor**: Fully connected layers with batch normalization and dropout
- **Embedding Layer**: Projects features to low-dimensional space
- **Contrastive Loss**: Learns to minimize distance for similar pairs

Architecture details:
- Input: Statistical and shape features from light curves
- Hidden layers: [256, 128, 64] neurons (configurable)
- Embedding dimension: 32
- Activation: ReLU
- Regularization: Dropout (0.3) and L2 weight decay

## 📈 Training

The model is trained using:
- **Optimizer**: Adam with learning rate scheduling
- **Loss**: Contrastive loss with configurable margin
- **Early stopping**: Prevents overfitting
- **Batch size**: 32 (default)
- **Epochs**: 100 (with early stopping)

Monitor training progress:
```bash
# Training logs are saved to outputs/logs/
# Visualizations are saved to outputs/plots/
```

## 📊 Evaluation Metrics

The system evaluates:
- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: For planet detection
- **F1 Score**: Balanced metric
- **ROC-AUC**: Classification performance
- **Confusion Matrix**: Detailed predictions
- **Embedding visualization**: t-SNE plots

## 📓 Notebooks

Explore the data and results using Jupyter notebooks:

1. **01_data_exploration.ipynb**: Visualize raw light curves and distributions
2. **02_feature_visualization.ipynb**: Analyze extracted features and importance

## 🎯 Results

Expected performance metrics:
- Accuracy: ~85-90%
- Precision: ~80-85%
- Recall: ~75-80%
- F1 Score: ~78-83%

Results are saved to `outputs/results/`:
- `evaluation_results.json`: Detailed metrics
- `test_predictions.csv`: Predictions on test set
- Visualization plots

## 🔧 Advanced Usage

### Custom Feature Extraction

Add new features in `src/feature_extraction.py`:
```python
def extract_custom_features(flux):
    # Your feature extraction logic
    return features
```

### Different Pair Generation Strategies

```python
# In src/pair_generation.py
generator = SiamesePairGenerator()

# Balanced pairs
X1, X2, labels = generator.create_balanced_pairs(features_df)

# Hard negative mining
X1, X2, labels = generator.create_hard_pairs(features_df)
```

### Model Customization

Modify architecture in `config.yaml`:
```yaml
model:
  hidden_dims: [512, 256, 128, 64]  # Deeper network
  embedding_dim: 64                  # Larger embedding
  dropout_rate: 0.5                  # More regularization
```

## 📚 References

- Siamese Neural Networks for One-shot Image Recognition (Koch et al., 2015)
- Exoplanet Detection using Machine Learning (Shallue & Vanderburg, 2018)
- Kepler Mission Data Products

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## 📝 License

This project is licensed under the MIT License.

## 👥 Contact

For questions or issues, please open an issue on GitHub.

---

**Happy Exoplanet Hunting! 🌟🪐**
