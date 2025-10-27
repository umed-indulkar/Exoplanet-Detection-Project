# 🏗️ UNIFIED EXOPLANET DETECTION SYSTEM - ARCHITECTURE DESIGN

**Version:** 2.0.0  
**Design Date:** October 26, 2025  
**Status:** Implementation Ready

---

## 🎯 SYSTEM OVERVIEW

The **Unified Exoplanet Detection System** combines the best features from all branches into a single, cohesive, production-ready platform for detecting exoplanets from light curve data.

### Design Principles

1. **Modularity:** Each component is independent and replaceable
2. **Scalability:** Handles single files to thousands of light curves
3. **Flexibility:** Multiple feature extraction modes (fast/standard/comprehensive)
4. **Performance:** Optimized for modern multi-core processors
5. **Usability:** CLI, Python API, and web dashboard interfaces
6. **Extensibility:** Easy to add new models, features, or visualizations

---

## 📦 PROJECT STRUCTURE

```
exoplanet_detection/
├── README.md                          # Main documentation
├── CHANGELOG.md                       # Version history
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Modern Python packaging
├── config.yaml                        # Default configuration
│
├── exodet/                           # Main package
│   ├── __init__.py                   # Package initialization
│   ├── __version__.py                # Version info
│   │
│   ├── core/                         # Core functionality
│   │   ├── __init__.py
│   │   ├── data_loader.py           # Unified data loading (NPZ/CSV)
│   │   ├── preprocessing.py          # Advanced preprocessing pipeline
│   │   ├── config.py                 # Configuration management
│   │   └── exceptions.py             # Custom exceptions
│   │
│   ├── features/                     # Feature extraction
│   │   ├── __init__.py
│   │   ├── basic_extractor.py       # Fast basic features (100+)
│   │   ├── ml_extractor.py          # ML-optimized features (50+)
│   │   ├── tsfresh_extractor.py     # Comprehensive TSFresh (350+)
│   │   ├── feature_selector.py      # Feature selection & pruning
│   │   └── feature_registry.py      # Feature catalog
│   │
│   ├── models/                       # Machine learning models
│   │   ├── __init__.py
│   │   ├── base_model.py            # Abstract base class
│   │   ├── siamese_network.py       # Siamese neural network
│   │   ├── fcnn_classifier.py       # Fully connected classifier
│   │   ├── ensemble.py              # Ensemble methods
│   │   └── hyperopt.py              # Hyperparameter optimization
│   │
│   ├── training/                     # Training pipeline
│   │   ├── __init__.py
│   │   ├── trainer.py               # Training orchestrator
│   │   ├── pair_generator.py        # Siamese pair generation
│   │   ├── data_splitter.py         # Train/val/test splitting
│   │   ├── callbacks.py             # Training callbacks
│   │   └── metrics.py               # Evaluation metrics
│   │
│   ├── visualization/                # Visualization tools
│   │   ├── __init__.py
│   │   ├── lightcurve_plots.py      # Light curve visualization
│   │   ├── feature_plots.py         # Feature analysis plots
│   │   ├── model_plots.py           # Model performance plots
│   │   ├── dashboard_components.py  # Dashboard widgets
│   │   └── report_generator.py      # Automated reporting
│   │
│   ├── utils/                        # Utilities
│   │   ├── __init__.py
│   │   ├── file_utils.py            # File I/O operations
│   │   ├── math_utils.py            # Mathematical helpers
│   │   ├── parallel.py              # Parallel processing
│   │   ├── logger.py                # Logging configuration
│   │   └── validation.py            # Data validation
│   │
│   └── cli/                          # Command-line interface
│       ├── __init__.py
│       ├── main.py                  # Main CLI entry point
│       ├── commands/                # CLI commands
│       │   ├── extract.py           # Feature extraction command
│       │   ├── train.py             # Training command
│       │   ├── evaluate.py          # Evaluation command
│       │   ├── batch.py             # Batch processing command
│       │   └── dashboard.py         # Launch dashboard command
│       └── args.py                  # Argument parsers
│
├── dashboard/                        # Web dashboard (Streamlit)
│   ├── app.py                       # Main dashboard app
│   ├── pages/                       # Multi-page dashboard
│   │   ├── 1_Data_Explorer.py      # Data exploration
│   │   ├── 2_Feature_Extraction.py  # Feature extraction interface
│   │   ├── 3_Model_Training.py     # Training interface
│   │   ├── 4_Model_Evaluation.py   # Evaluation interface
│   │   └── 5_Batch_Processing.py   # Batch operations
│   └── components/                  # Reusable components
│       ├── file_uploader.py
│       ├── plot_viewer.py
│       └── metrics_display.py
│
├── scripts/                          # Standalone scripts
│   ├── convert_to_excel.py          # CSV to Excel converter
│   ├── clean_data.py                # Data cleaning utility
│   ├── merge_datasets.py            # Dataset merging
│   └── benchmark_performance.py     # Performance testing
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_Quick_Start.ipynb        # Getting started guide
│   ├── 02_Data_Exploration.ipynb   # Data analysis
│   ├── 03_Feature_Engineering.ipynb # Feature development
│   ├── 04_Model_Training.ipynb     # Training examples
│   └── 05_Advanced_Topics.ipynb    # Advanced techniques
│
├── tests/                            # Unit & integration tests
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_integration.py
│
├── docs/                             # Documentation
│   ├── index.md                     # Documentation home
│   ├── installation.md              # Installation guide
│   ├── quickstart.md                # Quick start tutorial
│   ├── user_guide/                  # User documentation
│   │   ├── data_preparation.md
│   │   ├── feature_extraction.md
│   │   ├── model_training.md
│   │   └── visualization.md
│   ├── api_reference/               # API documentation
│   │   ├── core.md
│   │   ├── features.md
│   │   ├── models.md
│   │   └── visualization.md
│   └── developer_guide/             # Developer docs
│       ├── contributing.md
│       ├── architecture.md
│       └── extending.md
│
├── examples/                         # Example code
│   ├── basic_pipeline.py            # Simple example
│   ├── batch_processing.py          # Batch example
│   ├── custom_features.py           # Custom feature example
│   └── hyperparameter_tuning.py    # Optimization example
│
├── data/                             # Data directory (not in git)
│   ├── raw/                         # Raw NPZ/CSV files
│   ├── processed/                   # Processed data
│   ├── features/                    # Extracted features
│   └── models/                      # Saved models
│
└── outputs/                          # Output directory (not in git)
    ├── logs/                        # Log files
    ├── plots/                       # Generated plots
    ├── reports/                     # Generated reports
    └── checkpoints/                 # Model checkpoints
```

---

## 🔧 COMPONENT SPECIFICATIONS

### 1. Core Module (`exodet/core/`)

#### `data_loader.py`
```python
class UniversalDataLoader:
    """Load data from NPZ, CSV, or FITS files"""
    
    def load_npz(file_path) -> LightCurve
    def load_csv(file_path) -> LightCurve
    def load_batch(directory) -> List[LightCurve]
    def auto_detect_format(file_path) -> str
```

#### `preprocessing.py`
```python
class PreprocessingPipeline:
    """Configurable preprocessing pipeline"""
    
    def remove_nans()
    def detrend(method='polynomial', order=3)
    def sigma_clip(sigma=3, iterations=3)
    def normalize(method='zscore')
    def fold(period, epoch)
    def bin_data(bin_size)
```

#### `config.py`
```python
class Config:
    """Unified configuration management"""
    
    def load_from_yaml(path)
    def load_from_dict(config_dict)
    def validate()
    def save(path)
```

### 2. Features Module (`exodet/features/`)

#### Three-Tier Feature Extraction

**Tier 1: Fast (100+ features, <1s per curve)**
- Basic statistics
- Simple time-domain features
- Quick transit detection
- Use case: Initial screening, large datasets

**Tier 2: Standard (150+ features, 2-5s per curve)**
- All Tier 1 features
- Advanced statistical features
- Frequency-domain analysis
- ML-optimized features
- Use case: Standard analysis, model training

**Tier 3: Comprehensive (500+ features, 10-30s per curve)**
- All Tier 1 & 2 features
- Full TSFresh feature set
- Advanced astronomical features
- Use case: Final analysis, research

#### `feature_registry.py`
```python
class FeatureRegistry:
    """Central registry of all available features"""
    
    features: Dict[str, FeatureDefinition]
    
    def register(name, function, category, tier)
    def get_by_tier(tier) -> List[Feature]
    def get_by_category(category) -> List[Feature]
    def extract(data, features_list) -> DataFrame
```

### 3. Models Module (`exodet/models/`)

#### `siamese_network.py`
```python
class SiameseNetwork(BaseModel):
    """Siamese neural network for similarity learning"""
    
    architecture:
        - Feature Extractor: [256, 128, 64]
        - Embedding Layer: 32D
        - Distance Metric: Euclidean
        - Loss: Contrastive Loss
    
    def forward(x1, x2)
    def get_embedding(x)
    def compute_distance(x1, x2)
    def predict_similarity(x1, x2)
```

#### `ensemble.py`
```python
class EnsembleDetector:
    """Ensemble of multiple models"""
    
    models: List[BaseModel]
    
    def add_model(model, weight)
    def predict_ensemble(data)
    def optimize_weights(validation_data)
```

### 4. Training Module (`exodet/training/`)

#### `trainer.py`
```python
class Trainer:
    """Unified training orchestrator"""
    
    def train(model, train_data, val_data, config)
    def evaluate(model, test_data)
    def hyperparameter_search(search_space, n_trials)
    def save_checkpoint(epoch, model, optimizer)
    def load_checkpoint(path)
```

### 5. Visualization Module (`exodet/visualization/`)

#### Plot Types
- Light curve plots (raw, processed, folded)
- Feature distribution plots
- Feature correlation heatmaps
- Model performance curves (loss, metrics)
- Confusion matrices
- ROC curves
- t-SNE embeddings
- Comprehensive analysis dashboards

### 6. CLI Module (`exodet/cli/`)

#### Command Structure
```bash
exodet <command> [options]

Commands:
  extract     Extract features from light curves
  train       Train a detection model
  evaluate    Evaluate model performance
  predict     Make predictions on new data
  batch       Batch process multiple files
  dashboard   Launch web dashboard
  version     Show version info
  config      Configuration management
```

---

## 🎨 USER INTERFACES

### 1. Command-Line Interface (CLI)

**Example Usage:**
```bash
# Extract features (fast mode)
exodet extract --input data/raw/*.npz --output features.csv --mode fast

# Extract comprehensive features with parallel processing
exodet extract --input data/raw/ --output features.csv --mode comprehensive --workers 8

# Train Siamese network
exodet train --features features.csv --model siamese --config config.yaml

# Batch processing with visualization
exodet batch --input data/raw/ --output results/ --visualize --workers 4

# Launch dashboard
exodet dashboard --port 8501
```

### 2. Python API

**Example Usage:**
```python
from exodet import ExoplanetDetector
from exodet.features import FeatureExtractor
from exodet.models import SiameseNetwork

# Initialize detector
detector = ExoplanetDetector(config='config.yaml')

# Load and preprocess data
lc = detector.load('lightcurve.npz')
lc_clean = detector.preprocess(lc, detrend=True, sigma_clip=True)

# Extract features (multi-tier)
extractor = FeatureExtractor(tier='standard')
features = extractor.extract(lc_clean)

# Train model
model = SiameseNetwork()
detector.train(model, features, epochs=100)

# Evaluate
metrics = detector.evaluate(model, test_data)

# Predict
prediction = detector.predict('new_lightcurve.npz')
```

### 3. Web Dashboard (Streamlit)

**Pages:**
1. **Data Explorer** - Upload, visualize, and explore light curves
2. **Feature Extraction** - Interactive feature extraction with preview
3. **Model Training** - Configure and train models with live monitoring
4. **Model Evaluation** - Comprehensive evaluation with visualizations
5. **Batch Processing** - Process multiple files with progress tracking

---

## ⚙️ CONFIGURATION SYSTEM

### `config.yaml` Structure

```yaml
# System Configuration
system:
  seed: 42
  device: auto  # auto, cpu, cuda
  num_workers: 4
  verbose: true
  log_level: INFO

# Data Configuration
data:
  input_dir: data/raw/
  output_dir: data/processed/
  file_format: auto  # auto, npz, csv, fits
  
# Preprocessing Configuration
preprocessing:
  remove_nans: true
  detrend:
    enabled: true
    method: polynomial  # polynomial, savgol, gp
    order: 3
  sigma_clip:
    enabled: true
    sigma: 3.0
    iterations: 3
  normalize:
    enabled: true
    method: zscore  # zscore, minmax, robust
  fold:
    enabled: false
    period: null
    epoch: 0
  bin:
    enabled: false
    bin_size: 0.01

# Feature Extraction Configuration
features:
  tier: standard  # fast, standard, comprehensive
  custom_features: []
  parallel: true
  cache_enabled: true
  
  # Tier-specific settings
  fast:
    timeout: 1.0  # seconds per curve
  standard:
    timeout: 5.0
  comprehensive:
    timeout: 30.0
    tsfresh_params: ComprehensiveFCParameters

# Model Configuration
model:
  type: siamese  # siamese, fcnn, ensemble
  architecture:
    hidden_dims: [256, 128, 64]
    embedding_dim: 32
    dropout_rate: 0.3
    activation: relu
  
# Training Configuration
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.00001
  optimizer: adam  # adam, sgd, rmsprop
  scheduler:
    enabled: true
    patience: 5
    factor: 0.5
  early_stopping:
    enabled: true
    patience: 10
    min_delta: 0.0001
  
# Evaluation Configuration
evaluation:
  metrics: [accuracy, precision, recall, f1, auc]
  visualizations: [confusion_matrix, roc_curve, embedding]
  save_predictions: true

# Output Configuration
output:
  save_plots: true
  save_features: true
  save_models: true
  format: csv  # csv, excel, hdf5
  compression: gzip
```

---

## 🚀 PERFORMANCE OPTIMIZATION STRATEGIES

### 1. Parallel Processing
- Multi-core feature extraction
- Batch data loading
- Parallel model inference

### 2. Memory Management
- Lazy loading for large datasets
- Memory-mapped file I/O
- Garbage collection optimization
- Configurable batch sizes

### 3. Caching
- Feature extraction caching
- Preprocessed data caching
- Model checkpoint caching

### 4. Code Optimization
- Vectorized operations (NumPy)
- JIT compilation where applicable
- Efficient data structures

---

## 🔒 QUALITY ASSURANCE

### 1. Testing Strategy
- Unit tests (>80% coverage)
- Integration tests
- Performance benchmarks
- Regression tests

### 2. Code Quality
- Type hints throughout
- Comprehensive docstrings
- PEP 8 compliance
- Code review process

### 3. Documentation
- API reference (auto-generated)
- User guide (comprehensive)
- Developer guide
- Example gallery

---

## 📊 EXPECTED PERFORMANCE

### Processing Speed
- Fast Mode: 1000+ curves/hour (single core)
- Standard Mode: 500+ curves/hour (single core)
- Comprehensive Mode: 100+ curves/hour (single core)
- With 8 cores: 5x-8x speedup

### Model Performance
- Accuracy: 85-92%
- Precision: 80-88%
- Recall: 75-85%
- F1-Score: 78-86%
- ROC-AUC: 0.88-0.94

### Resource Usage
- Memory: 2-8 GB (depending on mode)
- CPU: Scales with cores
- Storage: 1-10 MB per processed curve

---

## 🔄 MIGRATION PATH

### From Main Branch
1. Import existing preprocessing pipeline
2. Integrate visualization tools
3. Port CLI commands
4. Adapt batch processing

### From newcode
1. Import Siamese network architecture
2. Integrate training pipeline
3. Port configuration system
4. Adapt dashboard components

### From newcode2
1. Import TSFresh extractor
2. Integrate performance optimizations
3. Port parallel processing logic
4. Adapt caching mechanisms

### From newcode3
1. Import utility scripts
2. Integrate Excel export
3. Port modern packaging

---

## 🎯 SUCCESS CRITERIA

### Functionality
✅ Supports NPZ, CSV, and FITS formats  
✅ Three-tier feature extraction (fast/standard/comprehensive)  
✅ Multiple ML models (Siamese, FCNN, ensemble)  
✅ Comprehensive visualization suite  
✅ CLI, Python API, and web dashboard

### Performance
✅ Process 500+ curves/hour (standard mode)  
✅ 85%+ accuracy on validation data  
✅ <8 GB memory usage  
✅ Scales to 8+ cores

### Usability
✅ Single command installation  
✅ <5 minutes to first prediction  
✅ Comprehensive documentation  
✅ Rich example gallery

### Quality
✅ 80%+ test coverage  
✅ PEP 8 compliant  
✅ Full type hints  
✅ No critical bugs

---

## 📅 IMPLEMENTATION TIMELINE

**Total Estimated Time: 4-5 weeks**

### Week 1: Foundation
- Project structure setup
- Core modules (data loader, preprocessing, config)
- Basic testing framework

### Week 2: Feature Extraction
- Three-tier feature extraction
- Feature registry and selection
- Performance optimization

### Week 3: Machine Learning
- Model implementations
- Training pipeline
- Hyperparameter optimization

### Week 4: Interfaces & Visualization
- CLI implementation
- Dashboard development
- Visualization tools

### Week 5: Polish & Documentation
- Comprehensive testing
- Documentation writing
- Examples and tutorials
- Performance optimization

---

**Status:** Ready for implementation  
**Next Action:** Begin core module development

