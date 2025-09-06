# Light Curve Analysis Project

## Overview

This is a comprehensive Python toolkit designed for processing, visualizing, and extracting features from astronomical light curves stored in `.npz` format. The project provides a complete pipeline from raw data loading through advanced feature extraction, with over 800 statistical, time-domain, and frequency-domain features. The system includes interactive visualization capabilities, manual feature pruning tools, and a command-line interface for batch processing. The project is specifically designed for astronomical data analysis, particularly for studying stellar variability, exoplanet transits, and other time-series phenomena in light curve data.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Core Module Structure
The system follows a modular architecture with clear separation of concerns:

- **Data Layer** (`data_loader.py`): Handles `.npz` file loading with flexible key detection and comprehensive preprocessing pipeline including detrending, sigma-clipping, folding, binning, and normalization
- **Feature Extraction** (`feature_extraction.py`): Implements 800+ features using statistical analysis, time-series methods, and astronomical-specific calculations
- **Visualization Engine** (`visualization.py`): Provides plotting capabilities for raw curves, folded curves, feature distributions, and comprehensive analysis dashboards
- **Feature Pruning** (`feature_pruning.py`): Interactive feature selection tools with manual pruning capabilities and feature reporting
- **CLI Interface** (`main.py`): Command-line entry point with comprehensive argument parsing for batch processing

### Data Processing Pipeline
The preprocessing pipeline follows a multi-stage approach:
1. NaN removal and initial validation
2. Stellar variability flattening using detrending algorithms
3. Quality point masking based on statistical criteria
4. Sigma-clipping for outlier removal (configurable sigma levels)
5. Optional period folding with epoch-based phasing
6. Time binning (5-minute default bins)
7. Flux normalization (zero mean, unit variance)

### Feature Architecture
Features are organized into categories:
- **Statistical Features**: Moments, percentiles, skewness, kurtosis
- **Time-Domain Features**: Variability indices, autocorrelation, trend analysis
- **Frequency-Domain Features**: Periodogram analysis, spectral features
- **Astronomical Features**: Transit-specific metrics, eclipse detection
- **Error-Based Features**: Signal-to-noise ratios, uncertainty propagation

### Visualization System
Multi-layered plotting system with:
- **Basic Plotting**: Simple time-series visualization with error bars
- **Folded Curves**: Phase-folded light curves with period highlighting
- **Feature Analysis**: Distribution plots, correlation matrices, feature importance
- **Comprehensive Dashboards**: Multi-panel analysis views with statistical summaries

## External Dependencies

### Core Scientific Stack
- **NumPy** (>=1.21.0): Array operations and mathematical functions
- **SciPy** (>=1.7.0): Statistical analysis, signal processing, and optimization
- **Pandas** (>=1.3.0): Data manipulation and CSV I/O operations

### Visualization Libraries
- **Matplotlib** (>=3.5.0): Primary plotting engine for all visualizations
- **Seaborn** (>=0.11.0): Statistical plotting and color schemes

### Astronomical Libraries
- **Astropy** (>=5.0.0): Astronomical constants, coordinate systems, and time handling
- **Lightkurve** (>=2.0.0): Specialized light curve analysis tools and algorithms

### Machine Learning and Statistics
- **Scikit-learn** (>=1.0.0): Feature preprocessing, dimensionality reduction, and statistical tools
- **Statsmodels** (>=0.13.0): Advanced statistical modeling and time series analysis
- **TSFresh** (>=0.19.0): Automated time series feature extraction (800+ features)

### Development and Interactive Tools
- **IPython** (>=7.0.0): Enhanced interactive computing environment
- **Jupyter** (>=1.0.0): Notebook interface for exploratory analysis
- **Notebook** (>=6.0.0): Web-based interactive development environment

### Optional Advanced Packages
- **PyMC3**: Bayesian statistical modeling for uncertainty quantification
- **emcee**: MCMC sampling for parameter estimation
- **corner**: Corner plots for multi-dimensional parameter visualization

The architecture prioritizes modularity and extensibility, allowing users to easily add new feature extraction methods, visualization types, or preprocessing steps without modifying core functionality.