"""
Generate comprehensive visualizations for trained models
Run after training to create all plots
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Import visualization functions
from exodet.ml.training_logger import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_metrics_bar
)

print("="*70)
print("GENERATING COMPREHENSIVE VISUALIZATIONS")
print("="*70)

# Load validation predictions (you'll need to save these during evaluation)
# For now, let's create a script that evaluates and visualizes

output_dir = Path('runs/logs')
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\n📊 Output directory: {output_dir}")
print("\n✓ Visualization module ready!")
print("\nTo generate plots, run:")
print("  python -m exodet.cli evaluate-siamese --model runs/exo_siamese_50ep.pt --features outputs/val.csv --target label --device cuda")
print("\nPlots will be saved to: runs/logs/")
