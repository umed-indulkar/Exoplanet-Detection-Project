#!/usr/bin/env python3
"""
Light Curve Analysis CLI

Command-line interface for processing, visualizing, and extracting features 
from astronomical light curves.
"""

import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from src.data_loader import load_npz_curve, preprocess_lightcurve
from src.visualization import (plot_lightcurve, plot_folded_curve, 
                              plot_feature_distribution, plot_comprehensive_analysis)
from src.feature_extraction import extract_features
from src.feature_pruning import (manual_prune, interactive_feature_selection,
                                save_selected_features, load_selected_features,
                                create_feature_report)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Light Curve Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --load data.npz --visualize
  %(prog)s --load data.npz --extract --save features.csv
  %(prog)s --load data.npz --extract --prune --save pruned_features.csv
  %(prog)s --load data.npz --period 2.5 --epoch 0 --visualize
        """
    )
    
    # File operations
    parser.add_argument('--load', type=str, metavar='FILE',
                       help='Load light curve from .npz file')
    parser.add_argument('--save', type=str, metavar='FILE',
                       help='Save results to file (CSV for features)')
    
    # Processing options
    parser.add_argument('--preprocess', action='store_true',
                       help='Apply preprocessing pipeline')
    parser.add_argument('--period', type=float, metavar='P',
                       help='Period for folding light curve')
    parser.add_argument('--epoch', type=float, metavar='T0', default=0,
                       help='Epoch time for folding (default: 0)')
    
    # Analysis options
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    parser.add_argument('--extract', action='store_true',
                       help='Extract features')
    parser.add_argument('--prune', action='store_true',
                       help='Interactive feature pruning')
    
    # Advanced options
    parser.add_argument('--features-file', type=str, metavar='FILE',
                       help='Load/save selected features list')
    parser.add_argument('--report', action='store_true',
                       help='Generate feature report')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for plots and reports')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Main workflow
    try:
        lc_data = None
        features_df = None
        selected_features = None
        
        # Load light curve
        if args.load:
            print(f"Loading light curve from {args.load}")
            lc_data = load_npz_curve(args.load)
            
            # Apply preprocessing if requested
            if args.preprocess:
                print("Applying preprocessing...")
                lc_data = preprocess_lightcurve(
                    lc_data, 
                    period=args.period, 
                    epoch_time=args.epoch,
                    apply_preprocessing=True
                )
            else:
                print("Skipping preprocessing (use --preprocess to enable)")
        
        # Visualization
        if args.visualize and lc_data is not None:
            print("Creating visualizations...")
            
            # Basic light curve plot
            fig, ax = plot_lightcurve(
                lc_data['time'], 
                lc_data['flux'], 
                lc_data['flux_err'],
                save_path=os.path.join(args.output_dir, 'lightcurve.png')
            )
            
            # Folded curve if period provided
            if args.period:
                fig, axes = plot_folded_curve(
                    lc_data['time'],
                    lc_data['flux'],
                    args.period,
                    epoch=args.epoch,
                    flux_err=lc_data['flux_err'],
                    save_path=os.path.join(args.output_dir, 'folded_curve.png')
                )
            
            # Comprehensive analysis plot
            fig, axes = plot_comprehensive_analysis(
                lc_data,
                period=args.period,
                save_path=os.path.join(args.output_dir, 'comprehensive_analysis.png')
            )
            
            print(f"Plots saved to {args.output_dir}/")
        
        # Feature extraction
        if args.extract and lc_data is not None:
            print("Extracting features...")
            features_df = extract_features(
                lc_data['time'],
                lc_data['flux'], 
                lc_data['flux_err']
            )
            
            print(f"Extracted {len(features_df.columns)} features")
            
            # Create feature distribution plots
            if not features_df.empty:
                plot_feature_distribution(
                    features_df,
                    save_path=os.path.join(args.output_dir, 'feature_distributions.png')
                )
        
        # Feature pruning
        if args.prune and features_df is not None:
            print("Starting interactive feature pruning...")
            
            # Load existing selection if file provided
            if args.features_file and os.path.exists(args.features_file):
                selected_features = load_selected_features(args.features_file)
                print(f"Loaded {len(selected_features)} pre-selected features")
            
            # Interactive selection
            selected_features = interactive_feature_selection(features_df)
            
            # Save selection if file provided
            if args.features_file and selected_features:
                save_selected_features(selected_features, args.features_file)
            
            # Create pruned dataframe
            if selected_features:
                features_df = manual_prune(features_df, selected_features)
        
        # Generate report
        if args.report and features_df is not None:
            print("Generating feature report...")
            create_feature_report(
                features_df,
                selected_features=selected_features,
                save_path=os.path.join(args.output_dir, 'feature_report.md')
            )
        
        # Save results
        if args.save and features_df is not None:
            print(f"Saving features to {args.save}")
            features_df.to_csv(args.save, index=False)
            print(f"Saved {len(features_df.columns)} features for {len(features_df)} objects")
        
        # Summary
        print("\n=== Summary ===")
        if lc_data:
            print(f"Light curve: {len(lc_data['time'])} data points")
            print(f"Time span: {np.max(lc_data['time']) - np.min(lc_data['time']):.2f}")
        if features_df is not None:
            print(f"Features: {len(features_df.columns)} extracted")
        if selected_features:
            print(f"Selected: {len(selected_features)} features")
        
        print("Analysis complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


def run_example_workflow():
    """Run example workflow for demonstration."""
    print("=== Light Curve Analysis Example ===\n")
    
    # This would normally load a real .npz file
    print("This is an example workflow.")
    print("To run with real data, use:")
    print("  python main.py --load your_data.npz --extract --visualize")
    print("")
    
    # Create synthetic data for demonstration
    print("Creating synthetic light curve data...")
    np.random.seed(42)
    time = np.linspace(0, 100, 1000)
    
    # Synthetic transit signal
    period = 10.0
    transit_depth = 0.02
    transit_duration = 0.5
    
    flux = np.ones_like(time)
    phase = (time / period) % 1.0
    
    # Add transits
    transit_mask = np.abs(phase - 0.5) < (transit_duration / period / 2)
    flux[transit_mask] -= transit_depth
    
    # Add noise
    flux += np.random.normal(0, 0.005, len(flux))
    flux_err = np.full_like(flux, 0.005)
    
    # Create light curve dictionary
    lc_data = {
        'time': time,
        'flux': flux,
        'flux_err': flux_err
    }
    
    print("Synthetic light curve created with:")
    print(f"  - {len(time)} data points")
    print(f"  - Period: {period} time units")
    print(f"  - Transit depth: {transit_depth}")
    
    # Extract features
    print("\nExtracting features...")
    features_df = extract_features(time, flux, flux_err)
    print(f"Extracted {len(features_df.columns)} features")
    
    # Show some example features
    print("\nExample features:")
    example_features = ['mean', 'std', 'amplitude', 'ls_peak_power', 'transit_ingress_slope']
    for feature in example_features:
        if feature in features_df.columns:
            value = features_df[feature].iloc[0]
            print(f"  {feature}: {value:.4f}")
    
    # Create visualizations
    print("\nCreating plots...")
    os.makedirs('example_output', exist_ok=True)
    
    plot_lightcurve(time, flux, flux_err, 
                   title="Synthetic Light Curve with Transits",
                   save_path='example_output/synthetic_lightcurve.png')
    
    plot_folded_curve(time, flux, period, flux_err=flux_err,
                     title="Folded Synthetic Light Curve",
                     save_path='example_output/synthetic_folded.png')
    
    plot_feature_distribution(features_df,
                             save_path='example_output/synthetic_features.png')
    
    # Save features
    features_df.to_csv('example_output/synthetic_features.csv', index=False)
    
    print(f"\nExample complete! Check 'example_output/' for results.")
    print("Files created:")
    print("  - synthetic_lightcurve.png")
    print("  - synthetic_folded.png") 
    print("  - synthetic_features.png")
    print("  - synthetic_features.csv")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        # No arguments provided, run example
        run_example_workflow()
    else:
        # Run with provided arguments
        sys.exit(main())