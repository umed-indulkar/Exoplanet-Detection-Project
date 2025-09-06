#!/usr/bin/env python3
"""
Batch Processing Script for Light Curves

Process multiple .npz light curve files in parallel and combine results.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

sys.path.append('src')
from src.data_loader import load_npz_curve, preprocess_lightcurve
from src.feature_extraction import extract_features
from src.visualization import plot_lightcurve, plot_folded_curve


def process_single_file(file_path, preprocess=True, period=None, epoch=0, output_dir=None):
    """
    Process a single .npz file and return features.
    
    Parameters:
    -----------
    file_path : str
        Path to .npz file
    preprocess : bool
        Whether to apply preprocessing
    period : float, optional
        Period for folding
    epoch : float
        Epoch time for folding
    output_dir : str, optional
        Directory to save plots
        
    Returns:
    --------
    dict
        Results including features and metadata
    """
    try:
        print(f"Processing: {os.path.basename(file_path)}")
        
        # Load light curve
        lc_data = load_npz_curve(file_path)
        
        # Apply preprocessing if requested
        if preprocess:
            lc_data = preprocess_lightcurve(
                lc_data, 
                period=period, 
                epoch_time=epoch,
                apply_preprocessing=True
            )
        
        # Extract features
        features_df = extract_features(
            lc_data['time'],
            lc_data['flux'], 
            lc_data['flux_err']
        )
        
        # Add metadata
        features_df['file_name'] = os.path.basename(file_path)
        features_df['file_path'] = file_path
        features_df['n_original_points'] = len(lc_data['time'])
        
        # Save plots if output directory specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Basic plot
            plot_lightcurve(
                lc_data['time'], 
                lc_data['flux'], 
                lc_data['flux_err'],
                title=f"Light Curve: {base_name}",
                save_path=os.path.join(output_dir, f"{base_name}_lightcurve.png")
            )
            
            # Folded plot if period provided
            if period:
                plot_folded_curve(
                    lc_data['time'],
                    lc_data['flux'],
                    period,
                    epoch=epoch,
                    flux_err=lc_data['flux_err'],
                    title=f"Folded: {base_name} (P={period})",
                    save_path=os.path.join(output_dir, f"{base_name}_folded.png")
                )
        
        result = {
            'success': True,
            'file_path': file_path,
            'features': features_df,
            'n_points': len(lc_data['time']),
            'error': None
        }
        
        print(f"✓ Completed: {os.path.basename(file_path)} ({len(lc_data['time'])} points, {len(features_df.columns)-3} features)")
        return result
        
    except Exception as e:
        error_msg = f"Error processing {file_path}: {str(e)}"
        print(f"✗ Failed: {os.path.basename(file_path)} - {str(e)}")
        
        result = {
            'success': False,
            'file_path': file_path,
            'features': None,
            'n_points': 0,
            'error': error_msg
        }
        return result


def find_npz_files(input_path):
    """
    Find all .npz files in a directory or return single file.
    
    Parameters:
    -----------
    input_path : str
        Path to directory or single .npz file
        
    Returns:
    --------
    list
        List of .npz file paths
    """
    if os.path.isfile(input_path):
        if input_path.endswith('.npz'):
            return [input_path]
        else:
            raise ValueError(f"File {input_path} is not a .npz file")
    
    elif os.path.isdir(input_path):
        # Find all .npz files recursively
        npz_files = []
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if file.endswith('.npz'):
                    npz_files.append(os.path.join(root, file))
        
        if not npz_files:
            raise ValueError(f"No .npz files found in directory {input_path}")
        
        return sorted(npz_files)
    
    else:
        raise ValueError(f"Path {input_path} does not exist")


def main():
    parser = argparse.ArgumentParser(
        description='Batch process multiple light curve .npz files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all .npz files in a directory
  python batch_process.py --input data/ --output batch_results.csv
  
  # Process with preprocessing and folding
  python batch_process.py --input data/ --preprocess --period 2.5 --output results.csv
  
  # Process single file
  python batch_process.py --input single_file.npz --output features.csv
  
  # Process in parallel with plots
  python batch_process.py --input data/ --output results.csv --plots plot_output/ --workers 4
        """
    )
    
    # Input/Output
    parser.add_argument('--input', '-i', required=True,
                       help='Input directory containing .npz files or single .npz file')
    parser.add_argument('--output', '-o', required=True,
                       help='Output CSV file for combined features')
    
    # Processing options
    parser.add_argument('--preprocess', action='store_true',
                       help='Apply preprocessing pipeline')
    parser.add_argument('--period', type=float,
                       help='Period for folding light curves')
    parser.add_argument('--epoch', type=float, default=0,
                       help='Epoch time for folding (default: 0)')
    
    # Performance options
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of parallel workers (default: 1)')
    parser.add_argument('--plots', type=str,
                       help='Directory to save plots (optional)')
    
    # Filtering options
    parser.add_argument('--limit', type=int,
                       help='Limit number of files to process (for testing)')
    parser.add_argument('--pattern', type=str,
                       help='Filename pattern to match (e.g., "*transit*")')
    
    args = parser.parse_args()
    
    try:
        # Find input files
        print(f"Searching for .npz files in: {args.input}")
        npz_files = find_npz_files(args.input)
        
        # Apply pattern filter if specified
        if args.pattern:
            import fnmatch
            npz_files = [f for f in npz_files if fnmatch.fnmatch(os.path.basename(f), args.pattern)]
            print(f"Applied pattern '{args.pattern}': {len(npz_files)} files match")
        
        # Apply limit if specified
        if args.limit:
            npz_files = npz_files[:args.limit]
            print(f"Limited to first {args.limit} files")
        
        print(f"Found {len(npz_files)} .npz files to process")
        
        if len(npz_files) == 0:
            print("No files to process!")
            return
        
        # Process files
        print(f"\nStarting batch processing with {args.workers} workers...")
        
        all_features = []
        failed_files = []
        
        if args.workers == 1:
            # Sequential processing
            for file_path in npz_files:
                result = process_single_file(
                    file_path, 
                    preprocess=args.preprocess,
                    period=args.period,
                    epoch=args.epoch,
                    output_dir=args.plots
                )
                
                if result['success']:
                    all_features.append(result['features'])
                else:
                    failed_files.append(result)
        
        else:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                # Submit all jobs
                future_to_file = {
                    executor.submit(
                        process_single_file, 
                        file_path,
                        args.preprocess,
                        args.period,
                        args.epoch,
                        args.plots
                    ): file_path for file_path in npz_files
                }
                
                # Collect results
                for future in as_completed(future_to_file):
                    result = future.result()
                    
                    if result['success']:
                        all_features.append(result['features'])
                    else:
                        failed_files.append(result)
        
        # Combine results
        print(f"\nCombining results...")
        if all_features:
            combined_df = pd.concat(all_features, ignore_index=True)
            combined_df.to_csv(args.output, index=False)
            
            print(f"✓ Successfully processed {len(all_features)} files")
            print(f"✓ Combined features saved to: {args.output}")
            print(f"✓ Dataset shape: {combined_df.shape}")
            print(f"✓ Features per object: {combined_df.shape[1]-3}")  # Subtract metadata columns
        else:
            print("✗ No files were successfully processed!")
        
        # Report failures
        if failed_files:
            print(f"\n✗ Failed to process {len(failed_files)} files:")
            for failed in failed_files:
                print(f"  - {os.path.basename(failed['file_path'])}: {failed['error']}")
            
            # Save failure log
            failure_log = args.output.replace('.csv', '_failures.txt')
            with open(failure_log, 'w') as f:
                for failed in failed_files:
                    f.write(f"{failed['file_path']}: {failed['error']}\n")
            print(f"✓ Failure log saved to: {failure_log}")
        
        print(f"\nBatch processing complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())