"""
Feature Pruning Module

This module provides utilities for manually selecting and pruning features
from the comprehensive feature set extracted from light curves.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def manual_prune(features_df, selected_features=None):
    """
    Manually prune features to keep only selected ones.
    
    Parameters:
    -----------
    features_df : pandas.DataFrame
        DataFrame containing all extracted features
    selected_features : list, optional
        List of feature names to keep. If None, returns all features.
        
    Returns:
    --------
    pandas.DataFrame
        Pruned DataFrame with only selected features
    """
    if features_df.empty:
        print("Warning: Input DataFrame is empty")
        return features_df.copy()
    
    if selected_features is None:
        print("No features selected, returning all features")
        return features_df.copy()
    
    # Validate selected features exist in DataFrame
    available_features = set(features_df.columns)
    selected_features = [f for f in selected_features if f in available_features]
    missing_features = set(selected_features) - available_features
    
    if missing_features:
        print(f"Warning: {len(missing_features)} selected features not found: {list(missing_features)[:5]}...")
    
    if not selected_features:
        print("Warning: No valid features selected, returning empty DataFrame")
        return pd.DataFrame()
    
    pruned_df = features_df[selected_features].copy()
    print(f"Pruned from {len(features_df.columns)} to {len(selected_features)} features")
    
    return pruned_df


def interactive_feature_selection(features_df, max_display=50):
    """
    Interactive feature selection interface for command line.
    
    Parameters:
    -----------
    features_df : pandas.DataFrame
        DataFrame containing all extracted features
    max_display : int, default=50
        Maximum number of features to display at once
        
    Returns:
    --------
    list
        List of selected feature names
    """
    if features_df.empty:
        print("No features available for selection")
        return []
    
    all_features = list(features_df.columns)
    selected_features = []
    
    print(f"\n=== Interactive Feature Selection ===")
    print(f"Total features available: {len(all_features)}")
    
    while True:
        print(f"\nOptions:")
        print("1. View all features")
        print("2. View features by category")
        print("3. Add features by name/pattern")
        print("4. Remove features") 
        print("5. View selected features")
        print("6. Clear selection")
        print("7. Finish selection")
        
        choice = input("\nEnter choice (1-7): ").strip()
        
        if choice == '1':
            view_all_features(features_df, max_display)
            
        elif choice == '2':
            view_features_by_category(features_df)
            
        elif choice == '3':
            new_features = add_features_interactive(all_features, selected_features)
            selected_features.extend(new_features)
            selected_features = list(set(selected_features))  # Remove duplicates
            
        elif choice == '4':
            selected_features = remove_features_interactive(selected_features)
            
        elif choice == '5':
            view_selected_features(selected_features, features_df)
            
        elif choice == '6':
            selected_features = []
            print("Selection cleared")
            
        elif choice == '7':
            break
            
        else:
            print("Invalid choice. Please enter 1-7.")
    
    print(f"\nFinal selection: {len(selected_features)} features")
    return selected_features


def view_all_features(features_df, max_display=50):
    """Display all features with their statistics."""
    features = list(features_df.columns)
    
    print(f"\n=== All Features ({len(features)} total) ===")
    
    # Show in chunks
    for i in range(0, len(features), max_display):
        chunk = features[i:i+max_display]
        print(f"\nFeatures {i+1}-{min(i+max_display, len(features))}:")
        
        for j, feature in enumerate(chunk, i+1):
            value = features_df[feature].iloc[0] if not features_df.empty else 'N/A'
            if isinstance(value, float):
                value_str = f"{value:.4f}" if not np.isnan(value) else "NaN"
            else:
                value_str = str(value)
            print(f"{j:3d}. {feature:<40} = {value_str}")
        
        if i + max_display < len(features):
            cont = input(f"\nPress Enter to continue or 'q' to stop: ").strip()
            if cont.lower() == 'q':
                break


def view_features_by_category(features_df):
    """Group and display features by category."""
    features = list(features_df.columns)
    
    # Define feature categories based on prefixes/patterns
    categories = {
        'Basic Statistics': [f for f in features if any(x in f for x in ['mean', 'median', 'std', 'var', 'min', 'max', 'percentile'])],
        'Time Domain': [f for f in features if any(x in f for x in ['time', 'autocorr', 'diff', 'trend', 'cadence'])],
        'Frequency Domain': [f for f in features if any(x in f for x in ['spectral', 'fft', 'freq', 'power', 'ls_'])],
        'Variability': [f for f in features if any(x in f for x in ['amplitude', 'rms', 'stetson', 'von_neumann'])],
        'Shape/Morphology': [f for f in features if any(x in f for x in ['asymmetry', 'run', 'frac_', 'kurtosis', 'skew'])],
        'Transit Features': [f for f in features if any(x in f for x in ['dip', 'transit', 'ingress', 'egress'])],
        'Error-based': [f for f in features if any(x in f for x in ['error', 'snr', 'chi2', 'weighted'])],
        'Astronomical': [f for f in features if any(x in f for x in ['mag', 'duty', 'concentration'])],
        'Other': []
    }
    
    # Categorize features
    categorized = set()
    for category, feature_list in categories.items():
        if category != 'Other':
            categorized.update(feature_list)
    
    # Put uncategorized features in 'Other'
    categories['Other'] = [f for f in features if f not in categorized]
    
    print(f"\n=== Features by Category ===")
    for category, feature_list in categories.items():
        if feature_list:
            print(f"\n{category} ({len(feature_list)} features):")
            for i, feature in enumerate(feature_list[:20], 1):  # Show first 20
                value = features_df[feature].iloc[0] if not features_df.empty else 'N/A'
                if isinstance(value, float):
                    value_str = f"{value:.4f}" if not np.isnan(value) else "NaN"
                else:
                    value_str = str(value)
                print(f"  {i:2d}. {feature:<35} = {value_str}")
            
            if len(feature_list) > 20:
                print(f"     ... and {len(feature_list)-20} more")


def add_features_interactive(all_features, selected_features):
    """Interactive feature addition."""
    new_features = []
    
    print(f"\n=== Add Features ===")
    print("Options:")
    print("1. Add by exact name")
    print("2. Add by pattern/wildcard")
    print("3. Add by category")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == '1':
        print("Enter feature names (comma-separated):")
        names = input().strip().split(',')
        for name in names:
            name = name.strip()
            if name in all_features and name not in selected_features:
                new_features.append(name)
                print(f"Added: {name}")
            elif name in selected_features:
                print(f"Already selected: {name}")
            else:
                print(f"Not found: {name}")
                
    elif choice == '2':
        pattern = input("Enter pattern (e.g., 'spectral', 'mean', etc.): ").strip()
        matching = [f for f in all_features if pattern.lower() in f.lower() and f not in selected_features]
        
        if matching:
            print(f"Found {len(matching)} matching features:")
            for i, feature in enumerate(matching[:20], 1):
                print(f"  {i}. {feature}")
            if len(matching) > 20:
                print(f"  ... and {len(matching)-20} more")
            
            response = input(f"Add all {len(matching)} features? (y/n): ").strip()
            if response.lower() == 'y':
                new_features.extend(matching)
                print(f"Added {len(matching)} features")
        else:
            print("No matching features found")
            
    elif choice == '3':
        # Add by category (simplified)
        category_patterns = {
            '1': ('Basic Statistics', ['mean', 'median', 'std', 'var', 'percentile']),
            '2': ('Time Domain', ['time', 'autocorr', 'diff', 'trend']),
            '3': ('Frequency Domain', ['spectral', 'freq', 'power', 'ls_']),
            '4': ('Variability', ['amplitude', 'rms', 'stetson']),
            '5': ('Transit Features', ['dip', 'transit'])
        }
        
        print("Categories:")
        for key, (name, patterns) in category_patterns.items():
            print(f"  {key}. {name}")
        
        cat_choice = input("Enter category number: ").strip()
        if cat_choice in category_patterns:
            name, patterns = category_patterns[cat_choice]
            matching = [f for f in all_features 
                       if any(p in f for p in patterns) and f not in selected_features]
            
            if matching:
                new_features.extend(matching)
                print(f"Added {len(matching)} features from {name}")
            else:
                print(f"No features found for {name}")
    
    return new_features


def remove_features_interactive(selected_features):
    """Interactive feature removal."""
    if not selected_features:
        print("No features selected")
        return selected_features
    
    print(f"\n=== Remove Features ===")
    print("Current selection:")
    for i, feature in enumerate(selected_features, 1):
        print(f"  {i:3d}. {feature}")
    
    print("\nOptions:")
    print("1. Remove by number")
    print("2. Remove by pattern")
    print("3. Keep only by pattern")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == '1':
        indices_str = input("Enter numbers to remove (comma-separated): ").strip()
        try:
            indices = [int(x.strip()) - 1 for x in indices_str.split(',') if x.strip()]
            indices = [i for i in indices if 0 <= i < len(selected_features)]
            
            # Remove in reverse order to maintain indices
            for i in sorted(indices, reverse=True):
                removed = selected_features.pop(i)
                print(f"Removed: {removed}")
        except ValueError:
            print("Invalid input")
            
    elif choice == '2':
        pattern = input("Enter pattern to remove: ").strip()
        to_remove = [f for f in selected_features if pattern.lower() in f.lower()]
        
        if to_remove:
            print(f"Will remove {len(to_remove)} features matching '{pattern}'")
            for feature in to_remove:
                selected_features.remove(feature)
                print(f"Removed: {feature}")
        else:
            print("No matching features found")
            
    elif choice == '3':
        pattern = input("Enter pattern to keep: ").strip()
        to_keep = [f for f in selected_features if pattern.lower() in f.lower()]
        
        if to_keep:
            removed_count = len(selected_features) - len(to_keep)
            selected_features[:] = to_keep  # Modify in place
            print(f"Kept {len(to_keep)} features, removed {removed_count}")
        else:
            print("No matching features found")
    
    return selected_features


def view_selected_features(selected_features, features_df):
    """Display currently selected features."""
    if not selected_features:
        print("\nNo features currently selected")
        return
    
    print(f"\n=== Selected Features ({len(selected_features)}) ===")
    for i, feature in enumerate(selected_features, 1):
        if not features_df.empty and feature in features_df.columns:
            value = features_df[feature].iloc[0]
            if isinstance(value, float):
                value_str = f"{value:.4f}" if not np.isnan(value) else "NaN"
            else:
                value_str = str(value)
            print(f"  {i:3d}. {feature:<40} = {value_str}")
        else:
            print(f"  {i:3d}. {feature}")


def save_selected_features(selected_features, filename):
    """Save selected feature names to a file."""
    try:
        with open(filename, 'w') as f:
            for feature in selected_features:
                f.write(feature + '\n')
        print(f"Selected features saved to {filename}")
    except Exception as e:
        print(f"Error saving features: {e}")


def load_selected_features(filename):
    """Load selected feature names from a file."""
    try:
        with open(filename, 'r') as f:
            features = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(features)} features from {filename}")
        return features
    except Exception as e:
        print(f"Error loading features: {e}")
        return []


def analyze_feature_importance(features_df, target=None, method='correlation'):
    """
    Analyze feature importance for pruning guidance.
    
    Parameters:
    -----------
    features_df : pandas.DataFrame
        DataFrame containing features
    target : array-like, optional
        Target variable for supervised importance
    method : str, default='correlation'
        Method for importance analysis
        
    Returns:
    --------
    pandas.Series
        Feature importance scores
    """
    if features_df.empty:
        return pd.Series()
    
    numeric_features = features_df.select_dtypes(include=[np.number])
    
    if method == 'variance':
        # Features with higher variance are potentially more informative
        importance = numeric_features.var()
        
    elif method == 'correlation' and target is not None:
        # Correlation with target
        importance = numeric_features.corrwith(pd.Series(target))
        importance = importance.abs()
        
    else:
        # Default: coefficient of variation
        importance = numeric_features.std() / (numeric_features.mean().abs() + 1e-10)
    
    return importance.sort_values(ascending=False)


def create_feature_report(features_df, selected_features=None, save_path=None):
    """
    Create a comprehensive report of features.
    
    Parameters:
    -----------
    features_df : pandas.DataFrame
        DataFrame containing features
    selected_features : list, optional
        List of selected features to highlight
    save_path : str, optional
        Path to save the report
    """
    if features_df.empty:
        print("No features to report")
        return
    
    report = []
    report.append("# Feature Analysis Report\n")
    
    # Summary statistics
    numeric_features = features_df.select_dtypes(include=[np.number])
    report.append(f"Total features: {len(features_df.columns)}")
    report.append(f"Numeric features: {len(numeric_features.columns)}")
    if selected_features:
        report.append(f"Selected features: {len(selected_features)}")
    report.append("")
    
    # Feature statistics
    report.append("## Feature Statistics")
    report.append("| Feature | Value | Type |")
    report.append("|---------|-------|------|")
    
    for feature in features_df.columns[:100]:  # Limit to first 100
        value = features_df[feature].iloc[0]
        dtype = str(features_df[feature].dtype)
        
        if isinstance(value, float):
            value_str = f"{value:.4f}" if not np.isnan(value) else "NaN"
        else:
            value_str = str(value)[:20]
        
        selected_mark = " ✓" if selected_features and feature in selected_features else ""
        report.append(f"| {feature}{selected_mark} | {value_str} | {dtype} |")
    
    if len(features_df.columns) > 100:
        report.append(f"| ... and {len(features_df.columns) - 100} more features | | |")
    
    report_text = "\n".join(report)
    
    if save_path:
        try:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Feature report saved to {save_path}")
        except Exception as e:
            print(f"Error saving report: {e}")
    else:
        print(report_text)