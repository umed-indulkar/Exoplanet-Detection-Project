"""
Modular CLI - Each team works independently
==========================================

CLI now imports from separate modules:
- preprocessing/: Data loading and cleaning
- feature_extraction/: Feature extraction
- models/: ML models (baseline and siamese)
- pipeline/: Integration and workflows
"""

import argparse
import glob
import os
from typing import List
from joblib import Parallel, delayed
import pandas as pd

# Import from separate modules (each team's work)
from ..preprocessing import load_lightcurve, load_batch_lightcurves, preprocess_lightcurve
from ..feature_extraction import extract_basic_features, extract_tsfresh_features, _HAS_TSFRESH
from ..models import train_baseline, evaluate_baseline, train_siamese_from_csv, evaluate_siamese_from_csv, _HAS_SIAMESE

# Optional TSFresh parameters
try:
    from tsfresh.feature_extraction import EfficientFCParameters, ComprehensiveFCParameters
    _HAS_TSFRESH_PARAMS = True
except Exception:
    _HAS_TSFRESH_PARAMS = False

def cmd_extract(args: argparse.Namespace) -> int:
    """Extract features from individual files"""
    inputs = []
    for pattern in args.input:
        inputs.extend(glob.glob(pattern))
    
    # Filter to supported files only
    supported_ext = {'.npz', '.csv', '.fits', '.fit'}
    inputs = [p for p in inputs if os.path.isfile(p) and os.path.splitext(p)[1].lower() in supported_ext]
    
    if not inputs:
        print("No input files matched.")
        return 1

    failures = []

    def process_path(path: str):
        try:
            # Preprocessing team's work
            lc = load_lightcurve(path)
            lc_clean = preprocess_lightcurve(lc)
            
            # Feature extraction team's work
            if args.tier == 'basic':
                feats = extract_basic_features(lc_clean, verbose=False)
            elif args.tier == 'tsfresh':
                if not _HAS_TSFRESH:
                    return (path, None, "tsfresh not available. Install: pip install tsfresh statsmodels")
                preset = (args.tsfresh_params or 'efficient').lower()
                if preset not in ('efficient','comprehensive'):
                    return (path, None, f"Unknown tsfresh preset: {preset}")
                if not _HAS_TSFRESH_PARAMS and preset == 'comprehensive':
                    return (path, None, "Comprehensive parameters unavailable. Install tsfresh.")
                default_fc_parameters = None
                if _HAS_TSFRESH_PARAMS:
                    default_fc_parameters = EfficientFCParameters() if preset == 'efficient' else ComprehensiveFCParameters()
                feats = extract_tsfresh_features(
                    lc_clean,
                    default_fc_parameters=default_fc_parameters,
                    n_jobs=getattr(args, 'workers', 0)
                )
            else:
                return (path, None, f"Unknown tier: {args.tier}")
            feats['source'] = path
            return (path, feats, None)
        except Exception as e:
            return (path, None, str(e))

    results = Parallel(n_jobs=args.file_workers)(
        delayed(process_path)(path) for path in inputs
    )
    
    # Collect successful results
    all_features = []
    for path, feats, err in results:
        if err:
            failures.append((path, err))
        else:
            all_features.append(feats)
    
    if failures:
        print(f"Failed to process {len(failures)} files:")
        for path, err in failures[:5]:
            print(f"  {path}: {err}")
        if len(failures) > 5:
            print(f"  ... and {len(failures)-5} more")
    
    if not all_features:
        print("No files processed successfully.")
        return 1
    
    # Save results
    df = pd.DataFrame(all_features)
    df.to_csv(args.output, index=False)
    print(f"Extracted features for {len(all_features)} files → {args.output}")
    return 0

def cmd_batch(args: argparse.Namespace) -> int:
    """Batch process a directory of files"""
    pattern = os.path.join(args.input, args.pattern)
    files = glob.glob(pattern)
    if not files:
        print(f"No files found matching {pattern}")
        return 1
    
    # Create temporary args for extract
    extract_args = argparse.Namespace()
    extract_args.input = files
    extract_args.output = args.output
    extract_args.tier = getattr(args, 'tier', 'tsfresh')
    extract_args.tsfresh_params = getattr(args, 'tsfresh_params', 'efficient')
    extract_args.workers = getattr(args, 'workers', 0)
    extract_args.file_workers = getattr(args, 'file_workers', 1)
    
    return cmd_extract(extract_args)

def cmd_train(args: argparse.Namespace) -> int:
    """Train baseline ML models"""
    # ML team's work
    model = train_baseline(
        args.features,
        target_col=args.target,
        model_type=args.model,
        output_path=args.output
    )
    print(f"Trained {args.model} model: {args.output}")
    return 0

def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate baseline ML models"""
    # ML team's work
    metrics = evaluate_baseline(
        args.model,
        args.features,
        target_col=args.target
    )
    print("Metrics:", metrics)
    return 0

def cmd_train_siamese(args: argparse.Namespace) -> int:
    """Train Siamese Neural Network"""
    if not _HAS_SIAMESE:
        print("Siamese trainer not available. Ensure torch is installed")
        return 2
    
    # ML team's work
    res = train_siamese_from_csv(
        args.features,
        target_col=args.target,
        embedding_dim=args.embedding,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        device=args.device,
        output_path=args.output,
        seed=args.seed,
    )
    print(f"Saved Siamese model: {res.model_path}")
    print("Metrics:", res.metrics)
    return 0

def cmd_evaluate_siamese(args: argparse.Namespace) -> int:
    """Evaluate Siamese Neural Network"""
    if not _HAS_SIAMESE:
        print("Siamese evaluator not available. Ensure torch is installed")
        return 2
    
    # ML team's work
    metrics = evaluate_siamese_from_csv(args.model, args.features, target_col=args.target, device=args.device)
    print("Metrics:", metrics)
    return 0

def cmd_organize(args: argparse.Namespace) -> int:
    """Organize dataset - Pipeline team's work"""
    from ..pipeline.data_organizer import organize_dataset
    organize_dataset()
    return 0

def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for modular CLI"""
    p = argparse.ArgumentParser(prog='exodet', description='Exoplanet Detection CLI - Modular')
    sub = p.add_subparsers(dest='cmd')

    # Feature extraction commands (Feature Extraction Team)
    sp = sub.add_parser('extract', help='Extract features from files')
    sp.add_argument('--input', nargs='+', required=True, help='Glob patterns')
    sp.add_argument('--output', required=True, help='Output CSV path')
    sp.add_argument('--tier', choices=['basic','tsfresh'], default='tsfresh')
    sp.add_argument('--tsfresh-params', choices=['efficient','comprehensive'], default='efficient')
    sp.add_argument('--workers', type=int, default=0, help='TSFresh workers')
    sp.add_argument('--file-workers', type=int, default=1, help='Parallel files')
    sp.set_defaults(func=cmd_extract)

    sp = sub.add_parser('batch', help='Batch process directory')
    sp.add_argument('--input', required=True, help='Directory path')
    sp.add_argument('--pattern', default='*.npz')
    sp.add_argument('--output', required=True)
    sp.add_argument('--tier', choices=['basic','tsfresh'], default='tsfresh')
    sp.add_argument('--workers', type=int, default=0)
    sp.add_argument('--file-workers', type=int, default=1)
    sp.set_defaults(func=cmd_batch)

    # ML model commands (ML Team)
    sp = sub.add_parser('train', help='Train baseline model')
    sp.add_argument('--features', required=True)
    sp.add_argument('--target', default='label')
    sp.add_argument('--model', choices=['logreg','rf'], default='rf')
    sp.add_argument('--output', required=True)
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser('evaluate', help='Evaluate baseline model')
    sp.add_argument('--model', required=True)
    sp.add_argument('--features', required=True)
    sp.add_argument('--target', default='label')
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser('train-siamese', help='Train Siamese model')
    sp.add_argument('--features', required=True)
    sp.add_argument('--target', default='label')
    sp.add_argument('--embedding', type=int, default=32)
    sp.add_argument('--epochs', type=int, default=10)
    sp.add_argument('--batch-size', type=int, default=256)
    sp.add_argument('--lr', type=float, default=1e-3)
    sp.add_argument('--val-split', type=float, default=0.2)
    sp.add_argument('--device', default='auto', choices=['auto','cpu','cuda'])
    sp.add_argument('--output', required=True)
    sp.add_argument('--seed', type=int, default=42)
    sp.set_defaults(func=cmd_train_siamese)

    sp = sub.add_parser('evaluate-siamese', help='Evaluate Siamese model')
    sp.add_argument('--model', required=True)
    sp.add_argument('--features', required=True)
    sp.add_argument('--target', default='label')
    sp.add_argument('--device', default='auto', choices=['auto','cpu','cuda'])
    sp.set_defaults(func=cmd_evaluate_siamese)

    # Pipeline commands (Pipeline Team)
    sp = sub.add_parser('organize', help='Organize dataset')
    sp.set_defaults(func=cmd_organize)

    return p

def main(argv=None) -> int:
    """Main entry point"""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, 'func'):
        parser.print_help()
        return 1
    return args.func(args)

if __name__ == '__main__':
    raise SystemExit(main())
