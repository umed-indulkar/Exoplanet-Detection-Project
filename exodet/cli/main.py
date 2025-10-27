import argparse
import glob
import os
import sys
import pandas as pd

from .. import load_lightcurve, load_batch_lightcurves, preprocess_lightcurve
from ..features import extract_basic_features

# Optional imports
try:
    from ..features.tsfresh_extractor import extract_tsfresh_features
    _HAS_TSFRESH = True
except Exception:
    _HAS_TSFRESH = False

# Optional ML
try:
    from ..ml.models import load_model, save_model, train_baseline, evaluate_baseline, predict_on_features
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

# Optional Siamese
try:
    from ..ml.siamese import train_siamese_from_csv, evaluate_siamese_from_csv
    _HAS_SIAMESE = True
except Exception:
    _HAS_SIAMESE = False


def cmd_extract(args: argparse.Namespace) -> int:
    inputs = []
    for pattern in args.input:
        inputs.extend(glob.glob(pattern))
    if not inputs:
        print("No input files matched.")
        return 1

    rows = []
    for path in inputs:
        lc = load_lightcurve(path)
        lc_clean = preprocess_lightcurve(lc)
        if args.tier == 'basic':
            feats = extract_basic_features(lc_clean, verbose=False)
        elif args.tier == 'tsfresh':
            if not _HAS_TSFRESH:
                print("tsfresh not available. Install: pip install tsfresh statsmodels")
                return 2
            feats = extract_tsfresh_features(lc_clean)
        else:
            print(f"Unknown tier: {args.tier}")
            return 3
        feats['source'] = path
        rows.append(feats)

    df = pd.concat(rows, ignore_index=True)
    df.to_csv(args.output, index=False)
    print(f"Saved: {args.output} ({df.shape})")
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    curves = load_batch_lightcurves(args.input, pattern=args.pattern)
    if not curves:
        print("No curves found.")
        return 1
    rows = []
    for lc in curves:
        lc_clean = preprocess_lightcurve(lc)
        feats = extract_basic_features(lc_clean, verbose=False)
        feats['source'] = lc.source_file
        rows.append(feats)
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(args.output, index=False)
    print(f"Saved: {args.output} ({df.shape})")
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.input)
    print(df.describe(include='all'))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    if not _HAS_SKLEARN:
        print("scikit-learn not available. Install: pip install scikit-learn joblib")
        return 2
    df = pd.read_csv(args.features)
    if args.target not in df.columns:
        print(f"Target column '{args.target}' not found in features.")
        return 3
    model, metrics = train_baseline(df, target_col=args.target, model_type=args.model)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    save_model(model, args.output)
    print(f"Saved model to: {args.output}")
    print("Metrics:", metrics)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    if not _HAS_SKLEARN:
        print("scikit-learn not available. Install: pip install scikit-learn joblib")
        return 2
    df = pd.read_csv(args.features)
    model = load_model(args.model)
    metrics = evaluate_baseline(model, df, target_col=args.target)
    print("Metrics:", metrics)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    if not _HAS_SKLEARN:
        print("scikit-learn not available. Install: pip install scikit-learn joblib")
        return 2
    model = load_model(args.model)
    # If user provided features CSV, use it; otherwise compute basic features from curves
    if args.features:
        df = pd.read_csv(args.features)
        preds = predict_on_features(model, df)
        out = df.copy()
        out['prediction'] = preds
        out.to_csv(args.output, index=False)
        print(f"Saved predictions: {args.output} ({out.shape})")
        return 0

    inputs = []
    for pattern in args.input:
        inputs.extend(glob.glob(pattern))
    if not inputs:
        print("No input files matched.")
        return 1
    rows = []
    for path in inputs:
        lc = load_lightcurve(path)
        lc_clean = preprocess_lightcurve(lc)
        feats = extract_basic_features(lc_clean, verbose=False)
        feats['source'] = path
        rows.append(feats)
    df = pd.concat(rows, ignore_index=True)
    preds = predict_on_features(model, df)
    df['prediction'] = preds
    df.to_csv(args.output, index=False)
    print(f"Saved predictions: {args.output} ({df.shape})")
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    try:
        import streamlit.web.cli as stcli
        from pathlib import Path
        app_path = Path(__file__).resolve().parent.parent / 'dashboard' / 'app.py'
        return stcli.main_run([str(app_path)])
    except Exception as e:
        print(f"Failed to launch dashboard: {e}")
        print("Install with: pip install streamlit")
        return 1


def cmd_train_siamese(args: argparse.Namespace) -> int:
    if not _HAS_SIAMESE:
        print("Siamese trainer not available. Ensure torch is installed: pip install torch --index-url https://download.pytorch.org/whl/cpu")
        return 2
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
    if not _HAS_SIAMESE:
        print("Siamese evaluator not available. Ensure torch is installed: pip install torch --index-url https://download.pytorch.org/whl/cpu")
        return 2
    metrics = evaluate_siamese_from_csv(args.model, args.features, target_col=args.target, device=args.device)
    print("Metrics:", metrics)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='exodet', description='Exoplanet Detection CLI')
    sub = p.add_subparsers(dest='cmd')

    sp = sub.add_parser('extract', help='Extract features from files')
    sp.add_argument('--input', nargs='+', required=True, help='Glob patterns')
    sp.add_argument('--output', required=True, help='Output CSV path')
    sp.add_argument('--tier', choices=['basic','tsfresh'], default='basic')
    sp.set_defaults(func=cmd_extract)

    sp = sub.add_parser('batch', help='Batch process a directory')
    sp.add_argument('--input', required=True, help='Directory path')
    sp.add_argument('--pattern', default='*.npz', help='Glob pattern')
    sp.add_argument('--output', required=True)
    sp.set_defaults(func=cmd_batch)

    sp = sub.add_parser('summary', help='Summarize a features CSV')
    sp.add_argument('--input', required=True)
    sp.set_defaults(func=cmd_summary)

    sp = sub.add_parser('train', help='Train a baseline model on features CSV')
    sp.add_argument('--features', required=True)
    sp.add_argument('--target', default='label')
    sp.add_argument('--model', choices=['logreg','rf'], default='rf')
    sp.add_argument('--output', required=True)
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser('evaluate', help='Evaluate a saved model on features CSV')
    sp.add_argument('--model', required=True)
    sp.add_argument('--features', required=True)
    sp.add_argument('--target', default='label')
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser('predict', help='Predict on features or raw curves')
    sp.add_argument('--model', required=True)
    sp.add_argument('--features', help='Optional features CSV')
    sp.add_argument('--input', nargs='+', help='Glob patterns for raw curves')
    sp.add_argument('--output', required=True)
    sp.set_defaults(func=cmd_predict)

    sp = sub.add_parser('dashboard', help='Launch Streamlit dashboard')
    sp.set_defaults(func=cmd_dashboard)

    sp = sub.add_parser('train-siamese', help='Train Siamese model on features CSV')
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

    sp = sub.add_parser('evaluate-siamese', help='Evaluate Siamese model on features CSV')
    sp.add_argument('--model', required=True)
    sp.add_argument('--features', required=True)
    sp.add_argument('--target', default='label')
    sp.add_argument('--device', default='auto', choices=['auto','cpu','cuda'])
    sp.set_defaults(func=cmd_evaluate_siamese)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, 'func'):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
