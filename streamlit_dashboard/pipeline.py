"""
Data Processing Pipeline for Exoplanet Light Curve Analysis
RAW CSV → cleaning → normalization → binning → smoothing → outlier removal → features
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import skew, kurtosis, iqr
import warnings
warnings.filterwarnings('ignore')


class DataPipeline:
    """Full data processing pipeline for light curve analysis."""

    def __init__(self, n_bins=500, smooth_window=11, smooth_poly=3, sigma=3.0):
        self.n_bins = n_bins
        self.smooth_window = smooth_window
        self.smooth_poly = smooth_poly
        self.sigma = sigma
        self.log = []
        self.raw_flux = None
        self.processed_flux = None
        self.features = None
        self.time_axis = None

    # ── Load & Validate ──────────────────────────────────────────
    def load_data(self, uploaded_file):
        try:
            df = pd.read_csv(uploaded_file, comment='#', header='infer')
            # If the file has no header row (all-numeric first row), reload headerless
            if df.shape[1] > 10:
                first_row_numeric = all(
                    str(v).replace('.', '', 1).replace('-', '', 1).replace('e', '', 1)
                    .replace('+', '', 1).isdigit()
                    for v in df.columns[:5]
                )
                if first_row_numeric:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, comment='#', header=None)
        except Exception as e:
            raise ValueError(f"Failed to parse CSV: {e}")
        if df.empty:
            raise ValueError("CSV file is empty.")

        # ── Named 'flux' column ─────────────────────────────────
        flux_cols = [c for c in df.columns if isinstance(c, str) and 'flux' in c.lower()]
        if flux_cols:
            flux = pd.to_numeric(df[flux_cols[0]], errors='coerce').values.astype(float)
            self.log.append(f"Loaded column '{flux_cols[0]}': {len(flux)} points")

        # ── Wide-format (>10 cols = binned light curve) ─────────
        elif df.shape[1] > 10:
            # Detect format: kepid+label+flux  OR  label+flux  OR  pure flux
            flux_start = 0
            if df.shape[1] in [502, 501]:
                # kepid + label + 500 bins  OR  label + 500 bins
                flux_start = df.shape[1] - 500
                self.log.append(f"Wide-format detected: skipping {flux_start} metadata col(s)")

            # Coerce flux columns to numeric (handles stray strings like 'CANDIDATE')
            flux_df = df.iloc[:, flux_start:].apply(pd.to_numeric, errors='coerce')

            # Remove flat/dead rows (std == 0 or all NaN)
            row_std = flux_df.std(axis=1)
            all_nan = flux_df.isnull().all(axis=1)
            is_flat = (row_std == 0) | all_nan
            flat_count = is_flat.sum()
            if flat_count > 0:
                self.log.append(f"Removed {flat_count} flat/dead row(s)")
                flux_df = flux_df[~is_flat]
                df = df[~is_flat]

            if flux_df.empty:
                raise ValueError("All rows are flat/dead after cleaning.")

            # Use first valid row
            flux = flux_df.iloc[0].values.astype(float)
            self.log.append(f"Wide-format loaded: {len(flux)} time bins")
        else:
            raise ValueError(
                "CSV must contain a 'flux' column or be wide-format (>10 columns).\n"
                "Supported: Kepler pipeline CSVs, phase-folded binned CSVs, or raw light curves."
            )

        self.raw_flux = flux.copy()
        self.time_axis = np.arange(len(flux))
        return flux, df

    # ── Clean ────────────────────────────────────────────────────
    def clean(self, flux):
        flux = np.where(np.isinf(flux), np.nan, flux)
        mask = np.isnan(flux)
        if mask.any():
            valid = ~mask
            if valid.sum() > 2:
                flux[mask] = np.interp(np.where(mask)[0], np.where(valid)[0], flux[valid])
            else:
                flux[mask] = 0
        self.log.append(f"Cleaned {mask.sum()} invalid values")
        return flux

    # ── Normalize ────────────────────────────────────────────────
    def normalize(self, flux):
        med = np.median(flux)
        if med != 0:
            flux = flux / med
        else:
            s = np.std(flux)
            flux = (flux - np.mean(flux)) / s if s > 0 else flux
        self.log.append(f"Normalized (median={med:.6f})")
        return flux

    # ── Bin ───────────────────────────────────────────────────────
    def bin_data(self, flux):
        if len(flux) <= self.n_bins:
            self.log.append(f"No binning needed ({len(flux)} pts)")
            return flux
        bin_size = len(flux) / self.n_bins
        binned = np.array([np.mean(flux[int(i * bin_size):int((i + 1) * bin_size)]) for i in range(self.n_bins)])
        self.log.append(f"Binned {len(flux)} → {self.n_bins} points")
        return binned

    # ── Smooth ───────────────────────────────────────────────────
    def smooth(self, flux):
        w = min(self.smooth_window, len(flux))
        if w % 2 == 0:
            w -= 1
        w = max(w, self.smooth_poly + 2)
        if w % 2 == 0:
            w += 1
        try:
            smoothed = savgol_filter(flux, w, self.smooth_poly)
            self.log.append(f"Savitzky-Golay applied (window={w})")
        except Exception:
            smoothed = flux
            self.log.append("Smoothing skipped")
        return smoothed

    # ── Outlier Removal ──────────────────────────────────────────
    def remove_outliers(self, flux):
        med = np.median(flux)
        mad = np.median(np.abs(flux - med))
        if mad == 0:
            mad = np.std(flux)
        z = 0.6745 * (flux - med) / (mad if mad > 0 else 1)
        mask = np.abs(z) > self.sigma
        count = mask.sum()
        if count > 0:
            fc = flux.copy()
            for idx in np.where(mask)[0]:
                s, e = max(0, idx - 5), min(len(flux), idx + 6)
                local_clean = flux[s:e][~mask[s:e]]
                fc[idx] = np.median(local_clean) if len(local_clean) > 0 else med
            flux = fc
        self.log.append(f"Removed {count} outliers (σ={self.sigma})")
        return flux

    # ── Feature Extraction ───────────────────────────────────────
    def extract_features(self, flux):
        f = {
            'mean': np.mean(flux), 'std': np.std(flux), 'median': np.median(flux),
            'min': np.min(flux), 'max': np.max(flux), 'range': np.ptp(flux),
            'skewness': float(skew(flux)), 'kurtosis': float(kurtosis(flux)),
            'iqr': float(iqr(flux)),
            'depth_estimate': np.mean(flux) - np.min(flux),
            'mad': np.median(np.abs(flux - np.median(flux))),
            'rms': np.sqrt(np.mean(flux ** 2)),
            'cv': np.std(flux) / np.mean(flux) if np.mean(flux) != 0 else 0,
        }
        self.features = f
        self.log.append(f"Extracted {len(f)} features")
        return f

    # ── Full Pipeline ────────────────────────────────────────────
    def run(self, uploaded_file, progress_cb=None):
        self.log = []
        steps = [
            (0.10, "Loading data…",       lambda _: self.load_data(uploaded_file)),
            (0.25, "Cleaning…",            lambda f: (self.clean(f), None)),
            (0.40, "Normalizing…",         lambda f: (self.normalize(f), None)),
            (0.55, "Binning to 500…",      lambda f: (self.bin_data(f), None)),
            (0.70, "Smoothing…",           lambda f: (self.smooth(f), None)),
            (0.85, "Removing outliers…",   lambda f: (self.remove_outliers(f), None)),
            (0.95, "Extracting features…", lambda f: (f, self.extract_features(f))),
        ]
        flux = None
        df = None
        for pct, msg, fn in steps:
            if progress_cb:
                progress_cb(pct, msg)
            result = fn(flux)
            if isinstance(result, tuple) and len(result) == 2:
                if df is None and isinstance(result[1], pd.DataFrame):
                    flux, df = result
                else:
                    flux = result[0]
            else:
                flux = result

        self.processed_flux = flux
        self.time_axis = np.arange(len(flux))
        if progress_cb:
            progress_cb(1.0, "✅ Pipeline complete!")
        return flux, self.features, df
