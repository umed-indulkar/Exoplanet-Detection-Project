"""
Data Processing Pipeline for Exoplanet Light Curve Analysis

Processing order (Kepler-style):
    load → clean NaN/Inf → outlier removal (percentile ∧ MAD, transit-safe)
         → detrending (adaptive median filter) → binning
         → robust MAD z-score normalization
         → (optional) Savitzky-Golay smoothing
         → feature extraction
         → BLS-style period search + phase folding

Two flux representations are stored:
    processed_flux  — z-score (MAD units). Dips sit at ~ -5σ … -20σ, visible.
    model_flux      — ratio-normalized (centered at 1.0). Feed this to the
                      sklearn / Siamese models that were trained on ratio flux.

Each stage writes into `self.steps[...]` for step-by-step visualization.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import median_filter
from scipy.stats import skew, kurtosis, iqr, median_abs_deviation
import warnings
warnings.filterwarnings('ignore')


# ────────────────────────────────────────────────────────────
# CSV Format Detection
# ────────────────────────────────────────────────────────────

def detect_csv_format(df):
    """Detect the CSV type. Returns (format_type, info)."""
    cols = list(df.columns)
    col_lower = [str(c).lower() for c in cols]
    n_cols = len(cols)

    # KOI / TCE metadata table — has kepid + koi_ or tce_ columns
    has_kepid = any('kepid' in c for c in col_lower)
    has_koi   = any('koi_' in c for c in col_lower)
    has_tce   = any('tce_' in c for c in col_lower)

    if has_kepid and has_koi:
        has_period = any('koi_period' in c for c in col_lower)
        has_epoch  = any('koi_time0bk' in c for c in col_lower)
        has_disp   = any('koi_disposition' in c for c in col_lower)
        return 'koi_table', {
            'n_rows': len(df),
            'has_period':   has_period,
            'has_epoch':    has_epoch,
            'has_disposition': has_disp,
        }

    if has_kepid and has_tce:
        return 'tce_table', {
            'message': 'NASA TCE parameter table — no flux time series. Upload a light curve CSV instead.',
            'n_rows': len(df),
        }

    feature_keywords = ['fft_coefficient', 'autocorrelation', 'agg_linear_trend',
                        'cwt_coefficients', 'partial_autocorrelation', 'augmented_dickey',
                        'lempel_ziv', 'permutation_entropy', 'binned_entropy']
    feature_hits = sum(1 for c in col_lower for kw in feature_keywords if kw in c)
    if feature_hits > 10:
        return 'feature_table', {
            'message': 'Pre-extracted feature table (FFT, autocorrelation, entropy, …) — '
                      'already-processed features, not a raw light curve.',
            'n_cols': n_cols, 'n_rows': len(df),
            'has_label': 'label' in col_lower or 'Label' in cols,
        }

    flux_cols = [c for c in cols if isinstance(c, str) and 'flux' in c.lower()]
    time_cols = [c for c in cols if isinstance(c, str)
                 and c.lower() in ('time', 't', 'bjd', 'bkjd', 'jd', 'mjd')]
    if flux_cols:
        return 'time_flux', {
            'flux_col': flux_cols[0],
            'time_col': time_cols[0] if time_cols else None,
            'n_points': len(df),
        }

    if n_cols > 10:
        flux_start = 0
        if n_cols in (502, 501):
            flux_start = n_cols - 500
        elif n_cols > 100:
            for i in range(min(3, n_cols)):
                try:
                    pd.to_numeric(df.iloc[:, i], errors='raise')
                except (ValueError, TypeError):
                    flux_start = i + 1
        return 'wide_lightcurve', {
            'n_cols': n_cols, 'n_rows': len(df),
            'flux_start': flux_start,
            'n_flux_bins': n_cols - flux_start,
        }

    return 'unknown', {
        'message': f'Unrecognized CSV format ({n_cols} cols, {len(df)} rows). '
                   'Expected: a flux column, or wide-format light curve (>10 cols).',
    }


# ────────────────────────────────────────────────────────────
# Pipeline
# ────────────────────────────────────────────────────────────

class DataPipeline:
    """Full light-curve processing pipeline (Kepler-reference style)."""

    def __init__(self, n_bins=500, smooth_window=7, smooth_poly=3,
                 sigma_pos=3.0, sigma_neg=8.0, detrend_window=201,
                 outlier_percentiles=(0.5, 99.9), min_data_points=30):
        self.n_bins = int(n_bins)
        self.smooth_window = int(smooth_window)
        self.smooth_poly = int(smooth_poly)
        # MAD thresholds (asymmetric): gentle on transits (sigma_neg), strict on flares (sigma_pos)
        self.sigma_pos = float(sigma_pos)
        self.sigma_neg = float(sigma_neg)
        self.detrend_window = int(detrend_window)
        self.outlier_percentiles = outlier_percentiles
        self.min_data_points = int(min_data_points)

        self.log = []

        # Inputs
        self.raw_flux = None
        self.raw_time = None         # Original time, for reference
        self.time_axis = None        # Post-bin time axis
        self.csv_format = None
        self.csv_info = None

        # Outputs
        self.steps = {}              # step_name → flux array
        self.processed_flux = None   # z-score flux (display / analysis)
        self.model_flux = None       # ratio-normalized flux (~1.0) for models
        self.features = None

        # Phase folding
        self.folded_phase = None
        self.folded_flux = None
        self.estimated_period = None
        self.period_power = None     # BLS-style power spectrum

    # ─── Load ────────────────────────────────────────────────
    def load_data(self, uploaded_file, row_index=0):
        try:
            df = pd.read_csv(uploaded_file, comment='#', header='infer')
            if df.shape[1] > 10:
                first_row_numeric = all(
                    str(v).replace('.', '', 1).replace('-', '', 1).replace('e', '', 1)
                    .replace('+', '', 1).isdigit() for v in df.columns[:5])
                if first_row_numeric:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, comment='#', header=None)
        except Exception as e:
            raise ValueError(f"Failed to parse CSV: {e}")

        if df.empty:
            raise ValueError("CSV file is empty.")

        fmt, info = detect_csv_format(df)
        self.csv_format, self.csv_info = fmt, info

        if fmt == 'tce_table':
            raise ValueError(f"Not a light curve file. {info['message']}")
        if fmt == 'feature_table':
            raise ValueError(f"Not a light curve file. {info['message']}")
        if fmt == 'unknown':
            raise ValueError(info['message'])

        if fmt == 'time_flux':
            flux_col = info['flux_col']
            time_col = info.get('time_col')
            flux = pd.to_numeric(df[flux_col], errors='coerce').values.astype(float)
            if time_col:
                t = pd.to_numeric(df[time_col], errors='coerce').values.astype(float)
                self.raw_time = t
                self.log.append(
                    f"Loaded time+flux: {len(flux)} pts, time range "
                    f"[{np.nanmin(t):.2f}, {np.nanmax(t):.2f}]")
            else:
                self.raw_time = np.arange(len(flux), dtype=float)
                self.log.append(f"Loaded flux '{flux_col}': {len(flux)} pts (no time col)")

        elif fmt == 'wide_lightcurve':
            flux_start = info['flux_start']
            flux_df = df.iloc[:, flux_start:].apply(pd.to_numeric, errors='coerce')
            row_std = flux_df.std(axis=1)
            is_flat = (row_std == 0) | flux_df.isnull().all(axis=1)
            flat_count = int(is_flat.sum())
            if flat_count:
                self.log.append(f"Removed {flat_count} flat/dead row(s)")
                flux_df = flux_df[~is_flat]
            if flux_df.empty:
                raise ValueError("All rows are flat/dead after cleaning.")
            n_valid = len(flux_df)
            row_index = min(max(0, int(row_index)), n_valid - 1)
            flux = flux_df.iloc[row_index].values.astype(float)
            self.raw_time = np.arange(len(flux), dtype=float)
            self.log.append(
                f"Wide-format: row {row_index}/{n_valid}, {len(flux)} bins "
                f"(skipped {flux_start} metadata col(s))")
        else:
            raise ValueError("Unsupported CSV format.")

        if len(flux) < self.min_data_points:
            raise ValueError(
                f"Only {len(flux)} points — need at least {self.min_data_points}.")

        self.raw_flux = flux.copy()
        self.steps['raw'] = flux.copy()
        return flux, df

    # ─── Clean (NaN/Inf interpolation) ───────────────────────
    def clean(self, flux):
        flux = np.where(np.isinf(flux), np.nan, flux)
        mask = np.isnan(flux)
        n_bad = int(mask.sum())
        if mask.any():
            valid = ~mask
            if valid.sum() > 2:
                flux = flux.copy()
                flux[mask] = np.interp(
                    np.where(mask)[0], np.where(valid)[0], flux[valid])
            else:
                flux = np.nan_to_num(flux, nan=np.nanmedian(flux) if valid.any() else 0.0)
        self.log.append(f"Cleaned {n_bad} invalid value(s)")
        self.steps['cleaned'] = flux.copy()
        return flux

    # ─── Outlier removal (percentile ∧ MAD, transit-safe) ────
    def clip_outliers(self, flux):
        """Remove positive outliers (flares, cosmic rays) only.

        Transit dips must NEVER be clipped — they are the signal we want to
        detect.  Only upward spikes (z > sigma_pos) are replaced with local
        median.  A very conservative low-end percentile (0.1th) catches
        genuinely corrupted negative values without touching transit dips.
        """
        n = len(flux)
        med = float(np.median(flux))
        mad = float(median_abs_deviation(flux, scale='normal'))
        if mad == 0:
            mad = float(np.std(flux))
        if mad == 0:
            self.log.append("Outlier clipping skipped (zero variance)")
            self.steps['clipped'] = flux.copy()
            return flux

        z = (flux - med) / mad

        # Only clip positive outliers (flares/spikes) via MAD threshold
        pos_bad = z > self.sigma_pos

        # Very conservative negative clip via percentile only (0.1th percentile)
        # — this removes genuinely corrupted points but not transit dips
        p_low = float(np.percentile(flux, 0.1))
        p_high = float(np.percentile(flux, self.outlier_percentiles[1]))
        neg_bad = flux < p_low
        pos_high_bad = flux > p_high

        bad = pos_bad | neg_bad | pos_high_bad

        # Build keep mask for local median replacement
        keep = ~bad
        n_pos = int(pos_bad.sum()) + int(pos_high_bad.sum())
        n_neg = int(neg_bad.sum())

        if bad.any():
            out = flux.copy()
            for idx in np.where(bad)[0]:
                s, e = max(0, idx - 5), min(n, idx + 6)
                local = flux[s:e][keep[s:e]]
                out[idx] = float(np.median(local)) if len(local) else med
            flux = out

        self.log.append(
            f"Clipped {n_pos} positive spike(s) (σ>{self.sigma_pos}) "
            f"and {n_neg} corrupted negative point(s) (<0.1th pct); "
            f"transits preserved")
        self.steps['clipped'] = flux.copy()
        return flux

    # ─── Detrend (adaptive median filter) ────────────────────
    def detrend(self, flux):
        """Remove stellar variability via a wide running-median baseline.

        The window must be LARGE enough that transit dips (typically <20% of
        the curve length) don't dominate the local median.  We use the user's
        detrend_window as a minimum and allow it to grow up to n//3 so that
        long transits in short light curves are never erased.
        """
        n = len(flux)
        if n < 20:
            self.log.append("Detrending skipped (too few points)")
            self.steps['detrended'] = flux.copy()
            return flux

        # Use user setting, but allow up to n//3 so window > any realistic transit width
        win = max(self.detrend_window, n // 3)
        win = min(win, n - 1)
        if win % 2 == 0:
            win -= 1
        win = max(win, 5)

        baseline = median_filter(flux, size=win, mode='reflect')
        # Guard against division by zero
        safe_val = float(np.median(flux[flux != 0])) if np.any(flux != 0) else 1.0
        baseline = np.where(baseline == 0, safe_val, baseline)

        detrended = flux / baseline
        variability = float(np.std(baseline) / np.median(baseline) * 100) \
            if np.median(baseline) != 0 else 0.0
        self.log.append(
            f"Detrended (window={win}, stellar variability removed: {variability:.3f}%)")
        self.steps['detrended'] = detrended.copy()
        return detrended

    # ─── Binning ─────────────────────────────────────────────
    def bin_data(self, flux, time=None):
        """Bin to n_bins points. Also downsample the time axis if provided."""
        n = len(flux)
        if n <= self.n_bins:
            self.log.append(f"No binning needed ({n} pts ≤ {self.n_bins})")
            return flux, time

        bin_size = n / self.n_bins
        binned = np.empty(self.n_bins, dtype=float)
        binned_t = np.empty(self.n_bins, dtype=float) if time is not None else None
        for i in range(self.n_bins):
            s, e = int(i * bin_size), int((i + 1) * bin_size)
            if e <= s:
                e = s + 1
            chunk = flux[s:e]
            binned[i] = np.nanmean(chunk) if chunk.size else np.nan
            if binned_t is not None:
                tc = time[s:e]
                binned_t[i] = np.nanmean(tc) if tc.size else np.nan
        # Fill any NaN bins with neighbors
        if np.isnan(binned).any():
            mask = np.isnan(binned)
            if (~mask).any():
                binned[mask] = np.interp(
                    np.where(mask)[0], np.where(~mask)[0], binned[~mask])
            else:
                binned = np.zeros_like(binned)
        self.log.append(f"Binned {n} → {self.n_bins} points")
        return binned, binned_t

    # ─── Robust MAD z-score normalization ────────────────────
    def normalize(self, flux):
        """Produce two representations:

          model_flux      — ratio form centered at 1.0 (for sklearn / Siamese).
          processed_flux  — robust z-score: (flux - median) / MAD.
                            Dips appear as deeply negative σ values.
        """
        med = float(np.median(flux))
        mad = float(median_abs_deviation(flux, scale='normal'))
        if mad == 0:
            mad = float(np.std(flux))

        # Ratio representation (for models trained on ~1.0 domain)
        if med not in (0.0,) and not np.isnan(med):
            model_flux = flux / med
        else:
            model_flux = flux.copy()

        # z-score representation (for display / dip visibility)
        if mad > 0:
            z = (flux - med) / mad
        else:
            z = flux - med

        if not np.all(np.isfinite(z)):
            z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.all(np.isfinite(model_flux)):
            model_flux = np.nan_to_num(model_flux, nan=1.0, posinf=1.0, neginf=1.0)

        self.log.append(
            f"Normalized — median={med:.6f}, MAD={mad:.6e}; "
            f"z-score range [{z.min():.2f}σ, {z.max():.2f}σ]")
        self.steps['normalized'] = z.copy()
        return z, model_flux

    # ─── Smoothing ───────────────────────────────────────────
    def smooth(self, flux):
        w = min(self.smooth_window, len(flux))
        if w % 2 == 0:
            w -= 1
        w = max(w, self.smooth_poly + 2)
        if w % 2 == 0:
            w += 1
        try:
            out = savgol_filter(flux, w, self.smooth_poly)
            self.log.append(f"Savitzky-Golay smoothing (win={w}, poly={self.smooth_poly})")
        except Exception:
            out = flux
            self.log.append("Smoothing skipped (error)")
        self.steps['smoothed'] = out.copy()
        return out

    # ─── Feature extraction (on z-score flux) ────────────────
    def extract_features(self, flux, model_flux=None):
        """Features for downstream analysis + models.

        Depth metrics use the ratio flux (model_flux) so they stay in
        physical units (% of stellar flux).
        """
        ratio = model_flux if model_flux is not None else flux

        mean_f = float(np.mean(flux))
        std_f = float(np.std(flux))
        med_f = float(np.median(flux))

        med_r = float(np.median(ratio))
        depth_abs = med_r - float(np.min(ratio))
        depth_pct = (depth_abs / med_r * 100.0) if med_r != 0 else 0.0

        frac_below = float(np.sum(flux < mean_f) / len(flux)) if len(flux) else 0.5
        diff = np.diff(flux)

        f = {
            'mean': mean_f,
            'std': std_f,
            'median': med_f,
            'min': float(np.min(flux)),
            'max': float(np.max(flux)),
            'range': float(np.ptp(flux)),
            'skewness': float(skew(flux)),
            'kurtosis': float(kurtosis(flux)),
            'iqr': float(iqr(flux)),
            'mad': float(median_abs_deviation(flux, scale='normal')),
            'rms': float(np.sqrt(np.mean(flux ** 2))),
            'depth_estimate': depth_abs,     # on ratio flux
            'depth_pct': depth_pct,          # on ratio flux, in %
            'cv': std_f / mean_f if mean_f != 0 else 0.0,
            'frac_below_mean': frac_below,
            'diff_std': float(np.std(diff)) if len(diff) else 0.0,
        }
        self.features = f
        self.log.append(f"Extracted {len(f)} features")
        return f

    # ─── BLS-style period search ─────────────────────────────
    def _bls_period_search(self, flux, time=None, n_periods=80):
        """Simple box-least-squares-like period grid search.

        For each candidate period, fold the flux, bin into phase bins, then
        score by  (out-of-transit median − in-transit minimum) / noise.
        Picks the period with the highest score.
        """
        n = len(flux)
        if n < 40:
            return None, None

        if time is None:
            time = np.arange(n, dtype=float)
        t = np.asarray(time, dtype=float) - float(time[0])

        # Period grid in *time* units (same as the time axis)
        min_period = max(2.0, (t[-1] - t[0]) / n * 5)      # at least 5 samples
        max_period = (t[-1] - t[0]) / 2.5                  # need ≥ 2.5 cycles
        if max_period <= min_period:
            return None, None

        periods = np.geomspace(min_period, max_period, n_periods)
        n_phase_bins = 40

        f = flux - np.median(flux)
        mad = float(median_abs_deviation(f, scale='normal'))
        noise = mad if mad > 0 else float(np.std(f)) or 1.0

        powers = np.zeros_like(periods)
        for i, P in enumerate(periods):
            phase = (t % P) / P
            edges = np.linspace(0.0, 1.0, n_phase_bins + 1)
            idx = np.clip(np.digitize(phase, edges) - 1, 0, n_phase_bins - 1)
            bin_means = np.full(n_phase_bins, np.nan)
            for b in range(n_phase_bins):
                chunk = f[idx == b]
                if chunk.size:
                    bin_means[b] = np.mean(chunk)
            if np.isnan(bin_means).all():
                continue
            dip = np.nanmin(bin_means)
            base = np.nanmedian(bin_means)
            # Power: how many σ the deepest bin sits below the baseline
            powers[i] = max(0.0, (base - dip) / noise)

        if not np.any(powers > 0):
            return None, None

        best_idx = int(np.argmax(powers))
        best_period = float(periods[best_idx])
        return best_period, {'periods': periods, 'powers': powers,
                             'best_power': float(powers[best_idx])}

    # ─── Phase fold ──────────────────────────────────────────
    def phase_fold(self, flux, time=None, period=None):
        if time is None or len(time) != len(flux):
            time = np.arange(len(flux), dtype=float)

        result = None
        if period is None:
            period, result = self._bls_period_search(flux, time=time)
            if result is not None:
                self.period_power = result

        if period is None or period < 2:
            self.log.append("Phase folding skipped (no reliable period found)")
            return None

        self.estimated_period = float(period)

        t0 = float(time[0])
        phase = ((time - t0) % period) / period

        # Center deepest bin at 0.5 for nicer visualization
        sort_idx = np.argsort(phase)
        sp = phase[sort_idx]
        sf = flux[sort_idx]

        n_phase_bins = min(60, max(20, len(flux) // 8))
        edges = np.linspace(0.0, 1.0, n_phase_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        binned = np.array([
            np.nanmedian(sf[(sp >= edges[i]) & (sp < edges[i + 1])])
            if np.any((sp >= edges[i]) & (sp < edges[i + 1])) else np.nan
            for i in range(n_phase_bins)
        ])
        valid = ~np.isnan(binned)
        centers = centers[valid]
        binned = binned[valid]

        # Re-center on deepest bin
        if len(binned):
            deepest = int(np.argmin(binned))
            shift = 0.5 - centers[deepest]
            centers = (centers + shift) % 1.0
            order = np.argsort(centers)
            centers = centers[order]
            binned = binned[order]

        self.folded_phase = centers
        self.folded_flux = binned
        self.log.append(
            f"Phase-folded (period={period:.3f}, "
            f"{len(binned)} phase bins, dip centered at 0.5)")
        return centers, binned, period

    # ─── Run the full pipeline ───────────────────────────────
    def run(self, uploaded_file, row_index=0, progress_cb=None, enable_phase_fold=True):
        self.log = []
        self.steps = {}

        def _tick(pct, msg):
            if progress_cb:
                progress_cb(pct, msg)

        _tick(0.05, "📂 Loading & detecting CSV format…")
        flux, df = self.load_data(uploaded_file, row_index)
        time_arr = self.raw_time.copy() if self.raw_time is not None \
            else np.arange(len(flux), dtype=float)

        _tick(0.15, "🧹 Cleaning invalid values…")
        flux = self.clean(flux)

        _tick(0.28, "✂️ Removing outliers (percentile ∧ MAD)…")
        flux = self.clip_outliers(flux)

        _tick(0.45, "📉 Detrending stellar variability…")
        flux = self.detrend(flux)

        _tick(0.58, f"📊 Binning to {self.n_bins} points…")
        flux, time_arr = self.bin_data(flux, time_arr)
        if time_arr is None:
            time_arr = np.arange(len(flux), dtype=float)

        _tick(0.70, "⚖️ Robust MAD z-score normalization…")
        z, model_flux = self.normalize(flux)

        _tick(0.78, "🔧 Smoothing…")
        z = self.smooth(z)
        # Apply the same smoothing to model_flux so features align
        w = min(self.smooth_window, len(model_flux))
        if w % 2 == 0:
            w -= 1
        w = max(w, self.smooth_poly + 2)
        if w % 2 == 0:
            w += 1
        try:
            model_flux = savgol_filter(model_flux, w, self.smooth_poly)
        except Exception:
            pass

        self.time_axis = time_arr
        self.processed_flux = z
        self.model_flux = model_flux

        _tick(0.87, "🔬 Extracting features…")
        self.extract_features(z, model_flux=model_flux)

        if enable_phase_fold:
            _tick(0.94, "🌀 Searching for period (BLS-style) & phase folding…")
            self.phase_fold(z, time=time_arr)

        _tick(1.0, "✅ Pipeline complete")
        return z, self.features, df

    # ─── Step helpers ────────────────────────────────────────
    def get_step_names(self):
        order = ['raw', 'cleaned', 'clipped', 'detrended', 'normalized', 'smoothed']
        return [s for s in order if s in self.steps]

    def get_step_data(self, step_name):
        return self.steps.get(step_name)
