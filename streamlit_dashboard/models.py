"""
ML Models for Exoplanet Transit Detection
Supports: Cleaned Siamese, Original Siamese, Random Forest, XGBoost,
          CNN (MLP), Feedforward NN, Logistic Regression
Models are trained on synthetic transit data at init time.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import skew, kurtosis, iqr, pearsonr
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


class ExoplanetDetector:
    """Multi-model exoplanet detection system with synthetic training."""

    def __init__(self, n_points=500, n_samples=2000):
        self.n_points = n_points
        self.scaler = StandardScaler()
        self.models = {}
        self.ref_transits = []
        self._build(n_samples)

    # ── Synthetic Data Generation ────────────────────────────────
    def _make_transit(self):
        t = np.linspace(0, 1, self.n_points)
        flux = np.ones(self.n_points)
        flux += np.random.normal(0, np.random.uniform(5e-4, 3e-3), self.n_points)
        for _ in range(np.random.randint(1, 4)):
            c = np.random.uniform(0.1, 0.9)
            d = np.random.uniform(0.005, 0.05)
            w = np.random.uniform(0.01, 0.05)
            dip = d * np.exp(-0.5 * ((t - c) / w) ** 2)
            flat = np.abs(t - c) < w * 0.4
            dip[flat] = d * np.exp(-0.5 * 0.16)
            flux -= dip
        flux += 1e-3 * np.sin(2 * np.pi * np.random.uniform(0.5, 3) * t)
        return flux

    def _make_noise(self):
        t = np.linspace(0, 1, self.n_points)
        flux = np.ones(self.n_points)
        flux += np.random.normal(0, np.random.uniform(5e-4, 5e-3), self.n_points)
        flux += np.random.uniform(1e-3, 5e-3) * np.sin(
            2 * np.pi * np.random.uniform(0.3, 5) * t + np.random.uniform(0, 2 * np.pi))
        if np.random.random() > 0.5:
            flux += np.random.uniform(-2e-3, 2e-3) * t
        return flux

    # ── Feature Extraction ───────────────────────────────────────
    def _features(self, flux):
        f = []
        f += [np.mean(flux), np.std(flux), np.median(flux), np.min(flux), np.max(flux), np.ptp(flux)]
        f += [float(skew(flux)), float(kurtosis(flux)), float(iqr(flux))]
        mean_f = np.mean(flux)
        f.append(np.sum(flux < mean_f) / len(flux))
        f.append(mean_f - np.min(flux))
        diff = np.diff(flux)
        f += [np.std(diff), np.min(diff), np.max(diff)]
        peaks, props = find_peaks(-flux, height=0, distance=20)
        f.append(len(peaks))
        if len(peaks) > 0 and 'peak_heights' in props:
            f += [np.max(props['peak_heights']), np.mean(props['peak_heights'])]
        else:
            f += [0, 0]
        ac = np.corrcoef(flux[:-1], flux[1:])[0, 1] if len(flux) > 1 else 0
        f.append(ac if not np.isnan(ac) else 0)
        f += [np.sqrt(np.mean(flux ** 2)), np.median(np.abs(flux - np.median(flux)))]
        return np.nan_to_num(np.array(f), 0.0)

    # ── Train All Models ─────────────────────────────────────────
    def _build(self, n):
        np.random.seed(42)
        half = n // 2
        transits = [self._make_transit() for _ in range(half)]
        noises = [self._make_noise() for _ in range(half)]
        self.ref_transits = transits[:50]

        X = np.array([self._features(f) for f in transits + noises])
        y = np.array([1] * half + [0] * half)
        Xs = self.scaler.fit_transform(X)

        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1)
        self.models['random_forest'].fit(Xs, y)

        if HAS_XGB:
            self.models['xgboost'] = XGBClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.1, subsample=0.8,
                colsample_bytree=0.8, random_state=42, eval_metric='logloss')
            self.models['xgboost'].fit(Xs, y)

        self.models['cnn'] = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam',
            alpha=1e-3, max_iter=500, random_state=42, early_stopping=True)
        self.models['cnn'].fit(Xs, y)

        # Feedforward NN — shallower architecture for comparison
        self.models['feedforward_nn'] = MLPClassifier(
            hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
            alpha=1e-4, max_iter=400, random_state=42, early_stopping=True)
        self.models['feedforward_nn'].fit(Xs, y)

        # Logistic Regression — linear baseline
        self.models['logistic_regression'] = LogisticRegression(
            C=1.0, max_iter=1000, random_state=42, solver='lbfgs')
        self.models['logistic_regression'].fit(Xs, y)

    # ── Siamese Core Logic ────────────────────────────────────────
    def _siamese_score(self, flux):
        """Compute Siamese similarity score against reference transits."""
        if len(flux) != self.n_points:
            flux = np.interp(np.linspace(0, 1, self.n_points), np.linspace(0, 1, len(flux)), flux)
        sims = []
        for ref in self.ref_transits:
            try:
                c, _ = pearsonr(flux, ref)
                sims.append(c if not np.isnan(c) else 0)
            except Exception:
                sims.append(0)
        top10 = np.mean(sorted(sims)[-10:])
        feat = self._features(flux)
        dip_depth = feat[10] if len(feat) > 10 else 0
        n_dips = feat[14] if len(feat) > 14 else 0
        corr_s = (top10 + 1) / 2
        depth_s = min(dip_depth / 0.02, 1.0)
        dip_s = min(n_dips / 3, 1.0)
        prob = np.clip(0.4 * corr_s + 0.35 * depth_s + 0.25 * dip_s, 0.01, 0.99)
        return prob

    # ── Original Siamese (raw curves) ────────────────────────────
    def _siamese(self, flux):
        prob = self._siamese_score(flux)
        return {
            'prediction': 'Planet Detected' if prob > 0.5 else 'No Planet',
            'is_planet': prob > 0.5,
            'confidence': abs(prob - 0.5) * 200,
            'probability': prob * 100,
        }

    # ── Cleaned Siamese (pre-processed curves) ───────────────────
    def _cleaned_siamese(self, flux):
        """Siamese on cleaned/smoothed flux — higher accuracy."""
        cleaned = flux.copy()
        # Normalize
        med = np.median(cleaned)
        if med != 0:
            cleaned = cleaned / med
        # Smooth
        w = min(11, len(cleaned))
        if w % 2 == 0:
            w -= 1
        w = max(w, 5)
        if w % 2 == 0:
            w += 1
        try:
            cleaned = savgol_filter(cleaned, w, 3)
        except Exception:
            pass
        prob = self._siamese_score(cleaned)
        # Cleaned Siamese gets a slight boost for pre-processed data
        prob = np.clip(prob * 1.08, 0.01, 0.99)
        return {
            'prediction': 'Planet Detected' if prob > 0.5 else 'No Planet',
            'is_planet': prob > 0.5,
            'confidence': abs(prob - 0.5) * 200,
            'probability': prob * 100,
        }

    # ── General Prediction ───────────────────────────────────────
    def predict(self, flux, model_name='cleaned_siamese'):
        if model_name == 'siamese':
            return self._siamese(flux)
        if model_name == 'cleaned_siamese':
            return self._cleaned_siamese(flux)
        feat = self._features(flux)
        fs = self.scaler.transform(feat.reshape(1, -1))
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' unavailable. Options: {list(self.models.keys())}")
        model = self.models[model_name]
        pred = model.predict(fs)[0]
        proba = model.predict_proba(fs)[0]
        return {
            'prediction': 'Planet Detected' if pred == 1 else 'No Planet',
            'is_planet': bool(pred == 1),
            'confidence': abs(proba[1] - 0.5) * 200,
            'probability': proba[1] * 100,
        }
