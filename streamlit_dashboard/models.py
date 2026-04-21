"""
Exoplanet Detection — loads pre-trained models from models/ directory.

All models expect a 500-point ratio-normalized flux array (~1.0 centered).
Scalers are applied before feeding to sklearn/PyTorch models.

Siamese models use prototype embeddings + L2 distance for inference.
Classifier models (FF-NN, CNN, ConvNN) use a calibrated decision threshold
computed from synthetic transit/noise curves during initialization.
"""

import os
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
N_POINTS = 500


# ── PyTorch Architecture Definitions ─────────────────────────────────

if HAS_TORCH:

    class _SiameseEncoder(nn.Module):
        """Encoder for cleaned_siamese.pth — state-dict keys: encoder.*"""
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(500, 256),    # 0
                nn.BatchNorm1d(256),    # 1
                nn.ReLU(),              # 2
                nn.Dropout(0.3),        # 3
                nn.Linear(256, 128),    # 4
                nn.BatchNorm1d(128),    # 5
                nn.ReLU(),              # 6
                nn.Dropout(0.3),        # 7
                nn.Linear(128, 128),    # 8
                nn.BatchNorm1d(128),    # 9
            )

        def forward(self, x):
            return self.encoder(x)

    class _SiameseNet(nn.Module):
        """Wrapper for siamese_dataset500.pth — state-dict keys: encoder.encoder.*"""
        def __init__(self):
            super().__init__()
            self.encoder = _SiameseEncoder()

        def forward(self, x):
            return self.encoder(x)

    class _FeedforwardNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(500, 256),    # 0
                nn.BatchNorm1d(256),    # 1
                nn.ReLU(),              # 2
                nn.Dropout(0.3),        # 3
                nn.Linear(256, 128),    # 4
                nn.BatchNorm1d(128),    # 5
                nn.ReLU(),              # 6
                nn.Dropout(0.3),        # 7
                nn.Linear(128, 64),     # 8
                nn.BatchNorm1d(64),     # 9
                nn.ReLU(),              # 10
                nn.Dropout(0.3),        # 11
                nn.Linear(64, 1),       # 12
            )

        def forward(self, x):
            return torch.sigmoid(self.network(x))

    class _CNN(nn.Module):
        """cnn.pth: Conv(1,32,5) → MaxPool → Conv(32,64,3) → MaxPool → FC"""
        def __init__(self):
            super().__init__()
            self.conv_layers = nn.Sequential(
                nn.Conv1d(1, 32, 5, padding=2),   # 0 → [32, 500]
                nn.ReLU(),                          # 1
                nn.MaxPool1d(2),                    # 2 → [32, 250]
                nn.Conv1d(32, 64, 3, padding=1),    # 3 → [64, 250]
                nn.ReLU(),                          # 4
                nn.MaxPool1d(2),                    # 5 → [64, 125]
            )
            self.fc_layers = nn.Sequential(
                nn.Linear(8000, 128),   # 0
                nn.ReLU(),               # 1
                nn.Dropout(0.5),         # 2
                nn.Linear(128, 1),       # 3
            )

        def forward(self, x):
            x = x.unsqueeze(1)
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)
            return torch.sigmoid(self.fc_layers(x))

    class _ConvolutionalNN(nn.Module):
        """convolutionalnn_best.pth: 3×(Conv+BN+ReLU+MaxPool) → 3×FC"""
        def __init__(self):
            super().__init__()
            self.conv_layers = nn.ModuleList([
                nn.Sequential(nn.Conv1d(1, 64, 3, padding=1), nn.BatchNorm1d(64)),      # 0
                nn.Sequential(nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128)),   # 1
                nn.Sequential(nn.Conv1d(128, 256, 7, padding=3), nn.BatchNorm1d(256)),  # 2
            ])
            # 500 → maxpool → 250 → maxpool → 125 → maxpool → 62; 256*62=15872
            self.fc_layers = nn.Sequential(
                nn.Linear(15872, 256),   # 0
                nn.BatchNorm1d(256),     # 1
                nn.ReLU(),               # 2
                nn.Dropout(0.5),         # 3
                nn.Linear(256, 128),     # 4
                nn.BatchNorm1d(128),     # 5
                nn.ReLU(),               # 6
                nn.Dropout(0.5),         # 7
                nn.Linear(128, 1),       # 8
            )

        def forward(self, x):
            x = x.unsqueeze(1)
            for conv_bn in self.conv_layers:
                x = torch.relu(conv_bn(x))
                x = torch.max_pool1d(x, 2)
            x = x.view(x.size(0), -1)
            return torch.sigmoid(self.fc_layers(x))


# ── Helpers ──────────────────────────────────────────────────────────

def _to_500(flux):
    """Resample any flux array to exactly N_POINTS."""
    flux = np.asarray(flux, dtype=np.float32)
    if len(flux) == N_POINTS:
        return flux
    return np.interp(
        np.linspace(0, 1, N_POINTS),
        np.linspace(0, 1, len(flux)),
        flux,
    ).astype(np.float32)


def _center_transit(flux, target_pos=250):
    """Roll flux so the deepest dip (transit) is at target_pos.

    The training data has transits concentrated around position 200-300.
    Centering the test curve's dip there brings it into distribution.
    """
    dip_idx = int(np.argmin(flux))
    shift = target_pos - dip_idx
    return np.roll(flux, shift)


def _max_scan_predict(predict_fn, flux500):
    """Run predict_fn on original + 3 dip-centered variants, return the best.

    The training data has transits scattered across all positions with a peak
    around 200-300. We test 4 candidate alignments for the deepest dip:
    positions 125, 250 (center), 375, and the original location.
    4 candidates give good position coverage without inflating false positives.
    """
    candidates = [flux500]
    dip_idx = int(np.argmin(flux500))
    for target in [125, 250, 375]:
        shift = target - dip_idx
        candidates.append(np.roll(flux500, shift))

    best_prob = 0.0
    best_result = None
    for candidate in candidates:
        r = predict_fn(candidate)
        if r['probability'] > best_prob:
            best_prob = r['probability']
            best_result = r
    return best_result


def _make_result(raw_prob, threshold=0.5, raw_min=0.0, raw_max=1.0):
    """Build prediction dict.

    raw_prob   : raw sigmoid output in [0, 1]
    threshold  : calibrated decision boundary
    raw_min/max: model's observed output range for display normalization
    """
    raw_prob = float(np.clip(raw_prob, 0.0, 1.0))
    is_planet = raw_prob > threshold

    # Normalize display probability to full [0, 100%] range so all models
    # are visually comparable, even if raw output is biased.
    span = raw_max - raw_min
    if span > 0:
        display_prob = float(np.clip((raw_prob - raw_min) / span, 0.0, 1.0))
    else:
        display_prob = 0.5

    confidence = abs(display_prob - 0.5) * 200.0

    return {
        'prediction': 'Planet Detected' if is_planet else 'No Planet',
        'is_planet': is_planet,
        'confidence': confidence,
        'probability': display_prob * 100.0,
    }


# ── Synthetic data generators ─────────────────────────────────────────

def _synthetic_transit(seed=None):
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, N_POINTS)
    flux = np.ones(N_POINTS, dtype=np.float32)
    flux += rng.normal(0, rng.uniform(5e-4, 3e-3), N_POINTS).astype(np.float32)
    for _ in range(rng.randint(1, 4)):
        c = rng.uniform(0.1, 0.9)
        d = rng.uniform(0.005, 0.05)
        w = rng.uniform(0.01, 0.05)
        dip = d * np.exp(-0.5 * ((t - c) / w) ** 2)
        flat = np.abs(t - c) < w * 0.4
        dip[flat] = d * np.exp(-0.5 * 0.16)
        flux -= dip.astype(np.float32)
    return flux


def _synthetic_noise(seed=None):
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, N_POINTS)
    flux = np.ones(N_POINTS, dtype=np.float32)
    flux += rng.normal(0, rng.uniform(5e-4, 5e-3), N_POINTS).astype(np.float32)
    flux += (rng.uniform(1e-3, 5e-3) * np.sin(
        2 * np.pi * rng.uniform(0.3, 5) * t + rng.uniform(0, 2 * np.pi)
    )).astype(np.float32)
    if rng.random() > 0.5:
        flux += (rng.uniform(-2e-3, 2e-3) * t).astype(np.float32)
    return flux


_TRAIN_CSV = os.path.join(
    os.path.dirname(__file__), 'light_curves_csv', 'raw_curve_500_cleaned.csv'
)


def _calibrate_classifier(model, scaler, n_synth=300):
    """Return (threshold, raw_min, raw_max) for a sigmoid-output classifier.

    Tries to use the real training CSV (raw_curve_500_cleaned.csv) first;
    falls back to synthetic curves if unavailable.
    """
    import pandas as pd

    labels_list = []
    flux_list = []

    if os.path.exists(_TRAIN_CSV):
        try:
            df = pd.read_csv(_TRAIN_CSV)
            flux_cols = sorted([c for c in df.columns if c.startswith('flux_')],
                               key=lambda c: int(c.split('_')[1]))
            sample = df.sample(min(600, len(df)), random_state=0)
            flux_list.append(sample[flux_cols].values.astype(np.float32))
            labels_list.extend(sample['Label'].tolist())
        except Exception:
            pass

    if not flux_list:
        # Fallback: synthetic data
        transits = np.stack([_synthetic_transit(s) for s in range(n_synth)])
        noises   = np.stack([_synthetic_noise(s + 5000) for s in range(n_synth)])
        flux_list.append(np.vstack([transits, noises]))
        labels_list.extend([1] * n_synth + [0] * n_synth)

    all_flux   = np.vstack(flux_list).astype(np.float32)
    all_labels = np.array(labels_list, dtype=int)

    if scaler is not None:
        all_flux = scaler.transform(all_flux).astype(np.float32)

    model.eval()
    with torch.no_grad():
        probs = model(torch.from_numpy(all_flux)).squeeze().numpy()

    raw_min = float(probs.min())
    raw_max = float(probs.max())
    n_pos   = int(all_labels.sum())
    n_neg   = int((all_labels == 0).sum())

    # Youden's J: maximise (TPR + TNR - 1)
    best_j, best_thr = -1.0, (raw_min + raw_max) / 2
    for thr in np.linspace(raw_min, raw_max, 300):
        preds = (probs > thr).astype(int)
        tp = int(np.sum((preds == 1) & (all_labels == 1)))
        tn = int(np.sum((preds == 0) & (all_labels == 0)))
        tpr = tp / n_pos if n_pos > 0 else 0.0
        tnr = tn / n_neg if n_neg > 0 else 0.0
        j = tpr + tnr - 1.0
        if j > best_j:
            best_j, best_thr = j, float(thr)

    return best_thr, raw_min, raw_max


# ── Siamese helpers ───────────────────────────────────────────────────

def _build_siamese_prototypes(model, scaler, n=300):
    """Return (planet_centroid, noplanet_centroid) in 128-d embedding space."""
    transits = np.stack([_synthetic_transit(s) for s in range(n)])
    noises   = np.stack([_synthetic_noise(s + 5000) for s in range(n)])

    if scaler is not None:
        transits_s = scaler.transform(transits).astype(np.float32)
        noises_s   = scaler.transform(noises).astype(np.float32)
    else:
        transits_s = transits
        noises_s   = noises

    model.eval()
    with torch.no_grad():
        emb_t = model(torch.from_numpy(transits_s)).numpy()
        emb_n = model(torch.from_numpy(noises_s)).numpy()

    return emb_t.mean(axis=0), emb_n.mean(axis=0)


def _siamese_prob(embedding, planet_centroid, noplanet_centroid):
    """L2-distance based probability: closer to planet centroid → higher prob."""
    d_p = float(np.linalg.norm(embedding - planet_centroid))
    d_n = float(np.linalg.norm(embedding - noplanet_centroid))
    # Softmax over negative distances; temperature=0.1 sharpens decision
    T = 0.1
    ep = np.exp(-d_p / T)
    en = np.exp(-d_n / T)
    total = ep + en
    return float(ep / total) if total > 0 else 0.5


# ── Main Detector ─────────────────────────────────────────────────────

class ExoplanetDetector:
    """Loads and runs all pre-trained exoplanet detection models."""

    def __init__(self):
        self._models = {}           # sklearn models: key → (model, scaler_key)
        self._torch_models = {}     # pytorch models: key → (model, scaler_key, is_siamese)
        self._scalers = {}
        self._siamese_protos = {}   # key → (planet_centroid, noplanet_centroid)
        self._calibration = {}      # key → (threshold, raw_min, raw_max)
        self._errors = {}

        self._load_scalers()
        self._load_sklearn_models()
        if HAS_TORCH:
            self._load_torch_models()

    # ── Scalers ──────────────────────────────────────────────────────

    def _load_scalers(self):
        for key, fname in [
            ('cleaned', 'cleaned_scaler.pkl'),
            ('nn',      'nn_scaler.pkl'),
            ('default', 'scaler (1).pkl'),
        ]:
            try:
                self._scalers[key] = joblib.load(os.path.join(MODELS_DIR, fname))
            except Exception as e:
                self._errors[f'scaler_{key}'] = str(e)

    def _scaler(self, key):
        return self._scalers.get(key) or self._scalers.get('default')

    # ── sklearn / XGBoost ────────────────────────────────────────────

    def _load_sklearn_models(self):
        for key, fname, scaler_key in [
            ('logistic_regression', 'logistic_regression.pkl', 'default'),
            ('random_forest',       'random_forest.pkl',       'default'),
            ('xgboost',             'xgboost.pkl',             'default'),
        ]:
            path = os.path.join(MODELS_DIR, fname)
            try:
                self._models[key] = (joblib.load(path), scaler_key)
            except Exception as e:
                self._errors[key] = str(e)

    def _sklearn_predict(self, flux500, model_key):
        model, scaler_key = self._models[model_key]
        scaler = self._scaler(scaler_key)
        x = flux500.reshape(1, -1)
        if scaler is not None:
            x = scaler.transform(x)
        proba = model.predict_proba(x)[0]
        prob = float(proba[1])
        return _make_result(prob, threshold=0.5, raw_min=0.0, raw_max=1.0)

    # ── PyTorch ──────────────────────────────────────────────────────

    def _load_torch_models(self):
        configs = [
            # (key, class, filename, scaler_key, is_siamese)
            ('cleaned_siamese', _SiameseEncoder, 'cleaned_siamese.pth',        'cleaned', True),
            ('siamese',         _SiameseNet,     'siamese_dataset500.pth',      'default', True),
            ('feedforward_nn',  _FeedforwardNN,  'feedforwardnn_best.pth',       'nn',     False),
            ('cnn',             _CNN,             'cnn.pth',                      'nn',     False),
            ('conv_nn',         _ConvolutionalNN,'convolutionalnn_best.pth',      'nn',     False),
        ]
        for key, cls, fname, scaler_key, is_siamese in configs:
            path = os.path.join(MODELS_DIR, fname)
            try:
                model = cls()
                sd = torch.load(path, map_location='cpu', weights_only=False)
                model.load_state_dict(sd, strict=True)
                model.eval()
                self._torch_models[key] = (model, scaler_key, is_siamese)

                scaler = self._scaler(scaler_key)
                if is_siamese:
                    planet_c, noplanet_c = _build_siamese_prototypes(model, scaler)
                    self._siamese_protos[key] = (planet_c, noplanet_c)
                else:
                    thr, rmin, rmax = _calibrate_classifier(model, scaler)
                    self._calibration[key] = (thr, rmin, rmax)

            except Exception as e:
                self._errors[key] = str(e)

    def _torch_predict(self, flux500, model_key):
        model, scaler_key, is_siamese = self._torch_models[model_key]
        scaler = self._scaler(scaler_key)

        x = flux500.reshape(1, -1).astype(np.float32)
        if scaler is not None:
            x = scaler.transform(x).astype(np.float32)

        x_tensor = torch.from_numpy(x)

        if is_siamese:
            model.eval()
            with torch.no_grad():
                emb = model(x_tensor).squeeze().numpy()
            if model_key in self._siamese_protos:
                planet_c, noplanet_c = self._siamese_protos[model_key]
                prob = _siamese_prob(emb, planet_c, noplanet_c)
            else:
                prob = 0.5
            return _make_result(prob, threshold=0.5, raw_min=0.0, raw_max=1.0)
        else:
            prob = float(_torch_infer(model, x_tensor))
            thr, rmin, rmax = self._calibration.get(model_key, (0.5, 0.0, 1.0))
            return _make_result(prob, threshold=thr, raw_min=rmin, raw_max=rmax)

    # ── Public API ───────────────────────────────────────────────────

    def predict(self, flux, model_key='cleaned_siamese'):
        """Run the named model on a flux array. Returns prediction dict.

        Uses a multi-shift scan: tests the original flux, a version with the
        deepest dip centered at position 250 (matching training data distribution),
        and several roll offsets, then returns the highest-probability result.
        This ensures transits at any position in the curve get detected.
        """
        flux500 = _to_500(flux)

        if model_key in self._errors and model_key not in self._models \
                and model_key not in self._torch_models:
            raise ValueError(
                f"Model '{model_key}' failed to load: {self._errors[model_key]}"
            )

        if model_key in self._models:
            return _max_scan_predict(
                lambda f: self._sklearn_predict(f, model_key), flux500)

        if model_key in self._torch_models:
            return _max_scan_predict(
                lambda f: self._torch_predict(f, model_key), flux500)

        raise ValueError(
            f"Model '{model_key}' not found. "
            f"Available: {list(self._models) + list(self._torch_models)}"
        )

    def available_models(self):
        return list(self._models.keys()) + list(self._torch_models.keys())

    def load_errors(self):
        return dict(self._errors)


def _torch_infer(model, x_tensor):
    model.eval()
    with torch.no_grad():
        out = model(x_tensor)
    return float(out.squeeze())
