"""
EXO-VISION AI LAB — Advanced Exoplanet Detection Dashboard
Streamlit application with ML pipeline, visualizations, and analytics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

from pipeline import DataPipeline, detect_csv_format
from models import ExoplanetDetector
from utils import calculate_snr, estimate_transit_depth, estimate_period, find_transit_dips
from styles import CSS
from kepler_fetch import fetch_folded_flux

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Page Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="EXO-VISION AI LAB",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Session State
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if "detector" not in st.session_state:
    with st.spinner("🧠 Loading pre-trained AI models from disk…"):
        st.session_state.detector = ExoplanetDetector()
    errs = st.session_state.detector.load_errors()
    if errs:
        st.warning(f"⚠️ Some models failed to load: {list(errs.keys())}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Plotly Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#e8e6f0"),
)

STEP_COLORS = {
    'raw': '#6366f1',
    'cleaned': '#8b5cf6',
    'clipped': '#f97316',
    'detrended': '#a855f7',
    'normalized': '#d946ef',
    'smoothed': '#00e5ff',
}

STEP_LABELS = {
    'raw': '① Raw flux',
    'cleaned': '② Cleaned (NaN / Inf filled)',
    'clipped': '③ Outliers clipped',
    'detrended': '④ Detrended (median baseline removed)',
    'normalized': '⑤ Normalized (MAD z-score)',
    'smoothed': '⑥ Smoothed (Savitzky-Golay)',
}


def _y_margin(y, frac=0.08):
    """Return a [lo, hi] range with a small margin so data never touches the edge."""
    y = np.asarray(y)
    if y.size == 0 or not np.isfinite(y).any():
        return None
    lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
    span = hi - lo
    if span == 0:
        span = max(abs(hi), 1.0) * 0.1
    return [lo - span * frac, hi + span * frac]


def make_main_light_curve(pipe, dips):
    """Processed light curve in MAD z-score units. Dips appear as deep negatives."""
    fig = go.Figure()

    proc = pipe.processed_flux
    time_ax = pipe.time_axis
    proc_x = time_ax if time_ax is not None and len(time_ax) == len(proc) \
        else np.arange(len(proc))

    # Shaded σ-bands for visual reference
    fig.add_hrect(y0=-1, y1=1, fillcolor='rgba(255,255,255,0.025)',
                  line_width=0, layer='below')
    fig.add_hrect(y0=-3, y1=-1, fillcolor='rgba(168,85,247,0.04)',
                  line_width=0, layer='below')

    # Baseline at 0 (median in z-score space)
    fig.add_hline(y=0, line_dash='dot', line_color='rgba(255,255,255,0.25)',
                  line_width=1,
                  annotation_text='Baseline (0σ)',
                  annotation_position='top left',
                  annotation_font_color='rgba(255,255,255,0.45)')

    # Processed flux (bold, NO fill-to-zero — y-axis auto-zooms to data)
    fig.add_trace(go.Scatter(
        x=proc_x, y=proc, mode='lines', name='Processed (z-score)',
        line=dict(color='#00e5ff', width=1.8)))

    # Dip markers
    if dips:
        dip_x = [proc_x[d['index']] if d['index'] < len(proc_x) else d['index']
                 for d in dips]
        dip_y = [d['value'] for d in dips]
        dip_text = [
            f"σ-depth: {abs(d['value']):.1f}σ · "
            f"phys depth: {d.get('depth_pct', 0):.3f}%"
            for d in dips
        ]
        fig.add_trace(go.Scatter(
            x=dip_x, y=dip_y,
            mode='markers+text', name='Transit Dips',
            marker=dict(color='#f87171', size=13, symbol='triangle-down',
                        line=dict(width=1.5, color='#fff')),
            text=['▼' for _ in dips],
            textposition='top center',
            textfont=dict(color='#f87171', size=13),
            hovertext=dip_text, hoverinfo='text'))

    # Auto-zoom y-range with margin (this is the fix for dips looking flat)
    y_range = _y_margin(proc, frac=0.12)

    fig.update_layout(**PLOTLY_LAYOUT, height=440,
                      margin=dict(l=60, r=30, t=40, b=50),
                      showlegend=True,
                      legend=dict(orientation='h', y=1.12, x=0.5, xanchor='center'),
                      xaxis_title='Time' if pipe.csv_format == 'time_flux' else 'Bin index',
                      yaxis_title='Flux (MAD σ from baseline)')
    if y_range is not None:
        fig.update_yaxes(range=y_range)
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.04)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.04)')
    return fig


def make_transit_zoom_plot(pipe, dips, pad=30):
    """Zoomed view centered on the deepest dip, in physical-ratio units."""
    if not dips or pipe.model_flux is None:
        return None

    deepest = max(dips, key=lambda d: abs(d['value']))
    idx = deepest['index']
    ratio = pipe.model_flux
    n = len(ratio)
    s, e = max(0, idx - pad), min(n, idx + pad + 1)
    if e - s < 5:
        return None

    x = np.arange(s, e)
    y = ratio[s:e]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines+markers',
        line=dict(color='#00e5ff', width=2),
        marker=dict(size=5, color='#00e5ff'),
        name='Flux (ratio)'))

    baseline = float(np.median(ratio))
    fig.add_hline(y=baseline, line_dash='dot',
                  line_color='rgba(255,255,255,0.3)',
                  annotation_text=f'Baseline {baseline:.5f}',
                  annotation_font_color='rgba(255,255,255,0.4)')
    fig.add_vline(x=idx, line_dash='dash', line_color='#f87171',
                  annotation_text=f'Dip @ {idx}',
                  annotation_font_color='#f87171')

    y_range = _y_margin(y, frac=0.2)
    fig.update_layout(**PLOTLY_LAYOUT, height=300,
                      margin=dict(l=60, r=30, t=40, b=40),
                      title='🔍 Transit Zoom — deepest dip in physical flux',
                      xaxis_title='Bin index', yaxis_title='Relative flux (median=1)')
    if y_range is not None:
        fig.update_yaxes(range=y_range)
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.04)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.04)')
    return fig


def make_step_comparison(pipe):
    """Create a stacked subplot showing each pipeline step."""
    steps = pipe.get_step_names()
    n = len(steps)
    if n == 0:
        return None

    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=[STEP_LABELS.get(s, s) for s in steps])

    for i, step in enumerate(steps):
        data = pipe.get_step_data(step)
        if data is None:
            continue
        color = STEP_COLORS.get(step, '#00e5ff')
        fig.add_trace(go.Scatter(
            y=data, mode='lines', name=STEP_LABELS.get(step, step),
            line=dict(color=color, width=1.2),
            showlegend=False), row=i + 1, col=1)

    fig.update_layout(**PLOTLY_LAYOUT, height=max(200, 120 * n),
                      margin=dict(l=60, r=30, t=30, b=30))
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.03)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.03)')

    # Style subplot titles
    for ann in fig.layout.annotations:
        ann.font.size = 11
        ann.font.color = 'rgba(232,230,240,0.6)'
        ann.font.family = 'JetBrains Mono, monospace'

    return fig


def make_phase_folded_plot(pipe):
    """Phase-folded curve in z-score units. Dip centered at phase=0.5."""
    if pipe.folded_phase is None or pipe.folded_flux is None:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pipe.folded_phase, y=pipe.folded_flux,
        mode='markers+lines', name='Phase-folded',
        marker=dict(color='#a855f7', size=6, opacity=0.85),
        line=dict(color='#a855f7', width=1.5)))

    fig.add_hline(y=0, line_dash='dot', line_color='rgba(255,255,255,0.2)',
                  annotation_text='Baseline (0σ)',
                  annotation_font_color='rgba(255,255,255,0.4)')

    period_label = f'period ≈ {pipe.estimated_period:.3f}' \
        if pipe.estimated_period else 'unknown'
    y_range = _y_margin(pipe.folded_flux, frac=0.15)

    fig.update_layout(**PLOTLY_LAYOUT, height=340,
                      margin=dict(l=60, r=30, t=50, b=50),
                      title=f'🌀 Phase-folded light curve ({period_label})',
                      xaxis_title='Phase (dip centered at 0.5)',
                      yaxis_title='Flux (MAD σ)')
    if y_range is not None:
        fig.update_yaxes(range=y_range)
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.04)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.04)')
    return fig


def make_bls_power_plot(pipe):
    """Show the BLS-style period search power spectrum, if available."""
    if pipe.period_power is None:
        return None
    p = pipe.period_power['periods']
    pw = pipe.period_power['powers']
    fig = go.Figure(go.Scatter(
        x=p, y=pw, mode='lines',
        line=dict(color='#00e5ff', width=1.5),
        fill='tozeroy', fillcolor='rgba(0,229,255,0.06)'))
    if pipe.estimated_period:
        fig.add_vline(x=pipe.estimated_period, line_dash='dash',
                      line_color='#f87171',
                      annotation_text=f'P = {pipe.estimated_period:.3f}',
                      annotation_font_color='#f87171')
    fig.update_layout(**PLOTLY_LAYOUT, height=260,
                      margin=dict(l=60, r=30, t=50, b=40),
                      title='📈 BLS-style period search',
                      xaxis_title='Trial period',
                      yaxis_title='Dip depth (σ)',
                      xaxis_type='log')
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.04)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.04)')
    return fig


def make_gauge(value, title, color='#00e5ff'):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, title={'text': title, 'font': {'size': 14}},
        number={'suffix': '%', 'font': {'size': 28, 'color': color}},
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor='rgba(255,255,255,0.2)'),
            bar=dict(color=color),
            bgcolor='rgba(255,255,255,0.03)',
            borderwidth=1, bordercolor='rgba(255,255,255,0.06)',
            steps=[dict(range=[0, 50], color='rgba(248,113,113,0.08)'),
                   dict(range=[50, 100], color='rgba(0,229,255,0.08)')],
            threshold=dict(line=dict(color='#fff', width=2), thickness=0.8, value=value))))
    fig.update_layout(**PLOTLY_LAYOUT, height=220, margin=dict(l=30, r=30, t=55, b=15))
    return fig


def make_feature_bars(features):
    keys = ['mean', 'std', 'skewness', 'kurtosis', 'iqr', 'depth_estimate', 'mad', 'rms', 'depth_pct']
    vals = [features.get(k, 0) for k in keys]
    labels = [k.replace('_', ' ').title() for k in keys]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation='h',
        marker=dict(color=vals, colorscale=[[0, '#a855f7'], [1, '#00e5ff']]),
        text=[f"{v:.6f}" for v in vals], textposition='auto'))
    fig.update_layout(**PLOTLY_LAYOUT, height=350, margin=dict(l=50, r=30, t=50, b=40),
                      title="Extracted Features", xaxis_title="Value", yaxis=dict(autorange='reversed'))
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model Registry
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL_REGISTRY = [
    {"name": "Cleaned Siamese", "key": "cleaned_siamese", "accuracy": 81.73, "category": "Deep Learning", "input": "500-pt flux (cleaned)", "file": "cleaned_siamese.pth"},
    {"name": "XGBoost", "key": "xgboost", "accuracy": 75.53, "category": "Ensemble", "input": "500-pt raw flux", "file": "xgboost.pkl"},
    {"name": "Original Siamese", "key": "siamese", "accuracy": 75.06, "category": "Deep Learning", "input": "500-pt raw flux", "file": "siamese_dataset500.pth"},
    {"name": "ConvNet Best", "key": "conv_nn", "accuracy": 74.50, "category": "Deep Learning", "input": "500-pt raw flux", "file": "convolutionalnn_best.pth"},
    {"name": "CNN", "key": "cnn", "accuracy": 73.17, "category": "Deep Learning", "input": "500-pt raw flux", "file": "cnn.pth"},
    {"name": "Random Forest", "key": "random_forest", "accuracy": 73.33, "category": "Ensemble", "input": "500-pt raw flux", "file": "random_forest.pkl"},
    {"name": "Feedforward NN", "key": "feedforward_nn", "accuracy": 70.97, "category": "Deep Learning", "input": "500-pt raw flux", "file": "feedforwardnn_best.pth"},
    {"name": "Logistic Regression", "key": "logistic_regression", "accuracy": 70.42, "category": "Linear", "input": "500-pt raw flux", "file": "logistic_regression.pkl"},
]
MODEL_NAMES = [m["name"] for m in MODEL_REGISTRY]
MODEL_MAP = {m["name"]: m["key"] for m in MODEL_REGISTRY}
MODEL_INFO = {m["key"]: m for m in MODEL_REGISTRY}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Header
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<div class="hero-header">
    <div class="hero-glow"></div>
    <h1 class="hero-title">🌌 EXO-VISION AI LAB</h1>
    <p class="hero-subtitle">Advanced Exoplanet Detection System · Neural Transit Analysis</p>
    <div class="hero-line"></div>
</div>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sidebar
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cat_colors = {"Deep Learning": "#00e5ff", "Ensemble": "#a855f7", "Linear": "#facc15"}

with st.sidebar:
    st.markdown("## 🔧 Configuration")
    st.markdown("---")

    selected = st.selectbox("🤖 Detection Model", MODEL_NAMES,
                            help="Models ranked by accuracy on test set.")
    model_key = MODEL_MAP[selected]
    info = MODEL_INFO[model_key]
    cat_color = cat_colors.get(info['category'], '#00e5ff')

    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);
                border-radius:12px;padding:1rem;margin:0.5rem 0;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
            <span style="color:{cat_color};font-size:0.75rem;font-weight:600;
                         text-transform:uppercase;letter-spacing:0.05em;">{info['category']}</span>
            <span style="color:#00e5ff;font-family:'JetBrains Mono',monospace;
                         font-size:1.1rem;font-weight:700;">{info['accuracy']}%</span>
        </div>
        <div style="color:rgba(232,230,240,0.5);font-size:0.8rem;">Input: {info['input']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Pipeline Parameters")
    n_bins = st.slider("📊 Binning Points", 100, 1000, 500, 50)
    smooth_w = st.slider("🔧 Smooth Window", 3, 51, 7, 2)
    smooth_p = st.slider("📐 Poly Order", 1, 5, 3)
    detrend_w = st.slider("📉 Detrend Window", 51, 401, 201, 10,
                           help="Running median window for detrending. Must be larger than any transit width. Larger = safer for wide transits.")
    sigma_pos = st.slider("✂️ Positive σ (flares)", 1.0, 5.0, 3.0, 0.5,
                           help="Clipping threshold for positive spikes/flares only. Transit dips are always preserved.")
    sigma_neg = 999.0  # effectively disabled — transits must not be clipped

    st.markdown("---")
    st.markdown("### 🏆 Model Leaderboard")
    for m in MODEL_REGISTRY:
        bar_pct = m['accuracy']
        c = cat_colors.get(m['category'], '#00e5ff')
        active = '▪' if m['key'] == model_key else ' '
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:0.4rem;margin-bottom:0.35rem;font-size:0.78rem;">
            <span style="color:{'#00e5ff' if m['key'] == model_key else 'rgba(232,230,240,0.3)'};
                         font-weight:{'700' if m['key'] == model_key else '400'};
                         min-width:110px;">{active} {m['name']}</span>
            <div style="flex:1;height:6px;background:rgba(255,255,255,0.04);border-radius:3px;overflow:hidden;">
                <div style="width:{bar_pct}%;height:100%;background:{c};border-radius:3px;transition:width 0.3s;"></div>
            </div>
            <span style="color:{c};font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                         min-width:45px;text-align:right;">{m['accuracy']}%</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="text-align:center;opacity:0.3;font-size:0.7rem;margin-top:2rem;">'
                'EXO-VISION AI LAB v3.0<br/>Stellar Intelligence Engine</div>', unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Upload Section
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""<div class="section-header">
    <span class="section-tag">// data::upload</span>
    <h2 class="section-title">📡 Stellar Data Upload</h2>
    <p class="section-subtitle">Upload light curve CSV files for transit analysis</p>
</div>""", unsafe_allow_html=True)

c1, c2 = st.columns([2, 1])
with c1:
    files = st.file_uploader("Upload CSV file(s)", type=["csv"],
                             accept_multiple_files=True,
                             help="Supports: time+flux CSVs, wide-format binned curves.")
with c2:
    st.markdown("""<div class="glass-card info-card"><h4>📋 Supported Formats</h4><ul>
        <li><code>time, flux</code> columns (Kepler/TESS)</li>
        <li>Wide-format (cols = time bins per star)</li>
        <li><strong>KOI table</strong> (kepid, koi_period, koi_time0bk, koi_disposition)</li>
        <li>Auto-detects format & rejects non-light-curves</li></ul></div>""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# KOI Table Renderer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_DISP_COLOR = {
    'CONFIRMED':     '#4ade80',
    'CANDIDATE':     '#facc15',
    'FALSE POSITIVE':'#f87171',
    'NOT DISPOSITIONED': '#94a3b8',
}

def _render_koi_table(file_obj, fi, model_key, model_name):
    """Download light curves for each KOI row and run the selected model."""
    file_obj.seek(0)
    try:
        koi_df = pd.read_csv(file_obj, comment='#')
    except Exception as e:
        st.error(f"Could not parse KOI table: {e}")
        return

    # Normalise column names
    koi_df.columns = [c.strip().lower() for c in koi_df.columns]
    required = ['kepid', 'koi_period', 'koi_time0bk']
    missing = [c for c in required if c not in koi_df.columns]
    if missing:
        st.error(f"KOI table missing required columns: {missing}")
        return

    has_disp    = 'koi_disposition' in koi_df.columns
    has_name    = 'kepoi_name'       in koi_df.columns
    n_rows      = len(koi_df)

    st.markdown(f"""<div class="section-header">
        <span class="section-tag">// koi::table</span>
        <h2 class="section-title">🔭 KOI Metadata Table — {n_rows} objects</h2>
        <p class="section-subtitle">Downloading Kepler light curves, phase-folding, and running <strong>{model_name}</strong></p>
    </div>""", unsafe_allow_html=True)

    if has_disp:
        counts = koi_df['koi_disposition'].value_counts()
        cols_c = st.columns(len(counts))
        for ci, (disp, cnt) in enumerate(counts.items()):
            col = _DISP_COLOR.get(str(disp).upper(), '#94a3b8')
            with cols_c[ci]:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">{disp}</div>
                    <div class="metric-value" style="color:{col}">{cnt}</div>
                    <div class="metric-unit">objects</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # --- Process each row ---
    rows_out = []
    overall_bar = st.progress(0, text="Starting…")

    for ri, (_, row) in enumerate(koi_df.iterrows()):
        overall_bar.progress((ri) / n_rows,
                             text=f"Processing {ri+1}/{n_rows}…")

        kepid  = int(row['kepid'])
        period = float(row['koi_period'])
        t0     = float(row['koi_time0bk'])
        disp   = str(row.get('koi_disposition', '')).upper().strip() if has_disp else '—'
        koi_name = str(row.get('kepoi_name', f'KIC {kepid}')) if has_name else f'KIC {kepid}'

        if period <= 0:
            rows_out.append({'KOI': koi_name, 'KIC': kepid,
                             'Known': disp, 'Period (d)': period,
                             'Model': '—', 'Prob %': 0,
                             'Verdict': 'Error', 'Dip %': 0,
                             'Transits': 0, 'Status': 'Invalid period'})
            continue

        flux, n_pts, n_transit, err = fetch_folded_flux(kepid, period, t0)

        if err or flux is None:
            rows_out.append({'KOI': koi_name, 'KIC': kepid,
                             'Known': disp, 'Period (d)': round(period, 4),
                             'Model': '—', 'Prob %': 0,
                             'Verdict': 'Download failed', 'Dip %': 0,
                             'Transits': 0, 'Status': err or 'No data'})
            continue

        # Run model
        try:
            result = st.session_state.detector.predict(flux, model_key)
        except Exception as e:
            rows_out.append({'KOI': koi_name, 'KIC': kepid,
                             'Known': disp, 'Period (d)': round(period, 4),
                             'Model': model_name, 'Prob %': 0,
                             'Verdict': 'Model error', 'Dip %': 0,
                             'Transits': n_transit, 'Status': str(e)})
            continue

        dip_pct = round((1.0 - float(np.min(flux))) * 100, 4)
        prob    = round(result['probability'], 1)
        verdict = result['prediction']

        # Agreement check
        if has_disp and disp in ('CONFIRMED', 'FALSE POSITIVE'):
            expected_planet = (disp == 'CONFIRMED')
            got_planet      = result['is_planet']
            status = 'Correct' if expected_planet == got_planet else 'Wrong'
        else:
            status = 'CANDIDATE — check result'

        rows_out.append({
            'KOI':        koi_name,
            'KIC':        kepid,
            'Known':      disp,
            'Period (d)': round(period, 4),
            'Model':      model_name,
            'Prob %':     prob,
            'Verdict':    verdict,
            'Dip %':      dip_pct,
            'Transits':   n_transit,
            'Status':     status,
        })

    overall_bar.progress(1.0, text="Done!")
    time.sleep(0.3)
    overall_bar.empty()

    if not rows_out:
        st.warning("No rows could be processed.")
        return

    result_df = pd.DataFrame(rows_out)

    # ── Summary metrics ───────────────────────────────────
    confirmed_detected = sum(
        1 for r in rows_out
        if r['Known'] == 'CONFIRMED' and r['Verdict'] == 'Planet Detected'
    )
    fp_rejected = sum(
        1 for r in rows_out
        if r['Known'] == 'FALSE POSITIVE' and r['Verdict'] == 'No Planet'
    )
    n_confirmed = sum(1 for r in rows_out if r['Known'] == 'CONFIRMED')
    n_fp        = sum(1 for r in rows_out if r['Known'] == 'FALSE POSITIVE')
    n_correct   = sum(1 for r in rows_out if r['Status'] == 'Correct')
    n_evaluated = sum(1 for r in rows_out if r['Status'] in ('Correct', 'Wrong'))

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Objects Processed</div>
            <div class="metric-value">{len(rows_out)}</div>
            <div class="metric-unit">KOI entries</div></div>""", unsafe_allow_html=True)
    with mc2:
        pct = f'{confirmed_detected/n_confirmed*100:.0f}%' if n_confirmed else '—'
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Confirmed Planets Found</div>
            <div class="metric-value" style="color:#4ade80">{confirmed_detected}/{n_confirmed}</div>
            <div class="metric-unit">recall {pct}</div></div>""", unsafe_allow_html=True)
    with mc3:
        pct = f'{fp_rejected/n_fp*100:.0f}%' if n_fp else '—'
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">False Positives Rejected</div>
            <div class="metric-value" style="color:#f87171">{fp_rejected}/{n_fp}</div>
            <div class="metric-unit">specificity {pct}</div></div>""", unsafe_allow_html=True)
    with mc4:
        pct = f'{n_correct/n_evaluated*100:.0f}%' if n_evaluated else '—'
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Overall Accuracy</div>
            <div class="metric-value">{n_correct}/{n_evaluated}</div>
            <div class="metric-unit">{pct} on confirmed/FP rows</div></div>""", unsafe_allow_html=True)

    # ── Results table ──────────────────────────────────────
    st.markdown("#### Detection Results")

    def _row_color(row):
        styles = [''] * len(row)
        col_list = list(row.index)
        if 'Verdict' in col_list:
            vi = col_list.index('Verdict')
            styles[vi] = 'color: #4ade80; font-weight:700' if row['Verdict'] == 'Planet Detected' \
                         else 'color: #f87171'
        if 'Known' in col_list:
            ki = col_list.index('Known')
            styles[ki] = f'color: {_DISP_COLOR.get(str(row["Known"]).upper(), "#94a3b8")}'
        if 'Status' in col_list:
            si = col_list.index('Status')
            if row['Status'] == 'Correct':
                styles[si] = 'color: #4ade80'
            elif row['Status'] == 'Wrong':
                styles[si] = 'color: #f87171; font-weight:700'
        return styles

    styled = result_df.style.apply(_row_color, axis=1).format(
        {'Prob %': '{:.1f}', 'Dip %': '{:.4f}', 'Period (d)': '{:.4f}'}
    )
    st.dataframe(styled, use_container_width=True,
                 height=min(600, 60 + 35 * len(result_df)),
                 key=f"koi_table_{fi}")

    # ── Probability bar chart ──────────────────────────────
    fig = go.Figure()
    colors = [
        '#4ade80' if r['Verdict'] == 'Planet Detected' else '#f87171'
        for r in rows_out
    ]
    fig.add_trace(go.Bar(
        x=[r['KOI'] for r in rows_out],
        y=[r['Prob %'] for r in rows_out],
        marker_color=colors,
        text=[f"{r['Prob %']:.1f}%" for r in rows_out],
        textposition='outside',
        customdata=[[r['Known'], r['Status']] for r in rows_out],
        hovertemplate='<b>%{x}</b><br>Prob: %{y:.1f}%<br>Known: %{customdata[0]}<br>Status: %{customdata[1]}<extra></extra>',
    ))
    fig.add_hline(y=50, line_dash='dash', line_color='rgba(255,255,255,0.35)',
                  annotation_text='Detection threshold (50%)')
    fig.update_layout(**PLOTLY_LAYOUT, height=420,
                      margin=dict(l=50, r=30, t=60, b=100),
                      title=f'Planet Probability — {model_name}',
                      xaxis_title='KOI', yaxis_title='Planet Probability %',
                      yaxis=dict(range=[0, 115]),
                      xaxis_tickangle=-40)
    st.plotly_chart(fig, use_container_width=True, key=f"koi_bar_{fi}")

    # ── Download ────────────────────────────────────────────
    csv_out = result_df.to_csv(index=False)
    st.download_button('📥 Download KOI Results (CSV)', csv_out,
                       f'koi_detection_{model_name}.csv', 'text/csv',
                       use_container_width=True, key=f"koi_dl_{fi}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Processing + Results
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
all_results = []

if files:
    for fi, f in enumerate(files):
        st.markdown("---")
        st.markdown(f'<div class="section-header"><span class="section-tag">// star::{fi+1}</span>'
                    f'<h2 class="section-title">⭐ {f.name}</h2></div>', unsafe_allow_html=True)

        try:
            # ── Pre-check format ─────────────────────────────
            f.seek(0)
            pre_df = pd.read_csv(f, comment='#', header='infer', nrows=5)
            fmt_type, fmt_info = detect_csv_format(pre_df)

            # ── KOI metadata table ────────────────────────────
            if fmt_type == 'koi_table':
                _render_koi_table(f, fi, model_key, selected)
                continue

            if fmt_type in ('tce_table', 'feature_table', 'unknown'):
                msg = fmt_info.get('message', 'Unrecognized format')
                st.error(f"🚫 **Cannot process this file:** {msg}")
                continue

            # Format badge
            badge_class = 'time-flux' if fmt_type == 'time_flux' else 'wide'
            badge_text = f"📊 {fmt_type.replace('_', ' ').upper()}"
            if fmt_type == 'time_flux':
                badge_text += f" · {fmt_info.get('flux_col', 'flux')} column"
            elif fmt_type == 'wide_lightcurve':
                badge_text += f" · {fmt_info.get('n_rows', '?')} stars × {fmt_info.get('n_flux_bins', '?')} bins"
            st.markdown(f'<span class="format-badge {badge_class}">{badge_text}</span>', unsafe_allow_html=True)

            # ── Ground-truth label (auto-detected from filename) ──────
            _fname_lower = f.name.lower()
            if any(k in _fname_lower for k in ('planet_clear', 'hot_jupiter', 'multi_transit',
                                                'subtle_planet', 'val_planet', 'val2_planet',
                                                'kepler10b', 'kepler7b', 'tres2b', 'real_')):
                _auto_gt = 'Planet (confirmed)'
            elif any(k in _fname_lower for k in ('no_planet', 'val_noplanet', 'val2_noplanet')):
                _auto_gt = 'No Planet (confirmed)'
            else:
                _auto_gt = 'Unknown'

            gt_options = ['Unknown', 'Planet (confirmed)', 'No Planet (confirmed)']
            ground_truth = st.selectbox(
                "🏷️ Ground truth (for accuracy scoring)",
                gt_options,
                index=gt_options.index(_auto_gt),
                key=f"gt_{fi}",
                help="Select the known classification to compute model accuracy for this file."
            )

            # Row selector for wide-format
            row_index = 0
            if fmt_type == 'wide_lightcurve' and fmt_info.get('n_rows', 0) > 1:
                row_index = st.number_input(
                    f"Select star row (0–{fmt_info['n_rows']-1})",
                    min_value=0, max_value=fmt_info['n_rows'] - 1, value=0,
                    key=f"row_{fi}")

            # ── Run Pipeline ─────────────────────────────────
            f.seek(0)
            pipe = DataPipeline(n_bins, smooth_w, smooth_p, sigma_pos, sigma_neg, detrend_w)
            bar = st.progress(0, text="Starting pipeline…")

            def _cb(v, t):
                bar.progress(v, text=t)
                time.sleep(0.08)

            processed, features, raw_df = pipe.run(f, row_index=row_index, progress_cb=_cb)
            bar.empty()
            st.success(f"✅ Pipeline complete — {len(processed)} processed points, "
                       f"format: {pipe.csv_format}")

            # ── TABS ─────────────────────────────────────────
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Light Curve",
                "🔬 Pipeline Steps",
                "🤖 Model Results",
                "📊 Features & Analytics",
                "🔍 Model Comparison",
            ])

            # ════════════════════════════════════════════════
            # TAB 1: Light Curve
            # ════════════════════════════════════════════════
            with tab1:
                # Dip detection on the z-score curve (threshold in σ units)
                dips = find_transit_dips(processed, sigma=2.5)
                # Recompute each dip's physical depth_pct from model_flux (ratio)
                if dips and pipe.model_flux is not None:
                    ratio = pipe.model_flux
                    ratio_baseline = float(np.median(ratio))
                    for d in dips:
                        i = d['index']
                        if 0 <= i < len(ratio) and ratio_baseline != 0:
                            d['depth_pct'] = (ratio_baseline - float(ratio[i])) / ratio_baseline * 100

                st.plotly_chart(make_main_light_curve(pipe, dips), use_container_width=True, key=f"lc_{fi}")

                if dips:
                    deepest = max(dips, key=lambda d: abs(d['value']))
                    st.info(
                        f"🔍 Detected **{len(dips)}** transit dip(s). "
                        f"Deepest at bin {deepest['index']}: "
                        f"**{abs(deepest['value']):.1f}σ** below baseline "
                        f"(physical depth ≈ {deepest.get('depth_pct', 0):.3f}%)"
                    )
                else:
                    st.caption("No significant dips detected (threshold: 2.5σ).")

                # Transit zoom around deepest dip (physical ratio)
                zoom_fig = make_transit_zoom_plot(pipe, dips)
                if zoom_fig is not None:
                    st.plotly_chart(zoom_fig, use_container_width=True, key=f"zoom_{fi}")

                # Phase-folded plot
                phase_fig = make_phase_folded_plot(pipe)
                if phase_fig:
                    st.plotly_chart(phase_fig, use_container_width=True, key=f"phase_{fi}")
                elif pipe.estimated_period:
                    st.caption(f"🌀 Estimated period: {pipe.estimated_period:.3f}")

            # ════════════════════════════════════════════════
            # TAB 2: Pipeline Steps
            # ════════════════════════════════════════════════
            with tab2:
                st.markdown('<div class="section-header"><span class="section-tag">// pipeline::steps</span>'
                            '<h2 class="section-title">🔬 Step-by-Step Processing</h2>'
                            '<p class="section-subtitle">See how each pipeline stage transforms the data</p></div>',
                            unsafe_allow_html=True)

                step_fig = make_step_comparison(pipe)
                if step_fig:
                    st.plotly_chart(step_fig, use_container_width=True, key=f"steps_{fi}")

                # Pipeline log
                with st.expander("📜 Pipeline Log", expanded=True):
                    for idx, log in enumerate(pipe.log):
                        st.markdown(
                            f'<div class="step-card">'
                            f'<span class="step-num">{idx+1}</span>'
                            f'<span class="step-name">{log}</span>'
                            f'</div>', unsafe_allow_html=True)

            # ════════════════════════════════════════════════
            # TAB 3: Model Results (selected model)
            # ════════════════════════════════════════════════
            with tab3:
                st.markdown('<div class="section-header"><span class="section-tag">// results::prediction</span>'
                            '<h2 class="section-title">🎯 Detection Results</h2></div>', unsafe_allow_html=True)

                # Models were trained on ratio-normalized flux (~1.0 baseline);
                # feed model_flux, not the z-score display curve.
                model_input = pipe.model_flux if pipe.model_flux is not None else processed
                with st.spinner("🧠 Running AI model…"):
                    result = st.session_state.detector.predict(model_input, model_key)

                rc = st.columns(3)
                is_p = result['is_planet']
                with rc[0]:
                    cls = "planet-detected" if is_p else "no-planet"
                    vcls = "success" if is_p else "danger"
                    icon = "🪐" if is_p else "🌑"
                    txt = "PLANET DETECTED" if is_p else "NO PLANET"
                    st.markdown(f'<div class="glass-card result-card {cls}"><div class="result-icon">{icon}</div>'
                                f'<div class="result-label">Prediction</div>'
                                f'<div class="result-value {vcls}">{txt}</div></div>', unsafe_allow_html=True)
                with rc[1]:
                    st.plotly_chart(make_gauge(result['confidence'], "Confidence"), use_container_width=True, key=f"gauge_conf_{fi}")
                with rc[2]:
                    st.plotly_chart(make_gauge(result['probability'], "Planet Probability", '#a855f7'),
                                    use_container_width=True, key=f"gauge_prob_{fi}")

            # ════════════════════════════════════════════════
            # TAB 4: Features & Analytics
            # ════════════════════════════════════════════════
            with tab4:
                st.markdown('<div class="section-header"><span class="section-tag">// analytics::advanced</span>'
                            '<h2 class="section-title">📊 Features & Advanced Analytics</h2></div>',
                            unsafe_allow_html=True)

                # SNR & transit depth are physical quantities — compute on ratio flux
                ratio_flux = pipe.model_flux if pipe.model_flux is not None else processed
                snr = calculate_snr(ratio_flux)
                td = estimate_transit_depth(ratio_flux)
                per = estimate_period(processed)  # autocorr works on z-score

                # Metrics row
                st.markdown(f"""<div class="metric-row">
                    <div class="metric-card"><div class="metric-label">Signal-to-Noise</div>
                        <div class="metric-value">{snr['snr']:.2f}</div>
                        <div class="metric-unit">{snr['snr_db']:.1f} dB</div></div>
                    <div class="metric-card"><div class="metric-label">Transit Depth</div>
                        <div class="metric-value">{td['depth_pct']:.4f}%</div>
                        <div class="metric-unit">Rp/Rs = {td['radius_ratio']:.4f}</div></div>
                    <div class="metric-card"><div class="metric-label">Est. Period</div>
                        <div class="metric-value">{per['period_idx'] or '—'}</div>
                        <div class="metric-unit">idx · strength {per['strength']:.3f}</div></div>
                    <div class="metric-card"><div class="metric-label">Dip Points</div>
                        <div class="metric-value">{td['n_dip_points']}</div>
                        <div class="metric-unit">below 2σ threshold</div></div>
                </div>""", unsafe_allow_html=True)

                fc1, fc2 = st.columns(2)
                with fc1:
                    st.plotly_chart(make_feature_bars(features), use_container_width=True, key=f"feat_{fi}")
                with fc2:
                    st.markdown('<div class="glass-card"><h4 style="font-family:Space Grotesk;color:#f1f0f5;">'
                                '📊 Statistical Summary</h4>', unsafe_allow_html=True)
                    for k, v in features.items():
                        st.markdown(
                            f'<div class="stat-row">'
                            f'<span class="stat-key">{k.replace("_"," ").title()}</span>'
                            f'<span class="stat-val">{v:.8f}</span></div>',
                            unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # BLS-style period search (primary) + autocorrelation (fallback view)
                bls_fig = make_bls_power_plot(pipe)
                if bls_fig is not None:
                    st.plotly_chart(bls_fig, use_container_width=True, key=f"bls_{fi}")

                if per.get('autocorr') is not None:
                    ac = per['autocorr']
                    ac_fig = go.Figure(go.Scatter(y=ac[:len(ac)//2], mode='lines',
                                                  line=dict(color='#a855f7', width=1.2),
                                                  fill='tozeroy', fillcolor='rgba(168,85,247,0.05)'))
                    ac_fig.update_layout(**PLOTLY_LAYOUT, height=240,
                                         margin=dict(l=50, r=30, t=50, b=40),
                                         title="Autocorrelation (lag-based period hint)",
                                         xaxis_title="Lag", yaxis_title="Correlation")
                    if per['period_idx']:
                        ac_fig.add_vline(x=per['period_idx'], line_dash='dash',
                                         line_color='#00e5ff',
                                         annotation_text=f"Lag ≈ {per['period_idx']}")
                    st.plotly_chart(ac_fig, use_container_width=True, key=f"autocorr_{fi}")

            # ════════════════════════════════════════════════
            # TAB 5: Model Comparison
            # ════════════════════════════════════════════════
            with tab5:
                st.markdown('<div class="section-header"><span class="section-tag">// models::comparison</span>'
                            '<h2 class="section-title">🔍 All-Model Comparison</h2>'
                            '<p class="section-subtitle">Run all 7 models and compare results</p></div>',
                            unsafe_allow_html=True)

                model_input = pipe.model_flux if pipe.model_flux is not None else processed
                with st.spinner("🧠 Running all models…"):
                    all_model_results = {}
                    for m in MODEL_REGISTRY:
                        try:
                            r = st.session_state.detector.predict(model_input, m['key'])
                            all_model_results[m['key']] = r
                        except Exception as e:
                            all_model_results[m['key']] = {
                                'prediction': 'Error', 'is_planet': False,
                                'confidence': 0, 'probability': 0,
                            }

                # Consensus
                planet_votes = sum(1 for r in all_model_results.values() if r['is_planet'])
                total_models = len(all_model_results)
                consensus_planet = planet_votes > total_models / 2
                avg_prob = np.mean([r['probability'] for r in all_model_results.values()])

                con_cls = "planet" if consensus_planet else "no-planet"
                con_icon = "🪐" if consensus_planet else "🌑"
                con_text = "PLANET DETECTED" if consensus_planet else "NO PLANET"
                con_color = "#00e5ff" if consensus_planet else "#f87171"

                st.markdown(f"""
                <div class="consensus-badge {con_cls}">
                    <div class="consensus-icon">{con_icon}</div>
                    <div class="consensus-label">Consensus ({planet_votes}/{total_models} models)</div>
                    <div class="consensus-value" style="color:{con_color}">{con_text}</div>
                    <div class="consensus-detail">Average probability: {avg_prob:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("")

                # Model cards grid
                cols = st.columns(3)
                for idx, m in enumerate(MODEL_REGISTRY):
                    r = all_model_results.get(m['key'], {})
                    is_planet = r.get('is_planet', False)
                    prob = r.get('probability', 0)
                    conf = r.get('confidence', 0)
                    pred = r.get('prediction', 'Error')
                    c = cat_colors.get(m['category'], '#00e5ff')
                    res_color = '#4ade80' if is_planet else '#f87171'

                    with cols[idx % 3]:
                        st.markdown(f"""
                        <div class="model-card">
                            <div class="model-card-header">
                                <span class="model-card-name">{m['name']}</span>
                                <span class="model-card-cat" style="background:{c}22;color:{c};border:1px solid {c}44;">{m['category']}</span>
                            </div>
                            <div class="model-card-result" style="color:{res_color}">
                                {'🪐' if is_planet else '🌑'} {pred}
                            </div>
                            <div class="model-card-bar">
                                <div class="model-card-bar-fill" style="width:{prob}%;background:{res_color};"></div>
                            </div>
                            <div class="model-card-meta">
                                Prob: {prob:.1f}% · Conf: {conf:.1f}% · Acc: {m['accuracy']}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            # ── Collect results for report ────────────────────
            gt_is_planet = (ground_truth == 'Planet (confirmed)')
            gt_is_nop    = (ground_truth == 'No Planet (confirmed)')
            if gt_is_planet or gt_is_nop:
                status = 'Correct' if (is_p == gt_is_planet) else 'Wrong'
            else:
                status = '—'

            all_results.append({
                'file':              f.name,
                'ground_truth':      ground_truth,
                'prediction':        result['prediction'],
                'is_planet':         is_p,
                'status':            status,
                'confidence':        result['confidence'],
                'probability':       result['probability'],
                'snr':               snr['snr'],
                'snr_db':            snr['snr_db'],
                'transit_depth_pct': td['depth_pct'],
                'radius_ratio':      td['radius_ratio'],
                'n_dips':            len(dips),
                'n_dip_points':      td['n_dip_points'],
                'period_idx':        per['period_idx'],
                'period_strength':   per['strength'],
                'data_points':       len(processed),
                'model_used':        selected,
            })

        except ValueError as e:
            st.error(f"❌ {e}")
        except Exception as e:
            st.error(f"❌ Processing Error: {e}")
            st.exception(e)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Detection Report — Summary of All Files
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if all_results:
        st.markdown("---")
        st.markdown('<div class="section-header"><span class="section-tag">// report::summary</span>'
                    '<h2 class="section-title">📋 Complete Detection Report</h2>'
                    '<p class="section-subtitle">Aggregated results from all analyzed light curves</p></div>',
                    unsafe_allow_html=True)

        total      = len(all_results)
        planets    = sum(1 for r in all_results if r['is_planet'])
        no_planets = total - planets
        avg_conf   = np.mean([r['confidence'] for r in all_results])
        avg_prob   = np.mean([r['probability'] for r in all_results])

        # Accuracy metrics (only for files with ground truth provided)
        evaluated   = [r for r in all_results if r['status'] != '—']
        n_correct   = sum(1 for r in evaluated if r['status'] == 'Correct')
        n_wrong     = sum(1 for r in evaluated if r['status'] == 'Wrong')
        n_evaluated = len(evaluated)

        tp_count = sum(1 for r in evaluated if r['status'] == 'Correct' and r['is_planet'])
        tn_count = sum(1 for r in evaluated if r['status'] == 'Correct' and not r['is_planet'])
        fp_count = sum(1 for r in evaluated if r['status'] == 'Wrong'   and r['is_planet'])
        fn_count = sum(1 for r in evaluated if r['status'] == 'Wrong'   and not r['is_planet'])
        n_planet_gt    = sum(1 for r in evaluated if r['ground_truth'] == 'Planet (confirmed)')
        n_noplanet_gt  = sum(1 for r in evaluated if r['ground_truth'] == 'No Planet (confirmed)')

        has_eval = n_evaluated > 0

        # ── Row 1: base metrics ──────────────────────────────
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Files Analyzed</div>'
                        f'<div class="metric-value">{total}</div>'
                        f'<div class="metric-unit">light curves</div></div>', unsafe_allow_html=True)
        with sc2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Planets Detected</div>'
                        f'<div class="metric-value" style="color:#4ade80;">🪐 {planets}</div>'
                        f'<div class="metric-unit">{planets/total*100:.0f}% detection rate</div></div>', unsafe_allow_html=True)
        with sc3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">No Planet</div>'
                        f'<div class="metric-value" style="color:#f87171;">🌑 {no_planets}</div>'
                        f'<div class="metric-unit">{no_planets/total*100:.0f}% negative</div></div>', unsafe_allow_html=True)
        with sc4:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Confidence</div>'
                        f'<div class="metric-value">{avg_conf:.1f}%</div>'
                        f'<div class="metric-unit">mean probability {avg_prob:.1f}%</div></div>', unsafe_allow_html=True)

        # ── Row 2: accuracy metrics (only when ground truth provided) ──
        if has_eval:
            st.markdown("---")
            st.markdown('<p style="color:rgba(232,230,240,0.5);font-size:0.85rem;margin-bottom:0.6rem;">'
                        '📊 Model Evaluation — based on labelled files</p>', unsafe_allow_html=True)

            acc_pct   = n_correct / n_evaluated * 100
            recall    = tp_count  / n_planet_gt   * 100 if n_planet_gt   else None
            spec      = tn_count  / n_noplanet_gt * 100 if n_noplanet_gt else None
            acc_color = '#4ade80' if acc_pct >= 70 else ('#facc15' if acc_pct >= 50 else '#f87171')

            ec1, ec2, ec3, ec4, ec5 = st.columns(5)
            with ec1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Correct</div>'
                            f'<div class="metric-value" style="color:{acc_color}">{n_correct}/{n_evaluated}</div>'
                            f'<div class="metric-unit">accuracy {acc_pct:.0f}%</div></div>', unsafe_allow_html=True)
            with ec2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">True Positives</div>'
                            f'<div class="metric-value" style="color:#4ade80">{tp_count}</div>'
                            f'<div class="metric-unit">planet correctly detected</div></div>', unsafe_allow_html=True)
            with ec3:
                st.markdown(f'<div class="metric-card"><div class="metric-label">True Negatives</div>'
                            f'<div class="metric-value" style="color:#4ade80">{tn_count}</div>'
                            f'<div class="metric-unit">no-planet correctly rejected</div></div>', unsafe_allow_html=True)
            with ec4:
                r_str = f'{recall:.0f}%' if recall is not None else '—'
                st.markdown(f'<div class="metric-card"><div class="metric-label">Recall</div>'
                            f'<div class="metric-value">{r_str}</div>'
                            f'<div class="metric-unit">of {n_planet_gt} confirmed planets</div></div>', unsafe_allow_html=True)
            with ec5:
                s_str = f'{spec:.0f}%' if spec is not None else '—'
                st.markdown(f'<div class="metric-card"><div class="metric-label">Specificity</div>'
                            f'<div class="metric-value">{s_str}</div>'
                            f'<div class="metric-unit">of {n_noplanet_gt} no-planet files</div></div>', unsafe_allow_html=True)

        # ── Results table ────────────────────────────────────
        report_df = pd.DataFrame(all_results)
        report_df = report_df.rename(columns={
            'file':              'File',
            'ground_truth':      'Ground Truth',
            'prediction':        'Result',
            'status':            'Status',
            'confidence':        'Confidence %',
            'probability':       'Probability %',
            'snr':               'SNR',
            'snr_db':            'SNR (dB)',
            'transit_depth_pct': 'Depth %',
            'radius_ratio':      'Rp/Rs',
            'n_dips':            'Dips',
            'n_dip_points':      'Dip Points',
            'period_idx':        'Period Idx',
            'period_strength':   'Period Str',
            'data_points':       'Data Pts',
            'model_used':        'Model',
        })
        report_df = report_df.drop(columns=['is_planet'], errors='ignore')

        def _style_report(row):
            styles = [''] * len(row)
            idx = list(row.index)
            def _ci(col):
                return idx.index(col) if col in idx else -1

            ri = _ci('Result')
            if ri >= 0:
                styles[ri] = ('color:#4ade80;font-weight:700' if row['Result'] == 'Planet Detected'
                              else 'color:#f87171')
            si = _ci('Status')
            if si >= 0:
                styles[si] = ('color:#4ade80;font-weight:700' if row['Status'] == 'Correct'
                              else ('color:#f87171;font-weight:700' if row['Status'] == 'Wrong'
                                    else 'color:rgba(232,230,240,0.4)'))
            return styles

        fmt_dict = {
            'Confidence %': '{:.1f}', 'Probability %': '{:.1f}',
            'SNR': '{:.2f}', 'SNR (dB)': '{:.1f}',
            'Depth %': '{:.4f}', 'Rp/Rs': '{:.4f}', 'Period Str': '{:.3f}',
        }
        st.markdown('<div class="glass-card" style="padding:1.5rem;">', unsafe_allow_html=True)
        st.dataframe(
            report_df.style.apply(_style_report, axis=1).format(fmt_dict, na_rep='—'),
            use_container_width=True,
            height=min(500, 60 + 35 * total),
            key="report_table"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Detection chart ──────────────────────────────────
        det_fig = go.Figure()
        bar_colors = []
        for r in all_results:
            if r['status'] == 'Correct':
                bar_colors.append('#4ade80')
            elif r['status'] == 'Wrong':
                bar_colors.append('#f87171')
            else:
                bar_colors.append('#00e5ff' if r['is_planet'] else '#94a3b8')

        det_fig.add_trace(go.Bar(
            x=[r['file'] for r in all_results],
            y=[r['probability'] for r in all_results],
            marker_color=bar_colors,
            text=[
                f"{r['prediction']}<br>{'✓' if r['status']=='Correct' else ('✗' if r['status']=='Wrong' else '')}"
                for r in all_results
            ],
            textposition='outside',
            textfont=dict(size=10),
            customdata=[[r['ground_truth'], r['status']] for r in all_results],
            hovertemplate='<b>%{x}</b><br>Prob: %{y:.1f}%<br>Truth: %{customdata[0]}<br>Status: %{customdata[1]}<extra></extra>',
        ))
        det_fig.add_hline(y=50, line_dash='dash', line_color='rgba(255,255,255,0.3)',
                          annotation_text='Detection Threshold (50%)')

        legend_note = ' | Green=Correct  Red=Wrong  Cyan=Planet(no GT)  Grey=NoPlanet(no GT)' if has_eval else ''
        det_fig.update_layout(**PLOTLY_LAYOUT, height=420,
                              margin=dict(l=50, r=30, t=60, b=100),
                              title=f'Planet Probability by File{legend_note}',
                              xaxis_title='Light Curve File', yaxis_title='Probability %',
                              yaxis=dict(range=[0, 115]),
                              xaxis_tickangle=-40)
        st.plotly_chart(det_fig, use_container_width=True, key="det_summary")

        csv_report = report_df.to_csv(index=False)
        st.download_button('📥 Download Full Report (CSV)', csv_report,
                           'exovision_detection_report.csv', 'text/csv',
                           use_container_width=True)

else:
    st.markdown("""<div style="text-align:center;padding:4rem 2rem;">
        <div style="font-size:4rem;margin-bottom:1rem;opacity:0.3;">📡</div>
        <p style="color:rgba(232,230,240,0.4);font-size:1.1rem;">
            Upload a light curve CSV to begin analysis</p>
        <p style="color:rgba(232,230,240,0.25);font-size:0.85rem;margin-top:0.5rem;">
            Supported: Kepler, TESS, or custom flux CSVs · Auto-detects format</p>
        <p style="color:rgba(232,230,240,0.2);font-size:0.75rem;margin-top:1rem;">
            Try uploading test files from the light_curves_csv folder:<br/>
            <code>test_hot_jupiter.csv</code> · <code>test_no_planet.csv</code> · <code>star_1.csv</code></p>
    </div>""", unsafe_allow_html=True)
