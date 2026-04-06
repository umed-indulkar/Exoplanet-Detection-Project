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

from pipeline import DataPipeline
from models import ExoplanetDetector
from utils import calculate_snr, estimate_transit_depth, estimate_period, find_transit_dips
from styles import CSS

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
# Session State — lazy-init the detector once
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if "detector" not in st.session_state:
    with st.spinner("🧠 Initializing AI models — training on synthetic transit data…"):
        st.session_state.detector = ExoplanetDetector()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Plotly Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#e8e6f0"),
)

def make_light_curve(raw, processed, dips):
    fig = make_subplots(rows=2, cols=1, row_heights=[0.65, 0.35],
                        shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=["Processed Light Curve", "Raw Data"])
    # processed
    fig.add_trace(go.Scatter(
        y=processed, mode='lines', name='Processed',
        line=dict(color='#00e5ff', width=1.5),
        fill='tozeroy', fillcolor='rgba(0,229,255,0.03)'), row=1, col=1)
    # dip markers
    if dips:
        fig.add_trace(go.Scatter(
            x=[d['index'] for d in dips], y=[d['value'] for d in dips],
            mode='markers', name='Transit Dips',
            marker=dict(color='#f87171', size=10, symbol='triangle-down',
                        line=dict(width=1, color='#fff'))), row=1, col=1)
    # raw
    raw_ds = raw
    if len(raw) > 2000:
        idx = np.linspace(0, len(raw) - 1, 2000).astype(int)
        raw_ds = raw[idx]
    fig.add_trace(go.Scatter(
        y=raw_ds, mode='lines', name='Raw',
        line=dict(color='#a855f7', width=0.8), opacity=0.6), row=2, col=1)
    fig.update_layout(**PLOTLY_LAYOUT, height=520, margin=dict(l=50, r=30, t=50, b=40),
                      showlegend=True, legend=dict(orientation='h', y=1.12, x=0.5, xanchor='center'))
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
    fig.update_layout(**PLOTLY_LAYOUT, height=250, margin=dict(l=30, r=30, t=60, b=20))
    return fig


def make_feature_bars(features):
    keys = ['mean', 'std', 'skewness', 'kurtosis', 'iqr', 'depth_estimate', 'mad', 'rms']
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
# ── Model Registry ──────────────────────────────────────────
MODEL_REGISTRY = [
    {"name": "Cleaned Siamese",    "key": "cleaned_siamese",    "accuracy": 81.73, "category": "Deep Learning", "input": "Cleaned curves",    "file": "cleaned_siamese.pth"},
    {"name": "XGBoost",            "key": "xgboost",            "accuracy": 75.53, "category": "Ensemble",      "input": "Extracted features", "file": "xgboost.pkl"},
    {"name": "Original Siamese",   "key": "siamese",            "accuracy": 75.06, "category": "Deep Learning", "input": "Raw curves",        "file": "siamese_dataset500.pth"},
    {"name": "CNN",                "key": "cnn",                "accuracy": 73.17, "category": "Deep Learning", "input": "Extracted features", "file": "cnn.pth"},
    {"name": "Random Forest",      "key": "random_forest",      "accuracy": 73.33, "category": "Ensemble",      "input": "Extracted features", "file": "random_forest.pkl"},
    {"name": "Feedforward NN",     "key": "feedforward_nn",     "accuracy": 70.97, "category": "Deep Learning", "input": "Extracted features", "file": "feedforward_nn.pth"},
    {"name": "Logistic Regression", "key": "logistic_regression", "accuracy": 70.42, "category": "Linear",       "input": "Extracted features", "file": "logistic_regression.pkl"},
]
MODEL_NAMES = [m["name"] for m in MODEL_REGISTRY]
MODEL_MAP = {m["name"]: m["key"] for m in MODEL_REGISTRY}
MODEL_INFO = {m["key"]: m for m in MODEL_REGISTRY}

with st.sidebar:
    st.markdown("## 🔧 Configuration")
    st.markdown("---")

    selected = st.selectbox("🤖 Detection Model", MODEL_NAMES,
                            help="Models ranked by accuracy on test set.")
    model_key = MODEL_MAP[selected]
    info = MODEL_INFO[model_key]

    # Show selected model card
    cat_colors = {"Deep Learning": "#00e5ff", "Ensemble": "#a855f7", "Linear": "#facc15"}
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
        <div style="color:rgba(232,230,240,0.35);font-size:0.7rem;margin-top:0.25rem;">File: {info['file']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Pipeline Parameters")
    n_bins = st.slider("📊 Binning Points", 100, 1000, 500, 50)
    smooth_w = st.slider("🔧 Smooth Window", 5, 51, 11, 2)
    smooth_p = st.slider("📐 Poly Order", 1, 5, 3)
    sigma = st.slider("✂️ Outlier Sigma", 1.0, 5.0, 3.0, 0.5)

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
                'EXO-VISION AI LAB v2.0<br/>Stellar Intelligence Engine</div>', unsafe_allow_html=True)

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
                             help="Must contain a 'flux' column or be wide-format.")
with c2:
    st.markdown("""<div class="glass-card info-card"><h4>📋 CSV Format</h4><ul>
        <li>Column named <code>flux</code></li>
        <li>Or wide-format (cols = time points)</li>
        <li>Single or multiple files</li></ul></div>""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Processing + Results Loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Store results for the summary report
all_results = []

if files:
    for fi, f in enumerate(files):
        st.markdown("---")
        st.markdown(f'<div class="section-header"><span class="section-tag">// star::{fi+1}</span>'
                    f'<h2 class="section-title">⭐ {f.name}</h2></div>', unsafe_allow_html=True)

        try:
            pipe = DataPipeline(n_bins, smooth_w, smooth_p, sigma)
            bar = st.progress(0, text="Starting pipeline…")
            def _cb(v, t):
                bar.progress(v, text=t)
                time.sleep(0.12)
            processed, features, raw_df = pipe.run(f, _cb)
            bar.empty()
            st.success(f"✅ Pipeline complete — {len(processed)} data points processed.")

            with st.expander("📜 Pipeline Log"):
                for log in pipe.log:
                    st.markdown(f'<div class="pipeline-step"><span class="step-icon">▸</span>{log}</div>',
                                unsafe_allow_html=True)

            # ── Prediction ───────────────────────────────────
            st.markdown('<div class="section-header"><span class="section-tag">// results::prediction</span>'
                        '<h2 class="section-title">🎯 Detection Results</h2></div>', unsafe_allow_html=True)

            with st.spinner("🧠 Running AI model…"):
                time.sleep(0.3)
                result = st.session_state.detector.predict(processed, model_key)

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
                st.plotly_chart(make_gauge(result['confidence'], "Confidence"), use_container_width=True)
            with rc[2]:
                st.plotly_chart(make_gauge(result['probability'], "Planet Probability", '#a855f7'),
                                use_container_width=True)

            # ── Light Curve ──────────────────────────────────
            st.markdown('<div class="section-header"><span class="section-tag">// viz::lightcurve</span>'
                        '<h2 class="section-title">📈 Light Curve Analysis</h2></div>', unsafe_allow_html=True)

            dips = find_transit_dips(processed)
            st.plotly_chart(make_light_curve(pipe.raw_flux, processed, dips), use_container_width=True)

            if dips:
                st.info(f"🔍 Detected **{len(dips)}** transit dip(s) — "
                        f"deepest at index {min(dips, key=lambda d: d['value'])['index']} "
                        f"(depth: {max(d['depth'] for d in dips):.6f})")

            # ── Features ─────────────────────────────────────
            fc1, fc2 = st.columns(2)
            with fc1:
                st.plotly_chart(make_feature_bars(features), use_container_width=True)
            with fc2:
                st.markdown('<div class="glass-card"><h4 style="font-family:Space Grotesk;color:#f1f0f5;">'
                            '📊 Statistical Summary</h4>', unsafe_allow_html=True)
                for k, v in features.items():
                    st.markdown(f'<div style="display:flex;justify-content:space-between;padding:4px 0;'
                                f'border-bottom:1px solid rgba(255,255,255,0.04);font-size:0.85rem;">'
                                f'<span style="color:rgba(232,230,240,0.5)">{k.replace("_"," ").title()}</span>'
                                f'<span style="color:#00e5ff;font-family:JetBrains Mono">{v:.8f}</span></div>',
                                unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # ── Advanced Analytics ───────────────────────────
            st.markdown('<div class="section-header"><span class="section-tag">// analytics::advanced</span>'
                        '<h2 class="section-title">🔬 Advanced Analytics</h2></div>', unsafe_allow_html=True)

            snr = calculate_snr(processed)
            td = estimate_transit_depth(processed)
            per = estimate_period(processed)

            st.markdown(f"""<div class="metric-row">
                <div class="metric-card"><div class="metric-label">Signal-to-Noise</div>
                    <div class="metric-value">{snr['snr']:.2f}</div>
                    <div class="metric-unit">{snr['snr_db']:.1f} dB</div></div>
                <div class="metric-card"><div class="metric-label">Transit Depth</div>
                    <div class="metric-value">{td['depth_pct']:.4f}%</div>
                    <div class="metric-unit">Rp/Rs = {td['radius_ratio']:.4f}</div></div>
                <div class="metric-card"><div class="metric-label">Est. Period</div>
                    <div class="metric-value">{per['period_idx'] or '—'}</div>
                    <div class="metric-unit">idx • strength {per['strength']:.3f}</div></div>
                <div class="metric-card"><div class="metric-label">Dip Points</div>
                    <div class="metric-value">{td['n_dip_points']}</div>
                    <div class="metric-unit">below 2σ threshold</div></div>
            </div>""", unsafe_allow_html=True)

            # Autocorrelation plot
            if per.get('autocorr') is not None:
                ac = per['autocorr']
                ac_fig = go.Figure(go.Scatter(y=ac[:len(ac)//2], mode='lines',
                                              line=dict(color='#a855f7', width=1.2),
                                              fill='tozeroy', fillcolor='rgba(168,85,247,0.05)'))
                ac_fig.update_layout(**PLOTLY_LAYOUT, height=250, margin=dict(l=50, r=30, t=50, b=40),
                                     title="Autocorrelation (Period Search)",
                                     xaxis_title="Lag", yaxis_title="Correlation")
                if per['period_idx']:
                    ac_fig.add_vline(x=per['period_idx'], line_dash='dash',
                                     line_color='#00e5ff', annotation_text=f"Period ≈ {per['period_idx']}")
                st.plotly_chart(ac_fig, use_container_width=True)

            # Collect for summary report
            all_results.append({
                'file': f.name,
                'prediction': result['prediction'],
                'is_planet': is_p,
                'confidence': result['confidence'],
                'probability': result['probability'],
                'snr': snr['snr'],
                'snr_db': snr['snr_db'],
                'transit_depth_pct': td['depth_pct'],
                'radius_ratio': td['radius_ratio'],
                'n_dips': len(dips),
                'n_dip_points': td['n_dip_points'],
                'period_idx': per['period_idx'],
                'period_strength': per['strength'],
                'data_points': len(processed),
                'model_used': selected,
            })

        except ValueError as e:
            st.error(f"❌ Validation Error: {e}")
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

        # Summary stats
        total = len(all_results)
        planets = sum(1 for r in all_results if r['is_planet'])
        no_planets = total - planets
        avg_conf = np.mean([r['confidence'] for r in all_results])
        avg_prob = np.mean([r['probability'] for r in all_results])

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

        # Build report dataframe
        report_df = pd.DataFrame(all_results)
        report_df = report_df.rename(columns={
            'file': 'File', 'prediction': 'Result', 'confidence': 'Confidence %',
            'probability': 'Probability %', 'snr': 'SNR', 'snr_db': 'SNR (dB)',
            'transit_depth_pct': 'Depth %', 'radius_ratio': 'Rp/Rs',
            'n_dips': 'Dips', 'n_dip_points': 'Dip Points',
            'period_idx': 'Period Idx', 'period_strength': 'Period Str',
            'data_points': 'Data Pts', 'model_used': 'Model',
        })
        report_df = report_df.drop(columns=['is_planet'], errors='ignore')

        # Styled table
        st.markdown('<div class="glass-card" style="padding:1.5rem;">', unsafe_allow_html=True)
        st.dataframe(report_df.style.format({
            'Confidence %': '{:.1f}', 'Probability %': '{:.1f}',
            'SNR': '{:.2f}', 'SNR (dB)': '{:.1f}',
            'Depth %': '{:.4f}', 'Rp/Rs': '{:.4f}', 'Period Str': '{:.3f}',
        }).map(
            lambda v: 'color: #4ade80; font-weight: 700' if v == 'Planet Detected'
            else ('color: #f87171' if v == 'No Planet' else ''),
            subset=['Result']
        ), use_container_width=True, height=min(400, 60 + 35 * total))
        st.markdown('</div>', unsafe_allow_html=True)

        # Detection chart
        det_fig = go.Figure()
        colors = ['#4ade80' if r['is_planet'] else '#f87171' for r in all_results]
        det_fig.add_trace(go.Bar(
            x=[r['file'] for r in all_results],
            y=[r['probability'] for r in all_results],
            marker_color=colors,
            text=[r['prediction'] for r in all_results],
            textposition='outside', textfont=dict(size=10),
        ))
        det_fig.add_hline(y=50, line_dash='dash', line_color='rgba(255,255,255,0.3)',
                          annotation_text='Detection Threshold (50%)')
        det_fig.update_layout(**PLOTLY_LAYOUT, height=380, margin=dict(l=50, r=30, t=50, b=80),
                              title='Planet Probability by File',
                              xaxis_title='Light Curve File', yaxis_title='Probability %',
                              yaxis=dict(range=[0, 110]))
        st.plotly_chart(det_fig, use_container_width=True)

        # Download report
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
            Supported: Kepler, TESS, or custom flux CSVs</p>
    </div>""", unsafe_allow_html=True)
