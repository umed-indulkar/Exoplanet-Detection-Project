"""Custom CSS for the EXO-VISION AI LAB Streamlit dashboard."""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Global ──────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(165deg, #0a0e27 0%, #111638 40%, #1a1f3a 100%) !important;
    color: #e8e6f0;
    font-family: 'Inter', sans-serif;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stMain"] > div { padding-top: 1rem; }

/* ── Sidebar ─────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(10,14,39,0.97) 0%, rgba(17,22,56,0.98) 100%) !important;
    border-right: 1px solid rgba(0,229,255,0.08);
}
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stSlider > div {
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
}

/* ── Hero Header ─────────────────────────────────────── */
.hero-header {
    text-align: center;
    padding: 2rem 1rem 1.2rem;
    position: relative;
    margin-bottom: 0.5rem;
}
.hero-glow {
    position: absolute; top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    width: 500px; height: 200px;
    background: radial-gradient(ellipse, rgba(0,229,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem; font-weight: 700;
    background: linear-gradient(135deg, #00e5ff 0%, #a855f7 50%, #00e5ff 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: none;
    letter-spacing: -0.02em;
    margin: 0;
    animation: glow-pulse 4s ease-in-out infinite alternate;
}
.hero-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem; color: rgba(232,230,240,0.5);
    letter-spacing: 0.1em; margin-top: 0.5rem;
}
.hero-line {
    margin: 1rem auto 0;
    width: 120px; height: 2px;
    background: linear-gradient(90deg, transparent, #00e5ff, #a855f7, transparent);
    border-radius: 2px;
}

/* ── Section Headers ─────────────────────────────────── */
.section-header { margin: 1.5rem 0 0.8rem; }
.section-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; letter-spacing: 0.12em;
    color: #00e5ff; text-transform: uppercase;
    padding: 4px 14px;
    border: 1px solid rgba(0,229,255,0.2);
    border-radius: 100px;
    background: rgba(0,229,255,0.05);
    display: inline-block; margin-bottom: 0.5rem;
}
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem; font-weight: 600;
    color: #f1f0f5; margin: 0.3rem 0;
}
.section-subtitle {
    font-size: 0.85rem; color: rgba(232,230,240,0.5);
    margin-top: 0.25rem;
}

/* ── Tabs ────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: rgba(255,255,255,0.02);
    border-radius: 14px;
    padding: 4px;
    border: 1px solid rgba(255,255,255,0.05);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 500;
    font-size: 0.85rem;
    color: rgba(232,230,240,0.5);
    border-radius: 10px;
    padding: 0.5rem 1.2rem;
    transition: all 0.3s ease;
    border: none !important;
    background: transparent;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #00e5ff;
    background: rgba(0,229,255,0.05);
}
.stTabs [aria-selected="true"] {
    color: #00e5ff !important;
    background: rgba(0,229,255,0.1) !important;
    border: 1px solid rgba(0,229,255,0.2) !important;
    font-weight: 600;
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}
.stTabs [data-baseweb="tab-border"] {
    display: none;
}

/* ── Glass Cards ─────────────────────────────────────── */
.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.glass-card:hover {
    border-color: rgba(0,229,255,0.15);
    box-shadow: 0 0 30px rgba(0,229,255,0.05);
}

/* ── Result Cards ────────────────────────────────────── */
.result-card {
    text-align: center; padding: 1.5rem 1rem;
    border-radius: 16px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
}
.result-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
.result-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; letter-spacing: 0.15em;
    text-transform: uppercase; color: rgba(232,230,240,0.4);
    margin-bottom: 0.3rem;
}
.result-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem; font-weight: 700;
}
.result-value.success { color: #00e5ff; text-shadow: 0 0 20px rgba(0,229,255,0.3); }
.result-value.danger { color: #f87171; }
.result-value.neutral { color: #a855f7; }
.planet-detected { border-color: rgba(0,229,255,0.2) !important; }
.no-planet { border-color: rgba(248,113,113,0.2) !important; }

/* ── Metric Cards ────────────────────────────────────── */
.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
.metric-card {
    flex: 1; min-width: 140px;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px; padding: 1rem 1.2rem;
}
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem; letter-spacing: 0.12em;
    text-transform: uppercase; color: rgba(232,230,240,0.4);
}
.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem; font-weight: 600; color: #00e5ff;
    margin-top: 0.25rem;
}
.metric-unit { font-size: 0.75rem; color: rgba(232,230,240,0.4); }

/* ── Pipeline Step Visualization ─────────────────────── */
.pipeline-step {
    display: flex; align-items: center; gap: 0.75rem;
    padding: 0.6rem 1rem; margin: 0.3rem 0;
    background: rgba(255,255,255,0.02);
    border-left: 2px solid rgba(0,229,255,0.3);
    border-radius: 0 8px 8px 0;
    font-size: 0.85rem; color: rgba(232,230,240,0.7);
}
.pipeline-step .step-icon { color: #00e5ff; font-size: 0.9rem; }

.step-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    margin: 0.3rem 0;
    display: flex;
    align-items: center;
    gap: 0.8rem;
    font-size: 0.82rem;
    transition: all 0.2s ease;
}
.step-card:hover {
    background: rgba(0,229,255,0.03);
    border-color: rgba(0,229,255,0.15);
}
.step-num {
    background: rgba(0,229,255,0.15);
    color: #00e5ff;
    border-radius: 50%;
    width: 24px; height: 24px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; font-weight: 600;
    flex-shrink: 0;
}
.step-name {
    color: rgba(232,230,240,0.8);
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 500;
}

/* ── Model Comparison Grid ───────────────────────────── */
.model-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}
.model-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1.2rem;
    transition: all 0.3s ease;
}
.model-card:hover {
    border-color: rgba(0,229,255,0.2);
    box-shadow: 0 0 20px rgba(0,229,255,0.04);
    transform: translateY(-2px);
}
.model-card-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 0.75rem;
}
.model-card-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.95rem; font-weight: 600; color: #f1f0f5;
}
.model-card-cat {
    font-size: 0.65rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em;
    padding: 2px 8px; border-radius: 6px;
}
.model-card-result {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem; font-weight: 700;
    margin: 0.5rem 0;
}
.model-card-bar {
    height: 6px; border-radius: 3px;
    background: rgba(255,255,255,0.04);
    overflow: hidden; margin-top: 0.5rem;
}
.model-card-bar-fill {
    height: 100%; border-radius: 3px;
    transition: width 0.5s ease;
}
.model-card-meta {
    font-size: 0.72rem; color: rgba(232,230,240,0.4);
    margin-top: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Format Badge ────────────────────────────────────── */
.format-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 4px 12px;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; font-weight: 500;
    letter-spacing: 0.03em;
}
.format-badge.time-flux {
    background: rgba(0,229,255,0.1);
    border: 1px solid rgba(0,229,255,0.25);
    color: #00e5ff;
}
.format-badge.wide {
    background: rgba(168,85,247,0.1);
    border: 1px solid rgba(168,85,247,0.25);
    color: #a855f7;
}

/* ── Consensus Badge ─────────────────────────────────── */
.consensus-badge {
    text-align: center;
    padding: 1.5rem;
    border-radius: 16px;
    border: 2px solid;
}
.consensus-badge.planet {
    background: rgba(0,229,255,0.05);
    border-color: rgba(0,229,255,0.3);
}
.consensus-badge.no-planet {
    background: rgba(248,113,113,0.05);
    border-color: rgba(248,113,113,0.3);
}
.consensus-icon { font-size: 3rem; margin-bottom: 0.5rem; }
.consensus-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; letter-spacing: 0.15em;
    text-transform: uppercase; color: rgba(232,230,240,0.4);
}
.consensus-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem; font-weight: 700;
    margin-top: 0.3rem;
}
.consensus-detail {
    font-size: 0.8rem; color: rgba(232,230,240,0.5);
    margin-top: 0.3rem;
}

/* ── Info Card ───────────────────────────────────────── */
.info-card ul {
    list-style: none; padding: 0; margin: 0.5rem 0 0 0;
}
.info-card li {
    padding: 0.3rem 0; font-size: 0.8rem;
    color: rgba(232,230,240,0.6);
    border-bottom: 1px solid rgba(255,255,255,0.03);
}
.info-card li:last-child { border: none; }
.info-card li::before { content: "→ "; color: #00e5ff; }
.info-card h4 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem; color: #f1f0f5;
    margin: 0 0 0.25rem 0;
}

/* ── Buttons ─────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,229,255,0.15), rgba(168,85,247,0.15)) !important;
    border: 1px solid rgba(0,229,255,0.3) !important;
    color: #00e5ff !important;
    border-radius: 12px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    border-color: rgba(0,229,255,0.6) !important;
    box-shadow: 0 0 25px rgba(0,229,255,0.15) !important;
    transform: translateY(-1px);
}

/* ── File Uploader ───────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02);
    border: 1px dashed rgba(0,229,255,0.2);
    border-radius: 16px; padding: 1rem;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,229,255,0.4);
    background: rgba(0,229,255,0.02);
}

/* ── Expander ────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 12px !important;
}

/* ── Scrollbar ───────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,229,255,0.2); border-radius: 3px; }

/* ── Animations ──────────────────────────────────────── */
@keyframes glow-pulse {
    0% { filter: brightness(1); }
    100% { filter: brightness(1.15); }
}

/* ── Divider ─────────────────────────────────────────── */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,229,255,0.15), rgba(168,85,247,0.15), transparent);
    margin: 1.5rem 0;
}

/* ── Stat Row ────────────────────────────────────────── */
.stat-row {
    display: flex; justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.85rem;
}
.stat-key { color: rgba(232,230,240,0.5); }
.stat-val { color: #00e5ff; font-family: 'JetBrains Mono', monospace; }
</style>
"""
