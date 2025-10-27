import streamlit as st
import pandas as pd
from pathlib import Path

from .. import load_lightcurve, preprocess_lightcurve
from ..features import extract_basic_features

try:
    from ..features.tsfresh_extractor import extract_tsfresh_features
    _HAS_TSFRESH = True
except Exception:
    _HAS_TSFRESH = False

st.set_page_config(page_title="Exodet Dashboard", layout="wide")

st.title("Exoplanet Light Curve Explorer")

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Upload NPZ/CSV/FITS", type=["npz","csv","fits","fit"])
    tier = st.selectbox("Feature tier", ["basic", "tsfresh"], index=0)
    detrend = st.selectbox("Detrend", ["polynomial","savgol","median"], index=0)
    normalize = st.selectbox("Normalize", ["zscore","minmax","robust","median"], index=0)
    run = st.button("Process")

if uploaded and run:
    # Save to temp
    tmp_path = Path(".st_tmp")
    tmp_path.mkdir(exist_ok=True)
    path = tmp_path / uploaded.name
    path.write_bytes(uploaded.getbuffer())

    lc = load_lightcurve(str(path))
    lc_clean = preprocess_lightcurve(
        lc,
        detrend={'enabled': True, 'method': detrend},
        sigma_clip={'enabled': True, 'sigma': 3.0},
        normalize={'enabled': True, 'method': normalize}
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Raw Light Curve")
        st.line_chart(pd.DataFrame({"time": lc.time, "flux": lc.flux}).set_index("time"))
    with col2:
        st.subheader("Processed Light Curve")
        st.line_chart(pd.DataFrame({"time": lc_clean.time, "flux": lc_clean.flux}).set_index("time"))

    st.subheader("Features")
    if tier == 'basic':
        feats = extract_basic_features(lc_clean, verbose=False)
    else:
        if not _HAS_TSFRESH:
            st.error("tsfresh not installed. Run: pip install tsfresh statsmodels")
            st.stop()
        feats = extract_tsfresh_features(lc_clean)
    st.dataframe(feats.T, use_container_width=True)

    st.success("Done")
else:
    st.info("Upload a file and click Process.")
