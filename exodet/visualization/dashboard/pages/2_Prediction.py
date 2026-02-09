import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go

st.title("⚙ Prediction Dashboard")
st.caption("Upload data or input manually for exoplanet prediction")

# ================= INPUT METHODS =================
input_method = st.radio("Choose Input Method:", ["Upload CSV", "Manual Input"])

if input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload Light Curve Data (CSV)", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.dataframe(data.head())
        st.success("Data loaded!")
    else:
        data = None
else:
    # Manual input form
    with st.form("manual_form"):
        flux1 = st.number_input("Flux 1", value=1.01)
        flux2 = st.number_input("Flux 2", value=0.99)
        submitted = st.form_submit_button("Submit Data")
        if submitted:
            data = pd.DataFrame({"FLUX_1": [flux1], "FLUX_2": [flux2]})
            st.write("Manual Data:", data)

# ================= PREDICTION =================
if st.button("🚀 Predict Exoplanet"):
    if data is not None:
        with st.spinner("Running prediction..."):
            time.sleep(2)  # Simulate model inference
        # Mock prediction
        prediction = np.random.choice(["Planet Detected", "No Planet"], p=[0.7, 0.3])
        confidence = np.random.uniform(0.8, 0.95)
        
        st.success(f"Prediction: {prediction} (Confidence: {confidence:.2f})")
        
        # Interactive chart for prediction visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data["FLUX_1"], mode='lines', name='Flux 1'))
        fig.add_trace(go.Scatter(x=data.index, y=data["FLUX_2"], mode='lines', name='Flux 2'))
        fig.update_layout(title="Light Curve Analysis", xaxis_title="Time", yaxis_title="Flux")
        st.plotly_chart(fig)
        
        # Export option
        if st.button("Export Results"):
            data.to_csv("prediction_results.csv")
            st.download_button("Download CSV", data.to_csv(), file_name="results.csv")
    else:
        st.error("Please provide data first.")

st.markdown("---")
st.caption("Prediction Page")