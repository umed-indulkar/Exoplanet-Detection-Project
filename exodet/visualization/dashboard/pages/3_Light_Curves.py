import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.title("🧠 Light Curves Visualization")
st.caption("Interactive graphs for stellar light curves and transit dips")

# ================= DATA SELECTION =================
data_option = st.selectbox("Select Data Source:", ["Sample Data", "Upload CSV", "Generate Synthetic"])

if data_option == "Sample Data":
    time_points = np.linspace(0, 10, 100)
    flux = 1 + 0.1 * np.sin(2 * np.pi * time_points / 5) + np.random.normal(0, 0.02, 100)
    data = pd.DataFrame({"Time": time_points, "Flux": flux})
elif data_option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload Light Curve CSV", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
    else:
        data = None
else:
    # Generate synthetic
    with st.form("synth_form"):
        amplitude = st.slider("Transit Amplitude", 0.01, 0.2, 0.1)
        period = st.slider("Period", 1, 10, 5)
        submitted = st.form_submit_button("Generate")
        if submitted:
            time_points = np.linspace(0, 20, 200)
            flux = 1 + amplitude * np.sin(2 * np.pi * time_points / period) + np.random.normal(0, 0.01, 200)
            data = pd.DataFrame({"Time": time_points, "Flux": flux})

# ================= VISUALIZATION =================
if data is not None:
    st.subheader("Light Curve Plot")
    
    # Filters
    time_range = st.slider("Select Time Range", float(data["Time"].min()), float(data["Time"].max()), (0.0, 10.0))
    filtered_data = data[(data["Time"] >= time_range[0]) & (data["Time"] <= time_range[1])]
    
    # Interactive plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_data["Time"], y=filtered_data["Flux"], mode='lines+markers', name='Flux'))
    fig.update_layout(title="Stellar Light Curve", xaxis_title="Time", yaxis_title="Normalized Flux", 
                      xaxis=dict(rangeslider=dict(visible=True)))
    st.plotly_chart(fig, use_container_width=True)
    
    # Add transit dip annotation
    if st.checkbox("Highlight Transit Dip"):
        dip_time = st.slider("Dip Time", float(data["Time"].min()), float(data["Time"].max()), 5.0)
        fig.add_vline(x=dip_time, line_dash="dash", annotation_text="Transit Dip")
        st.plotly_chart(fig)
    
    # Compare multiple curves
    if st.button("Add Comparison Curve"):
        comp_flux = 1 + 0.05 * np.sin(2 * np.pi * filtered_data["Time"] / 3) + np.random.normal(0, 0.02, len(filtered_data))
        fig.add_trace(go.Scatter(x=filtered_data["Time"], y=comp_flux, mode='lines', name='Comparison'))
        st.plotly_chart(fig)
else:
    st.info("Select or upload data to visualize.")

st.markdown("---")
st.caption("Light Curves Page")