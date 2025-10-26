import streamlit as st
import torch
import pandas as pd
import numpy as np
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="PyTorch Model Inference",
    page_icon="🧠",
    layout="wide"
)

# Title and description
st.title("🧠 PyTorch Model Inference Interface")
st.markdown("Upload your `.pth` model and `.csv` features to get predictions")

# Create two columns for file uploads
col1, col2 = st.columns(2)

with col1:
    st.subheader("📦 1. Upload Model")
    model_file = st.file_uploader(
        "Choose your PyTorch model file",
        type=['pth'],
        help="Upload your trained .pth model file"
    )
    
    if model_file is not None:
        st.success(f"✅ Model loaded: {model_file.name}")
        st.info(f"Size: {model_file.size / (1024*1024):.2f} MB")

with col2:
    st.subheader("📊 2. Upload Features")
    csv_file = st.file_uploader(
        "Choose your CSV file with features",
        type=['csv'],
        help="Upload CSV file containing features for prediction"
    )
    
    if csv_file is not None:
        df = pd.read_csv(csv_file)
        st.success(f"✅ CSV loaded: {csv_file.name}")
        st.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Show preview
        with st.expander("📋 Preview Data"):
            st.dataframe(df.head(10))

# Divider
st.divider()

# Prediction section
if model_file is not None and csv_file is not None:
    
    # Additional options
    col1, col2 = st.columns([2, 1])
    
    with col1:
        device = st.selectbox("Select Device", ["cpu", "cuda"], index=0)
    
    with col2:
        batch_size = st.number_input("Batch Size", min_value=1, max_value=1000, value=32)
    
    st.divider()
    
    # Predict button
    if st.button("🚀 Run Predictions", type="primary", use_container_width=True):
        
        with st.spinner("Loading model and running predictions..."):
            try:
                # Load model
                model = torch.load(model_file, map_location=device)
                model.eval()
                
                # Prepare data
                df = pd.read_csv(csv_file)
                
                # Convert to tensor
                # Adjust this based on your data preprocessing needs
                features = torch.tensor(df.values, dtype=torch.float32)
                
                # Run predictions
                predictions_list = []
                
                with torch.no_grad():
                    # Process in batches
                    for i in range(0, len(features), batch_size):
                        batch = features[i:i+batch_size].to(device)
                        outputs = model(batch)
                        predictions_list.append(outputs.cpu().numpy())
                
                # Combine all predictions
                predictions = np.vstack(predictions_list)
                
                # Create results dataframe
                if predictions.shape[1] == 1:
                    # Single output
                    results_df = pd.DataFrame({
                        'Sample': range(1, len(predictions) + 1),
                        'Prediction': predictions.flatten()
                    })
                else:
                    # Multiple outputs
                    pred_cols = {f'Output_{i+1}': predictions[:, i] for i in range(predictions.shape[1])}
                    results_df = pd.DataFrame({
                        'Sample': range(1, len(predictions) + 1),
                        **pred_cols
                    })
                
                # Display success
                st.success(f"✅ Predictions completed! Generated {len(results_df)} predictions")
                
                # Show results
                st.subheader("📈 Prediction Results")
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Samples", len(results_df))
                with col2:
                    st.metric("Output Dimensions", predictions.shape[1])
                with col3:
                    if predictions.shape[1] == 1:
                        st.metric("Mean Prediction", f"{predictions.mean():.4f}")
                
                # Show results table
                st.dataframe(results_df, use_container_width=True, height=400)
                
                # Download button
                csv_buffer = BytesIO()
                results_df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv_buffer,
                    file_name="predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                # Visualization (if single output)
                if predictions.shape[1] == 1:
                    st.subheader("📊 Prediction Distribution")
                    st.bar_chart(results_df.set_index('Sample')['Prediction'])
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.exception(e)

else:
    st.info("👆 Please upload both model and CSV files to start predictions")

# Footer with instructions
st.divider()
st.subheader("📝 Instructions")

with st.expander("How to use this app"):
    st.markdown("""
    ### Steps:
    1. **Upload Model (.pth)**: Upload your trained PyTorch model file
    2. **Upload CSV**: Upload your CSV file containing features for prediction
    3. **Configure Settings**: Select device (CPU/CUDA) and batch size
    4. **Run Predictions**: Click the predict button to generate results
    5. **Download Results**: Export predictions as CSV file
    
    ### Requirements:
    Your CSV file should contain only numerical features in the format your model expects.
    
    ### Model Format:
    - Model should be saved using `torch.save(model, 'model.pth')`
    - Model should accept input tensors of shape `(batch_size, num_features)`
    
    ### Example Usage:
    ```python
    # To run this app:
    streamlit run app.py
    ```
    """)

with st.expander("Installation Requirements"):
    st.code("""
pip install streamlit torch pandas numpy
    """, language="bash")

with st.expander("Model Training Example"):
    st.code("""
import torch
import torch.nn as nn

# Example model
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Save model
model = MyModel(10, 20, 1)
torch.save(model, 'model.pth')
    """, language="python")