import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from  exoplanet_model_nosiamese import (
    ImprovedExoplanetClassifier, 
    train_model, 
    hyperparameter_tuning,
    create_balanced_dataloader
)
import torch
import io

st.set_page_config(page_title='Exoplanet Classification Dashboard', layout='wide')

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

st.title('🌟 Exoplanet Classification with tsfresh Features')
st.markdown('---')

# Sidebar configuration
st.sidebar.header('⚙️ Configuration')

# Data loading section
st.sidebar.subheader('Data Input')
uploaded_train = st.sidebar.file_uploader("Upload Training Data (CSV)", type=['csv'])
uploaded_val = st.sidebar.file_uploader("Upload Validation Data (CSV)", type=['csv'])

# Training parameters
st.sidebar.subheader('Training Parameters')
epochs = st.sidebar.slider('Training Epochs', 10, 200, 100, 10)
batch_size = st.sidebar.select_slider('Batch Size', options=[16, 32, 64, 128], value=32)
lr = st.sidebar.number_input('Learning Rate', 1e-5, 1e-1, 1e-3, format="%.6f")
patience = st.sidebar.slider('Early Stopping Patience', 5, 30, 15, 5)

# Hyperparameter tuning
st.sidebar.subheader('Hyperparameter Tuning')
run_tuning = st.sidebar.checkbox('Enable Optuna Tuning', value=False)
n_trials = st.sidebar.slider('Number of Trials', 10, 100, 30, 10) if run_tuning else 0

# Architecture settings
st.sidebar.subheader('Model Architecture')
dropout = st.sidebar.slider('Dropout Rate', 0.0, 0.7, 0.3, 0.05)
use_batch_norm = st.sidebar.checkbox('Use Batch Normalization', value=True)
hidden_layers = st.sidebar.multiselect(
    'Hidden Layer Sizes',
    [768, 512, 384, 256, 192, 128, 64, 32],
    default=[512, 256, 128, 64]
)

# Main content area
col1, col2, col3 = st.columns(3)

# Initialize session state
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
    st.session_state.history = None
    st.session_state.best_metrics = None

# Data loading function
@st.cache_data
def load_data(train_file, val_file):
    if train_file is not None and val_file is not None:
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        
        # Assume last column is label
        X_train = train_df.iloc[:, :-1].values
        y_train = train_df.iloc[:, -1].values
        X_val = val_df.iloc[:, :-1].values
        y_val = val_df.iloc[:, -1].values
        
        return X_train, y_train, X_val, y_val, X_train.shape[1]
    return None, None, None, None, None

# Load data
if uploaded_train and uploaded_val:
    X_train, y_train, X_val, y_val, input_dim = load_data(uploaded_train, uploaded_val)
    
    if X_train is not None:
        # Display data statistics
        st.subheader('📊 Dataset Statistics')
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Training Samples", len(X_train))
        with col2:
            st.metric("Validation Samples", len(X_val))
        with col3:
            st.metric("Features", input_dim)
        with col4:
            positive_ratio = np.mean(y_train) * 100
            st.metric("Positive Class %", f"{positive_ratio:.2f}%")
        
        # Class distribution
        st.subheader('📈 Class Distribution')
        fig_dist = go.Figure()
        train_counts = pd.Series(y_train).value_counts()
        fig_dist.add_trace(go.Bar(
            x=['Non-Exoplanet', 'Exoplanet'],
            y=[train_counts.get(0, 0), train_counts.get(1, 0)],
            marker_color=['#FF6B6B', '#4ECDC4']
        ))
        fig_dist.update_layout(
            title="Training Set Distribution",
            xaxis_title="Class",
            yaxis_title="Count",
            height=300
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Training button
        st.markdown('---')
        if st.button('🚀 Start Training', type='primary', use_container_width=True):
            # Create dataloaders
            train_loader = create_balanced_dataloader(
                X_train, y_train, batch_size=batch_size, is_train=True
            )
            val_loader = create_balanced_dataloader(
                X_val, y_val, batch_size=batch_size*2, is_train=False
            )
            
            # Initialize model
            model = ImprovedExoplanetClassifier(
                input_dim=input_dim,
                hidden_dims=sorted(hidden_layers, reverse=True),
                dropout_rate=dropout,
                use_batch_norm=use_batch_norm
            )
            
            # Training progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner('Training in progress...'):
                history, val_loss, val_f1 = train_model(
                    model, train_loader, val_loader,
                    epochs=epochs, patience=patience, lr=lr
                )
            
            # Store results
            st.session_state.training_complete = True
            st.session_state.history = history
            st.session_state.best_metrics = {
                'val_loss': val_loss,
                'val_f1': val_f1,
                'best_auc': max(history['auc']),
                'best_precision': max(history['precision']),
                'best_recall': max(history['recall'])
            }

            # Store the trained model for download
            st.session_state.model = model
            
            progress_bar.progress(100)
            status_text.success('✅ Training completed!')

        
        # Display results if training is complete
        if st.session_state.training_complete and st.session_state.history:
            st.markdown('---')
            st.subheader('🎯 Training Results')
            
            # Best metrics
            metrics = st.session_state.best_metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Best F1 Score", f"{metrics['val_f1']:.4f}")
            with col2:
                st.metric("Best AUC", f"{metrics['best_auc']:.4f}")
            with col3:
                st.metric("Best Precision", f"{metrics['best_precision']:.4f}")
            with col4:
                st.metric("Best Recall", f"{metrics['best_recall']:.4f}")
            with col5:
                st.metric("Val Loss", f"{metrics['val_loss']:.4f}")
            
            # Training curves
            history = st.session_state.history
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Loss Curves', 'Accuracy & F1', 'Precision & Recall', 'ROC-AUC'),
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )
            
            epochs_range = list(range(1, len(history['train_loss']) + 1))
            
            # Loss curves
            fig.add_trace(
                go.Scatter(x=epochs_range, y=history['train_loss'], 
                          name='Train Loss', line=dict(color='#FF6B6B')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=epochs_range, y=history['val_loss'], 
                          name='Val Loss', line=dict(color='#4ECDC4')),
                row=1, col=1
            )
            
            # Accuracy & F1
            fig.add_trace(
                go.Scatter(x=epochs_range, y=history['accuracy'], 
                          name='Accuracy', line=dict(color='#95E1D3')),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=epochs_range, y=history['f1'], 
                          name='F1 Score', line=dict(color='#F38181')),
                row=1, col=2
            )
            
            # Precision & Recall
            fig.add_trace(
                go.Scatter(x=epochs_range, y=history['precision'], 
                          name='Precision', line=dict(color='#AA96DA')),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=epochs_range, y=history['recall'], 
                          name='Recall', line=dict(color='#FCBAD3')),
                row=2, col=1
            )
            
            # ROC-AUC
            fig.add_trace(
                go.Scatter(x=epochs_range, y=history['auc'], 
                          name='ROC-AUC', line=dict(color='#A8D8EA')),
                row=2, col=2
            )
            
            fig.update_xaxes(title_text="Epoch", row=2, col=1)
            fig.update_xaxes(title_text="Epoch", row=2, col=2)
            fig.update_layout(height=700, showlegend=True)
            
            st.plotly_chart(fig, use_container_width=True)
            
           # Download trained model
            st.markdown('---')
            st.subheader('💾 Export Model')
            if st.session_state.training_complete and 'model' in st.session_state:
                buffer = io.BytesIO()
                torch.save(st.session_state.model.state_dict(), buffer)
                buffer.seek(0)
                
                st.download_button(
                    label="⬇️ Download Trained Model",
                    data=buffer,
                    file_name="best_exoplanet_model.pth",
                    mime="application/octet-stream"
                )
        # Hyperparameter tuning section
        if run_tuning:
            st.markdown('---')
            st.subheader('🔬 Hyperparameter Optimization')
            
            if st.button('🔍 Run Optuna Study', type='secondary'):
                train_loader = create_balanced_dataloader(
                    X_train, y_train, batch_size=batch_size, is_train=True
                )
                val_loader = create_balanced_dataloader(
                    X_val, y_val, batch_size=batch_size*2, is_train=False
                )
                
                with st.spinner(f'Running {n_trials} optimization trials...'):
                    study = hyperparameter_tuning(
                        train_loader, val_loader, input_dim, n_trials=n_trials
                    )
                
                st.success('Optimization complete!')
                
                # Display best parameters
                st.json(study.best_trial.params)
                
                st.info(f"Best F1 Score: {-study.best_trial.value:.4f}")

else:
    st.info('👆 Please upload training and validation datasets to begin')
    
    # Display example data format
    with st.expander("📋 Expected Data Format"):
        st.markdown("""
        **CSV Requirements:**
        - Features: All tsfresh extracted features (columns 1 to N-1)
        - Label: Binary classification label (0 or 1) in the last column
        - No missing values
        - No index column
        
        **Example:**
        ```
        feature_1, feature_2, ..., feature_N, label
        0.123,     0.456,     ..., 0.789,     0
        0.234,     0.567,     ..., 0.890,     1
        ```
        """)

# Footer
st.markdown('---')
st.markdown(
    '<div style="text-align: center; color: #888;">Built with ❤️ for Exoplanet Discovery</div>',
    unsafe_allow_html=True
)
