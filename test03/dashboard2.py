import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import torch
import io
from model2 import (
    SiameseContrastiveNetwork,
    train_full_pipeline,
    pretrain_siamese,
    finetune_classifier,
    inference_single_sample
)

st.set_page_config(page_title='Siamese Exoplanet Classifier', layout='wide')

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .phase-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

st.title('🌌 Siamese Network for Exoplanet Classification')
st.markdown("""
<div style='background-color: #e3f2fd; padding: 15px; border-radius: 10px; border-left: 5px solid #2196F3;'>
<b>Two-Phase Training Strategy:</b><br>
<b>Phase 1:</b> Contrastive Learning - Train Siamese network to learn discriminative embeddings<br>
<b>Phase 2:</b> Fine-tune Classifier - Adapt learned features for binary classification<br>
<b>Inference:</b> Use single input for prediction (one-shot capability)
</div>
""", unsafe_allow_html=True)

st.markdown('---')

# Sidebar configuration
st.sidebar.header('⚙️ Training Configuration')

# Data loading
st.sidebar.subheader('📁 Data Input')
uploaded_train = st.sidebar.file_uploader("Training Data (CSV)", type=['csv'], key='train')
uploaded_val = st.sidebar.file_uploader("Validation Data (CSV)", type=['csv'], key='val')

# Training mode selection
st.sidebar.subheader('🎯 Training Mode')
training_mode = st.sidebar.radio(
    "Select Training Mode:",
    ["Full Pipeline (Both Phases)", "Phase 1 Only", "Phase 2 Only"],
    help="Full Pipeline recommended for first-time training"
)

# Phase 1 parameters
st.sidebar.subheader('🔵 Phase 1: Contrastive Learning')
pretrain_epochs = st.sidebar.slider('Pretraining Epochs', 20, 200, 100, 10)
pretrain_lr = st.sidebar.number_input('Pretrain LR', 1e-5, 1e-2, 1e-3, format="%.6f")
margin = st.sidebar.slider('Contrastive Margin', 0.5, 2.0, 1.0, 0.1)

# Phase 2 parameters
st.sidebar.subheader('🟢 Phase 2: Classifier Fine-tuning')
finetune_epochs = st.sidebar.slider('Fine-tuning Epochs', 10, 100, 50, 5)
finetune_lr = st.sidebar.number_input('Finetune LR', 1e-6, 1e-3, 1e-4, format="%.6f")
freeze_backbone = st.sidebar.checkbox('Freeze Backbone in Phase 2', value=False)

# Common parameters
st.sidebar.subheader('🔧 Common Parameters')
batch_size = st.sidebar.select_slider('Batch Size', options=[16, 32, 64, 128], value=32)
embedding_dim = st.sidebar.slider('Embedding Dimension', 64, 256, 128, 32)
dropout = st.sidebar.slider('Dropout Rate', 0.0, 0.5, 0.3, 0.05)

# Architecture
st.sidebar.subheader('🏗️ Architecture')
hidden_layers = st.sidebar.multiselect(
    'Hidden Layer Sizes',
    [768, 512, 384, 256, 192, 128, 64],
    default=[512, 256, 128]
)

# Initialize session state
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
    st.session_state.pretrain_history = None
    st.session_state.finetune_history = None
    st.session_state.model = None
    st.session_state.best_f1 = None

# Load data function
@st.cache_data
def load_data(train_file, val_file):
    if train_file is not None and val_file is not None:
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        
        X_train = train_df.iloc[:, :-1].values
        y_train = train_df.iloc[:, -1].values
        X_val = val_df.iloc[:, :-1].values
        y_val = val_df.iloc[:, -1].values
        
        return X_train, y_train, X_val, y_val, X_train.shape[1]
    return None, None, None, None, None

# Main content
if uploaded_train and uploaded_val:
    X_train, y_train, X_val, y_val, input_dim = load_data(uploaded_train, uploaded_val)
    
    if X_train is not None:
        # Display dataset info
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<p class="big-font">Dataset Statistics</p>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class='metric-card'>
            🔢 Training Samples: {len(X_train)}<br>
            📊 Input Features: {X_train.shape[1]}<br>
            🎯 Validation Samples: {len(X_val)}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown('<p class="big-font">Class Distribution</p>', unsafe_allow_html=True)
            train_pos = (y_train == 1).sum()
            train_neg = (y_train == 0).sum()
            val_pos = (y_val == 1).sum()
            val_neg = (y_val == 0).sum()
            
            st.markdown(f"""
            <div class='metric-card'>
            Training: {train_pos} Exoplanets, {train_neg} Non-Exoplanets<br>
            Validation: {val_pos} Exoplanets, {val_neg} Non-Exoplanets
            </div>
            """, unsafe_allow_html=True)

        # Training button and progress
        if st.button('🚀 Start Training'):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Initialize model
            model = SiameseContrastiveNetwork(
                input_dim=input_dim,
                embedding_dim=embedding_dim,
                hidden_dims=hidden_layers,
                dropout_rate=dropout
            )
            
            try:
                if training_mode == "Full Pipeline (Both Phases)":
                    status_text.text("Starting full training pipeline...")
                    model, pretrain_hist, finetune_hist, best_f1 = train_full_pipeline(
                        X_train, y_train, X_val, y_val, 
                        input_dim=input_dim,
                        pretrain_epochs=pretrain_epochs,
                        finetune_epochs=finetune_epochs,
                        batch_size=batch_size,
                        pretrain_lr=pretrain_lr,
                        finetune_lr=finetune_lr
                    )
                    st.session_state.pretrain_history = pretrain_hist
                    st.session_state.finetune_history = finetune_hist
                    
                elif training_mode == "Phase 1 Only":
                    status_text.text("Starting Phase 1: Contrastive Learning...")
                    pretrain_hist = pretrain_siamese(
                        model, X_train, y_train, X_val, y_val,
                        epochs=pretrain_epochs,
                        batch_size=batch_size,
                        lr=pretrain_lr
                    )
                    st.session_state.pretrain_history = pretrain_hist
                    
                else:  # Phase 2 Only
                    status_text.text("Starting Phase 2: Classifier Fine-tuning...")
                    finetune_hist, best_f1 = finetune_classifier(
                        model, X_train, y_train, X_val, y_val,
                        epochs=finetune_epochs,
                        batch_size=batch_size,
                        lr=finetune_lr
                    )
                    st.session_state.finetune_history = finetune_hist
                
                st.session_state.model = model
                st.session_state.best_f1 = best_f1 if 'best_f1' in locals() else None
                st.session_state.training_complete = True
                
                status_text.text("Training completed successfully! 🎉")
                progress_bar.progress(100)
                
            except Exception as e:
                st.error(f"Training failed with error: {str(e)}")
                progress_bar.empty()
                status_text.empty()
        
        # Display training results if available
        if st.session_state.training_complete:
            st.markdown("---")
            st.markdown('<p class="big-font">Training Results</p>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.session_state.pretrain_history is not None:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=st.session_state.pretrain_history['train_loss'],
                        name='Train Loss',
                        line=dict(color='#1f77b4')
                    ))
                    fig.add_trace(go.Scatter(
                        y=st.session_state.pretrain_history['val_loss'],
                        name='Val Loss',
                        line=dict(color='#ff7f0e')
                    ))
                    fig.update_layout(
                        title="Phase 1: Contrastive Learning Loss",
                        xaxis_title="Epoch",
                        yaxis_title="Loss",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if st.session_state.finetune_history is not None:
                    # Create subplots for metrics
                    fig = make_subplots(rows=2, cols=1, 
                                      subplot_titles=("Classification Metrics", "Loss Curves"))
                    
                    # Metrics plot
                    fig.add_trace(
                        go.Scatter(y=st.session_state.finetune_history['f1'], 
                                 name='F1', line=dict(color='#2ca02c')), row=1, col=1)
                    fig.add_trace(
                        go.Scatter(y=st.session_state.finetune_history['precision'], 
                                 name='Precision', line=dict(color='#d62728')), row=1, col=1)
                    fig.add_trace(
                        go.Scatter(y=st.session_state.finetune_history['recall'], 
                                 name='Recall', line=dict(color='#9467bd')), row=1, col=1)
                    
                    # Loss plot
                    fig.add_trace(
                        go.Scatter(y=st.session_state.finetune_history['train_loss'], 
                                 name='Train Loss', line=dict(color='#1f77b4')), row=2, col=1)
                    fig.add_trace(
                        go.Scatter(y=st.session_state.finetune_history['val_loss'], 
                                 name='Val Loss', line=dict(color='#ff7f0e')), row=2, col=1)
                    
                    fig.update_layout(height=600, showlegend=True)
                    fig.update_xaxes(title_text="Epoch", row=2, col=1)
                    fig.update_yaxes(title_text="Metrics", row=1, col=1)
                    fig.update_yaxes(title_text="Loss", row=2, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Display best metrics
            if st.session_state.best_f1 is not None:
                st.markdown(f"""
                <div class='phase-card'>
                <h3>🎯 Best Model Performance</h3>
                F1 Score: {st.session_state.best_f1:.4f}
                </div>
                """, unsafe_allow_html=True)
            
            # Inference section
            st.markdown("---")
            st.markdown('<p class="big-font">Model Inference</p>', unsafe_allow_html=True)
            
            uploaded_inference = st.file_uploader("Upload sample for inference (CSV)", type=['csv'])
            if uploaded_inference is not None:
                try:
                    inference_data = pd.read_csv(uploaded_inference)
                    if inference_data.shape[1] == X_train.shape[1]:
                        prob = inference_single_sample(st.session_state.model, 
                                                    inference_data.values[0])
                        prediction = "Exoplanet" if prob > 0.5 else "Non-Exoplanet"
                        confidence = prob if prob > 0.5 else 1 - prob
                        
                        st.markdown(f"""
                        <div class='metric-card'>
                        <h3>Prediction Results:</h3>
                        Classification: {prediction}<br>
                        Confidence: {confidence:.2%}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("Invalid input dimensions. Please ensure the sample has the same number of features as the training data.")
                except Exception as e:
                    st.error(f"Error during inference: {str(e)}")
        
        # After displaying the training results, add download buttons
        if st.session_state.training_complete:
            st.markdown("---")
            st.markdown('<p class="big-font">Download Results</p>', unsafe_allow_html=True)
            
            # Create CSV for training history
            if st.session_state.pretrain_history is not None:
                pretrain_df = pd.DataFrame({
                    'epoch': range(1, len(st.session_state.pretrain_history['train_loss']) + 1),
                    'train_loss': st.session_state.pretrain_history['train_loss'],
                    'val_loss': st.session_state.pretrain_history['val_loss']
                })
                st.download_button(
                    label="📥 Download Phase 1 Training History",
                    data=pretrain_df.to_csv(index=False).encode('utf-8'),
                    file_name="pretrain_history.csv",
                    mime="text/csv"
                )
            
            if st.session_state.finetune_history is not None:
                finetune_df = pd.DataFrame({
                    'epoch': range(1, len(st.session_state.finetune_history['train_loss']) + 1),
                    'train_loss': st.session_state.finetune_history['train_loss'],
                    'val_loss': st.session_state.finetune_history['val_loss'],
                    'accuracy': st.session_state.finetune_history['accuracy'],
                    'precision': st.session_state.finetune_history['precision'],
                    'recall': st.session_state.finetune_history['recall'],
                    'f1_score': st.session_state.finetune_history['f1'],
                    'auc': st.session_state.finetune_history['auc']
                })
                st.download_button(
                    label="📥 Download Phase 2 Training History",
                    data=finetune_df.to_csv(index=False).encode('utf-8'),
                    file_name="finetune_history.csv",
                    mime="text/csv"
                )
            
            # Add model download button
            if st.session_state.model is not None:
                model_buffer = io.BytesIO()
                torch.save(st.session_state.model.state_dict(), model_buffer)
                model_buffer.seek(0)
                st.download_button(
                    label="📥 Download Trained Model",
                    data=model_buffer,
                    file_name="exoplanet_model.pth",
                    mime="application/octet-stream"
                )