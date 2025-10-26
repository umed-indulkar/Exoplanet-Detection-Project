import streamlit as st
from test02.exoplanet_model_nosiamese import SiameseFCNN, train_model, hyperparameter_tuning, plot_study_analytics, train_loader, val_loader

st.set_page_config(page_title='Siamese FCNN Dashboard', layout='wide')
st.title('Siamese FCNN Hyperparameter Tuning & Training Dashboard')

# Sidebar settings
st.sidebar.header('Settings')
n_trials = st.sidebar.slider('Number of Optuna Trials', min_value=5, max_value=50, value=10, step=5)
epochs = st.sidebar.slider('Number of Training Epochs', min_value=1, max_value=50, value=10, step=1)
lr = st.sidebar.number_input('Learning Rate', min_value=1e-5, max_value=1.0, value=1e-3, format="%.6f")

# Cache model initialization
@st.cache_resource
def get_model(input_dim=None):
    return SiameseFCNN(input_dim=input_dim)

# Cache training results
@st.cache_data
def get_training_results(model, train_loader, val_loader, epochs, lr):
    history, best_val_loss = train_model(model, train_loader, val_loader, epochs=epochs, lr=lr)
    return history, best_val_loss

# Cache Optuna hyperparameter tuning
@st.cache_data
def get_optuna_study(train_loader, val_loader, n_trials):
    study = hyperparameter_tuning(train_loader, val_loader, n_trials=n_trials)
    return study

# Initialize model
model = get_model(input_dim=None)  # input_dim taken from locked hyperparams

# Run training and get metrics
with st.spinner('Training the model...'):
    history, best_val_loss = get_training_results(model, train_loader, val_loader, epochs, lr)

# Display training metrics
st.subheader('Training & Validation Metrics')
st.line_chart({
    'Train Loss': history['train_loss'],
    'Validation Loss': history['val_loss'],
    'Accuracy': history['accuracy'],
    'Precision': history['precision'],
    'Recall': history['recall'],
    'ROC-AUC': history['auc']
})

# Run hyperparameter tuning and show Optuna graphs
with st.spinner('Running hyperparameter tuning with Optuna...'):
    study = get_optuna_study(train_loader, val_loader, n_trials)

st.subheader('Optuna Optimization History')
plot_study_analytics(study)
st.pyplot()

st.subheader('Optuna Parameter Importance')
st.pyplot(study.plot_param_importances())

st.success('Dashboard fully loaded!')
