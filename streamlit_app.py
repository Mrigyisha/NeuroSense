"""
Streamlit Web Application for EEG Motor Imagery Classification

This app allows users to:
1. Upload their own EEG dataset (MATLAB .mat files)
2. View data information and visualizations
3. Classify trials using pretrained models
4. Generate classification reports
"""
import streamlit as st
import numpy as np
import scipy.io
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import tempfile

from dataset_classes import MotorImageryBcic4, MotorImageryOpenBmi
from signal_processing import bandpass
from visualization import plot_psd, plot_logvar

# Page configuration
st.set_page_config(
    page_title="EEG Motor Imagery Classifier",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model(model_type='combined'):
    """Load pretrained model and CSP matrix."""
    model_path = f'models/{model_type}_model.pkl'
    csp_path = f'models/{model_type}_csp.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(csp_path):
        return None, None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(csp_path, 'rb') as f:
        csp_matrix = pickle.load(f)
    
    return model, csp_matrix

def get_dataset_info(dataset):
    """Extract information about the dataset."""
    info = {}
    try:
        info['Sample Rate'] = f"{dataset.sample_rate} Hz"
        info['Number of Channels'] = dataset.nchannels
        info['Channel Names'] = ', '.join(dataset.channel_names[:10]) + ('...' if len(dataset.channel_names) > 10 else '')
        info['Class 1'] = dataset.cl1
        info['Class 2'] = dataset.cl2
        if dataset.trials is not None:
            info['Trials Class 1'] = dataset.trials[dataset.cl1].shape[2] if dataset.cl1 in dataset.trials else 0
            info['Trials Class 2'] = dataset.trials[dataset.cl2].shape[2] if dataset.cl2 in dataset.trials else 0
    except:
        pass
    return info

def process_uploaded_file(uploaded_file, dataset_type):
    """Process uploaded MATLAB file."""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mat') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        if dataset_type == 'BCIC4':
            dataset = MotorImageryBcic4(tmp_path)
            dataset.load_mat()
            dataset.setup_training_trials()
            dataset.filter(8, 15)
            dataset.feature_extract_trials()
        else:  # OpenBMI
            dataset = MotorImageryOpenBmi(tmp_path)
            dataset.load_mat()
            dataset.setup_training_trials()
            dataset.feature_extract_trials()
        
        return dataset, None
    except Exception as e:
        return None, str(e)
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def generate_classification_report(predictions, dataset, model_type):
    """Generate a classification report."""
    class_names = {1: dataset.cl1, -1: dataset.cl2}
    
    # Count predictions
    pred_counts = {}
    for pred in predictions:
        class_name = class_names.get(pred, 'Unknown')
        pred_counts[class_name] = pred_counts.get(class_name, 0) + 1
    
    # Create report
    report = f"""
# EEG Motor Imagery Classification Report

## Dataset Information
- **Dataset Type**: {model_type}
- **Sample Rate**: {dataset.sample_rate} Hz
- **Number of Channels**: {dataset.nchannels}
- **Total Trials Classified**: {len(predictions)}

## Classification Results

### Prediction Distribution
"""
    for class_name, count in pred_counts.items():
        percentage = (count / len(predictions)) * 100
        report += f"- **{class_name}**: {count} trials ({percentage:.1f}%)\n"
    
    report += f"""
### Detailed Predictions

| Trial # | Predicted Class |
|---------|---------------|
"""
    for i, pred in enumerate(predictions, 1):
        class_name = class_names.get(pred, 'Unknown')
        report += f"| {i} | {class_name} |\n"
    
    return report

def main():
    """Main Streamlit application."""
    st.markdown('<h1 class="main-header">🧠 EEG Motor Imagery Classifier</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Upload & Classify", "About"])
    
    if page == "About":
        st.header("About This Application")
        st.markdown("""
        This application allows you to classify EEG motor imagery data using pretrained machine learning models.
        
        ### Features:
        - Upload your own MATLAB (.mat) EEG dataset
        - View dataset information and statistics
        - Visualize EEG signals and features
        - Classify trials as left hand, right hand, or foot movements
        - Generate detailed classification reports
        
        ### Supported Datasets:
        - **BCI Competition IV Dataset 1** (100Hz)
        - **OpenBMI Dataset** (1000Hz)
        
        ### How to Use:
        1. Go to "Upload & Classify" page
        2. Select your dataset type
        3. Upload your MATLAB file
        4. Choose a pretrained model
        5. View results and download report
        """)
        return
    
    # Main content
    st.header("Upload & Classify EEG Data")
    
    # Model selection
    st.subheader("1. Select Pretrained Model")
    model_type = st.selectbox(
        "Choose a pretrained model:",
        ["combined", "bcic4", "openbmi"],
        help="Combined model is trained on both datasets and generally performs best"
    )
    
    # Load model
    model, csp_matrix = load_model(model_type)
    
    if model is None:
        st.error(f"❌ Pretrained model '{model_type}' not found. Please run `python train_models.py` first to train models.")
        st.info("""
        To train models:
        1. Make sure you have dataset files in the `data/` directory
        2. Run: `python train_models.py`
        3. Refresh this page
        """)
        return
    
    st.success(f"✅ Loaded {model_type} model")
    
    # Dataset type selection
    st.subheader("2. Select Dataset Type")
    dataset_type = st.radio(
        "What type of dataset are you uploading?",
        ["BCIC4", "OpenBMI"],
        help="BCIC4: BCI Competition IV format\nOpenBMI: OpenBMI dataset format"
    )
    
    # File upload
    st.subheader("3. Upload Your Dataset")
    uploaded_file = st.file_uploader(
        "Choose a MATLAB file (.mat)",
        type=['mat'],
        help="Upload a .mat file in BCIC4 or OpenBMI format"
    )
    
    if uploaded_file is not None:
        with st.spinner("Processing uploaded file..."):
            dataset, error = process_uploaded_file(uploaded_file, dataset_type)
        
        if error:
            st.error(f"Error processing file: {error}")
            return
        
        if dataset is None:
            st.error("Failed to process dataset")
            return
        
        st.success("✅ Dataset loaded successfully!")
        
        # Dataset Information
        st.subheader("📊 Dataset Information")
        info = get_dataset_info(dataset)
        
        col1, col2 = st.columns(2)
        with col1:
            for key, value in list(info.items())[:len(info)//2]:
                st.metric(key, value)
        with col2:
            for key, value in list(info.items())[len(info)//2:]:
                st.metric(key, value)
        
        # Visualizations
        st.subheader("📈 Visualizations")
        
        viz_tabs = st.tabs(["Power Spectral Density", "Log-Variance Features", "EEG Signal"])
        
        with viz_tabs[0]:
            st.write("Power Spectral Density (PSD) Analysis")
            if dataset.trials is not None:
                try:
                    from signal_processing import psd
                    psd_r, freqs = psd(dataset.trials[dataset.cl1], dataset.sample_rate)
                    psd_l, freqs = psd(dataset.trials[dataset.cl2], dataset.sample_rate)
                    trials_PSD = {dataset.cl1: psd_r, dataset.cl2: psd_l}
                    
                    # Create PSD plot
                    channels = ['C3', 'Cz', 'C4']
                    available_channels = [ch for ch in channels if ch in dataset.channel_names]
                    
                    if available_channels:
                        n_chans = len(available_channels)
                        fig, axes = plt.subplots(1, n_chans, figsize=(5*n_chans, 4))
                        if n_chans == 1:
                            axes = [axes]
                        
                        for i, ch in enumerate(available_channels):
                            ch_idx = dataset.channel_names.index(ch)
                            axes[i].plot(freqs, np.mean(trials_PSD[dataset.cl1][ch_idx, :, :], axis=1), label=dataset.cl1)
                            axes[i].plot(freqs, np.mean(trials_PSD[dataset.cl2][ch_idx, :, :], axis=1), label=dataset.cl2)
                            axes[i].set_xlabel('Frequency (Hz)')
                            axes[i].set_ylabel('PSD')
                            axes[i].set_title(f'Channel {ch}')
                            axes[i].legend()
                            axes[i].grid(True)
                            axes[i].set_xlim(1, 35)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.warning("Required channels (C3, Cz, C4) not found in dataset")
                except Exception as e:
                    st.warning(f"Could not generate PSD plot: {e}")
        
        with viz_tabs[1]:
            st.write("Log-Variance Features")
            if dataset.trials_csp is not None:
                try:
                    from signal_processing import logvar
                    logvar_cl1 = logvar(dataset.trials_csp[dataset.cl1])
                    logvar_cl2 = logvar(dataset.trials_csp[dataset.cl2])
                    
                    fig, ax = plt.subplots(figsize=(12, 5))
                    nchannels = logvar_cl1.shape[0]
                    x0 = np.arange(nchannels)
                    x1 = np.arange(nchannels) + 0.4
                    
                    y0 = np.mean(logvar_cl1, axis=1)
                    y1 = np.mean(logvar_cl2, axis=1)
                    
                    ax.bar(x0, y0, width=0.5, color='b', label=dataset.cl1)
                    ax.bar(x1, y1, width=0.4, color='r', label=dataset.cl2)
                    ax.set_xlim(-0.5, nchannels + 0.5)
                    ax.set_xlabel('Components')
                    ax.set_ylabel('Log-Variance')
                    ax.set_title('Log-Variance of CSP Components')
                    ax.legend()
                    ax.grid(True, axis='y')
                    
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not generate log-variance plot: {e}")
        
        with viz_tabs[2]:
            st.write("Raw EEG Signal (Channel C3)")
            if dataset.EEG is not None and 'C3' in dataset.channel_names:
                try:
                    c3_idx = dataset.channel_names.index('C3')
                    # Plot a sample of the signal
                    sample_length = min(5000, dataset.EEG.shape[1])
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.plot(dataset.EEG[c3_idx, :sample_length], linewidth=0.5)
                    ax.set_xlabel('Sample')
                    ax.set_ylabel('Voltage (μV)')
                    ax.set_title('EEG Signal - Channel C3')
                    ax.grid(True)
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not generate EEG signal plot: {e}")
        
        # Classification
        st.subheader("🔮 Classification")
        
        if st.button("Classify Trials", type="primary"):
            with st.spinner("Classifying trials..."):
                try:
                    # Combine all trials for prediction
                    if dataset_type == 'BCIC4':
                        all_trials = np.concatenate([
                            dataset.trials_filt[dataset.cl1],
                            dataset.trials_filt[dataset.cl2]
                        ], axis=2)
                    else:
                        all_trials = np.concatenate([
                            dataset.trials[dataset.cl1],
                            dataset.trials[dataset.cl2]
                        ], axis=2)
                    
                    # Extract features using the pretrained CSP
                    features = dataset.extract_features_for_prediction(all_trials, csp_matrix)
                    
                    # Predict
                    predictions = model.predict(features)
                    probabilities = model.predict_proba(features) if hasattr(model, 'predict_proba') else None
                    
                    # Store in session state
                    st.session_state['predictions'] = predictions
                    st.session_state['probabilities'] = probabilities
                    st.session_state['features'] = features
                    
                    st.success("✅ Classification complete!")
                    
                except Exception as e:
                    st.error(f"Classification error: {e}")
                    st.exception(e)
        
        # Display results
        if 'predictions' in st.session_state:
            predictions = st.session_state['predictions']
            probabilities = st.session_state.get('probabilities', None)
            
            st.subheader("📋 Classification Results")
            
            # Summary statistics
            class_names = {1: dataset.cl1, -1: dataset.cl2}
            pred_counts = {}
            for pred in predictions:
                class_name = class_names.get(pred, 'Unknown')
                pred_counts[class_name] = pred_counts.get(class_name, 0) + 1
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Trials", len(predictions))
            with col2:
                st.metric(f"{dataset.cl1} Predictions", pred_counts.get(dataset.cl1, 0))
            with col3:
                st.metric(f"{dataset.cl2} Predictions", pred_counts.get(dataset.cl2, 0))
            
            # Results table
            st.write("### Detailed Results")
            results_data = []
            for i, pred in enumerate(predictions, 1):
                class_name = class_names.get(pred, 'Unknown')
                row = {"Trial": i, "Predicted Class": class_name}
                if probabilities is not None:
                    row["Confidence"] = f"{probabilities[i-1].max()*100:.1f}%"
                results_data.append(row)
            
            df_results = pd.DataFrame(results_data)
            st.dataframe(df_results, use_container_width=True)
            
            # Visualization
            st.write("### Prediction Distribution")
            fig, ax = plt.subplots(figsize=(8, 5))
            classes = list(pred_counts.keys())
            counts = list(pred_counts.values())
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            ax.bar(classes, counts, color=colors[:len(classes)])
            ax.set_ylabel('Number of Trials')
            ax.set_xlabel('Predicted Class')
            ax.set_title('Classification Results Distribution')
            ax.grid(True, axis='y')
            st.pyplot(fig)
            
            # Download report
            st.subheader("📥 Download Report")
            report = generate_classification_report(predictions, dataset, model_type)
            st.download_button(
                label="Download Classification Report (Markdown)",
                data=report,
                file_name="classification_report.md",
                mime="text/markdown"
            )
            
            # Download results as CSV
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="Download Results (CSV)",
                data=csv,
                file_name="classification_results.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()

