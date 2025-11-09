# Brain-Computer Interface : 2-Class Motor Imagery Classification using CSP/SVM

A comprehensive EEG motor imagery classification system with both command-line and web interface capabilities.

## Features

- **Signal Processing**: Bandpass filtering, Common Spatial Patterns (CSP), feature extraction
- **Machine Learning**: Support Vector Machine (SVM) classification
- **Pretrained Models**: Train and save models for quick inference
- **Web Interface**: Streamlit app for easy data upload and classification
- **Visualization**: Power Spectral Density, log-variance features, EEG signal plots
- **Report Generation**: Automatic classification reports

## Project Structure

```
eeg/
├── BCI_Project.py          # Main command-line execution script
├── streamlit_app.py        # Streamlit web application
├── train_models.py          # Script to train and save pretrained models
├── dataset_classes.py       # Dataset handler classes (BCIC4, OpenBMI)
├── signal_processing.py    # Signal processing functions
├── visualization.py        # Plotting functions
├── requirements.txt        # Python dependencies
└── data/                   # Dataset files directory
    └── models/            # Saved pretrained models (created after training)
```

## Installation

1. Clone or download this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Datasets

### BCI Competition IV Dataset 1
- Format: `data/BCICIV_calib_ds1{letter}.mat`
- Download: https://www.bbci.de/competition/iv/
- Sample rate: 100Hz
- Classes: Left hand, Right hand, Foot (2 classes per file)

### OpenBMI Dataset
- Format: `data/sess01_subj{num}_EEG_MI.mat`
- Download: http://dx.doi.org/10.5524/100542
- Sample rate: 1000Hz
- Classes: Left hand, Right hand

## Usage

### 1. Command-Line Interface

Run the main script interactively:
```bash
python BCI_Project.py
```

This will prompt you to:
- Select dataset type (BCIC4 or OpenBMI)
- Enter file identifiers
- View accuracy scores and visualizations

### 2. Train Pretrained Models

Before using the Streamlit app, train models on your available datasets:
```bash
python train_models.py
```

This will:
- Process all available datasets in the `data/` directory
- Train models for BCIC4, OpenBMI, and a combined model
- Save models and CSP matrices to `models/` directory

### 3. Streamlit Web Application

Launch the web interface:
```bash
streamlit run streamlit_app.py
```

The app provides:
- **Upload Interface**: Upload your own MATLAB dataset files
- **Data Visualization**: View PSD, log-variance features, and raw EEG signals
- **Classification**: Classify trials using pretrained models
- **Reports**: Download classification results as Markdown or CSV

#### Using the Streamlit App:

1. **Select Model**: Choose from combined, bcic4, or openbmi pretrained models
2. **Upload Dataset**: Select your dataset type and upload a .mat file
3. **View Information**: See dataset statistics and visualizations
4. **Classify**: Click "Classify Trials" to get predictions
5. **Download Results**: Export reports and CSV files

## Code Modules

### `dataset_classes.py`
- `MotorImageryBcic4`: Handler for BCI Competition IV datasets
- `MotorImageryOpenBmi`: Handler for OpenBMI datasets
- Methods: `load_mat()`, `setup_training_trials()`, `filter()`, `feature_extract_trials()`, `train_model()`, `predict_trials()`

### `signal_processing.py`
- `bandpass()`: Bandpass filtering
- `csp()`: Common Spatial Patterns algorithm
- `logvar()`: Log-variance feature extraction
- `psd()`: Power Spectral Density calculation

### `visualization.py`
- `plot_psd()`: Plot Power Spectral Density
- `plot_logvar()`: Plot log-variance features

## Classification Pipeline

1. **Data Loading**: Load MATLAB files with EEG recordings
2. **Trial Extraction**: Extract time windows around motor imagery events
3. **Filtering**: Apply bandpass filter (8-15 Hz for BCIC4, 8-30 Hz for OpenBMI)
4. **CSP Feature Extraction**:
   - Apply Common Spatial Patterns
   - Select first 2 and last 2 components
   - Compute log-variance → 4 features per trial
5. **Classification**: Predict using trained SVM model

## Model Training

The system supports two approaches:

1. **Subject-Specific**: Train a model on each dataset individually (original approach)
2. **Pretrained Models**: Train on multiple datasets and save for reuse (new approach)

Pretrained models are saved as:
- `models/bcic4_model.pkl` + `models/bcic4_csp.pkl`
- `models/openbmi_model.pkl` + `models/openbmi_csp.pkl`
- `models/combined_model.pkl` + `models/combined_csp.pkl`

## Output

- **Accuracy Scores**: Cross-validation accuracy for model evaluation
- **Visualizations**: 
  - Power Spectral Density plots
  - Log-variance feature comparisons
  - Raw EEG signal plots
- **Classification Reports**: Detailed predictions per trial
- **CSV Exports**: Results in tabular format

## Notes

- EEG patterns vary significantly between individuals, so subject-specific models often perform better
- The pretrained models provide a starting point but may need fine-tuning for new subjects
- Ensure your uploaded datasets match the expected format (BCIC4 or OpenBMI)

## References

- Marijn van Vliet, neuroscience_tutorials/eeg-bci/ (GitHub)
- BCI Competition IV: https://www.bbci.de/competition/iv/
- OpenBMI Dataset: http://dx.doi.org/10.5524/100542
