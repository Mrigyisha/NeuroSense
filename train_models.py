"""
Script to train and save pretrained models on available datasets.

This script trains models on all available BCIC4 and OpenBMI datasets,
combines the training data, and saves pretrained models and CSP matrices.
"""
import os
import pickle
import numpy as np
from sklearn.svm import SVC
from dataset_classes import MotorImageryBcic4, MotorImageryOpenBmi
import glob

def train_bcic4_models():
    """Train models on all available BCIC4 datasets."""
    print("Training BCIC4 models...")
    
    # Find all BCIC4 files
    bcic4_files = glob.glob('data/BCICIV_calib_ds1*.mat')
    
    if not bcic4_files:
        print("No BCIC4 files found in data/ directory")
        return None, None
    
    all_features = []
    all_labels = []
    all_csp_matrices = []
    
    for file_path in bcic4_files:
        try:
            print(f"Processing {file_path}...")
            d = MotorImageryBcic4(file_path)
            d.load_mat()
            d.setup_training_trials()
            d.filter(8, 15)
            d.feature_extract_trials()
            
            # Collect features and labels
            features = d.train_cl1_cl2[:, :4]
            labels = d.train_cl1_cl2[:, 4]
            all_features.append(features)
            all_labels.append(labels)
            all_csp_matrices.append(d.w)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    if not all_features:
        print("No valid BCIC4 data processed")
        return None, None
    
    # Combine all features and labels
    X_combined = np.vstack(all_features)
    y_combined = np.hstack(all_labels)
    
    # Train model on combined data
    model = SVC(kernel='rbf', probability=True)
    model.fit(X_combined, y_combined)
    
    # Use the first CSP matrix as reference (or average them)
    # For simplicity, we'll use the first one
    # In production, you might want to average or use ensemble
    csp_matrix = all_csp_matrices[0] if all_csp_matrices else None
    
    print(f"BCIC4 Model trained on {len(X_combined)} samples")
    print(f"Class distribution: {np.bincount((y_combined + 1).astype(int))}")
    
    return model, csp_matrix

def train_openbmi_models():
    """Train models on all available OpenBMI datasets."""
    print("Training OpenBMI models...")
    
    # Find all OpenBMI files
    openbmi_files = glob.glob('data/sess01_subj*_EEG_MI.mat')
    
    if not openbmi_files:
        print("No OpenBMI files found in data/ directory")
        return None, None
    
    all_features = []
    all_labels = []
    all_csp_matrices = []
    
    for file_path in openbmi_files:
        try:
            print(f"Processing {file_path}...")
            d = MotorImageryOpenBmi(file_path)
            d.load_mat()
            d.setup_training_trials()
            d.feature_extract_trials()
            
            # Collect features and labels
            features = d.train_cl1_cl2[:, :4]
            labels = d.train_cl1_cl2[:, 4]
            all_features.append(features)
            all_labels.append(labels)
            all_csp_matrices.append(d.w)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    if not all_features:
        print("No valid OpenBMI data processed")
        return None, None
    
    # Combine all features and labels
    X_combined = np.vstack(all_features)
    y_combined = np.hstack(all_labels)
    
    # Train model on combined data
    model = SVC(kernel='rbf', probability=True)
    model.fit(X_combined, y_combined)
    
    # Use the first CSP matrix as reference
    csp_matrix = all_csp_matrices[0] if all_csp_matrices else None
    
    print(f"OpenBMI Model trained on {len(X_combined)} samples")
    print(f"Class distribution: {np.bincount((y_combined + 1).astype(int))}")
    
    return model, csp_matrix

def main():
    """Main function to train and save all models."""
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Train BCIC4 models
    bcic4_model, bcic4_csp = train_bcic4_models()
    if bcic4_model is not None:
        with open('models/bcic4_model.pkl', 'wb') as f:
            pickle.dump(bcic4_model, f)
        with open('models/bcic4_csp.pkl', 'wb') as f:
            pickle.dump(bcic4_csp, f)
        print("BCIC4 model saved to models/bcic4_model.pkl")
        print("BCIC4 CSP matrix saved to models/bcic4_csp.pkl")
    
    # Train OpenBMI models
    openbmi_model, openbmi_csp = train_openbmi_models()
    if openbmi_model is not None:
        with open('models/openbmi_model.pkl', 'wb') as f:
            pickle.dump(openbmi_model, f)
        with open('models/openbmi_csp.pkl', 'wb') as f:
            pickle.dump(openbmi_csp, f)
        print("OpenBMI model saved to models/openbmi_model.pkl")
        print("OpenBMI CSP matrix saved to models/openbmi_csp.pkl")
    
    # Train a combined model
    if bcic4_model is not None and openbmi_model is not None:
        print("\nTraining combined model...")
        # Load features from both datasets
        bcic4_files = glob.glob('data/BCICIV_calib_ds1*.mat')
        openbmi_files = glob.glob('data/sess01_subj*_EEG_MI.mat')
        
        all_features = []
        all_labels = []
        
        for file_path in bcic4_files:
            try:
                d = MotorImageryBcic4(file_path)
                d.load_mat()
                d.setup_training_trials()
                d.filter(8, 15)
                d.feature_extract_trials()
                all_features.append(d.train_cl1_cl2[:, :4])
                all_labels.append(d.train_cl1_cl2[:, 4])
            except:
                continue
        
        for file_path in openbmi_files:
            try:
                d = MotorImageryOpenBmi(file_path)
                d.load_mat()
                d.setup_training_trials()
                d.feature_extract_trials()
                all_features.append(d.train_cl1_cl2[:, :4])
                all_labels.append(d.train_cl1_cl2[:, 4])
            except:
                continue
        
        if all_features:
            X_combined = np.vstack(all_features)
            y_combined = np.hstack(all_labels)
            
            combined_model = SVC(kernel='rbf', probability=True)
            combined_model.fit(X_combined, y_combined)
            
            with open('models/combined_model.pkl', 'wb') as f:
                pickle.dump(combined_model, f)
            print(f"Combined model saved to models/combined_model.pkl")
            print(f"Combined model trained on {len(X_combined)} samples")
    
    print("\nModel training complete!")

if __name__ == "__main__":
    main()

