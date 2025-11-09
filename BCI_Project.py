"""
Main execution script for EEG Motor Imagery Classification.

Marijn van Vliet, neuroscience_tutorials/eeg-bci/
3. Imagined movement.ipynb, GitHub repository,
https://github.com/wmvanvliet/neuroscience_tutorials/blob/master/eeg-bci/3.%20Imagined%20movement.ipynb

The above file in the GitHub repository acted as the building block for this script,
especially for functions and the 'MotorImageryBcic4' class.

This script provides an interactive interface to process EEG data from:
- BCI Competition IV Dataset 1
- OpenBMI Dataset

The code has been refactored into separate modules:
- signal_processing.py: Signal processing functions (filtering, CSP, etc.)
- visualization.py: Plotting functions
- dataset_classes.py: Dataset handler classes
"""
from dataset_classes import MotorImageryBcic4, MotorImageryOpenBmi
from sklearn.svm import SVC
import matplotlib.pyplot as plt


if __name__ == "__main__":
    while True:
        try:
            dataset_option = int(input('Enter 1 for BCIC4 dataset, and 2 for OpenBMI dataset\n'))
            if not (dataset_option == 1 or dataset_option == 2):
                raise ValueError
        except ValueError:
            print('Input provided is not 1 or 2! Please try again.\n')
            continue
        if (dataset_option == 1):
            print('MATLAB files from BCI Competition IV have to be in format data/BCICIV_calib_ds1(letter).mat in the same directory as this file.\n')
            file_list = [letter for letter in input('Enter list of letter(s) for the BCIC4 files seprated by commas\n').split(',')]
            try:
                for letter in file_list:
                    d = MotorImageryBcic4(f'data/BCICIV_calib_ds1{letter}.mat')
                    model = SVC(kernel='rbf')
                    d.load_mat()
                    d.setup_training_trials()
                    d.filter(8,15)
                    d.feature_extract_trials()
                    SCORE = d.eval_model(model)
                    print(f'Accuracy of prediction for data/BCICIV_calib_ds1{letter}.mat is {round(SCORE,2)}')
                    d.plot_trials()
                    d.plot_raw_filt_csp()
                    d.plot_filt_csp_logvar()
                    plt.show()
                break
            except FileNotFoundError:
                print('Requested file(s) not found\n')
        if (dataset_option == 2):
            print('MATLAB files from OpenBMI dataset have to be in format sess01_subj(num)_EEG_MI.mat in the same directory as this file.\n')
            print('num is an integer with two digits (add leading zero for single digit numbers) ranging from 01 to 49')
            file_list = [num for num in input('enter list of integers for the openBMI files seprated by comma\n').split(',')]
            try:
                for num in file_list:
                    d = MotorImageryOpenBmi(f'data/sess01_subj{num}_EEG_MI.mat')
                    model = SVC(kernel='rbf')
                    d.load_mat()
                    d.setup_training_trials()
                    d.feature_extract_trials()
                    SCORE = d.eval_model(model)
                    print(f'Accuracy of prediction for data/sess01_subj{num}_EEG_MI.mat is {round(SCORE,2)}')
                    d.plot_filt_csp()
                    d.plot_logvar_filt_csp()
                    plt.show()
                break
            except FileNotFoundError:
                print('Requested file(s) not found\n')
