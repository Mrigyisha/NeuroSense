# Main script for EEG Motor Imagery Classification
# This script uses MotorImageryBcic4 and MotorImageryOpenBmi classes to run classification
from dataset_classes import MotorImageryBcic4, MotorImageryOpenBmi
from sklearn.svm import SVC
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Ask the user which dataset to use (BCIC4 or OpenBMI)
    while True:
        try:
            dataset_option = int(input('Enter 1 for BCIC4 dataset, and 2 for OpenBMI dataset\n'))
            if not (dataset_option == 1 or dataset_option == 2):
                raise ValueError
        except ValueError:
            print('Input provided is not 1 or 2! Please try again.\n')
            continue
        if (dataset_option == 1):
            # Let user know required BCIC4 files location/format
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
            # Let user know required OpenBMI files location/format
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
