"""
Dataset handler classes for EEG motor imagery data.

This module contains classes for handling different EEG datasets:
- MotorImageryBcic4: BCI Competition IV Dataset 1
- MotorImageryOpenBmi: OpenBMI Dataset
"""
import numpy as np
import scipy.io
import scipy.signal
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from matplotlib import patches

from signal_processing import bandpass, csp, apply_mix, logvar, psd
from visualization import plot_psd, plot_logvar


class MotorImageryBcic4:
    """
    Class is used to handle MATLAB files from BCI Competition IV
    Data set 1 100Hz data which can be found here: https://www.bbci.de/competition/iv/
    along with a description of the data.
    They have to be of format 'data/BCICIV_calib_ds1{letter}.mat' and
    be downloaded and placed in a folder named 'data' in the same directory
    as this file.
    The files contain EEG recordings of users while doing two classes of motor imagery
    selected from the three classes left hand, right hand, and foot.
    Class has methods that pre-process the EEG recording by applying bandpass filter
    as well as feature exctraction and selection using Common Spatial Patterns.
    Trains a Support Vector Machines model to classify between the two classes of
    movement.
    Can evaluate the model and give an accuracy score.
    """
    def __init__(self, mat_path):
        """
        Instantiates a MotorImageryBcic4 object that handles
        a MATLAB file.
        Args:
            mat_path (str): relative path of MATLAB file in the format
            'data/BCICIV_calib_ds1{letter}.mat'
            downloaded from BCI Competition IV Data set 1 100Hz dataset.
        """
        self.mat_path = mat_path
        self.nsamples = None
        self.nsamples_win = None
        self.cl1 = None
        self.cl2 = None
        self.nchannels = None
        self.sample_rate = None
        self.train_cl1_cl2 = None
        self.trials = None
        self.trials_filt = None
        self.trials_csp = None
        self.event_codes = None
        self.w = None
        self.event_onsets = None
        self.EEG = None
        self.channel_names = None

    def load_mat(self):
        """
        Loads the important data from the MATLAB file into class attributes
        """
        m = scipy.io.loadmat(self.mat_path, struct_as_record=True)

        # SciPy.io.loadmat does not deal well with Matlab structures, resulting in lots of
        # extra dimensions in the arrays. This makes the code a bit more cluttered
        self.sample_rate = m['nfo']['fs'][0][0][0][0]
        self.channel_names = [s[0] for s in m['nfo']['clab'][0][0][0]]
        self.EEG = m['cnt'].T
        self.nchannels, self.nsamples = self.EEG.shape
        self.event_onsets = m['mrk'][0][0][0]
        self.event_codes = m['mrk'][0][0][1]
        [self.cl1, self.cl2] = [s[0] for s in m['nfo']['classes'][0][0][0]]

    def setup_training_trials(self):
        # Dictionary to store the trials in, each class gets an entry
        trials = {}

        # The time window (in samples) to extract for each trial, here 0.5 -- 2.5 seconds
        win = np.arange(int(0.5*self.sample_rate), int(2.5*self.sample_rate))
        # Length of the time window
        self.nsamples_win = len(win)
        # Loop over the classes (right, foot)
        for cl, code in zip([self.cl1, self.cl2], np.unique(self.event_codes)):

            # Extract the onsets for the class
            cl_onsets = self.event_onsets[self.event_codes == code]

            # Allocate memory for the trials
            trials[cl] = np.zeros((self.nchannels, self.nsamples_win, len(cl_onsets)))

            # Extract each trial
            for i, onset in enumerate(cl_onsets):
                trials[cl][:,:,i] = self.EEG[:, win+onset]

        #trials : 3d-array (channels x samples x trials) The EEG signal
        self.trials = trials

    def filter(self, lo, hi):
        """
        Applies a bandpass filter between lo and hi frequencies
        on the EEG data in the 'trials' attribute
        Args:
            lo (float): Lower frequency bound (in Hz)
            hi (float): Upper frequency bound (in Hz)
        """
        self.trials_filt = {self.cl1: bandpass(self.trials[self.cl1], lo, hi, self.sample_rate, 6),
                            self.cl2: bandpass(self.trials[self.cl2], lo, hi, self.sample_rate, 6)}

    def feature_extract_trials(self):
        # Calculate the number of trials for each class the above percentage boils down to
        ntrain_r = int(self.trials_filt[self.cl1].shape[2] )
        ntrain_l = int(self.trials_filt[self.cl2].shape[2] )


        # Train the CSP on the training set
        w = csp(self.trials_filt[self.cl1], self.trials_filt[self.cl2], self.nsamples_win)
        self.w = w

        self.trials_csp = {self.cl1: np.ones_like(self.trials_filt[self.cl1]),
                           self.cl2: np.ones_like(self.trials_filt[self.cl2])}

        # Apply the CSP on the training
        self.trials_csp[self.cl1] = apply_mix(w, self.trials_filt[self.cl1])
        self.trials_csp[self.cl2] = apply_mix(w, self.trials_filt[self.cl2])

        trials_csp_cl1 = self.trials_csp[self.cl1]
        trials_csp_cl2 = self.trials_csp[self.cl2]

        # Select only the first and last two components for classification
        comp = np.array([0,1,-2,-1])
        trials_csp_cl1 = trials_csp_cl1[comp,:,:]
        trials_csp_cl2 = trials_csp_cl2[comp,:,:]

        # Calculate the log-var
        train_cl1 = logvar(trials_csp_cl1)
        train_cl2 = logvar(trials_csp_cl2)

        train_cl1 = np.append(train_cl1.transpose(),
                            np.ones([ntrain_r,1],dtype = int),
                            axis = 1)
        train_cl2 = np.append(train_cl2.transpose(),
                            (-1* np.ones([ntrain_l,1],dtype = int)),
                            axis = 1)

        self.train_cl1_cl2 = np.vstack([train_cl1, train_cl2])

    def plot_raw_filt_csp(self):


        psd_r, freqs = psd(self.trials[self.cl1],self.sample_rate)
        psd_l, freqs = psd(self.trials[self.cl2],self.sample_rate)
        trials_PSD = {self.cl1: psd_r, self.cl2: psd_l}

        plot_psd(
            trials_PSD,
            freqs,
            [self.channel_names.index(ch) for ch in ['C3', 'Cz', 'C4']],
            chan_lab=['C3', 'Cz', 'C4'],
            maxy=500)

        psd_r, freqs = psd(self.trials_filt[self.cl1],self.sample_rate)
        psd_l, freqs = psd(self.trials_filt[self.cl2],self.sample_rate)
        trials_filt_PSD = {self.cl1: psd_r, self.cl2: psd_l}

        plot_psd(
            trials_filt_PSD,
            freqs,
            [self.channel_names.index(ch) for ch in ['C3', 'Cz', 'C4']],
            chan_lab=['C3', 'Cz', 'C4'],
            maxy=500)

        psd_r, freqs = psd(self.trials_csp[self.cl1],self.sample_rate)
        psd_l, freqs = psd(self.trials_csp[self.cl2],self.sample_rate)
        trials_csp_PSD = {self.cl1: psd_r, self.cl2: psd_l}

        plot_psd(
            trials_csp_PSD,
            freqs,
            [0,1,-2,-1],
            chan_lab=['First Component','Second Component','Second to Last Component','Last Component'])

    def plot_filt_csp_logvar(self):
        logvar_filt_cl1 = np.ones_like(self.trials_filt[self.cl1])
        logvar_filt_cl2 = np.ones_like(self.trials_filt[self.cl2])

        logvar_filt_cl1 = logvar(self.trials_filt[self.cl1])
        logvar_filt_cl2 = logvar(self.trials_filt[self.cl2])

        logvar_csp_cl1 = np.ones_like(self.trials_csp[self.cl1])
        logvar_csp_cl2 = np.ones_like(self.trials_csp[self.cl2])

        logvar_csp_cl1 = logvar(self.trials_csp[self.cl1])
        logvar_csp_cl2 = logvar(self.trials_csp[self.cl2])

        plot_logvar(logvar_filt_cl1,logvar_filt_cl2)
        plot_logvar(logvar_csp_cl1,logvar_csp_cl2)

    def train_model(self, model):
        """
        Trains a model on the extracted features.
        Args:
            model (sklearn.svm.SVC): initialized model with kernel='rbf'
        Returns:
            trained model
        """
        x = self.train_cl1_cl2[:,:4]
        y = self.train_cl1_cl2[:,4]
        model.fit(x, y)
        return model

    def eval_model(self, model):
        """
        Takes in a Support Vector Machine model and trains it on the
        self.train_cl1_cl2 object attribute using K-fold cross validation
        K == 10
        Args:
            model (sklearn.svm.SVC): initialized model with kernel='rbf'
        Returns:
            float: accuracy of the model at predicting right and left hand movement
        """
        score = 0
        x = self.train_cl1_cl2[:,:4]
        y = self.train_cl1_cl2[:,4]
        kf = StratifiedKFold(n_splits=10)
        for (train, test) in (kf.split(x,y)):
            model.fit(x[train,:],y[train])
            score += model.score(x[test,:],y[test])
        return score*100/kf.get_n_splits()

    def extract_features_for_prediction(self, trials_data, csp_matrix=None):
        """
        Extract features from new trial data for prediction.
        Args:
            trials_data: 3d-array (channels x samples x trials) of filtered EEG data
            csp_matrix: Optional CSP matrix. If None, uses self.w
        Returns:
            2d-array (trials x features) of extracted features
        """
        if csp_matrix is None:
            if self.w is None:
                raise ValueError("CSP matrix not available. Run feature_extract_trials() first or provide csp_matrix.")
            csp_matrix = self.w
        
        # Apply CSP transformation
        trials_csp = apply_mix(csp_matrix, trials_data)
        
        # Select first and last 2 components
        comp = np.array([0,1,-2,-1])
        trials_csp_selected = trials_csp[comp,:,:]
        
        # Calculate log-variance features
        features = logvar(trials_csp_selected).transpose()
        
        return features

    def predict_trials(self, model, trials_data, csp_matrix=None):
        """
        Predict classes for new trial data.
        Args:
            model: Trained sklearn model
            trials_data: 3d-array (channels x samples x trials) of filtered EEG data
            csp_matrix: Optional CSP matrix. If None, uses self.w
        Returns:
            array: Predicted classes for each trial
        """
        features = self.extract_features_for_prediction(trials_data, csp_matrix)
        predictions = model.predict(features)
        return predictions

    def plot_trials(self):

        fig, ax = plt.subplots()
        fig.suptitle('EEG Recording of Channel C3 while performing Motor Imagery')
        fig.supxlabel('time samples at 100 Hz')
        fig.supylabel('Voltage (microV)')

        try:
            if self.EEG is None:
                raise TypeError
        except TypeError:
            print('Please use load_mat() first.\n')
        else:
            EEG = self.EEG
            ax.set_xlim([0, EEG.shape[1]])
            EEG[self.channel_names.index('C3'),:] = np.divide(EEG[self.channel_names.index('C3'),:],10)
            ax.set_ylim([np.min(EEG[self.channel_names.index('C3'),:]), np.max(EEG[self.channel_names.index('C3'),:])])
            x = range(EEG.shape[1])
            ax.plot(x,EEG[self.channel_names.index('C3'),:], linewidth = 0.2, color = '#7c8594')

            ax.scatter(self.event_onsets[self.event_codes == 1],EEG[self.channel_names.index('C3'),self.event_onsets[self.event_codes == 1]],
                        color = 'red',
                        marker = 'x',
                        linewidth = 2,
                        zorder = 2.5)
            ax.scatter(self.event_onsets[self.event_codes == -1],EEG[self.channel_names.index('C3'),self.event_onsets[self.event_codes == -1]],
                        color = 'black',
                        marker = 'x',
                        linewidth = 2,
                        zorder = 2.5)
            plt.legend([None,'Right Hand','Left Hand'])

            win = np.arange(int(0.5*self.sample_rate), int(2.5*self.sample_rate))
            for cl, code in zip([self.cl1, self.cl2], np.unique(self.event_codes)):
                cl_onsets = self.event_onsets[self.event_codes == code]
                for _, onset in enumerate(cl_onsets):
                    x = win+onset
                    y = self.EEG[self.channel_names.index('C3'),x]
                    if(cl == self.cl1):
                        ax.plot(x,y,color = 'black',zorder = 2.5)
                        rect = patches.Rectangle([x[0],np.max(y)],
                                                (x[-1]-x[0]),
                                                (np.min(y)-np.max(y)),
                                                edgecolor = 'black',
                                                alpha = 0.3)
                        ax.add_patch(rect)
                    else:
                        ax.plot(x,y,color = 'red',zorder = 2.5)
                        rect = patches.Rectangle([x[0],np.max(y)],
                                                (x[-1]-x[0]),
                                                (np.min(y)-np.max(y)),
                                                edgecolor = 'red',
                                                alpha = 0.3)
                        ax.add_patch(rect)
            plt.show(block = False)


class MotorImageryOpenBmi:
    """
    Class is used to handle MATLAB files from the OpenBMI dataset
    which can be found here: http://dx.doi.org/10.5524/100542.
    They have to be of format 'sess01_subj0{num}_EEG_MI.mat' and
    be downloaded and placed in a folder named 'data' in the same directory
    as this file.
    The files contain EEG recordings of users while doing motor imagery tasks of
    right and left hand movement.
    Class has methods that pre-process the EEG recording by applying bandpass filter
    as well as feature exctraction and selection using Common Spatial Patterns.
    Trains a Support Vector Machines model to classify between the two classes of
    movement.
    Can evaluate the model and give an accuracy score.

    """
    def __init__(self, mat_path):
        """
        Instantiates a MotorImageryOpenBmi object that handles
        a MATLAB file
        Args:
            mat_path (str): relative path of MATLAB file in the format
            'data/sess01_subj0{num}_EEG_MI.mat'
            downloaded from the OpenBMI data set from
            http://dx.doi.org/10.5524/100542.
        """
        self.mat_path = mat_path
        self.nsamples = None
        self.nsamples_win = None
        self.cl1 = None
        self.cl2 = None
        self.nchannels = None
        self.sample_rate = None
        self.test_cl1_cl2 = None
        self.train_cl1_cl2 = None
        self.trials = None
        self.test_dict = None
        self.trials_filt = None
        self.channel_names = None
        self.w = None
        self.EEG_train = None
        self.event_codes_train = None
        self.event_onsets_train = None
        self.trials_csp = None

    def load_mat(self):
        """
        Loads the important data from the MATLAB file into class attributes
        """
        m = scipy.io.loadmat(self.mat_path, struct_as_record=True)

        # SciPy.io.loadmat does not deal well with Matlab structures, resulting in lots of
        # extra dimensions in the arrays. This makes the code a bit more cluttered

        #rate of EEG data sampling, 1000Hz
        self.sample_rate = m['EEG_MI_train']['fs'][0][0][0][0]

        #array of shape (number of channels, number of data samples) == (62, 1341200),
        # EEG data during training phase
        self.EEG_train = m['EEG_MI_train']['x'][0][0].T

        #array of shape (number of channels, number of data samples) == (62, 1545960),
        # EEG data during testing phase
        EEG_test = m['EEG_MI_test']['x'][0][0].T

        #array continaing names of channels used in EEG recording
        self.channel_names = [s[0] for s in m['EEG_MI_train']['chan'][0][0][0]]

        self.nchannels, self.nsamples = self.EEG_train.shape

        #array of shape(1, number of event onsets) == (1,100),
        # its values are the indicies of the EEG_train array when
        #motor imagery events onsetted
        self.event_onsets_train = m['EEG_MI_train']['t'][0][0]

        #array of shape(1, number of event onsets) == (1,100),
        # its values are the indicies of the EEG_test array for event onsets
        event_onsets_test = m['EEG_MI_test']['t'][0][0]

        #array of shape(1, number of event onsets) == (1,100),
        #its values are the class of movement for the event
        #with the same index in the event_onset array
        #value of 1 indicating imagined right hand movement,
        #while 2 indicates imagined left hand movement.
        self.event_codes_train = m['EEG_MI_train']['y_dec'][0][0]

        #same as event_codes_train,
        #however value of -1 now indicates left hand movement instead of 2
        event_codes_test = m['EEG_MI_test']['y_dec'][0][0].astype('int8')
        event_codes_test[event_codes_test == 2] = -1

        #class labels
        [self.cl1, self.cl2] = ['right', 'left']

        #Dictionary to store relevant arrays for use in online testing phase
        test_dict = {'EEG' : EEG_test,
                    'event_onsets' : event_onsets_test,
                    'event_codes' : event_codes_test}
        self.test_dict = test_dict

    def setup_training_trials(self):
        """
        The offline EEG data is setup according to instructions from
        OpenBMI literature found here
        https://doi.org/10.1093/gigascience/giz002.
        20 electrodes in the motor cortex region were selected:
        (FC-5/3/1/2/4/6, C-5/3/1/z/2/4/6, and CP-5/3/1/z/2/4/6)
        The EEG data were band-pass filtered between 8 and 30 Hz
        with a 5th order Butterworth digital filter.
        Continuous EEG data were then segmented
        from 1,000 to 3,500 ms with respect to stimulus onset.
        EEG epochs were therefore constituted as
        20 (channels) x 2500 (samples) x 100 (trials).
        """
        required_channels = ['FC1','FC2','FC3','FC4','FC5','FC6',
                            'C1','C2','C3','C4','C5','C6','Cz',
                            'CP1','CP2','CP3','CP4','CP5','CP6','CPz']

        # Save original channel names before overwriting
        original_channel_names = self.channel_names.copy()
        self.channel_names = required_channels

        EEG_train_20_channels = np.zeros((20,self.EEG_train.shape[1]))

        for i, _ in enumerate(required_channels):
            index = int(original_channel_names.index(required_channels[i]))
            EEG_train_20_channels[i,:] = self.EEG_train[index,:]

        a, b = scipy.signal.butter(5,
                                [8,30],
                                'bandpass',
                                analog = False,
                                fs=1000)

        EEG_train_20_filt = np.zeros_like(EEG_train_20_channels)
        for i in range(20):
            EEG_train_20_filt[i,:] = scipy.signal.filtfilt(a, b, EEG_train_20_channels[i,:])

        self.trials_filt = EEG_train_20_filt

        # Dictionary to store the trials in, each class gets an entry, used for training phase
        trials = {}

        #Continous EEG data was epoched from 1000 to 3500 ms with respect to simulus onset
        #therefore each epoch
        win = np.arange(int(1*self.sample_rate), int(3.5*self.sample_rate))
        # Length of the time window
        self.nsamples_win = len(win)
        # Loop over the classes (right, left)
        for cl, code in zip([self.cl1, self.cl2], np.unique(self.event_codes_train)):

            # Extract the onsets for the class
            cl_onsets = self.event_onsets_train[self.event_codes_train == code]

            # Allocate memory for the trials
            trials[cl] = np.zeros((20, self.nsamples_win, len(cl_onsets)))

            # Extract each trial
            for i, onset in enumerate(cl_onsets):
                trials[cl][:,:,i] = EEG_train_20_filt[:, win+onset]

        #trials : 3d-array (channels x samples x trials) The EEG signal
        self.trials = trials

    def feature_extract_trials(self):
        """
        Common Spatial Pattern technique is used to dsicriminate between
        two classes of motor Imagery by optimizing the variances between them.
        Log of the variance is passed to the classifier model.
        """
        # Calculate the number of trials for each class
        ntrain_r = int(self.trials[self.cl1].shape[2] )
        ntrain_l = int(self.trials[self.cl2].shape[2] )


        # Train the CSP on the training set, w is the CSP projection matrix
        w = csp(self.trials[self.cl1], self.trials[self.cl2], self.nsamples_win)
        self.w = w
        # Apply the CSP on the training set, gives the spatial filtered signal
        # where the first and last rows contain the filters that maximize the
        # variance between the 2 classes

        self.trials_csp = {self.cl1: np.ones_like(self.trials[self.cl1]),
                           self.cl2: np.ones_like(self.trials[self.cl2])}

        self.trials_csp[self.cl1] = apply_mix(w, self.trials[self.cl1])
        self.trials_csp[self.cl2] = apply_mix(w, self.trials[self.cl2])

        trials_csp_cl1 = self.trials_csp[self.cl1]
        trials_csp_cl2 = self.trials_csp[self.cl2]

        # Select only the first 2 and last 2 components for classification
        comp = np.array([0,1,-2,-1])
        trials_csp_cl1 = trials_csp_cl1[comp,:,:]
        trials_csp_cl2 = trials_csp_cl2[comp,:,:]

        # Calculate the log-var
        train_cl1 = logvar(trials_csp_cl1)
        train_cl2 = logvar(trials_csp_cl2)

        train_cl1 = np.append(train_cl1.transpose(),
                              np.ones([ntrain_r,1],dtype = int),
                              axis = 1)
        train_cl2 = np.append(train_cl2.transpose(),
                              (-1* np.ones([ntrain_l,1],dtype = int)),
                              axis = 1)
        # train_cl1_cl2: 2d array of shape (trials x (features + 1))
        # 4 features for each trial along with label of trial
        self.train_cl1_cl2 = np.vstack([train_cl1, train_cl2])

    def plot_filt_csp(self):
        psd_r, freqs = psd(self.trials[self.cl1],self.sample_rate)
        psd_l, freqs = psd(self.trials[self.cl2],self.sample_rate)
        trials_PSD = {self.cl1: psd_r, self.cl2: psd_l}

        plot_psd(
            trials_PSD,
            freqs,
            [self.channel_names.index(ch) for ch in ['C3', 'Cz', 'C4']],
            chan_lab=['C3', 'Cz', 'C4'])

        psd_r, freqs = psd(self.trials_csp[self.cl1],self.sample_rate)
        psd_l, freqs = psd(self.trials_csp[self.cl2],self.sample_rate)
        trials_csp_PSD = {self.cl1: psd_r, self.cl2: psd_l}

        plot_psd(
            trials_csp_PSD,
            freqs,
            [0,1,-2,-1],
            chan_lab=['First Component','Second Component','Second to Last Component','Last Component'])

    def plot_logvar_filt_csp(self):
        logvar_filt_cl1 = np.ones_like(self.trials[self.cl1])
        logvar_filt_cl2 = np.ones_like(self.trials[self.cl2])

        logvar_filt_cl1 = logvar(self.trials[self.cl1])
        logvar_filt_cl2 = logvar(self.trials[self.cl2])

        logvar_csp_cl1 = np.ones_like(self.trials_csp[self.cl1])
        logvar_csp_cl2 = np.ones_like(self.trials_csp[self.cl2])

        logvar_csp_cl1 = logvar(self.trials_csp[self.cl1])
        logvar_csp_cl2 = logvar(self.trials_csp[self.cl2])

        plot_logvar(logvar_filt_cl1,logvar_filt_cl2)
        plot_logvar(logvar_csp_cl1,logvar_csp_cl2)

    def train_model(self, model):
        """
        Trains a model on the extracted features.
        Args:
            model (sklearn.svm.SVC): initialized model with kernel='rbf'
        Returns:
            trained model
        """
        x = self.train_cl1_cl2[:,:4]
        y = self.train_cl1_cl2[:,4]
        model.fit(x, y)
        return model

    def eval_model(self, model):
        """
        Takes in a Support Vector Machine model and trains it on the
        self.train_cl1_cl2 object attribute using K-fold cross validation
        K == 10

        Args:
            model (sklearn.svm.SVC): initialized model with kernel='rbf'

        Returns:
            float: accuracy of the model at predicting right and left hand movement
        """
        score = 0
        x = self.train_cl1_cl2[:,:4]
        y = self.train_cl1_cl2[:,4]
        kf = StratifiedKFold(n_splits=10)
        for (train, test) in (kf.split(x,y)):
            model.fit(x[train,:],y[train])
            score += model.score(x[test,:],y[test])
        return score*100/kf.get_n_splits()

    def extract_features_for_prediction(self, trials_data, csp_matrix=None):
        """
        Extract features from new trial data for prediction.
        Args:
            trials_data: 3d-array (channels x samples x trials) of filtered EEG data
            csp_matrix: Optional CSP matrix. If None, uses self.w
        Returns:
            2d-array (trials x features) of extracted features
        """
        if csp_matrix is None:
            if self.w is None:
                raise ValueError("CSP matrix not available. Run feature_extract_trials() first or provide csp_matrix.")
            csp_matrix = self.w
        
        # Apply CSP transformation
        trials_csp = apply_mix(csp_matrix, trials_data)
        
        # Select first and last 2 components
        comp = np.array([0,1,-2,-1])
        trials_csp_selected = trials_csp[comp,:,:]
        
        # Calculate log-variance features
        features = logvar(trials_csp_selected).transpose()
        
        return features

    def predict_trials(self, model, trials_data, csp_matrix=None):
        """
        Predict classes for new trial data.
        Args:
            model: Trained sklearn model
            trials_data: 3d-array (channels x samples x trials) of filtered EEG data
            csp_matrix: Optional CSP matrix. If None, uses self.w
        Returns:
            array: Predicted classes for each trial
        """
        features = self.extract_features_for_prediction(trials_data, csp_matrix)
        predictions = model.predict(features)
        return predictions

