# EEG motor imagery dataset handlers for BCIC IV-1 and OpenBMI
import numpy as np
import scipy.io
import scipy.signal
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from matplotlib import patches

from signal_processing import bandpass, csp, apply_mix, logvar, psd
from visualization import plot_psd, plot_logvar


class MotorImageryBcic4:
    # Handler for BCI Competition IV, Dataset 1 (100 Hz)
    def __init__(self, mat_path):
        """Initialize with path to a .mat file, e.g., data/BCICIV_calib_ds1a.mat."""
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
        """Load metadata, EEG, and event markers from the MATLAB file."""
        m = scipy.io.loadmat(self.mat_path, struct_as_record=True)

        # Unpack fields and squeeze extra nesting from MATLAB structs
        self.sample_rate = m['nfo']['fs'][0][0][0][0]
        self.channel_names = [s[0] for s in m['nfo']['clab'][0][0][0]]
        self.EEG = m['cnt'].T
        self.nchannels, self.nsamples = self.EEG.shape
        self.event_onsets = m['mrk'][0][0][0]
        self.event_codes = m['mrk'][0][0][1]
        [self.cl1, self.cl2] = [s[0] for s in m['nfo']['classes'][0][0][0]]

    def setup_training_trials(self):
        # Turn EEG into individual trials for each class
        trials = {}

        # Use data from 0.5s to 2.5s after the cue
        win = np.arange(int(0.5*self.sample_rate), int(2.5*self.sample_rate))
        self.nsamples_win = len(win)
        for cl, code in zip([self.cl1, self.cl2], np.unique(self.event_codes)):

            cl_onsets = self.event_onsets[self.event_codes == code]

            trials[cl] = np.zeros((self.nchannels, self.nsamples_win, len(cl_onsets)))

            # Fill in the trials for this class
            for i, onset in enumerate(cl_onsets):
                trials[cl][:,:,i] = self.EEG[:, win+onset]

        # Store it as channels × samples × trials
        self.trials = trials

    def filter(self, lo, hi):
        """Bandpass trials between lo and hi (Hz)."""
        self.trials_filt = {self.cl1: bandpass(self.trials[self.cl1], lo, hi, self.sample_rate, 6),
                            self.cl2: bandpass(self.trials[self.cl2], lo, hi, self.sample_rate, 6)}

    def feature_extract_trials(self):
        # Number of trials per class
        ntrain_r = int(self.trials_filt[self.cl1].shape[2] )
        ntrain_l = int(self.trials_filt[self.cl2].shape[2] )

        # Compute CSP projection
        w = csp(self.trials_filt[self.cl1], self.trials_filt[self.cl2], self.nsamples_win)
        self.w = w

        self.trials_csp = {self.cl1: np.ones_like(self.trials_filt[self.cl1]),
                           self.cl2: np.ones_like(self.trials_filt[self.cl2])}

        # Project trials to CSP space
        self.trials_csp[self.cl1] = apply_mix(w, self.trials_filt[self.cl1])
        self.trials_csp[self.cl2] = apply_mix(w, self.trials_filt[self.cl2])

        trials_csp_cl1 = self.trials_csp[self.cl1]
        trials_csp_cl2 = self.trials_csp[self.cl2]

        # Use first and last two components
        comp = np.array([0,1,-2,-1])
        trials_csp_cl1 = trials_csp_cl1[comp,:,:]
        trials_csp_cl2 = trials_csp_cl2[comp,:,:]

        # Log-variance features
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

        # PSD before filtering, by class
        psd_r, freqs = psd(self.trials[self.cl1],self.sample_rate)
        psd_l, freqs = psd(self.trials[self.cl2],self.sample_rate)
        trials_PSD = {self.cl1: psd_r, self.cl2: psd_l}

        plot_psd(
            trials_PSD,
            freqs,
            [self.channel_names.index(ch) for ch in ['C3', 'Cz', 'C4']],
            chan_lab=['C3', 'Cz', 'C4'],
            maxy=500)

        # PSD after filtering
        psd_r, freqs = psd(self.trials_filt[self.cl1],self.sample_rate)
        psd_l, freqs = psd(self.trials_filt[self.cl2],self.sample_rate)
        trials_filt_PSD = {self.cl1: psd_r, self.cl2: psd_l}

        plot_psd(
            trials_filt_PSD,
            freqs,
            [self.channel_names.index(ch) for ch in ['C3', 'Cz', 'C4']],
            chan_lab=['C3', 'Cz', 'C4'],
            maxy=500)

        # PSD of CSP components used for classification
        psd_r, freqs = psd(self.trials_csp[self.cl1],self.sample_rate)
        psd_l, freqs = psd(self.trials_csp[self.cl2],self.sample_rate)
        trials_csp_PSD = {self.cl1: psd_r, self.cl2: psd_l}

        plot_psd(
            trials_csp_PSD,
            freqs,
            [0,1,-2,-1],
            chan_lab=['First Component','Second Component','Second to Last Component','Last Component'])

    def plot_filt_csp_logvar(self):
        # Compare log-variance before/after CSP
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
        """Fit classifier on 4 CSP log-variance features per trial."""
        x = self.train_cl1_cl2[:,:4]
        y = self.train_cl1_cl2[:,4]
        model.fit(x, y)
        return model

    def eval_model(self, model):
        """10-fold CV accuracy on the training feature matrix."""
        score = 0
        x = self.train_cl1_cl2[:,:4]
        y = self.train_cl1_cl2[:,4]
        kf = StratifiedKFold(n_splits=10)
        for (train, test) in (kf.split(x,y)):
            model.fit(x[train,:],y[train])
            score += model.score(x[test,:],y[test])
        return score*100/kf.get_n_splits()

    def extract_features_for_prediction(self, trials_data, csp_matrix=None):
        """Extract 4 CSP log-variance features per trial for inference."""
        if csp_matrix is None:
            if self.w is None:
                raise ValueError("CSP matrix not available. Run feature_extract_trials() first or provide csp_matrix.")
            csp_matrix = self.w
        
        # Transform to CSP space
        trials_csp = apply_mix(csp_matrix, trials_data)
        
        # Keep first/last two components
        comp = np.array([0,1,-2,-1])
        trials_csp_selected = trials_csp[comp,:,:]
        
        # Log-variance features (trials × features)
        features = logvar(trials_csp_selected).transpose()
        
        return features

    def predict_trials(self, model, trials_data, csp_matrix=None):
        """Predict class labels for provided trials using a trained model."""
        features = self.extract_features_for_prediction(trials_data, csp_matrix)
        predictions = model.predict(features)
        return predictions

    def plot_trials(self):

        # Plot raw C3 with event markers and trial windows
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
    # Handler for OpenBMI motor imagery dataset
    def __init__(self, mat_path):
        """Initialize with path to OpenBMI .mat, e.g., data/sess01_subj01_EEG_MI.mat."""
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
        """Load training/test EEG, channels, and event meta from MATLAB file."""
        m = scipy.io.loadmat(self.mat_path, struct_as_record=True)

        # 1000 Hz sampling
        self.sample_rate = m['EEG_MI_train']['fs'][0][0][0][0]

        # Training EEG (channels × samples)
        self.EEG_train = m['EEG_MI_train']['x'][0][0].T

        # Testing EEG (channels × samples)
        EEG_test = m['EEG_MI_test']['x'][0][0].T

        self.channel_names = [s[0] for s in m['EEG_MI_train']['chan'][0][0][0]]

        self.nchannels, self.nsamples = self.EEG_train.shape

        # Event onsets (train) in sample indices
        self.event_onsets_train = m['EEG_MI_train']['t'][0][0]

        # Event onsets (test)
        event_onsets_test = m['EEG_MI_test']['t'][0][0]

        # Event codes (train): 1=right, 2=left
        self.event_codes_train = m['EEG_MI_train']['y_dec'][0][0]

        # Event codes (test): map 2->-1 for left
        event_codes_test = m['EEG_MI_test']['y_dec'][0][0].astype('int8')
        event_codes_test[event_codes_test == 2] = -1

        # Class label names
        [self.cl1, self.cl2] = ['right', 'left']

        # Pack test artifacts for later use
        test_dict = {'EEG' : EEG_test,
                    'event_onsets' : event_onsets_test,
                    'event_codes' : event_codes_test}
        self.test_dict = test_dict

    def setup_training_trials(self):
        """Select 20 motor-cortex channels, bandpass 8–30 Hz, epoch 1.0–3.5 s."""
        required_channels = ['FC1','FC2','FC3','FC4','FC5','FC6',
                            'C1','C2','C3','C4','C5','C6','Cz',
                            'CP1','CP2','CP3','CP4','CP5','CP6','CPz']

        # Reorder/select channels
        original_channel_names = self.channel_names.copy()
        self.channel_names = required_channels

        EEG_train_20_channels = np.zeros((20,self.EEG_train.shape[1]))

        for i, _ in enumerate(required_channels):
            index = int(original_channel_names.index(required_channels[i]))
            EEG_train_20_channels[i,:] = self.EEG_train[index,:]

        # 5th-order Butterworth bandpass (8–30 Hz at 1000 Hz)
        a, b = scipy.signal.butter(5,
                                [8,30],
                                'bandpass',
                                analog = False,
                                fs=1000)

        EEG_train_20_filt = np.zeros_like(EEG_train_20_channels)
        for i in range(20):
            EEG_train_20_filt[i,:] = scipy.signal.filtfilt(a, b, EEG_train_20_channels[i,:])

        self.trials_filt = EEG_train_20_filt

        # Epoch training data per class
        trials = {}

        # 1.0–3.5 s post-cue window
        win = np.arange(int(1*self.sample_rate), int(3.5*self.sample_rate))
        self.nsamples_win = len(win)
        for cl, code in zip([self.cl1, self.cl2], np.unique(self.event_codes_train)):

            cl_onsets = self.event_onsets_train[self.event_codes_train == code]

            trials[cl] = np.zeros((20, self.nsamples_win, len(cl_onsets)))

            # Fill trials for this class
            for i, onset in enumerate(cl_onsets):
                trials[cl][:,:,i] = EEG_train_20_filt[:, win+onset]

        # channels × samples × trials
        self.trials = trials

    def feature_extract_trials(self):
        """Compute CSP, derive 4 log-var features per trial, stack with labels."""
        # Counts per class
        ntrain_r = int(self.trials[self.cl1].shape[2] )
        ntrain_l = int(self.trials[self.cl2].shape[2] )

        # CSP projection matrix
        w = csp(self.trials[self.cl1], self.trials[self.cl2], self.nsamples_win)
        self.w = w

        self.trials_csp = {self.cl1: np.ones_like(self.trials[self.cl1]),
                           self.cl2: np.ones_like(self.trials[self.cl2])}

        self.trials_csp[self.cl1] = apply_mix(w, self.trials[self.cl1])
        self.trials_csp[self.cl2] = apply_mix(w, self.trials[self.cl2])

        trials_csp_cl1 = self.trials_csp[self.cl1]
        trials_csp_cl2 = self.trials_csp[self.cl2]

        # Use first/last two components
        comp = np.array([0,1,-2,-1])
        trials_csp_cl1 = trials_csp_cl1[comp,:,:]
        trials_csp_cl2 = trials_csp_cl2[comp,:,:]

        # Log-variance features
        train_cl1 = logvar(trials_csp_cl1)
        train_cl2 = logvar(trials_csp_cl2)

        train_cl1 = np.append(train_cl1.transpose(),
                              np.ones([ntrain_r,1],dtype = int),
                              axis = 1)
        train_cl2 = np.append(train_cl2.transpose(),
                              (-1* np.ones([ntrain_l,1],dtype = int)),
                              axis = 1)
        self.train_cl1_cl2 = np.vstack([train_cl1, train_cl2])

    def plot_filt_csp(self):
        # PSD before CSP (filtered trials)
        psd_r, freqs = psd(self.trials[self.cl1],self.sample_rate)
        psd_l, freqs = psd(self.trials[self.cl2],self.sample_rate)
        trials_PSD = {self.cl1: psd_r, self.cl2: psd_l}

        plot_psd(
            trials_PSD,
            freqs,
            [self.channel_names.index(ch) for ch in ['C3', 'Cz', 'C4']],
            chan_lab=['C3', 'Cz', 'C4'])

        # PSD of CSP components used for classification
        psd_r, freqs = psd(self.trials_csp[self.cl1],self.sample_rate)
        psd_l, freqs = psd(self.trials_csp[self.cl2],self.sample_rate)
        trials_csp_PSD = {self.cl1: psd_r, self.cl2: psd_l}

        plot_psd(
            trials_csp_PSD,
            freqs,
            [0,1,-2,-1],
            chan_lab=['First Component','Second Component','Second to Last Component','Last Component'])

    def plot_logvar_filt_csp(self):
        # Compare log-variance before/after CSP
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
        """Fit classifier on 4 CSP log-variance features per trial."""
        x = self.train_cl1_cl2[:,:4]
        y = self.train_cl1_cl2[:,4]
        model.fit(x, y)
        return model

    def eval_model(self, model):
        """10-fold CV accuracy on the training feature matrix."""
        score = 0
        x = self.train_cl1_cl2[:,:4]
        y = self.train_cl1_cl2[:,4]
        kf = StratifiedKFold(n_splits=10)
        for (train, test) in (kf.split(x,y)):
            model.fit(x[train,:],y[train])
            score += model.score(x[test,:],y[test])
        return score*100/kf.get_n_splits()

    def extract_features_for_prediction(self, trials_data, csp_matrix=None):
        """Extract 4 CSP log-variance features per trial for inference."""
        if csp_matrix is None:
            if self.w is None:
                raise ValueError("CSP matrix not available. Run feature_extract_trials() first or provide csp_matrix.")
            csp_matrix = self.w
        
        # Transform to CSP space
        trials_csp = apply_mix(csp_matrix, trials_data)
        
        # Keep first/last two components
        comp = np.array([0,1,-2,-1])
        trials_csp_selected = trials_csp[comp,:,:]
        
        # Log-variance features (trials × features)
        features = logvar(trials_csp_selected).transpose()
        
        return features

    def predict_trials(self, model, trials_data, csp_matrix=None):
        """Predict class labels for provided trials using a trained model."""
        features = self.extract_features_for_prediction(trials_data, csp_matrix)
        predictions = model.predict(features)
        return predictions

