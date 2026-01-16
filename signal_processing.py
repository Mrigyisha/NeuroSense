# Signal processing functions for EEG data analysis.
import numpy as np
import scipy.signal
from numpy import linalg


def bandpass(trials, lo, hi, sample_rate, filter_order):
    # Apply a bandpass filter to the input signal
    a, b = scipy.signal.iirfilter(filter_order, [lo/(sample_rate/2.0), hi/(sample_rate/2.0)])

    nchannels = trials.shape[0]
    nsamples = trials.shape[1]
    ntrials = trials.shape[2]

    trials_filt = np.zeros((nchannels, nsamples, ntrials))
    for i in range(ntrials):
        trials_filt[:,:,i] = scipy.signal.filtfilt(a, b, trials[:,:,i], axis=1)

    return trials_filt


def logvar(trials):
    # Calculate the log variance of the data for each channel
    return np.log(np.var(trials, axis=1))


def cov(trials, nsamples):
    # Return the average covariance across trials
    ntrials = trials.shape[2]
    covs = [ trials[:,:,i].dot(trials[:,:,i].T) / nsamples for i in range(ntrials) ]
    return np.mean(covs, axis=0)


def whitening(sigma):
    # Get a whitening matrix for sigma
    u, l, _ = linalg.svd(sigma)
    return u.dot( np.diag(l ** -0.5) )


def csp(trials_r, trials_l, nsamples):
    # Compute the CSP spatial filter matrix W
    cov_r = cov(trials_r, nsamples)
    cov_l = cov(trials_l, nsamples)
    p = whitening(cov_r + cov_l)
    b, _, _ = linalg.svd( p.T.dot(cov_l).dot(p) )
    w = p.dot(b)
    return w


def apply_mix(w, trials):
    # Apply the mix matrix to all trials

    nchannels = trials.shape[0]
    nsamples = trials.shape[1]
    ntrials = trials.shape[2]

    trials_csp = np.zeros((nchannels, nsamples, ntrials))
    for i in range(ntrials):
        trials_csp[:,:,i] = w.T.dot(trials[:,:,i])
    return trials_csp


def psd(trials, sample_rate):
    # Compute Power Spectral Density (PSD) for every trial

    nchannels = trials.shape[0]
    nsamples = trials.shape[1]
    ntrials = trials.shape[2]

    npoints = (nsamples / 2) + 1 # Only need half because PSD is symmetric
    trials_PSD = np.zeros((nchannels, int(npoints), ntrials))

    # Go through all trials and channels
    from matplotlib import mlab
    for trial in range(ntrials):
        for ch in range(nchannels):
            # Get the PSD for this spot
            (PSD, freqs) = mlab.psd(trials[ch,:,trial], NFFT=int(nsamples), Fs=sample_rate)
            trials_PSD[ch, :, trial] = PSD.ravel()

    return trials_PSD, freqs

