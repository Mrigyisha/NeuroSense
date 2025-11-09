"""
Signal processing functions for EEG data analysis.

This module contains functions for filtering, feature extraction, and
Common Spatial Patterns (CSP) analysis used in motor imagery classification.
"""
import numpy as np
import scipy.signal
from numpy import linalg


def bandpass(trials, lo, hi, sample_rate, filter_order):
    '''
    Designs and applies a bandpass filter to the signal.

    Parameters
    ----------
    trials : 3d-array (channels x samples x trials)
        The EEG signal
    lo : float
        Lower frequency bound (in Hz)
    hi : float
        Upper frequency bound (in Hz)
    sample_rate : float
        Sample rate of the signal (in Hz)
    filter_order : uint
        Order of the filter

    Returns
    -------
    trials_filt : 3d-array (channels x samples x trials)
        The bandpassed signal
    '''

    # The iirfilter() function takes the filter order: higher numbers mean a sharper frequency cutoff,
    # but the resulting signal might be shifted in time, lower numbers mean a soft frequency cutoff,
    # but the resulting signal less distorted in time. It also takes the lower and upper frequency bounds
    # to pass, divided by the niquist frequency, which is the sample rate divided by 2:
    a, b = scipy.signal.iirfilter(filter_order, [lo/(sample_rate/2.0), hi/(sample_rate/2.0)])

    # Applying the filter to each trial

    nchannels = trials.shape[0]
    nsamples = trials.shape[1]
    ntrials = trials.shape[2]

    trials_filt = np.zeros((nchannels, nsamples, ntrials))
    for i in range(ntrials):
        trials_filt[:,:,i] = scipy.signal.filtfilt(a, b, trials[:,:,i], axis=1)

    return trials_filt


def logvar(trials):
    '''
    Calculate the log-var of each channel.

    Parameters
    ----------
    trials : 3d-array (channels x samples x trials)
        The EEG signal.

    Returns
    -------
    logvar - 2d-array (channels x trials)
        For each channel the logvar of the signal
    '''
    # Calculate the log(var) of the trials
    return np.log(np.var(trials, axis=1))


def cov(trials, nsamples):
    ''' Calculate the covariance for each trial and return their average '''
    ntrials = trials.shape[2]
    covs = [ trials[:,:,i].dot(trials[:,:,i].T) / nsamples for i in range(ntrials) ]
    return np.mean(covs, axis=0)


def whitening(sigma):
    ''' Calculate a whitening matrix for covariance matrix sigma. '''
    u, l, _ = linalg.svd(sigma)
    return u.dot( np.diag(l ** -0.5) )


def csp(trials_r, trials_l, nsamples):
    '''
    Calculate the CSP transformation matrix W.
    arguments:
        trials_r - Array (channels x samples x trials) containing right hand movement trials
        trials_l - Array (channels x samples x trials) containing left hand movement trials
    returns:
        Mixing matrix W
    '''
    cov_r = cov(trials_r, nsamples)
    cov_l = cov(trials_l, nsamples)
    p = whitening(cov_r + cov_l)
    b, _, _ = linalg.svd( p.T.dot(cov_l).dot(p) )
    w = p.dot(b)
    return w


def apply_mix(w, trials):
    ''' Apply a mixing matrix to each trial (basically multiply w with the EEG signal matrix)
        arguments:
        w - Mixing matrix
        trials - Array (channels x samples x trials)
    returns:
        trials_csp - Array (components x samples x trials)
    '''

    nchannels = trials.shape[0]
    nsamples = trials.shape[1]
    ntrials = trials.shape[2]

    trials_csp = np.zeros((nchannels, nsamples, ntrials))
    for i in range(ntrials):
        trials_csp[:,:,i] = w.T.dot(trials[:,:,i])
    return trials_csp


def psd(trials, sample_rate):
    '''
    Calculates for each trial the Power Spectral Density (PSD).

    Parameters
    ----------
    trials : 3d-array (channels x samples x trials)
        The EEG signal
    sample_rate : float
        Sample rate of the signal (in Hz)

    Returns
    -------
    trial_PSD : 3d-array (channels x PSD x trials)
        the PSD for each trial.
    freqs : list of floats
        The frequencies for which the PSD was computed (useful for plotting later)
    '''

    nchannels = trials.shape[0]
    nsamples = trials.shape[1]
    ntrials = trials.shape[2]

    npoints = (nsamples / 2) + 1
    trials_PSD = np.zeros((nchannels, int(npoints), ntrials))

    # Iterate over trials and channels
    from matplotlib import mlab
    for trial in range(ntrials):
        for ch in range(nchannels):
            # Calculate the PSD
            (PSD, freqs) = mlab.psd(trials[ch,:,trial], NFFT=int(nsamples), Fs=sample_rate)
            trials_PSD[ch, :, trial] = PSD.ravel()

    return trials_PSD, freqs

