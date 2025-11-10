"""Signal processing utilities: filtering, features, and CSP."""
import numpy as np
import scipy.signal
from numpy import linalg


def bandpass(trials, lo, hi, sample_rate, filter_order):
    """Zero-phase IIR bandpass per trial (channels × samples × trials)."""
    # Design digital IIR bandpass (normalized by Nyquist)
    a, b = scipy.signal.iirfilter(filter_order, [lo/(sample_rate/2.0), hi/(sample_rate/2.0)])

    nchannels = trials.shape[0]
    nsamples = trials.shape[1]
    ntrials = trials.shape[2]

    trials_filt = np.zeros((nchannels, nsamples, ntrials))
    for i in range(ntrials):
        trials_filt[:,:,i] = scipy.signal.filtfilt(a, b, trials[:,:,i], axis=1)

    return trials_filt


def logvar(trials):
    """Log-variance per channel/component (channels × trials)."""
    return np.log(np.var(trials, axis=1))


def cov(trials, nsamples):
    """Trial-wise covariance averaged across trials."""
    ntrials = trials.shape[2]
    covs = [ trials[:,:,i].dot(trials[:,:,i].T) / nsamples for i in range(ntrials) ]
    return np.mean(covs, axis=0)


def whitening(sigma):
    """Whitening matrix for covariance sigma."""
    u, l, _ = linalg.svd(sigma)
    return u.dot( np.diag(l ** -0.5) )


def csp(trials_r, trials_l, nsamples):
    """Compute CSP projection matrix W from right/left class trials."""
    cov_r = cov(trials_r, nsamples)
    cov_l = cov(trials_l, nsamples)
    p = whitening(cov_r + cov_l)
    b, _, _ = linalg.svd( p.T.dot(cov_l).dot(p) )
    w = p.dot(b)
    return w


def apply_mix(w, trials):
    """Apply mixing matrix W to trials → components × samples × trials."""

    nchannels = trials.shape[0]
    nsamples = trials.shape[1]
    ntrials = trials.shape[2]

    trials_csp = np.zeros((nchannels, nsamples, ntrials))
    for i in range(ntrials):
        trials_csp[:,:,i] = w.T.dot(trials[:,:,i])
    return trials_csp


def psd(trials, sample_rate):
    """Power Spectral Density per channel and trial (periodogram-style)."""

    nchannels = trials.shape[0]
    nsamples = trials.shape[1]
    ntrials = trials.shape[2]

    npoints = (nsamples / 2) + 1
    trials_PSD = np.zeros((nchannels, int(npoints), ntrials))

    from matplotlib import mlab
    for trial in range(ntrials):
        for ch in range(nchannels):
            (PSD, freqs) = mlab.psd(trials[ch,:,trial], NFFT=int(nsamples), Fs=sample_rate)
            trials_PSD[ch, :, trial] = PSD.ravel()

    return trials_PSD, freqs

