"""
Visualization functions for EEG data analysis.

This module contains plotting functions for Power Spectral Density (PSD)
and log-variance features.
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_psd(trials_PSD, freqs, chan_ind, chan_lab=None, maxy=None):
    '''
    Plots PSD data calculated with psd().

    Parameters
    ----------
    trials_PSD : dict
        The PSD data, as returned by psd()
    freqs : list of floats
        The frequencies for which the PSD is defined, as returned by psd()
    chan_ind : list of integers
        The indices of the channels to plot
    chan_lab : list of strings
        (optional) List of names for each channel
    maxy : float
        (optional) Limit the y-axis to this value
    '''
    plt.figure(figsize=(12,5))

    nchans = len(chan_ind)

    # Maximum of 3 plots per row
    if nchans == 4:
        nrows = 2
        ncols = 2
    else:
        nrows = int(np.ceil(nchans / 3))
        ncols = min(4, nchans)

    # Enumerate over the channels
    for i,ch in enumerate(chan_ind):
        # Figure out which subplot to draw to
        plt.subplot(nrows,ncols,i+1)

        # Plot the PSD for each class
        for cl in trials_PSD.keys():
            plt.plot(freqs, np.mean(trials_PSD[cl][ch,:,:], axis=1), label=cl)

        # All plot decoration below...

        plt.xlim(1,35)

        if maxy != None:
            plt.ylim(0,maxy)

        plt.grid()

        plt.xlabel('Frequency (Hz)')

        if chan_lab == None:
            plt.title('Channel %d' % (ch+1))
        else:
            plt.title(chan_lab[i])

        plt.legend()

    plt.tight_layout()
    plt.show(block = False)


def plot_logvar(trials_cl1, trials_cl2):
    '''
    Plots the log-var of each channel/component.
    arguments:
        trials_cl1 - Array (channels x trials) containing log-vars for class 1
        trials_cl2 - Array (channels x trials) containing log-vars for class 2
    '''
    plt.figure(figsize=(12,5))

    nchannels = trials_cl1.shape[0]

    x0 = np.arange(nchannels)
    x1 = np.arange(nchannels) + 0.4

    y0 = np.mean(trials_cl1, axis=1)
    y1 = np.mean(trials_cl2, axis=1)

    plt.bar(x0, y0, width=0.5, color='b')
    plt.bar(x1, y1, width=0.4, color='r')

    plt.xlim(-0.5, nchannels+0.5)

    plt.gca().yaxis.grid(True)
    plt.title('log-var of each channel/component')
    plt.xlabel('channels/components')
    plt.ylabel('log-var')
    plt.legend(['right','left'])
    plt.show(block = False)

