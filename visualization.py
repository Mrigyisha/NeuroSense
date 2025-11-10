"""Plotting utilities for PSD and log-variance features."""
import numpy as np
import matplotlib.pyplot as plt


def plot_psd(trials_PSD, freqs, chan_ind, chan_lab=None, maxy=None):
    """Plot mean PSD per class for selected channels."""
    plt.figure(figsize=(12,5))

    nchans = len(chan_ind)

    # Grid shape
    if nchans == 4:
        nrows = 2
        ncols = 2
    else:
        nrows = int(np.ceil(nchans / 3))
        ncols = min(4, nchans)

    for i,ch in enumerate(chan_ind):
        plt.subplot(nrows,ncols,i+1)

        # Mean PSD per class
        for cl in trials_PSD.keys():
            plt.plot(freqs, np.mean(trials_PSD[cl][ch,:,:], axis=1), label=cl)

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
    """Plot mean log-variance per component for two classes."""
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

