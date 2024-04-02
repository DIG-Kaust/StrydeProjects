import numpy as np
import matplotlib.pyplot as plt


def mag2db(x):
    """Magnitude to Decibel

    Parameters
    ----------
    x : :obj:`float` or :obj:`np.ndarray`
        Input
    
    Returns
    -------
    xdb : :obj:`float` or :obj:`np.ndarray`
         Output in dB

    """
    return 20 * np.log10(x)


def db2mag(x):
    """Decibel to Magnitude
    
    Parameters
    ----------
    x : :obj:`float` or :obj:`np.ndarray`
        Input in dB
    
    Returns
    -------
    xmag : :obj:`float` or :obj:`np.ndarray`
         Output in normal magnitude

    """
    return 10 ** (x / 20)


def customplot(ax, x, y, color, plottype):
    """Customized plotting

    Add trace to a given axis based on user-defined plottype

    Parameters
    ----------
    ax : :obj:`plt.axes`
        Axes handle
    x : :obj:`np.ndarray`
        X-axis values
    y : :obj:`np.ndarray`
        Y-axis values
    color : :obj:`str`
        Color to plot trace
    plottype : :obj:`matplotlib.pyplot`
        Plotting handle (plot, semilogy, loglog)
         
    """
    if plottype == plt.plot:
        ax.plot(x, y, color)
    elif plottype == plt.semilogy:
        ax.semilogy(x, y, color)
    elif plottype == plt.loglog:
        ax.loglog(x, y, color)