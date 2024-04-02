import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import stft
from pystryde.utils import mag2db


def _wiggletrace(ax, tz, trace, center=0, cpos='r', cneg='b'):
    """Seismic wiggle
    
    Plot a seismic wiggle trace onto an axis

    Parameters
    ----------
    ax : :obj:`plt.axes`
        Axes handle
    tz : :obj:`np.ndarray`
        Depth (or time) axis
    trace : :obj:`np.ndarray`
        Wiggle trace
    center : :obj:`float`, optional
        Center of x-axis where to switch color
    cpos : :obj:`str`, optional
        Color of positive filling
    cneg : :obj:`str`, optional
        Color of negative filling

    Returns
    -------
    ax : :obj:`plt.axes`, optional
         Axes handle

    """
    ax.fill_betweenx(tz, center, trace, where=(trace > center),
                     color=cpos, interpolate=True)
    ax.fill_betweenx(tz, center, trace, where=(trace <= center),
                     color=cneg, interpolate=True)
    ax.plot(trace, tz, 'k', lw=1)
    return ax


def wiggletracecomb(ax, tz, x, traces, scaling=None,
                    cpos='r', cneg='b'):
    """Seismic wiggles
    
    Plot a comb of seismic wiggle traces onto an axis

    Parameters
    ----------
    ax : :obj:`plt.axes`
        Axes handle
    tz : :obj:`np.ndarray`
        Depth (or time) axis
    x : :obj:`np.ndarray`
        Lateral axis
    traces : :obj:`np.ndarray`
        Wiggle traces
    scaling : :obj:`float`, optional
        Scaling to apply to each trace
    cpos : :obj:`str`, optional
        Color of positive filling
    cneg : :obj:`str`, optional
        Color of negative filling

    Returns
    -------
    ax : :obj:`plt.axes`, optional
         Axes handle

    """
    dx = np.abs(x[1]-x[0])
    tracesmax = np.max(traces, axis=1)
    tracesmin = np.min(traces, axis=1)
    dynrange = tracesmax - tracesmin
    maxdynrange = np.max(dynrange)
    if scaling is None:
        scaling = 2*dx/maxdynrange
    else:
        scaling = scaling*2*dx/maxdynrange

    for ix, xx in enumerate(x):
        trace = traces[ix]
        _wiggletrace(ax, tz, xx + trace * scaling, center=xx,
                     cpos=cpos, cneg=cneg)
    ax.set_xlim(x[0]-1.5*dx, x[-1]+1.5*dx)
    ax.set_ylim(tz[-1], tz[0])
    ax.set_xticks(x)
    ax.set_xticklabels([str(xx) for xx in x])

    return ax


def spectrogram(ax, data, dt, twin, overlap=0.5, nfft=None, db=True, clim=None, cmap='jet', tlims=None):
    """Spectrogram

    Compute and display spectrogram using :func:`scipy.signal.stft`

    Parameters
    ----------
    ax : :obj:`plt.axes`, optional
        Axes handle
    data : :obj:`np.ndarray`
        Trace to compute stft
    dt : :obj:`float`
        Time sampling
    twin : :obj:`float`
        Time extent of a single window
    overlap : :obj:`float`, optional
        Percentage of overlap between consecutive windows
    nfft : :obj:`int`, optional
        Number of samples in fft (if ``None``, use same samples of time window)
    db : :obj:`bool`, optional
        Display power spectrogram in normal or dB scale
    clim : :obj:`tuple`, optional
        Limits of colorscale
    cmap : :obj:`tuple`, optional
        Colormap
    tlims : :obj:`tuple`, optional
        Limits of time axis

    Returns
    -------
    ax : :obj:`plt.axes`, optional
         Axes handle
    
    """

    # Compute window size
    ntwin = int(np.round(twin / dt))
    if nfft is None:
        nfft = ntwin
    overlap = int(ntwin * overlap)

    # Compute stft
    f_stft, t_stft, data_stft = stft(data, 1. / dt, nperseg=ntwin, noverlap=overlap, nfft=nfft)

    if db:
        data_stft = mag2db(np.abs(data_stft)+1e-10)
    else:
        data_stft = np.abs(data_stft) ** 2
    if clim is None:
        cmin, cmax = data_stft.min(), data_stft.max()
    else:
        cmin, cmax = clim
    im = ax.imshow(data_stft, vmin=cmax, vmax=cmin, cmap=cmap, 
                   extent=(tlims[0] if tlims is not None else 0, 
                           tlims[1] if tlims is not None else data.size * dt, 
                           f_stft[-1], f_stft[0]))
    ax.set_title('STFT Magnitude')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    ax.axis('tight')
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax)
    
    return ax