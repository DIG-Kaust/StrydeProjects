import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, correlate
from pylops.signalprocessing import Shift
from pystryde.visual import wiggletracecomb


def filterdata(nfilt, fmin, fmax, dt, inp):
    """Filter data
    
    Apply Butterworth  band-pass filter to data
    
    Parameters
    ----------
    nfilt : :obj:`int`
        Size of filter
    fmin : :obj:`float`
        Minimum frequency
    fmax : :obj:`float`
        Maximum frequency
    dt : :obj:`float`
        Time sampling
    inp : :obj:`numpy.ndarray`
        Data of size `nx x nt`
        
    Returns
    -------
    b : :obj:`numpy.ndarray`
        Filter numerator coefficients
    b : :obj:`numpy.ndarray`
        Filter denominator coefficients
    sos : :obj:`numpy.ndarray`
        Filter sos 
    filtered : :obj:`numpy.ndarray`
        Filtered data of size `nx x nt`
    
    """
    if fmin is None:
        b, a = butter(nfilt, fmax, 'low', analog=True)
        sos = butter(nfilt, fmax, 'low', fs=1/dt, output='sos')
    else:
        b, a = butter(nfilt, [fmin, fmax], 'bandpass', analog=True)
        sos = butter(nfilt, [fmin, fmax], 'bandpass', fs=1/dt, output='sos')
    filtered = sosfiltfilt(sos, inp, axis=-1)
    return b, a, sos, filtered


def averagespectrum(data, dt):
    """Average spectrum
    
    Compute average spectrum over traces
    
    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        Data of size `nt x nx`
    dt : :obj:`float`
        Time sampling
        
    Returns
    -------
    Dave : :obj:`numpy.ndarray`
        Average spectrum of size `nt`
    f : :obj:`numpy.ndarray`
        Frequency axis
    
    """
    nfft = data.shape[0]
    DATA = np.fft.rfft(data, axis=0, n=nfft)
    f = np.fft.rfftfreq(nfft, dt)
    Dave = np.mean(np.abs(DATA), axis=1)
    return Dave, f


def parkdispersion(data, dx, dt, cmin, cmax, dc, fmax):
    """Dispersion panel
    
    Calculate dispersion curves using the method of
    Park et al. 1998
    
    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        Data of size `nx x nt`
    dx : :obj:`float`
        Spatial sampling
    dt : :obj:`float`
        Time sampling
    cmin : :obj:`float`
        Minimum velocity
    cmax : :obj:`float`
        Maximum velocity
    dc : :obj:`float`
        Velocity sampling
    fmax : :obj:`float`
        Maximum frequency
        
    Returns
    -------
    f : :obj:`numpy.ndarray`
        Frequency axis
    c : :obj:`numpy.ndarray`
        Velocity axis`
    disp : :obj:`numpy.ndarray`
        Dispersion panel of size `nc x nf`

    """
    nr, nt = data.shape
    
    # Axes
    t = np.linspace(0.0, nt*dt, nt)

    f = sp.fftpack.fftfreq(nt, dt)[:nt//2]
    df = f[1] - f[0]
    fmax_idx = int(fmax//df)

    c = np.arange(cmin, cmax, dc)  # set phase velocity range
    x = np.linspace(0.0, (nr-1)*dx, nr)

    # Fx spectrum
    U = sp.fftpack.fft(data)[:, :nt//2]
    
    # Dispersion panel
    disp = np.zeros((len(c), fmax_idx))
    for fi in range(fmax_idx):
        for ci in range(len(c)):
            k = 2.0*np.pi*f[fi]/(c[ci])
            disp[ci, fi] = np.abs(
                np.dot(dx * np.exp(1.0j*k*x), U[:, fi]/np.abs(U[:, fi])))

    return f[:fmax_idx], c, disp


def aligndata(data, irec, t, tlims, plotflag=False):
    """Align data by cross-correlation

    Align shot gathers generated multiple times at the same source
    by finding the best shift by cross-correlation that aligns traces
    at a given receiver location
    
    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        Data of size `ns x nr x nt`
    irec : :obj:`int`
        Index of receiver to select traces for alignment
    t : :obj:`numpy.ndarray`
        Time axis
    tlims : :obj:`tuple`
        Indices of first and last time samples used to extract 
        portion of data for cross-correlation
    plotflag : :obj:`bool`, optional
        Plotting flag

    Returns
    -------
    datashift : :obj:`numpy.ndarray`
        Shifted data of size `ns x nr x nt`
    figs : :obj:`tuple`, optional
        Figure handles (when ``plotflag=True``)

    """
    nshots_src, nr, nt = data.shape
    itzero = tlims[1]-tlims[0] - 1

    # compute correlations
    tcorr = np.hstack((-t[:tlims[1]-tlims[0]][::-1], t[:tlims[1]-tlims[0]][1:]))
    corr = np.vstack([correlate(data[0, irec, tlims[0]:tlims[1]], data[i, irec, tlims[0]:tlims[1]], mode='full') for i in range(nshots_src)])
    print('Indices of max aligment:', np.argmax(corr, axis=1))

    Sop = Shift(dims=(nshots_src*nr, nt), shift=np.repeat(np.argmax(corr, axis=1)-itzero, nr), axis=-1, real=True)
    datashift = (Sop @ data.reshape(nshots_src * nr, nt)).reshape(nshots_src, nr, nt)

    corr = np.vstack([correlate(datashift[0, irec, tlims[0]:tlims[1]], datashift[i, irec, tlims[0]:tlims[1]], mode='full') for i in range(nshots_src)])
    print('Indices of max aligment after shifts:', np.argmax(corr, axis=1))

    if plotflag:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        wiggletracecomb(ax, t[tlims[0]:tlims[1]], np.arange(nshots_src), data[:, irec, tlims[0]:tlims[1]], scaling=0.6, cpos='r', cneg='b')
        ax.set_title(f'Original data (at rec={irec})')

        fig1, ax = plt.subplots(1, 1, figsize=(12, 3))
        for i in range(nshots_src):
            ax.plot(t[tlims[0]:tlims[1]], data[i, irec, tlims[0]:tlims[1]], 'k', lw=0.5)
        ax.set_title(f'Original data (at rec={irec})')

        fig2, ax = plt.subplots(1, 1, figsize=(12, 3))
        ax.plot(t[tlims[0]:tlims[1]], data[0, irec, tlims[0]:tlims[1]], 'r', label='First trace')
        ax.plot(t[tlims[0]:tlims[1]], np.mean(data[:, irec, tlims[0]:tlims[1]], axis=0), 'k', label='Stack')
        ax.set_title(f'Original stacked data (at rec={irec})')
        ax.legend()

        fig3, ax = plt.subplots(1, 1, figsize=(12, 5))
        wiggletracecomb(ax, tcorr, np.arange(nshots_src), corr, scaling=0.6, cpos='r', cneg='b')
        ax.set_title(f'Trace Correlations')

        fig4, ax = plt.subplots(1, 1, figsize=(12, 5))
        wiggletracecomb(ax, t[tlims[0]:tlims[1]], np.arange(nshots_src), datashift[:, irec, tlims[0]:tlims[1]], scaling=0.6, cpos='r', cneg='b')
        ax.set_title(f'Shifted data (at rec={irec})')

        fig5, ax = plt.subplots(1, 1, figsize=(12, 3))
        for i in range(nshots_src):
            ax.plot(t[tlims[0]:tlims[1]], datashift[i, irec, tlims[0]:tlims[1]], 'k', lw=0.5)
        ax.set_title(f'Shifted data (at rec={irec})')

        fig6, ax = plt.subplots(1, 1, figsize=(12, 3))
        ax.plot(t[tlims[0]:tlims[1]], datashift[0, irec, tlims[0]:tlims[1]], 'r', label='First trace')
        ax.plot(t[tlims[0]:tlims[1]], np.mean(datashift[:, irec, tlims[0]:tlims[1]], axis=0), 'k', label='Stack')
        ax.set_title(f'Shifted stacked data (at rec={irec})')
        ax.legend()

    return datashift, (fig, fig1, fig2, fig3, fig4, fig5, fig6) if plotflag else None


