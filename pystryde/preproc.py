import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, correlate, filtfilt
from pylops.signalprocessing import Shift
from pystryde.utils import mag2db, db2mag, customplot
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


def aligndata(data, irec, t, tlims, ishotmaster=0, otherdata=None, plotflag=False):
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
    ishotmaster : :obj:`int`, optional
        Index of shot to use as master for correlation
    otherdata : :obj:`numpy.ndarray`, optional
        Additional data of size `ns x nr x nt` to align based on the shifts
        found with data
    plotflag : :obj:`bool`, optional
        Plotting flag

    Returns
    -------
    datashift : :obj:`numpy.ndarray` or :obj:`tuple`
        Shifted data of size `ns x nr x nt` or tuple of shifted data of size `ns x nr x nt` (if ``otherdata`` is not None)
    figs : :obj:`tuple`, optional
        Figure handles (when ``plotflag=True``)

    """
    nshots_src, nr, nt = data.shape
    itzero = tlims[1]-tlims[0] - 1

    # compute correlations
    tcorr = np.hstack((-t[:tlims[1]-tlims[0]][::-1], t[:tlims[1]-tlims[0]][1:]))
    corr = np.vstack([correlate(data[ishotmaster, irec, tlims[0]:tlims[1]], data[i, irec, tlims[0]:tlims[1]], mode='full') for i in range(nshots_src)])
    print('Indices of max aligment:', np.argmax(corr, axis=1))

    Sop = Shift(dims=(nshots_src*nr, nt), shift=np.repeat(np.argmax(corr, axis=1)-itzero, nr), axis=-1, real=True)
    datashift = (Sop @ data.reshape(nshots_src * nr, nt)).reshape(nshots_src, nr, nt)

    corr = np.vstack([correlate(datashift[ishotmaster, irec, tlims[0]:tlims[1]], datashift[i, irec, tlims[0]:tlims[1]], mode='full') for i in range(nshots_src)])
    print('Indices of max aligment after shifts:', np.argmax(corr, axis=1))

    if otherdata is not None:
        otherdatashift = (Sop @ otherdata.reshape(nshots_src * nr, nt)).reshape(nshots_src, nr, nt)

        othercorr = np.vstack([correlate(otherdatashift[ishotmaster, irec, tlims[0]:tlims[1]], datashift[i, irec, tlims[0]:tlims[1]], mode='full') for i in range(nshots_src)])
        print('Indices of max aligment after shifts (for otherdata):', np.argmax(othercorr, axis=1))

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

    if otherdata is not None:
        datashift = (datashift, otherdatashift)

    return datashift, (fig, fig1, fig2, fig3, fig4, fig5, fig6) if plotflag else None


def correct_sensor_response(data, sensor_response, t_response,
                            nfft=None, fcorners=None, nsmooth=1, 
                            waterlevel=60, waterlevelkind='add',
                            plotflag=False, plottype=plt.plot):
    """Correct sensor response

    Apply correction to data based on sensor response. Optionally, one can choose to flatten
    the response in the entire bandwidth of the acquired signal, or in a selected bandwidth.

    The sensor correction process is carried out in the frequency domain in the following 3 steps:

    - A flat frequency spectrum is created based on ``fcorners`` amd ``nsmooth`` parameters, which
      will be used as target output for the sensor correction
    - A stable inverse response is created for sensor response using a user-defined ``waterlevel``
    - The product of the output frequency spectrum and inverse sensor respose are applied to the
      frequency-domain version of the input data

    Two options can be chosen for the inverse:

    - ``waterlevelkind=add``: the water level is added to the denominator of the inverse response
    - ``waterlevelkind=thresh``: the water level is used as threshold and set for any frequency whose response is
        below this value.

    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        Data of size `nr x nt`
    sensor_response : :obj:`numpy.ndarray`
        Sensor response of size `ntresp`
    t_response : :obj:`numpy.ndarray`
        Time axis of sensor response of size `ntresp`
    nfft : :obj:`int`, optional
        Number of samples in the Fourier domain
    fcorners : :obj:`tuple` or :obj:`list`, optional
        Corner frequencies (``flow, fmidlow, fmidhigh, fhigh``) used to create the ideal output response
        with 0 before ``flow`` and after ``fhigh``, 1 in between ``fmidlow, fmidhigh``, and linear ramps
        from ``flow`` to ``fmidlow`` and from ``fmidhigh`` to ``fhigh``
    nsmooth : :obj:`int`, optional
        Number of samples for smoothing of the ideal output response
    waterlevel : :obj:`float`, optional
        Water level in dB added to inverse of sensor response (note, if X is
        chosen this will be will be -XdB of peak amplitude)
    waterlevelkind :obj:`str`, optional
        Water level kind (``add`` or ``thresh``). See above for details.
    plotflag : :obj:`bool`, optional
        Plotting flag
    plottype : :obj:`matplotlib.pyplot`, optional
        Plotting method used for ploting

    Returns
    -------
    datafilt : :obj:`numpy.ndarray`
        Filtered data of size `nr x nt`
    OUTeffective : :obj:`numpy.ndarray`
        Effective output response (response divided by its shaped inverse)
    f : :obj:`numpy.ndarray`
        Frequency axis

    """
    nr, nt = data.shape
    dt = t_response[1] - t_response[0]
    t = np.arange(nt) * dt

    if nfft is None:
        nfft = nt

    # find center of sensor response from time axis
    itcenter = np.argmin(np.abs(t_response))

    # compute and remove mean from data
    data_mean = np.mean(data)
    data -= data_mean

    # convert response and data to frequency domain
    f = np.fft.rfftfreq(nfft, dt)
    df = f[1]

    Response = np.fft.rfft(sensor_response, n=nfft)
    D = np.fft.rfft(data, n=nfft, axis=-1)

    # design output frequency spectrum
    OUT = np.ones_like(f)
    if fcorners is not None:
        ifcorners = [None if fcorner is None else int(np.round(fcorner / df)) for fcorner in fcorners]
        if ifcorners[0] is not None:
            OUT[:ifcorners[0]] = 0.
            OUT[ifcorners[0]:ifcorners[1]] = np.arange(ifcorners[1] - ifcorners[0]) / (ifcorners[1] - ifcorners[0])
        if ifcorners[3] is not None:
            OUT[ifcorners[2]:ifcorners[3]] = np.arange(ifcorners[3] - ifcorners[2], 0, -1) / (
                        ifcorners[3] - ifcorners[2])
            OUT[ifcorners[3]:] = 0.
        if nsmooth > 1:
            OUT = filtfilt(np.ones(nsmooth) / nsmooth, 1, OUT)

    # define water level
    waterlevel = db2mag(mag2db(np.abs(Response).max()) - waterlevel)

    # compute inverse response
    if waterlevelkind == 'add':
        Response_inv = np.conj(Response) / (np.conj(Response) * Response + waterlevel)
    elif waterlevelkind == 'thresh':
        Response_inv = 1. / Response
        Response_mask = np.abs(Response) < waterlevel
        Response_inv[Response_mask] = waterlevel / np.exp(1j*np.angle(Response)[Response_mask])
    else: 
        raise NotImplementedError('waterlevelkind must be add or threshold...')

    Response_inv_shaped = Response_inv * OUT

    # compute effective output response
    OUTeffective = np.abs(Response * Response_inv_shaped)

    # apply inverse response to data
    Sop = Shift((nr, nfft), itcenter * dt, axis=-1, sampling=dt, real=True)
    Dfilt = (Response_inv_shaped * D)
    datafilt = (Sop @ np.real(np.fft.irfft(Dfilt, axis=-1)))[..., :nt]

    # add mean back
    datafilt += data_mean

    if plotflag:

        fig, axs = plt.subplots(2, 2, figsize=(18, 8))
        customplot(axs[0, 0], f, np.abs(D[nr // 2]), 'k', plottype)
        axs[0, 0].set_ylabel('Raw data spectrum')
        ax2 = axs[0, 0].twinx()
        ax2.plot(f, OUT, 'r')
        ax2.set_ylabel('Desired sensor response', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        customplot(axs[0, 1], f, OUT * np.abs(D[nr // 2]), 'k', plottype)
        axs[0, 1].set_ylabel('Filtered data spectrum')
        ax2 = axs[0, 1].twinx()
        customplot(ax2, f, np.abs(Response), 'r', plottype)
        ax2.set_ylabel('Sensor Response', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        customplot(axs[1, 0], f, np.abs(Dfilt[nr // 2]), 'k', plottype)
        axs[1, 0].set_ylabel('Sensor corrected data spectrum')
        ax2 = axs[1, 0].twinx()
        customplot(ax2, f, np.abs(Response_inv_shaped), 'r', plottype)
        ax2.set_ylabel('Inverse Sensor Response', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        axs[1, 1].plot(t, data[nr // 2], 'k', lw=2)
        axs[1, 1].plot(t, datafilt[nr // 2], 'r', lw=1)
        axs[1, 1].set_ylabel('Time domain data')
        plt.tight_layout()

    return datafilt, OUTeffective, f
