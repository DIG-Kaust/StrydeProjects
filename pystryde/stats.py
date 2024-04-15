import numpy as np

from scipy.signal import filtfilt
from pylops.signalprocessing.sliding1d import sliding1d_design


def rms(x):
    """Root-mean square
    
    Compute running root-mean square of a time series ``x``
    
    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Data of size `nx`
        
    Returns
    -------
    x2ave : :obj:`float`
        Root-mean square
    
    """
    return np.sqrt(np.mean(x ** 2, axis=-1))


def sliding_summary(x, t, nwin, nover, fun):
    """Summary attribute 

    Compute a summary attribute (based on ``fun``) over possibly
    overlapping time windows

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Data of size `nx x nt`
    t : :obj:`numpy.ndarray`
        Time axis size `nt`
    nwin : :obj:`int`
        Window size used to compute the attibute
    nover : :obj:`int`
        Window overlap
    fun : :obj:`funct`
        Function handle computing an attribute over the last axis
    
    Returns
    -------
    tproc : :obj:`numpy.ndarray`
        Time axis of processed data
    
    """
    # sliding design
    nt = t.size
    wins_inends = sliding1d_design(nt, nwin, nover, nwin)[3]

    xproc = np.zeros(list(x.shape[:-1]) + [len(wins_inends[0]), ])
    for iwin, (win, wend) in enumerate(zip(wins_inends[0], wins_inends[1])):
        # extract data in window
        xwin = x[..., win:wend]

        # apply function
        xwinfun = fun(xwin)
        
        # recombine
        xproc[..., iwin] = xwinfun

    # truncate start and end
    tproc = t[(np.array(wins_inends[1]) + np.array(wins_inends[0])) // 2]
    
    return tproc, xproc, wins_inends