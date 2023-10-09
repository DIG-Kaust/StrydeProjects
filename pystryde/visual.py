import numpy as np


def _wiggletrace(ax, tz, trace, center=0, cpos='r', cneg='b'):
    """Plot a seismic wiggle trace onto an axis

    Parameters
    ----------
    ax : :obj:`plt.axes`, optional
         Axes handle
    tz : :obj:`np.ndarray `
        Depth (or time) axis
    trace : :obj:`np.ndarray `
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
    """Plot a comb of seismic wiggle traces onto an axis

    Parameters
    ----------
    ax : :obj:`plt.axes`, optional
         Axes handle
    tz : :obj:`np.ndarray `
        Depth (or time) axis
    x : :obj:`np.ndarray `
        Lateral axis
    traces : :obj:`np.ndarray `
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