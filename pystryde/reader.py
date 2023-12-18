import os
import glob
import datetime
import numpy as np
import segyio
import struct
import matplotlib.pyplot as plt

from astropy.time import Time
from pystryde.preproc import filterdata


# Format converters... thanks STRYDE for not following SEG-Y header formatting :) 
def float_to_bin(num):
    return format(struct.unpack('!I', struct.pack('!f', num))[0], '032b')

def int_to_bin(num):
    return format(struct.unpack('!I', struct.pack('!I', num))[0], '032b')

def short_to_bin(num):
    return format(struct.unpack('!H', struct.pack('!h', num))[0], '016b')

def bin_to_float(binary):
    return struct.unpack('!f',struct.pack('!I', int(binary, 2)))[0]

def bin_to_double(binary):
    return struct.unpack('!d',struct.pack('!q', int(binary, 2)))[0]

def shortbin_to_float(binary):
    return struct.unpack('!f',struct.pack('!b', binary))[0]

def gps_to_utc(gpstime):
    gpstime = gpstime - 37 # seems like I need to remove 37 seconds
    t = Time(gpstime, format='gps')
    t = Time(t, format='iso')
    return t.value

def bin_to_utc(head):
    h1 = head[169]
    h2 = head[171]
    h3 = head[173]
    h4 = head[175]
    gpstime = str(int(bin_to_double(short_to_bin(h1) + short_to_bin(h2) + short_to_bin(h3) + short_to_bin(h4))))
    gpstime = float(gpstime)
    utctime = datetime.datetime.strptime(gps_to_utc(gpstime),'%Y-%m-%d %H:%M:%S.000')
    return utctime


class strydecont():
    """Continuous recording

    Container for continuous recording of a single Stryde node
    
    Parameters
    ----------
    file : :obj:`str`
        Filename

    """
    def __init__(self, file):
        self.file = file

    def interpret(self):
        """Interpret segy file
        """
        with segyio.open(self.file, ignore_geometry=True) as f:
            self.recidx = f.attributes(185)[:]
            self.recpoint = np.array([int(bin_to_float(bin(recpoint))) for recpoint in f.attributes(181)[:]])
            self.recline = np.array([bin_to_float(short_to_bin(recline1) + short_to_bin(recline2)) 
                                     for recline1, recline2 in zip(f.attributes(177)[:], f.attributes(179)[:])])
            self.utctime = [bin_to_utc(f.header[i]) for i in range(f.tracecount)]
            
            self.recx = f.attributes(segyio.TraceField.GroupX)[:]
            self.recy = f.attributes(segyio.TraceField.GroupY)[:]
            self.recz = f.attributes(segyio.TraceField.ReceiverGroupElevation)[:]

            # scaling factor
            self.sc = f.header[0][segyio.TraceField.SourceGroupScalar]
            if (self.sc < 0):
                self.sc = 1. / abs(self.sc)

            self.ntraces = f.tracecount
            self.t = f.samples * 1e-3
            self.dt = self.t[1] - self.t[0]
            
    def getrecord(self):
        """Read data
        """ 
        with segyio.open(self.file, ignore_geometry=True) as f:
            self.data = segyio.collect(f.trace[:])

    def plotrecord(self, jt=1, clip=None, cmap='gray', title=None, figsize=(10, 10)):
        """Display data recording

        Parameters
        ----------
        jt : :obj:`int`
            Jump in time axis
        clip : :obj:`float`, optional
            Max value to use in colorscale
        cmap : :obj:`str`, optional
            Colormap
        title : :obj:`str`, optional
            Figure title (if `None`, use filename)
        figsize : :obj:`tuple`, optional
            Figure size

        Returns
        -------
        fig : :obj:`plt.figure`
            Figure handle
        ax : :obj:`plt.axes`, optional
            Axes handle

        """
        clip = np.max(np.abs(self.data)) if clip is None else clip
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(self.data[:, ::jt].T, cmap=cmap, vmin=-clip, vmax=clip,
                  extent=(0, self.ntraces, self.t[-1], self.t[0]))
        ax.axis('tight')
        ax.set_xlabel('60sec sequence number')
        ax.set_ylabel('T [s]')
        ax.set_title(self.file if title is None else title)
        return fig, ax


class strydeconts():
    """Continuous recording for entire survey

    Container for continuous recording of a collection of Stryde nodes
    
    Parameters
    ----------
    directory : :obj:`str`
        Directory where files are stored
    filereg : :obj:`str`, optional
        Regexp to select files in directory

    """
    def __init__(self, directory, filereg='CR_*'):
        self.directory = directory
        self.files = glob.glob(os.path.join(directory, filereg))
        self.nfiles = len(self.files)

    def interpret(self):
        """Interpret segy files

        Interpret collection of files and extract common information (e.g., t)
        and file-wise information (e.g., receiver coordinates)

        """
        # Define common parameters
        with segyio.open(self.files[0], ignore_geometry=True) as f:

            # scaling factor
            self.sc = f.header[0][segyio.TraceField.SourceGroupScalar]
            if (self.sc < 0):
                self.sc = 1. / abs(self.sc)

            self.t = f.samples * 1e-3
            self.dt = self.t[1] - self.t[0]
            self.nt = self.t.size
            
        # Define receiver grid
        self.recx = np.zeros(self.nfiles)
        self.recy = np.zeros(self.nfiles)
        self.ntraces = np.zeros(self.nfiles)
        
        self.recidx = np.zeros(self.nfiles)
        self.recline = np.zeros(self.nfiles)
        self.recpoint = np.zeros(self.nfiles)
        self.utctimes = []
        for ifile, file in enumerate(self.files):
            with segyio.open(file, ignore_geometry=True) as f:
                self.recidx[ifile] = f.header[0][185]
                self.recline[ifile] = bin_to_float(short_to_bin(f.header[0][177]) + short_to_bin(f.header[0][179])) 
                self.recpoint[ifile] = int(bin_to_float(bin(f.header[0][181])))
                
                self.recx[ifile] = f.header[0][segyio.TraceField.GroupX]
                self.recy[ifile] = f.header[0][segyio.TraceField.GroupY]

                self.ntraces[ifile] = f.tracecount
                self.utctimes.append([bin_to_utc(f.header[i]) for i in range(f.tracecount)])
            
        # Indentify unique srcidx, srcline, srcpoint
        self.recidx_axis = np.sort(np.unique(self.recidx))
        self.recline_axis = np.sort(np.unique(self.recline))
        self.recpoint_axis = np.sort(np.unique(self.recpoint))

    def getrecords(self, utctime_start=None, utctime_end=None):
        """Read data
        
        Reas portion of data from files, for all receivers within a given time window

        Parameters
        ----------
        utctime_start : :obj:`datatime.date`, optional
            Start time in UTC time zone
        utctime_end : :obj:`datatime.date`, optional
            End time in UTC time zone

        """ 
        if utctime_start is None:
            # find maximum utctime start time from records and use it as origin
            utctime_start = max([self.utctimes[i][0] for i in range(self.nfiles)])
        if utctime_end is None:
            # find minimum utctime end time from records and use it as end time
            utctime_end = min([self.utctimes[i][-1] for i in range(self.nfiles)])
        
        # identify lenght of time axis
        it_starts = np.array([np.where([utctime_start == time for time in self.utctimes[i]])[0][0] for i in range(self.nfiles)])
        it_ends = np.array([np.where([utctime_end == time for time in self.utctimes[i]])[0][0] for i in range(self.nfiles)])
        nsections = min(it_ends - it_starts)

        # find common utctime axis
        nsections_stations = np.where((it_ends - it_starts) == nsections)[0]
        if len(nsections_stations) == 0:
            raise NotImplementedError('Select more restringent utctime_start and utctime_end times such that at least one station ' +
                                      'covers the entire duration')
        else:
            self.utctime = self.utctimes[nsections_stations[0]][it_starts[nsections_stations[0]]: it_ends[nsections_stations[0]]]

        # extract data
        self.data = np.zeros((len(self.recidx_axis), len(self.recline_axis), len(self.recpoint_axis), nsections, self.nt))
        self.nrecs = len(self.recidx_axis) * len(self.recline_axis) * len(self.recpoint_axis)
        for ifile, file in enumerate(self.files):
            with segyio.open(file, ignore_geometry=True) as f:
                self.data[np.where(self.recidx[ifile] == self.recidx_axis)[0][0],
                          np.where(self.recline[ifile] == self.recline_axis)[0][0],
                          np.where(self.recpoint[ifile] == self.recpoint_axis)[0][0]] = segyio.collect(f.trace[:])[it_starts[ifile]:it_ends[ifile]]

    def extract(self, utctime_start=None, utctime_end=None, nsamples=None):
        """Extract data
        
        Extract portion of the locally stored data (already read from file by the `getrecords` method)

        Parameters
        ----------
        utctime_start : :obj:`datatime.date`, optional
            Start time in UTC time zone
        utctime_end : :obj:`datatime.date`, optional
            End time in UTC time zone
        nsamples : :obj:`int`, optional
            Number of samples (will overwrite `utctime_end` and allow selecting time 
            windows which are smaller than multiples of 60mins)

        """ 
        if utctime_start is None:
            iutc_start = 0
        else:
            iutc_start = np.where([utctime_start == time for time in self.utctime])[0][0]
        if utctime_end is None:
            iutc_end = -1
        else:
            iutc_end = np.where([utctime_end == time for time in self.utctime])[0][0]
        # extract data
        data = self.data[:, :, :, iutc_start:iutc_end, :].reshape(self.nrecs, (iutc_end-iutc_start) *  self.nt).T
        if not nsamples:
            tlims = [self.utctime[iutc_start], self.utctime[iutc_end]]
        else:
            data = data[:nsamples]
            tlims = [self.utctime[iutc_start], self.utctime[iutc_start] + datetime.timedelta(seconds=int(self.dt * nsamples))]
        return data, tlims

    def plotgeom(self, coords=True, local=False, figsize=(10, 10)):
        """Display geometry
        
        Quick display of geometry of acquisition in global or local coordinates

        Parameters
        ----------
        coords : :obj:`bool`, optional
            Display line-point (`False`) or physical coordinates (`True`)
        local : :obj:`bool`, optional
            Use local (`True`) or global (`False`) coordinates
        figsize : :obj:`tuple`, optional
            Figure size

        Returns
        -------
        fig : :obj:`plt.figure`
            Figure handle
        ax : :obj:`plt.axes`, optional
            Axes handle

        """ 
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if not coords:
            ax.scatter(self.recline, self.recpoint, c='b', s=6)
            ax.set_xlabel('Src/Rec Line')
            ax.set_ylabel('Src/Rec Point')
        else:
            if not local:
                ax.scatter(self.recx * self.sc, self.recy * self.sc, c='b', s=6)
            else:
                ox = np.min(self.recx)
                oy = np.min(self.recy)
                ax.scatter((self.recx - ox) * self.sc, (self.recy - oy) * self.sc, c='b', s=6)
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Z [m]')
        return fig, ax

    def plotrecord(self, utctime_start=None, utctime_end=None, nsamples=None, jt=1, 
                   clip=None, cmap='gray', title=None, figsize=(10, 10)):
        """Display recording
        
        Quick display of recording

        Parameters
        ----------
        utctime_start : :obj:`datatime.date`, optional
            Start time in UTC time zone
        utctime_end : :obj:`datatime.date`, optional
            End time in UTC time zone
        nsamples : :obj:`int`, optional
            Number of samples (will overwrite `utctime_end` and allow selecting time 
            windows which are smaller than multiples of 60mins)
        jt : :obj:`int`
            Jump in time axis
        clip : :obj:`float`, optional
            Max value to use in colorscale
        cmap : :obj:`str`, optional
            Colormap
        title : :obj:`str`, optional
            Figure title (if `None`, use filename)
        figsize : :obj:`tuple`, optional
            Figure size

        Returns
        -------
        fig : :obj:`plt.figure`
            Figure handle
        ax : :obj:`plt.axes`, optional
            Axes handle

        """ 
        data, tlims = self.extract(utctime_start, utctime_end, nsamples)
        clip = np.max(np.abs(self.data)) if clip is None else clip
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(data[::jt], cmap=cmap, vmin=-clip, vmax=clip, extent=(0, self.nrecs, tlims[-1], tlims[0]))
        ax.axis('tight')
        ax.set_xlabel('Receivers')
        ax.set_ylabel('T [s]')
        ax.set_title(self.directory if title is None else title)
        return fig, ax


class strydeshot():
    """Shot recording

    Container for shot-gather recording
    
    Parameters
    ----------
    file : :obj:`str`
        Filename

    """
    def __init__(self, file):
        self.file = file
    
    def interpret(self):
        """Interpret segy file
        """
        with segyio.open(self.file, ignore_geometry=True) as f:
            self.srcidx = f.attributes(197)[:]
            self.srcline = np.array([int(bin_to_float(bin(srcline))) for srcline in f.attributes(189)[:]])
            self.srcpoint = np.array([int(bin_to_float(bin(srcpoint))) for srcpoint in f.attributes(193)[:]])
            self.recidx = f.attributes(185)[:]
            self.recpoint = np.array([int(bin_to_float(bin(recpoint))) for recpoint in f.attributes(181)[:]])
            self.recline = np.array([bin_to_float(short_to_bin(recline1) + short_to_bin(recline2)) 
                                     for recline1, recline2 in zip(f.attributes(177)[:], f.attributes(179)[:])])

            self.srcx = f.attributes(segyio.TraceField.SourceX)[:]
            self.srcy = f.attributes(segyio.TraceField.SourceY)[:]
            self.recx = f.attributes(segyio.TraceField.GroupX)[:]
            self.recy = f.attributes(segyio.TraceField.GroupY)[:]
            self.recz = f.attributes(segyio.TraceField.ReceiverGroupElevation)[:]

            # scaling factor
            self.sc = f.header[0][segyio.TraceField.SourceGroupScalar]
            if (self.sc < 0):
                self.sc = 1. / abs(self.sc)

            self.t = f.samples * 1e-3
            self.dt = self.t[1] - self.t[0]
            
    def getshot(self):
        """Read data
        """ 
        with segyio.open(self.file, ignore_geometry=True) as f:
            self.data = segyio.collect(f.trace[:])

    def filter(self, nfilt, fmin, fmax, dt, plotflag=False, itmax_plot=-1, 
               figsize=(10, 10), title=None, **kwargs_plot):
        """Filter data

        Apply band-pass filter to data

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
        plotflag : :obj:`bool`, optional
            Plotting flag
        itmax_plot : :obj:`int`, optional
            Index of maximum sample to plot
    
        Returns
        -------
        datafilt : :obj:`numpy.ndarray`
            Filtered data of size `nx x nt`

        """ 
        datafilt = filterdata(nfilt, fmin, fmax, dt, self.data)[-1]
        if plotflag:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.imshow(datafilt[:, :itmax_plot].T, **kwargs_plot)
            ax.axis('tight')
            ax.set_xlabel('Rec point')
            ax.set_ylabel('T [s]')
            ax.set_title(self.file if title is None else title)
        return datafilt

    def plotgeom(self, coords=True, local=False, figsize=(10, 10)):
        """Display geometry
        
        Quick display of geometry of acquisition in global or local coordinates

        Parameters
        ----------
        coords : :obj:`bool`, optional
            Display line-point (`False`) or physical coordinates (`True`)
        local : :obj:`bool`, optional
            Use local (`True`) or global (`False`) coordinates
        figsize : :obj:`tuple`, optional
            Figure size

        Returns
        -------
        fig : :obj:`plt.figure`
            Figure handle
        ax : :obj:`plt.axes`, optional
            Axes handle

        """ 
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if not coords:
            ax.scatter(self.recline, self.recpoint, c='b', s=6)
            ax.scatter(self.srcline, self.srcpoint, c='r', s=6)
            ax.set_xlabel('Src/Rec Line')
            ax.set_ylabel('Src/Rec Point')
        else:
            if not local:
                ax.scatter(self.recx * self.sc, self.recy * self.sc, c='b', s=6)
                ax.scatter(self.srcx * self.sc, self.srcy * self.sc, c='r', s=9)
            else:
                ox = min(np.min(self.srcx), np.min(self.recx)) 
                oy = min(np.min(self.srcy), np.min(self.recy))
                ax.scatter((self.recx - ox) * self.sc, (self.recy - oy) * self.sc, c='b', s=6)
                ax.scatter((self.srcx - ox) * self.sc, (self.srcy - oy) * self.sc, c='r', s=9)
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Z [m]')
        return fig, ax
        
    def plotshot(self, clip=None, itmax=None, cmap='gray', title=None, figsize=(10, 10)):
        """Display shot gather
        
        Quick display of shot gather

        Parameters
        ----------
        clip : :obj:`float`, optional
            Max value to use in colorscale
        itmax : :obj:`int`, optional
            Index of las time sample to display
        cmap : :obj:`str`, optional
            Colormap
        title : :obj:`str`, optional
            Figure title (if `None`, use filename)
        figsize : :obj:`tuple`, optional
            Figure size

        Returns
        -------
        fig : :obj:`plt.figure`
            Figure handle
        ax : :obj:`plt.axes`, optional
            Axes handle

        """ 
        clip = np.max(np.abs(self.data)) if clip is None else clip
        itmax = self.data.shape[1] if itmax is None else itmax
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(self.data[:, :itmax].T, cmap=cmap, vmin=-clip, vmax=clip, extent=(self.recpoint[0], self.recpoint[-1], self.t[itmax], self.t[0]))
        ax.axis('tight')
        ax.set_xlabel('Rec point')
        ax.set_ylabel('T [s]')
        ax.set_title(self.file if title is None else title)
        return fig, ax


class strydeshots(strydeshot):
    """Shot recordings

    Container for shot-gather recordings from all sources in a survey
    
    Parameters
    ----------
    directory : :obj:`str`
        Directory where files are stored
    filereg : :obj:`str`, optional
        Regexp to select files in directory
    """
    def __init__(self, directory, filereg='S_*'):
        self.directory = directory
        self.files = glob.glob(os.path.join(directory, filereg))
        self.nfiles = len(self.files)
        
    def interpret(self):
        """Interpret segy files

        Interpret collection of files and extract common information (e.g., t, recx, recy, 
        recpoint, recline) and file-wise information (e.g., source coordinates)

        """
        # Define receiver grid (assumed to be fixed throughout the survey)
        with segyio.open(self.files[0], ignore_geometry=True) as f:
            self.recidx = f.attributes(185)[:]
            self.recpoint = np.array([int(bin_to_float(bin(recpoint))) for recpoint in f.attributes(181)[:]])
            self.recline = np.array([bin_to_float(short_to_bin(recline1) + short_to_bin(recline2)) 
                                     for recline1, recline2 in zip(f.attributes(177)[:], f.attributes(179)[:])])

            self.recx = f.attributes(segyio.TraceField.GroupX)[:]
            self.recy = f.attributes(segyio.TraceField.GroupY)[:]
            self.recz = f.attributes(segyio.TraceField.ReceiverGroupElevation)[:]

            # scaling factor
            self.sc = f.header[0][segyio.TraceField.SourceGroupScalar]
            if (self.sc < 0):
                self.sc = 1. / abs(self.sc)

            self.t = f.samples * 1e-3
            self.dt = self.t[1] - self.t[0]
            self.nt = self.t.size
            self.nr = f.tracecount
            
        # Define source grid
        self.srcx = np.zeros(self.nfiles)
        self.srcy = np.zeros(self.nfiles)

        self.srcidx = np.zeros(self.nfiles)
        self.srcline = np.zeros(self.nfiles)
        self.srcpoint = np.zeros(self.nfiles)

        for ifile, file in enumerate(self.files):
            with segyio.open(file, ignore_geometry=True) as f:
                self.srcidx[ifile] = f.header[0][197]
                self.srcline[ifile] = int(bin_to_float(bin(f.header[0][189])))
                self.srcpoint[ifile] = int(bin_to_float(bin(f.header[0][193])))
                
                self.srcx[ifile] = f.header[0][segyio.TraceField.SourceX]
                self.srcy[ifile] = f.header[0][segyio.TraceField.SourceY]

        # Indentify unique srcidx, srcline, srcpoint
        self.srcidx_axis = np.sort(np.unique(self.srcidx))
        self.srcline_axis = np.sort(np.unique(self.srcline))
        self.srcpoint_axis = np.sort(np.unique(self.srcpoint))
    
    def getshot(self):
        """Read data
        
        Read entire survey ordered in shot-gather mode

        """ 
        self.data = np.zeros((len(self.srcidx_axis), len(self.srcline_axis), len(self.srcpoint_axis), self.nr, self.nt))
        for ifile, file in enumerate(self.files):
            with segyio.open(file, ignore_geometry=True) as f:
                self.data[np.where(self.srcidx[ifile] == self.srcidx_axis)[0][0],
                          np.where(self.srcline[ifile] == self.srcline_axis)[0][0],
                          np.where(self.srcpoint[ifile] == self.srcpoint_axis)[0][0]] = segyio.collect(f.trace[:])
    