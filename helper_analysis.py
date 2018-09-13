# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 18:58:04 2017

@author: Razib
"""

import numpy as np
import matplotlib.pyplot as plt
from fast_histogram import histogram1d, histogram2d
from scipy.ndimage import center_of_mass
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.optimize as optimize
from scipy.interpolate import interp1d

def integrators(x, y, xlim):
    
    '''
    This definition return the index and the integrated y value setforth by
    xlim.
    Usage: index, ysum = integrators(xarray, yarray, xlim)
    
    * xarray and yarray must either be a list or a numpy array
    * xlim must be a list with LOWER RANGE and UPPER RANGE
    '''
    x = np.array(x)
    y = np.array(y)
    
    if x.shape != y.shape:
        print('x and y must be of same sized numpy array. No accumulation done!')
        return None, None
    
    if not isinstance(xlim, list):
        print('xlim must be a list' )
        return None, None

    else:
        idx = np.where((x > xlim[0]) & (x < xlim[1]))
        yIntegration = y[idx].sum()
        return idx, yIntegration
    
    
def pipico(xArray):
    xFirstSlice = []
    xSecondSlice = []
    start = 1
    for i in xArray:
        for j in xArray[start:]:
            xFirstSlice.append(i)
            xSecondSlice.append(j)
        start = start + 1 
    return np.array(xFirstSlice), np.array(xSecondSlice)


def tripico(array):
   list1 = []
   list2 = []
   list3 = []
   
      # c=1
   for i in range(0,len(array)-2):
       for j in range(i+1,len(array)-1):
           for k in range(i+2,len(array)):
               if k<=j:
                   continue
               list1.append(array[i])
               list2.append(array[j])
               list3.append(array[k])
     #  c=c+1
   x=np.asarray(list1)
   y=np.asarray(list2)
   z=np.asarray(list3)
   return x,y,z

def masked(arr, lessThan):
    arr = np.ma.masked_where(arr < lessThan, arr)
    return arr
    

def plotTushTush_old(arr2D):
    
    xAxis, yAxis, zAxis = arr2D.centerData
    zAxis += 1
    zAxis = np.transpose(np.log10(zAxis))
    zAxis_masked = masked(zAxis, np.log10(1.1))
    
    
    fig, ax = plt.subplots()
    
    cmaps = plt.cm.viridis
    cmaps.set_bad(color = 'white')
    ax.pcolormesh(xAxis, yAxis, zAxis_masked, cmap = cmaps)
    
    
def plotTushTush(xAxis, yAxis, zAxis):
        
    zAxis += 1
    zAxis = np.transpose(np.log10(zAxis))
    zAxis_masked = masked(zAxis, np.log10(1.1))
    
    
    fig, ax = plt.subplots()
    
    cmaps = plt.cm.jet
    cmaps.set_bad(color = 'white')
    img = ax.pcolormesh(xAxis, yAxis, zAxis_masked, cmap = cmaps)
    plt.colorbar(img)


def mqConversion(testmq, testTOF, actualTOF):
    #for 170 eV
    #tof = [505, 2235, 6825, 20415]
    #testmq = [1, 18, 165, 1469]
    #
    tof = np.array(testTOF)
    sqrtMQ = np.sqrt(testmq)
    k, t0 = np.polyfit(sqrtMQ, tof, 1)
    
    convMQ = ((np.array(actualTOF) - t0)/ k )**2
    return convMQ

def tofConversion(testmq, testTOF, mq):
    tof = np.array(testTOF)
    sqrtMQ = np.sqrt(testmq)
    k, t0 = np.polyfit(sqrtMQ, tof, 1)
    
    convTOF = k * np.sqrt(mq) + t0
    return convTOF


def plotproj(x, y, z, xlabel='xlabel', ylabel='ylabel', cmaps='jet', n=100):
    fig = plt.figure(n,figsize=(10, 10))

    grid = plt.GridSpec(8, 8, hspace=0., wspace=0.)
    main_ax = fig.add_subplot(grid[:-1, 1:])
    y_hist = fig.add_subplot(grid[:-1, 0], sharey=main_ax)
    x_hist = fig.add_subplot(grid[-1, 1:], sharex=main_ax)
    
    divider = make_axes_locatable(main_ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    
    # scatter points on the main axes
    im = main_ax.pcolormesh(x, y, z.T, norm=LogNorm(), cmap = cmaps)
    fig.colorbar(im, cax=cax, orientation='horizontal')
    #main_ax.set_xticklabels([])
    #main_ax.set_yticklabels([])
    
    sumy = z.sum(0)
    sumx = z.sum(1)
    
    x_hist.plot(x, sumx)
    x_hist.set_xlabel(xlabel)
     
    y_hist.plot(sumy, y)
    y_hist.set_xlim(sumy.max(), 0)
    y_hist.set_ylabel(ylabel)


class hist1d:
    def __init__(self, xmin, xmax, nbins = 10):
        
        self.nbins = nbins
        self.edges = np.linspace(xmin, xmax, nbins + 1)
        self.centers = (self.edges[:-1] + self.edges[1:])/2.0
        
        self.delta = 0.0
        self.range = (xmin, xmax + self.delta)
        self.hists = histogram1d([], nbins, self.range)
    
    def fill(self, arr):
        hists = histogram1d(arr, self.nbins, self.range)
        self.hists += hists
        
    @property
    def edgeData(self):
        return self.edges, np.int64(self.hists)
    
    @property
    def centerData(self):
        return self.centers, self.hists
    
    
class hist2d:
    def __init__(self, xmin, xmax, ymin, ymax, nxbins = 10, nybins = 10):
        self.nxbins = nxbins
        self.nybins = nybins
        
        self.xedges = np.linspace(xmin, xmax, nxbins + 1)
        self.xcenters = (self.xedges[:-1] + self.xedges[1:])/2.0
        
        self.yedges = np.linspace(ymin, ymax, nybins + 1)
        self.ycenters = (self.yedges[:-1] + self.yedges[1:])/2.0
        
        self.delta = 0.0
        self.xrange = (xmin, xmax + self.delta)
        self.yrange = (ymin, ymax + self.delta)
        self.hists = histogram2d([], [], [self.nxbins, self.nybins], [self.xrange, self.yrange])
        
        
    def fill(self, xarr, yarr):
        hists = histogram2d(xarr, yarr, [self.nxbins, self.nybins], [self.xrange, self.yrange] )
        self.hists += hists
        
    @property
    def centerData(self):
        return self.xcenters, self.ycenters, self.hists
        
        
        
def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = optimize.leastsq(errorfunction, params)
    return p

def center_by_moments(x_data, y_data, z_data, need_plot=True):
    """Returns xcenter and ycenter of the detector image"""
    data = z_data
    params = fitgaussian(data)
    fit = gaussian(*params)
    (height, x, y, width_x, width_y) = params
    
    if need_plot==True:
        plt.figure()
        
        plt.pcolormesh(data.T, norm=LogNorm(), cmap=plt.cm.gist_earth_r)    
        plt.contour(fit(*np.indices(data.shape)), cmap=plt.cm.copper)
        ax = plt.gca()
    
        
        plt.text(0.95, 0.05, """
        x : %.1f
        y : %.1f
        width_x : %.1f
        width_y : %.1f""" %(x, y, width_x, width_y),
                fontsize=16, horizontalalignment='right',
                verticalalignment='bottom', transform=ax.transAxes)
    
    indx, indy = data.shape
    x_index = np.linspace(0, indx-1, indx)
    y_index = np.linspace(0, indy-1, indy)

    fx = interp1d(x_index, x_data)
    fy = interp1d(y_index, y_data)

    center_x = fx(x)
    center_y = fy(y)
    
    return center_x.item(), center_y.item()