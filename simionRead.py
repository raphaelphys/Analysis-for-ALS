# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 15:28:50 2018

@author: Razib
"""

import numpy as np
from scipy.interpolate import Rbf


def simionReader(filename):

    f = open(filename, 'r')
    
    ignoredLine = f.readline()
    
    evtMarker = 0
    
    simBucket1 = np.zeros(10)
    
    for line in f:
        dat = line.strip().split(',') #strips the \n character and then splits
        evtMarker += 1
        if evtMarker%2 != 0:
            velIon = dat[-4:]
        if evtMarker%2 == 0:
            tof = dat[2]
            posIonAndMQ = dat[3:8]
            combinedList = [tof] + posIonAndMQ + velIon
            simBucket1 = np.vstack((simBucket1, np.float32(combinedList)))
    
    f.close()
    simFilename = 'SIM' + filename.split('.csv')[0] + '.npy'
    np.save(simFilename, simBucket1)
    
    

if __name__ == "__main__":
    
    filename = 'N2_plus_170eV_0p1.csv'
    simionReader(filename)
    