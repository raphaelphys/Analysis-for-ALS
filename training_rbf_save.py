# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 18:40:59 2018

@author: Razib Obaid
"""

from scipy.interpolate import Rbf
import numpy as np
from sklearn.externals import joblib
import warnings

def training_datasets(filename):
    simBucket = np.load(filename)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rbke = Rbf((simBucket[1:,0])**2, simBucket[1:,4], simBucket[1:,5], simBucket[1:,9], function='quintic')
        rbvx = Rbf(simBucket[1:,0],simBucket[1:,4], simBucket[1:,5], simBucket[1:,7], function='quintic')
        rbvy = Rbf(simBucket[1:,0],simBucket[1:,4], simBucket[1:,5], simBucket[1:,8], function='quintic')
        rbvz = Rbf(simBucket[1:,0]**2,simBucket[1:,4], simBucket[1:,5], simBucket[1:,6], function='quintic')
        
    foldername='C:/Users/Coincidence/Documents/Ho_analysis/Ho_analysis/simion_training/'
    
    joblib.dump(rbke, foldername+'ke'+filename+'.pkl')
    joblib.dump(rbvx, foldername+'vx'+filename+'.pkl')
    joblib.dump(rbvy, foldername+'vy'+filename+'.pkl')
    joblib.dump(rbvz, foldername+'vz'+filename+'.pkl')
    
# =============================================================================
#     Later on to read, you need to use:
#        from sklearn.externals import joblib
#     
#        rbKE = joblib.load('keHo_rbf_170eV.pkl')
# =============================================================================

    
if __name__ == '__main__':
    
    #training_datasets('simBucket_Nplus_170eV.npy')
    #training_datasets('simBucket_Ho_170eV.npy')
    training_datasets('simBucket_HoCN_170eV_191amu.npy')