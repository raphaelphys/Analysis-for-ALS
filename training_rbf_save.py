# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 18:40:59 2018

@author: Razib Obaid
"""

from scipy.interpolate import Rbf
import numpy as np
from sklearn.externals import joblib

def training_datasets(filename='simBucket_Ho_170eV.npy'):
    simBucket = np.load(filename)
        
    rbke = Rbf((simBucket[1:,0])**2, simBucket[1:,4], simBucket[1:,5], simBucket[1:,9], function='cubic')
    rbvx = Rbf(simBucket[1:,0],simBucket[1:,4], simBucket[1:,5], simBucket[1:,7], function='cubic')
    rbvy = Rbf(simBucket[1:,0],simBucket[1:,4], simBucket[1:,5], simBucket[1:,8], function='cubic')
    rbvz = Rbf(simBucket[1:,0]**2,simBucket[1:,4], simBucket[1:,5], simBucket[1:,6], function='cubic')
    
    
    joblib.dump(rbke, 'keHo_rbf_170eV.pkl')
    joblib.dump(rbvx, 'vxHo_rbf_170eV.pkl')
    joblib.dump(rbvy, 'vyHo_rbf_170eV.pkl')
    joblib.dump(rbvz, 'vzHo_rbf_170eV.pkl')
    
# =============================================================================
#     Later on to read, you need to use:
#        from sklearn.externals import joblib
#     
#        rbKE = joblib.load('keHo_rbf_170eV.pkl')
# =============================================================================

    
if __name__ == '__main__':
    
    training_datasets(filename='simBucket_Ho_170eV.npy')
    