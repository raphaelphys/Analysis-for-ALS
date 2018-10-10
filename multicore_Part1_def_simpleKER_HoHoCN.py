# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 16:45:48 2018

@author: Razib
"""

import numpy as np
import matplotlib.pyplot as plt
from helper_analysis import hist2d, pipico, hist1d, init_hist1d, init_hist2d,cosineAnglefromMomenta, twoConditions

from matplotlib.colors import LogNorm
import math as ma
from scipy.interpolate import Rbf
from sklearn.externals import joblib
   
# Define analysis for single process --------------------------------------------------------

def SingleThread(ChunkData): #Full Data divided in chunks for parallel process
    
    # Gates on TOF
    #ionGate = [6500, 7100]
    ionGate1 = [6700, 6950]
    ionGate2 = [7200, 7500]
    t12Gate = [350, 670, 14010, 14280]
    #ionGate = [1935, 2005] #Previous
    #ionGate = [1925, 2050] #for N+
    pos_offset_1 = [-0.046, -0.676]
    pos_offset_2 = [-0.0542, -0.486]
    mass_species = [165, 191]    
    #t12_range = [0, 200, 2, 3800, 4100, 2] #For t2-t1, t2+t1 cut
    
    rotPipi_range = [200, 850, 2, 13950, 14450, 2]
    keP_range = [0, 800, 1, 0, 40, 0.1]
    
    kerRange = [0, 20, 0.1]
    kekeRange = [0, 10, 0.1, 0, 10, 0.1]
    
    angleRange = [0, 180, 1]
    
    angle2DRange = [-400, 400, 2, -100, 500, 2]
    
    pRange = [-200, 200, 1]
    mass_species = np.array(mass_species) * 1.66E-27
    
        
    rotPipico = hist2d(*init_hist2d(*rotPipi_range))
    kerBuck = hist1d(*init_hist1d(*kerRange))
    kekeBuck = hist2d(*init_hist2d(*kekeRange))
    kePBuck = hist2d(*init_hist2d(*keP_range))
    angleBuck1D = hist1d(*init_hist1d(*angleRange))
    pxBuck1D = hist1d(*init_hist1d(*pRange))
    pyBuck1D = hist1d(*init_hist1d(*pRange))
    pzBuck1D = hist1d(*init_hist1d(*pRange))
    keTotBuck = hist1d(*init_hist1d(*kekeRange[0:3]))
    kercut1 = hist1d(*init_hist1d(*angleRange))
    kercut2 = hist1d(*init_hist1d(*angleRange))
    kercut3 = hist1d(*init_hist1d(*angleRange))
    angle2D = hist2d(*init_hist2d(*angle2DRange))
    angle2Dcut1 = hist2d(*init_hist2d(*angle2DRange))
    angle2Dcut2 = hist2d(*init_hist2d(*angle2DRange))
    angle2Dcut3 = hist2d(*init_hist2d(*angle2DRange))
    
    #########################################################################
    
    tof = [505.5, 2231.8, 6822, 20408]
    testmq = [1, 18, 165, 1469]
    _, t0 =  np.polyfit(np.sqrt(testmq), np.array(tof), 1)
    
    #if tweaking in t0 is needed
# =============================================================================
    t0 = t0 + 0.058*t0
# =============================================================================
    
    #rbKER = joblib.load('../../simion_training/keHo_rbf_170eV.pkl')
#    rbvx = joblib.load('../../simion_training/vxHo_rbf_170eV.pkl')
#    rbvy = joblib.load('../../simion_training/vyHo_rbf_170eV.pkl')
#    rbvz = joblib.load('../../simion_training/vzHo_rbf_170eV.pkl')
#    rbKER = joblib.load('../../simion_training/keHo_rbf_170eV.pkl')
    rbvx1 = joblib.load('../simion_training/vxsimBucket_Ho_170eV.npy.pkl')
    rbvy1 = joblib.load('../simion_training/vysimBucket_Ho_170eV.npy.pkl')
    rbvz1 = joblib.load('../simion_training/vzsimBucket_Ho_170eV.npy.pkl')
    rbKER1 = joblib.load('../simion_training/kesimBucket_Ho_170eV.npy.pkl')
    
    rbvx2 = joblib.load('../simion_training/vxsimBucket_HoCN_170eV_191amu.npy.pkl')
    rbvy2 = joblib.load('../simion_training/vysimBucket_HoCN_170eV_191amu.npy.pkl')
    rbvz2 = joblib.load('../simion_training/vzsimBucket_HoCN_170eV_191amu.npy.pkl')
    rbKER2 = joblib.load('../simion_training/kesimBucket_HoCN_170eV_191amu.npy.pkl')
    print('Done reading the rbKER')
    

 
# Initialize variables
    numberOfCoincidentEvent = 0
    event_counter = 0
    LineNum = 0
    TotalNumLines = ChunkData.size
    

    while True:
    
        #checks if end of file is reached
        if LineNum >= TotalNumLines: 
            break

        #reading the hits in terms of x, y, tof
        numberOfHits =  ChunkData[LineNum]
        LineNum = LineNum +1
        numberOfElectrons = ChunkData[LineNum]        
        LineNum = LineNum + 1    
        
        ion = ChunkData[LineNum:LineNum + numberOfHits * 3]  
        
        LineNum = LineNum + numberOfHits * 3;
        #electron = ChunkData[LineNum:LineNum + 4 * 3]
        LineNum = LineNum + numberOfElectrons * 3
        
        if numberOfHits < 2:
            continue
        
        
        #dividing by factor since thats how the numbers were saved in terms of int32
        ion = ion.reshape((numberOfHits, 3 )).T/1000

        ionX = ion[0]
        ionY = ion[1]
        ionTOF = ion[2]
        
        
        checkCondition = twoConditions(ionTOF, ionGate1, ionGate2)
        
        if checkCondition.sum() == 2:
            
            ionTOF = ionTOF[checkCondition]
            ionX = ionX[checkCondition]
            ionY = ionY[checkCondition]
            
            ionX[0] = ionX[0] - pos_offset_1[0]
            ionX[1] = ionX[1] - pos_offset_2[0]
            
            ionY[0] = ionY[0] - pos_offset_1[1]
            ionY[1] = ionY[1] - pos_offset_2[1]
            
            t12min = ionTOF[1] - ionTOF[0]
            t12plus = ionTOF[1] + ionTOF[0]
            
            rotPipico.fill(t12min, t12plus)
            
            if (t12Gate[0] < t12min < t12Gate[1]) and (t12Gate[2] < t12plus < t12Gate[3]):
                
                rKER1 = rbKER1(((ionTOF[0])/1e3 - t0/1e3)**2, ionX[0], ionY[0])
                vx1 = rbvx1(((ionTOF[0])/1e3 - t0/1e3), ionX[0], ionY[0])
                vy1 = rbvy1(((ionTOF[0])/1e3 - t0/1e3), ionX[0], ionY[0])
                vz1 = rbvz1(((ionTOF[0])/1e3 - t0/1e3)**2, ionX[0], ionY[0])
                
                rKER2 = rbKER2(((ionTOF[1])/1e3 - t0/1e3)**2, ionX[1], ionY[1])
                vx2 = rbvx2(((ionTOF[1])/1e3 - t0/1e3), ionX[1], ionY[1])
                vy2 = rbvy2(((ionTOF[1])/1e3 - t0/1e3), ionX[1], ionY[1])
                vz2 = rbvz2(((ionTOF[1])/1e3 - t0/1e3)**2, ionX[1], ionY[1])
                
                rKER = np.array([rKER1, rKER2])
                
                 ####for projection plot #########################
                V1 = np.array([vx1, vy1, vz1])
                V2 = np.array([vx2, vy2, vz2])
                
                P1 = 1e3 * 0.5E24 * mass_species[0] * V1
                P2 = 1e3 * 0.5E24* mass_species[1] * V2
                
                PX = np.zeros(2)
                PY = np.zeros(2)
                
                PX[0] = np.linalg.norm(P1)
                PY[0] = 0
                
                PX[1] = 1e3 * 0.5E24 * mass_species[1] * np.dot(V1, V2)/np.linalg.norm(V1)
                PX[1] = PX[1]
                p2square = P2.dot(P2)
                PY[1] = np.sqrt(p2square - PX[1]**2) 
                
                cosAngle = cosineAnglefromMomenta(*P1, *P2)
                
                
                ################################################
                
                px, py, pz = np.vstack((P1, P2)).T
                
                pTot = px.sum() + py.sum() + pz.sum()
                arcCosAngle = np.rad2deg(np.arccos(cosAngle))
                #histogram filling
#                kerBuck.fill(rKER.sum())
#                kekeBuck.fill(*rKER)
                kePBuck.fill(np.abs(pTot), rKER.sum())
                pxBuck1D.fill(px.sum())
                pyBuck1D.fill(py.sum())
                pzBuck1D.fill(pz.sum())
                
                
                angleBuck1D.fill(arcCosAngle)
                kerBuck.fill(rKER.sum())
                kekeBuck.fill(*rKER)
                keTotBuck.fill(rKER)
                
                angle2D.fill(PX[1], PY[1])
                
                
                if (0.05 < rKER.sum() < 1.9):
                    kercut1.fill(arcCosAngle)
                    angle2Dcut1.fill(PX[1], PY[1])
                elif (2.4 < rKER.sum() < 4.8):
                    kercut2.fill(arcCosAngle)
                    angle2Dcut2.fill(PX[1], PY[1])
                elif (5.0 < rKER.sum() < 20):
                    kercut3.fill(arcCosAngle)
                    angle2Dcut3.fill(PX[1], PY[1])
                
                # Flags to check run progress
                event_counter += 1
                if event_counter%5000 == 0:
                    print(event_counter)
                    
                numberOfCoincidentEvent += 1 
                
    print("Number of coincident evt: {}" .format(numberOfCoincidentEvent))            
    #+++++++++ Specify the histogram output here +++++#                
    return [kekeBuck, kePBuck, kerBuck, pxBuck1D, pyBuck1D, pzBuck1D, angleBuck1D, rotPipico, keTotBuck, kercut1, kercut2, kercut3, angle2D, angle2Dcut1, angle2Dcut2, angle2Dcut3]
    #+++++++++++++++++++++++++++++++++++++++++++++++++#