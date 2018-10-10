# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 16:49:20 2018

@author: Razib
"""

import numpy as np
from helper_analysis import *
from matplotlib.colors import LogNorm
import math as ma
import glob
from multicore_Part1_def_simpleKER_HoHoCN import SingleThread
import matplotlib.pyplot as plt



if __name__ == '__main__': # All code outside of this if statement will be
    # run by all the subprocesses if Windows is operating system. Put code 
    # that should be run once only here.

#Number of processes for multiprocessing
    
    NumCores = 3 #Adjust for best performance
    import multiprocessing as mp    
    fileNum = 0
    path = '160eV-1000/*.txt'
    fileLim = 0 


    for fname in glob.glob(path):
        filename = fname
        print(filename)
    
        f = open(filename)

################################################### Prepare Multiprocessing
#(no need to make changes here)

        #Read file all at once
        print('Reading file...')
        LineNum = 0
        DataFile = np.fromfile(f, dtype = 'int32', count = -1)
        f.close()
        TotalNumLines = DataFile.size
        print('File read successfully!')
    
        # get total hit number
        print('Getting number of hits...')
        TotalEventNum = 0
    
        while True:
    
            if LineNum >= TotalNumLines:
                break
        
            numberOfHits =  DataFile[LineNum]   
            TotalEventNum = TotalEventNum + numberOfHits
            LineNum += 1
            numberOfElectrons = DataFile[LineNum]
            
            LineNum = LineNum + numberOfHits * 3
            LineNum = LineNum + numberOfElectrons * 3 + 1
        
        print('Total number of hits in file = ',TotalEventNum)
        print('Determine cuts for multiprocessing...')
         
        #Multiprocessing: define positions where data should be cut into chunks    
        Cuts = np.zeros(NumCores+1)
        Cuts[NumCores] = TotalNumLines
        Cut = TotalEventNum / NumCores
    
        LineNum = 0
        TotalEventNum = 0
        CutNum = 1
        ApproxCutPos = Cut
    
        while True:
    
            if LineNum >= TotalNumLines:
                break
        
            numberOfHits =  DataFile[LineNum]   
            TotalEventNum = TotalEventNum + numberOfHits
            LineNum += 1
            numberOfElectrons = DataFile[LineNum]
            LineNum = LineNum + numberOfHits * 3
            LineNum = LineNum + numberOfElectrons * 3 + 1
        
            if TotalEventNum > ApproxCutPos: #Cut only at position where an event ends
                Cuts[CutNum]=LineNum
                CutNum = CutNum + 1
                ApproxCutPos = CutNum * Cut
                
        print('Completed!')
        print('Cutting data into chunks...')
        
        #Multiprocessing: slice data into chunks at positions specified in cuts
        chunks = []
    
        ChunkNum = 1
        while True:
      
            if ChunkNum > NumCores:
                break
        
            chunks.append(DataFile[int(Cuts[ChunkNum-1]):int(Cuts[ChunkNum])])    
        
            ChunkNum = ChunkNum + 1       
    
        # Start Multiprocessing
        print('Multiprocessing starts...')
        ProcPool = mp.Pool(NumCores)
        PartResults = ProcPool.map(SingleThread, chunks)
        print('Multiprocessing completed!')
    ################################################### Prepare Multiprocessing end
                
        # Join histograms (sums histograms to zeroth entry in PartResults)
        ChunkNum = 1
        while True:
      
            if ChunkNum > NumCores-1:
                break
            
            #+++++++++ Add Hisograms here +++++#
            for i in range(0,16):
                PartResults[0][i].hists += PartResults[ChunkNum][i].hists
            #++++++++++++++++++++++++++++++++++#
            
            ChunkNum = ChunkNum + 1   
                    
    # Further processing of histograms
                    

        
        ProcPool.close()
        if fileNum == 0:          
            en1_center, en2_center, count_ke = PartResults[0][0].edgeData
            tof_coinc, ke_coinc, count_tofKe = PartResults[0][1].edgeData
            ker, countKer = PartResults[0][2].edgeData
            p, countPx = PartResults[0][3].edgeData
            p, countPy = PartResults[0][4].edgeData
            p, countPz = PartResults[0][5].edgeData
            cosAngle, countAngle = PartResults[0][6].edgeData
            t12m, t12p, countT12 = PartResults[0][7].edgeData
            en_keTot, countkeTot = PartResults[0][8].edgeData
            cosAngle, countKERcut1 = PartResults[0][9].edgeData
            cosAngle, countKERcut2 = PartResults[0][10].edgeData
            cosAngle, countKERcut3 = PartResults[0][11].edgeData
            P1, P2, countP1P2 = PartResults[0][12].edgeData
            _, _, countP1P2cut1 = PartResults[0][13].edgeData
            _, _, countP1P2cut2 = PartResults[0][14].edgeData
            _, _, countP1P2cut3 = PartResults[0][15].edgeData
            
        else:
            count_ke += PartResults[0][0].hists
            count_tofKe += PartResults[0][1].hists
            countKer += PartResults[0][2].hists
            countPx += PartResults[0][3].hists
            countPy += PartResults[0][4].hists
            countPz += PartResults[0][5].hists
            countAngle += PartResults[0][6].hists
            countT12 += PartResults[0][7].hists
            countkeTot += PartResults[0][8].hists
            countKERcut1 += PartResults[0][9].hists
            countKERcut2 += PartResults[0][10].hists
            countKERcut3 += PartResults[0][11].hists
            countP1P2 += PartResults[0][12].hists
            countP1P2cut1 += PartResults[0][13].hists
            countP1P2cut2 += PartResults[0][14].hists
            countP1P2cut3 += PartResults[0][15].hists
            
                   
        fileNum = fileNum + 1
        print('Going to the next file or wrapping up')
        
        if fileLim > 0:
            if fileLim == fileNum:
                break
#        
    filename = 'Results/Temp/Ho_HoCN_160eV_cutKER_withmod_FinalQuintic_res0p1.npz'
    np.savez_compressed(filename, en1_center=en1_center, en2_center=en2_center,\
                        count_ke=count_ke, ker = ker, countKer = countKer, p = p, countPx=countPx,\
                        countPy=countPy, countPz=countPz, cosAngle=cosAngle, countAngle=countAngle,
                        en_keTot=en_keTot, countkeTot=countkeTot, countKERcut1=countKERcut1, \
                        countKERcut2=countKERcut2, countKERcut3=countKERcut3, P1=P1, P2=P2, countP1P2=countP1P2,\
                        countP1P2cut1=countP1P2cut1, countP1P2cut2=countP1P2cut2, countP1P2cut3=countP1P2cut3)
    
    #Xgx,Xgy,Xgz=XY.centerData
    
    # Plots
           
    print('Going to plot')
    
    plt.figure(11)
    plt.subplot(221)
    plt.plot(ker[:-1], countKer)
    plt.xlabel('ker (eV)')
    plt.subplot(222)
    plt.plot(p[:-1], countPx)
    plt.xlabel('px')
    plt.subplot(223)
    plt.plot(p[:-1], countPy)
    plt.xlabel('py')
    plt.subplot(224)
    plt.plot(p[:-1], countPz)
    plt.xlabel('pz')
    
    plt.figure(1)
    plt.plot(cosAngle[:-1], countAngle)
    plt.xlabel('cos theta')
    
    plt.figure(2)
    plt.pcolormesh(en1_center[:-1], en2_center[:-1], np.transpose(count_ke), cmap='jet')
    plt.title('KE in coincidence', fontsize=20)
    plt.xlabel('kinetic energy of ion1 (eV)')
    plt.ylabel('kinetic energy of ion2 (eV)')
    
    
    plt.figure(3)
    plt.pcolormesh(tof_coinc[:-1], ke_coinc[:-1],np.transpose(count_tofKe), cmap='jet')
    plt.title('tof1-tof2 vs tof1+tof2', fontsize=20)
    
    # Pipico Rotated
    
    plt.figure(4)
    plt.pcolormesh(t12m[:-1], t12p[:-1], np.transpose(countT12), cmap='jet')
    plt.title('tof1+tof2-tof3 vs tof1+tof2+tof3', fontsize=20)
    plt.colorbar()
    
    plt.figure(5)
    plt.plot(en_keTot[:-1], countkeTot)
    
    plt.figure(6)
    plt.subplot(311)
    plt.plot(cosAngle[:-1], countKERcut1)
    
    plt.subplot(312)
    plt.plot(cosAngle[:-1], countKERcut2)
    
    plt.subplot(313)
    plt.plot(cosAngle[:-1], countKERcut3)
    
    plt.figure(7)
    plt.pcolormesh(P1[:-1], P2[:-1], countP1P2.T, norm=LogNorm())
    plt.colorbar()
    
    plt.figure(8)
    plt.pcolormesh(P1[:-1], P2[:-1], countP1P2cut1.T, norm=LogNorm())
    plt.colorbar()
    
    plt.figure(9)
    plt.pcolormesh(P1[:-1], P2[:-1], countP1P2cut2.T, norm=LogNorm())
    plt.colorbar()
    
    plt.figure(10)
    plt.pcolormesh(P1[:-1], P2[:-1], countP1P2cut3.T, norm=LogNorm())
    plt.colorbar()
    
    plt.show()
    print('Done plotting')
    