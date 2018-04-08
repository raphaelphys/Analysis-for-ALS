# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 19:09:19 2018

@author: Razib
"""

import numpy as np

foreText = "particles {coordinates = 0,"

f = open('Test_Shashank.fly2', 'w')

f.write(foreText)

for i in np.arange(1, 10, 0.5):
    midText = "standard_beam {n = 180, tob = 0, mass = 17, charge = 1, \
ke ="+ str(i) + ", cwf = 1, color = 0, direction = cone_direction_distribution {\
axis = vector(1, 0, 0), half_angle = 180, fill = true}, position = vector(0, 0, 0)},"
    f.write(midText)
    

f.write( "}")
f.close()