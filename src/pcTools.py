#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
@author: mxl
@date: 04/29/2020
'''

import numpy as np
import open3d as o3d
import cv2
from matplotlib import pyplot as plt
from preprocessing import DataLoader
d1 = DataLoader.DataLoader('../imgs/left/', downsize=3)
d2 = DataLoader.DataLoader('../imgs/right/', downsize=3)
stereo = cv2.StereoBM_create(numDisparities=10*16, blockSize=15)
stereo.setDisp12MaxDiff(200)
stereo.setMinDisparity(7)
stereo.setUniquenessRatio(5)
stereo.setSpeckleWindowSize(3)
stereo.setSpeckleRange(5)
left = d1.getItem(1180)
right = d2.getItem(1180)
left1 = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
right1 = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
disp = stereo.compute(left1, right1).astype(np.float32)/16.0
b = 3.57
f = 702
disp = (1/disp)*b*f
disp = disp/disp.max()*10000
plt.subplot(1,2,1)
plt.imshow(disp, cmap='jet')
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(left)
plt.show()