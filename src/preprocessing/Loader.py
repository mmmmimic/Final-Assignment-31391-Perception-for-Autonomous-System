#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
@author: mxl
@date: 04/16/2020
'''
import cv2
import numpy as np
import math
import os
import glob

class DataLoader(object):
    def __init__(self, path, idx=0, cvt=None, size=None, downsize=None):
        super(DataLoader).__init__()
        self.cvt = cvt
        self.idx = idx
        self.size = size
        self.downsize = downsize
        self.file = glob.glob(path+'*')
        self._len = len(self.file)

    def cvtImg(self, im):
        if isinstance(self.cvt, np.int):
            im = cv2.cvtColor(im, self.cvt)
        if isinstance(self.size, tuple):
            im = cv2.resize(im, self.size)
        if isinstance(self.downsize, np.int):
            new_size =  (int(im.shape[1]/self.downsize),int(im.shape[0]/self.downsize))
            im = cv2.resize(im, dsize=new_size)
        return im

    def getItem(self, idx):
        im = self.cvtImg(cv2.imread(self.file[idx]))
        return im
    
    def __next__(self):
        im = self.getItem(self.idx)
        self.idx+=1
        return im
    
    def getRest(self):
        return [self.__next__() for i in range(self.idx, self.len)]

    @property
    def len(self):
        return self._len

        
class DualLoader(DataLoader):
    def __init__(self, path, idx=0, cvt=None, size=None, downsize=None):
        super(DualLoader, self).__init__(path, idx=idx, cvt=cvt, size=size, downsize=downsize) 
        self.file_l = glob.glob(path+'left/*')
        self.file_r = glob.glob(path+'right/*')
        assert len(self.file_l)==len(self.file_r)
        self._len = len(self.file_l)

    def getItem(self, idx):
        iml = self.cvtImg(cv2.imread(self.file_l[idx]))
        imr = self.cvtImg(cv2.imread(self.file_r[idx]))
        return iml, imr
    
    def __next__(self):
        iml, imr = self.getItem(self.idx)
        self.idx+=1
        return iml, imr






