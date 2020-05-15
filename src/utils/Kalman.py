#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
@author: mxl
@date: 04/20/2020
'''
import cv2
import numpy as np
import math
import random

class Kalman(object):
    def __init__(self):
        super(Kalman).__init__()
        # The initial state (6x1).
        # displacement, velocity, acceleration
        # including x, v_x, a_x, y, v_y, a_y 
        self.X = np.zeros((6,1))
        # The initial uncertainty (6x6).
        # let's assume a large initial value since at the very first the uncertainty is high
        uc = 500
        self.P = uc*np.eye(6)
        # The external motion (6x1).
        self.u = np.zeros((6,1))
        # The transition matrix (6x6).
        self.F = np.array([[1, 1, 0.5, 0, 0, 0],
              [0, 1, 1, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 1, 0.5],
              [0, 0, 0, 0, 1, 1],
              [0, 0, 0, 0, 0, 1]])
        # The observation matrix (2x6).
        self.H = np.array([[1,0,0,0,0,0],[0,0,0,1,0,0]])
        # The measurement uncertainty.
        self.R = np.array([[20],[20]])
        self.I = np.eye(6)

    def update(self, Z):
        y = Z-np.dot(self.H, self.X)
        S = np.dot(np.dot(self.H, self.P),np.transpose(self.H))+self.R
        K = np.dot(np.dot(self.P, np.transpose(self.H)),np.linalg.pinv(S))
        self.X = self.X+np.dot(K, y)
        self.P = np.dot((self.I-np.dot(K, self.H)),self.P)
    
    def predict(self):
        self.X = np.dot(self.F, self.X)+self.u
        self.P = np.dot(np.dot(self.F, self.P),np.transpose(self.F))

    def filt(self, Z):
        self.update(Z)
        pose = np.array([self.X[0],self.X[3]])
        self.predict()
        return pose