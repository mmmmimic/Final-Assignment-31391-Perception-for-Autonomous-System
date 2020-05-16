#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
@author: mxl
@date: 04/30/2020
'''
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import imutils
import math
import copy
from numpy import linalg
import scipy.signal as signal
import glob
import random
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as op
from torch import nn
from torchvision import models

transform = transforms.Compose((transforms.ToTensor(), transforms.Normalize(mean=(0,), std=(1,))))

'''
#Build a New ResNet
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=3):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(8192, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(1, 8192)
        out = self.fc(out)
        return out
'''
# A pretty simple cnn(the less the trainable parameters are, the faster the net is)
class MyNet(nn.Module):
    def __init__(self, num_classes=3):
        super(MyNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(65536, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(1, 65536)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

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

class Footprint(object):
    def __init__(self):
        super(Footprint, self).__init__()
        self.data = []
        self.delta = 0
        self.var = []
        self.sx = 0
        self.sy = 0

    def push(self, point):
        if self.isNewObj(point):
            self.data.append(point)

    def motionModel(self, order, x, point, t):
        time = [int(i*t/order) for i in range(1, order+1)]
        S = np.array([self.getDist(point[list(x).index(min(x)), :2], 
        point[list(x).index(sorted(x)[t]), :2]) for t in time]).reshape(-1,1)
        T = np.zeros((order, order))
        for i in range(order):
            for j in range(order):
                T[i,j] = (time[i])**(j+1)/(j+1)
        var = linalg.pinv(T)@S
        return var

    def update(self):
        # update the motion model variables
        order = len(self.var)
        for i in range(order-1):
            for j in range(i+1, order):
                self.var[i]+=self.var[j]/(j)

    def move(self):
        # return delta_s
        order = len(self.var)
        delta_s = 0
        for i in range(order):
            delta_s += (self.var[i])/(i+1)
        return delta_s
             
    def regression(self, order=2):
        # return the predicted delta_x and delta_y, vel, acc and beta
        t = len(self.data)-1
        if not t:
            return None
        point = np.array([np.array([data[0],data[1], 1]) for data in self.data]).reshape(-1, 3)
        u, w, vt = linalg.svd(point)
        # denoising: median filter, kernel_size:5
        point[:,0] = signal.medfilt(point[:,0], 5)
        point[:,1] = signal.medfilt(point[:,1], 5)
        v = vt[-1, :]
        #y = point[:,1]
        #assert v[0]
        #x = -(v[1]*y+v[2])/v[0]
        x = point[:,0]
        assert v[1]
        y = -(v[0]*x+v[2])/v[1]
        # see the fitting line
        #plt.plot(y,-x)
        #plt.show()
        if order==1:
            dx = (max(x)-min(x))/t
            dy = (max(y)-min(y))/t
            theta = math.atan(dy/dx)
            var = [np.sqrt(dx**2+dy**2)]
        else:
            var = self.motionModel(order, x, point, t)
            theta = math.atan(-v[0]/v[1])
        symbol_x = int(x[-1]-x[0])
        symbol_y = int(y[-1]-y[0])
        symbol_x = symbol_x/np.abs(symbol_x) if symbol_x else 0
        symbol_y = symbol_y/np.abs(symbol_y) if symbol_y else 0
        self.sx = symbol_x
        self.sy = symbol_y
        self.var = var
        if math.isnan(theta):
            theta = math.pi
        self.theta = theta

    def predict(self, point):
        if np.min(point)<0:
            return point
        if len(self.data):
            self.regression()
        delta_s = self.move()
        dx, dy = np.abs(delta_s*math.cos(self.theta)), np.abs(delta_s*math.sin(self.theta))
        dx *= self.sx
        dy *= self.sy
        point[0] = int(point[0]+dx)
        point[1] = int(point[1]+dy)
        self.update()
        return point

    @staticmethod
    def getDist(start, stop):
        # get the distance between two points format::[1x2]ndarray
        return np.sqrt(((start-stop)@(start-stop).T))
        
    def isNewObj(self, point, metric=100):
        if len(self.data)>1:
            # metric: Euclian distance
            self.delta = self.getDist(point.T, self.data[-1].T)
            if self.delta>metric:
                self.data = []
                return False
        return True

# define a tracker based on background subtractor
class BS(object):
    def __init__(self, trainNum):
        super(BS, self).__init__()
        # the number of the frames used for training
        self.trainNum = trainNum
        self.bs = cv2.createBackgroundSubtractorKNN(detectShadows=False)
        self.bs.setHistory(trainNum)
        self.counter = 0

    def train(self, img):
        self.bs.apply(img)
        self.counter += 1
    
    def feed(self, img):
        if self.counter<=self.trainNum:
            self.train(img)
            return False, [], []
        else:
            mask = self.bs.apply(img)
            ret, roi, center = self.analysis(mask)
            return ret, roi, center
    
    def analysis(self, mask, debug=False):
        if debug: 
            cv2.imshow('mask', mask)
            cv2.waitKey(1)
        # morphological operation
        mask = cv2.dilate(mask, None, iterations=12)
        mask = cv2.erode(mask, None, iterations=20)
        # find the largest blob
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        area = [cv2.contourArea(c) for c in cnts]
        maxArea = np.array(area)[np.array(area)>2000]
        object = []
        ret = False
        center = []
        roi = []
        for i in range(len(maxArea)):
            ret = True # flag: is the object found?
            idx = area.index(maxArea[i])
            obj = cnts[idx]
            object.append(obj)
            #fetch boundingbox
            (x, y, w, h) = cv2.boundingRect(obj)
            #calculate the geometry center
            grid = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
            X = grid[0,...]
            Y = grid[1,...]
            m = np.zeros_like(mask)
            m[y:y+h, x:x+w] = mask[y:y+h, x:x+w]/255.
            cent = np.array([int(np.sum(m*Y)/np.sum(m)), int(np.sum(m*X)/np.sum(m))])
            #cent = np.array([int(x+w/2),int(y+h/2)])
            center.append(cent)
            roi.append((x,y,x+w, y+h))
        return ret, roi, center

class Kalman(object):
    def __init__(self, ini_pose):
        super(Kalman).__init__()
        # The initial state (6x1).
        # displacement, velocity, acceleration
        # including x, v_x, a_x, y, v_y, a_y 
        self.X = np.zeros((6,1))
        self.X[0] = ini_pose[0]
        self.X[3] = ini_pose[1]
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
        Z = Z.reshape((2,-1))
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
        pose = np.array([int(self.X[0]),int(self.X[3])])
        self.predict()
        return pose

class Tracker(object):
    def __init__(self, loader):
        super(Tracker, self).__init__()
        self.bs = BS(70) #BS tracker
        self.loader = loader #to load frames
        self.crop() #assign the entry and exit of the conveyor manually
        self.isOcc = False #check if the object is blocked
        self.center = None #the center of the target in the current frame
        self.roi = None #boundingbox
        self.fp = Footprint() #save the track of the object(front view)
        self.dp = Footprint() #save the track of the object(top view)
        self.ratio = 0.5 #the threshold of the area of the object
        self.area = [] #the area of the object
        self.flag = False #flag: if there is object on the conveyor
        self.depth = None #the depth of the frame, pregenerated
        self.im = [] #current frame
        self.counter = 0 #the accumulative number of the objects
        self.model = torch.load('model.h5') #载入神经网络
        self.model.to('cpu') #put the CNN on cpu
        self.label = 'None' #the label of the object

    def crop(self):
        # assign the entry and exit of the conveyor manually
        im = self.loader.getItem(0)
        cv2.namedWindow('im')
        cv2.resizeWindow('resized',778, 1787)
        x1,y1,w1,h1 = cv2.selectROI('im', im)
        x2,y2,w2,h2 = cv2.selectROI('im', im)
        #draw the "conveyor" in white lines
        cv2.rectangle(im, (x1,y1), (x1+w1,y1+h1), (255, 255, 255), 5)
        cv2.rectangle(im, (x2,y2), (x2+w2,y2+h2), (255, 255, 255), 5)
        cv2.line(im, (x1,y1), (x2,y2), (255, 255, 255), 5)
        cv2.line(im, (x1+w1,y1), (x2+w2,y2), (255, 255, 255), 5)
        cv2.line(im, (x1,y1+h1), (x2,y2+h2), (255, 255, 255), 5)
        cv2.line(im, (x1+w1,y1+h1), (x2+w2,y2+h2), (255, 255, 255), 5)
        plt.imshow(im)
        plt.show()
        self.rect_in = (x1,y1,w1,h1)
        self.rect_out = (x2,y2,w2,h2)
    
    def inspect(self, center, rect):
        # check if the object is entering or leaving the conveyor
        if (rect[0]<=center[0]<=rect[0]+rect[2]) and (rect[1]<=center[1]<=rect[1]+rect[3]):
            return 1
        else:
            return 0
        
    def check(self, center, roi):
        # check if the object is entering or leaving the conveyor
        # check if the object is enteringthe conveyor
        status_in = [self.inspect(cent, self.rect_in) for cent in center]
        # check if the object is leaving the conveyor
        status_out = [self.inspect(cent, self.rect_out) for cent in center]
        # if there is an object entering the conveyor area and there is nothing on the conveyor currently
        if np.sum(np.array(status_in))==1 and not self.flag:
            # get the index of the object
            idx = status_in.index(1)
            # save center and roi
            self.center = center[idx]
            self.roi = roi[idx]
            # now there is an object on the conveyor
            self.flag = True
            # get the object for classification
            im = self.im[self.roi[1]-30:self.roi[3]+30, self.roi[0]-30:self.roi[2]+30,:]
            # object number +1
            self.counter+=1
            # classification
            label = ['Box', 'Book', 'Cup']
            # to accelerate the calculation, use gray image 
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            # normalize input size 128*128
            im = cv2.resize(im, (128,128))
            im = im.reshape((128, 128, 1))
            im = transform(im)
            im.resize_(1,1,128,128)
            output = self.model(im)
            # print the label
            _, predicted = torch.max(output.data, 1)
            self.label = label[int(predicted)]

        #if there is an object leaving the conveyor area and there is an object on the conveyor currently
        if np.sum(np.array(status_out))==1 and self.flag:
            # clear the position of the object
            self.center = None
            self.roi = None
            # now there is no object on the conveyor
            self.flag = False
            # clear the saved areas
            self.area = []
            # clear the object label
            self.label = 'None'

    @staticmethod
    def findNest(plist, p):
        # find the nearest point to p in plist
        dist = [math.sqrt((pl-p)@(pl-p).T) for pl in plist]
        idx = dist.index(min(dist))
        return idx

    def loop(self):
        # loop for 3d tracking 
        for i in range(self.loader.len):
            self.depth = np.load('depthOcc/'+f'{i+1:04d}.npy')
            # adjust the depth(because the former depth data is not so exact, the focal length is 700 and basiline is 3.5)
            # focal length of the camera
            fl = 704.4922
            # baseline
            bl = math.sqrt(((-120.0307)**2+(-0.2427)**2+(-0.4453)**2))
            self.depth = self.depth/(700*3.5)*fl*bl
            #the max depth is set to be 100
            self.depth[self.depth>100] = 0
            im,dm = self.track(i) #track each frame
            self.im = im
            # to show the image, downsampling the image
            # draw front view
            im = cv2.resize(im, (690, 300))
            cv2.imshow('Front View', im)
            # draw top view
            dm = cv2.resize(dm, (690, 100))
            cv2.imshow('Top View', dm)
            cv2.waitKey(1)
            # combining the top view with the front one, we can have a 3d loook at the object
        cv2.destroyAllWindows()

    def track(self, i):
        # track objects in a frame 
        # fetch the frame
        im = self.loader.getItem(i)
        # generate a blank image for the top view
        dm = np.zeros((100,im.shape[1],3))
        # if there is moving objects in the frame, ret should be True
        # ret represents the existence of the moving objects in the image
        ret, roi, center = self.bs.feed(im)
        # there are 5 possible status in the frame 
        # 1. Original Frame: there is no moving object in the frame, then ret and flag are all False.
        # 2. Green Box: there are moving objects in the frame, but they are not on the conveyor, 
        # they should be considered as noise. ret is Ture while flag is False.
        # 3. Red Box: there are moving objects in the frame and one of them on the conveyor. In addition, the area of the 
        # object on the conveyor is larger than the threshold. ret is True and flag is True. 
        # 4. Yellow Point: there are moving objects in the frame and one of them on the conveyor whereas the area is smaller
        # than the threshold. ret is True and flag is True. 
        # 5. Yellow Point: there is no moving objects detected in the frame. But there is somthing on the conveyor. ret is False
        # while flag is True. 
        if ret:
            self.isOcc = False
            # if there is moving object detected in the frame, the frame should be at one of status 2-4
            # check if the object is on the conveyor, update flag
            self.check(center, roi)
            if not self.flag:
                # when flag is False, status 2, green box
                for j in range(len(roi)):
                    ro = roi[j]
                    cent = center[j]
                    cv2.rectangle(im, ro[:2], ro[2:], (0, 255, 0), 5)
            else:
                # if flag is True, the status should be 3 or 4. We should check the area
                # find the object first
                idx = self.findNest(center, self.center)
                # remove the obejct from candidates
                self.center = center.pop(idx)
                self.roi = roi.pop(idx)
                # calculate the area
                area = (self.roi[2]-self.roi[0])*(self.roi[3]-self.roi[1])
                # save the area
                self.area.append(area)
                if area>=max(self.area)*self.ratio:
                    # if the area exceeds the threshold, the frame should be at status 3, red box. 
                    cv2.circle(im, (self.center[0], self.center[1]), 5, (255, 0, 0), -1)
                    cv2.rectangle(im, self.roi[:2], self.roi[2:], (0, 0, 255), 5)
                    # show the object label
                    cv2.putText(im, self.label, self.roi[:2], 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    # save the coordinate of the center 
                    self.fp.push(self.center.T)
                    # get the depth of the center of the object, conbine it with the x coordinate got before
                    depth = np.mean(self.depth[self.center[1]-5:self.center[1]+5, self.center[0]-5:self.center[0]+5])
                    dp = np.array([self.center[0],depth])
                    # save the center
                    self.dp.push(dp)
                    # draw the top view
                    cv2.circle(dm, (int(dp[0]), int(dp[1])), 10, (0, 0, 255), -1)
                    # draw other candidates 
                    for j in range(len(roi)):
                        ro = roi[j]
                        cent = center[j]
                        cv2.rectangle(im, ro[:2], ro[2:], (0, 255, 0), 5)
                else:
                    # if the area is less than the threshold, the status should be 4. predict the center 
                    self.center = (self.fp.predict(self.center.T)).T
                    # only draw the center 
                    cv2.putText(im, self.label, (self.center[0]+10, self.center[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    cv2.circle(im, (self.center[0], self.center[1]), 5, (0, 255, 255), -1)
                    # because of the occlusion, clear the saved path before
                    self.fp.data = []
                    # predict the depth
                    depth = np.mean(self.depth[self.center[1]-5:self.center[1]+5, self.center[0]-5:self.center[0]+5])
                    dp = np.array([self.center[0],depth])
                    dp = (self.dp.predict(dp.T)).T
                    cv2.circle(dm, (self.center[0], int(dp[1])), 10, (0, 255, 255), -1)
        else:
            # if there is no moving object in the frame, the status ought to be 1 or 5
            # check if there is something on the conveyor
            if self.flag and not self.isOcc:
                # if the conveyor is activated, then the frame should be at status 5. 
                # from this frame, the object is under occlusion
                # change the status flag, now the object is under occlusion
                self.isOcc = True
                # predict the center
                self.center = (self.fp.predict(self.center.T)).T
                # because of the occlusion, clear the saved path before
                self.fp.data = []
                # draw the center
                cv2.putText(im, self.label, (self.center[0]+10, self.center[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.circle(im, (self.center[0], self.center[1]), 5, (0, 255, 255), -1) 
                # predict the depth
                depth = np.mean(self.depth[self.center[1]-5:self.center[1]+5, self.center[0]-5:self.center[0]+5])
                dp = np.array([self.center[0],depth])
                dp = (self.dp.predict(dp.T)).T
                self.dp.data = []
                cv2.circle(dm, (self.center[0], int(dp[1])), 10, (0, 255, 255), -1)

            elif  self.flag:
                # if the object is still on the conveyor and it is under occlusion before
                self.center = (self.fp.predict(self.center.T)).T
                # draw the center
                cv2.putText(im, self.label, (self.center[0]+10, self.center[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.circle(im, (self.center[0], self.center[1]), 5, (0, 255, 255), -1) 
                # predict the depth
                depth = np.mean(self.depth[self.center[1]-5:self.center[1]+5, self.center[0]-5:self.center[0]+5])
                dp = np.array([self.center[0],depth])
                dp = (self.dp.predict(dp.T)).T
                cv2.circle(dm, (self.center[0], int(dp[1])), 10, (0, 255, 255), -1)
            else:
                # if there is noting on the conveyor, the status should be 1
                self.center = None
                self.roi = None
                self.isOcc = False
        return im,dm

def getLeftDepth(left, right):
    h,w,c = left.shape
    winSize = 5
    numberOfDisparities = int((w / 8) + 15) & -16
    stereo = cv2.StereoSGBM_create(0,16,3)
    stereo.setPreFilterCap(32)
    stereo.setP1(8*c*winSize**2)
    stereo.setP2(32*c*winSize**2)
    stereo.setMinDisparity(0)
    stereo.setBlockSize(winSize)
    stereo.setNumDisparities(numberOfDisparities)
    stereo.setDisp12MaxDiff(100)
    stereo.setUniquenessRatio(10)
    stereo.setSpeckleRange(32)
    stereo.setSpeckleWindowSize(0)
    disp = stereo.compute(left, right).astype(np.float32) / 16.0
    disp = cv2.medianBlur(disp, 5)
    return disp

def getRightDepth(left, right):
    h,w,c = right.shape
    winSize = 5
    numberOfDisparities = int((w / 8) + 15) & -16
    # use SGBM to get the disparity map
    stereo = cv2.StereoSGBM_create(0,16,3)
    stereo.setPreFilterCap(32)
    stereo.setP1(16*c*winSize**2)
    stereo.setP2(32*c*winSize**2)
    stereo.setMinDisparity(-numberOfDisparities)
    stereo.setBlockSize(winSize)
    stereo.setNumDisparities(numberOfDisparities)
    stereo.setDisp12MaxDiff(100)
    stereo.setUniquenessRatio(10)
    stereo.setSpeckleRange(32)
    stereo.setSpeckleWindowSize(0)
    disp = stereo.compute(right, left).astype(np.float32) / 16.0
    # turn all the disparity value to positive
    disp = numberOfDisparities+disp
    depth = disp.copy()
    disp[disp<0] = 0
    w,h = disp.shape
    # make up the "holes" in the map
    # use integration map to accelerate the calculation
    inte = cv2.integral2(disp)[0]
    for i in range(w):
        for j in range(h):
            size = 30
            if not disp[i,j]:
                idx1 = max([i-size,0])
                idx2 = min([w,i+size])
                idx3 = max([j-size,0])
                idx4 = min([h,j+size])
                arr = disp[idx1:idx2,idx3:idx4]
                if np.sum(arr>0):
                    num = np.sum(arr>0)
                else:
                    num = np.inf
                depth[i,j] = (inte[idx1,idx3] + inte[idx2,idx4]-inte[idx2,idx3]-inte[idx1,idx4])/num
    disp = depth
    depth[depth==0] = -1
    # the focal length is 700 and the baseline length is 3.5
    depth = 700*3.5/depth
    # the minimum depth should be 0, in line with the basic knowledge 
    depth[depth<0] = 0
    return disp, depth


if __name__=="__main__":
    #leftLoader = DataLoader('imgsOcc/left/')
    rightLoader = DataLoader('imgsOcc/right/')

    t = Tracker(rightLoader)
    t.loop()
    '''
    for i in range(rightLoader.len):
        left = leftLoader.getItem(i)
        right = rightLoader.getItem(i)
        disp, depth = getRightDepth(left, right)
        np.save('depthOcc/'+f'{(i+1):04d}.npy', depth)
    '''










