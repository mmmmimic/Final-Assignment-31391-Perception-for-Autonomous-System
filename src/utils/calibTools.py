#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
@author: mxl
@date: 04/07/2020
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt
from preprocessing import DataLoader

def calibrate(lpath, rpath, nbh, nbv, show=False):
    '''
    get camera information
    '''
    nb_horizontal = 9
    nb_vertical = 6
    objp = np.zeros((nbh*nbv,3), np.float32) # set x and y coordinates
    objp[:,:2] = np.mgrid[0:nbv,0:nbh].T.reshape(-1,2) # set z coordinates
    # first read images, then extract image points 
    left = DataLoader.DataLoader(lpath) #left images
    right = DataLoader.DataLoader(rpath)
    imgpl = [] #image points
    imgpr = []
    img_left = [] #image
    img_right = []
    objpoints = []
    for i in range(left.len):
        iml = left.__next__()
        imr = right.__next__()
        retl, corners_l = cv2.findChessboardCorners(iml,(nb_vertical,nb_horizontal))
        retr, corners_r = cv2.findChessboardCorners(imr,(nb_vertical,nb_horizontal))
        if retl and retr: # only when both of the images in a pair has the corners
            imgpl.append(corners_l)
            imgpr.append(corners_r)
            img_left.append(iml)
            img_right.append(imr)
            objpoints.append(objp)
        if show:
            iiml, iimr = iml.copy(), imr.copy()
            cv2.drawChessboardCorners(iml.copy(), (nbv,nbh), corners_l, retl)
            cv2.drawChessboardCorners(imr.copy(), (nbv,nbh), corners_r, retr)
            cv2.imshow('left', iiml)
            cv2.waitKey(1)
            cv2.imshow('right', iimr)
            cv2.waitKey(1)
    gray = cv2.cvtColor(iml, cv2.COLOR_BGR2GRAY)
    # determine the camera matirx 1
    mtx1, dist1= cv2.calibrateCamera(objpoints, imgpl, gray.shape[::-1], None, None)[1:3]
    gray = cv2.cvtColor(imr, cv2.COLOR_BGR2GRAY)
    # determine the camera matirx 2
    mtx2, dist2 = cv2.calibrateCamera(objpoints, imgpr, gray.shape[::-1], None, None)[1:3]
    cv2.destroyAllWindows()
    mtx1, dist1, mtx2, dist2, R, T= cv2.stereoCalibrate(objpoints, imgpl, imgpr, mtx1, dist1, mtx2, dist2, 
    gray.shape[::-1], flags=cv2.CALIB_FIX_INTRINSIC & cv2.CALIB_SAME_FOCAL_LENGTH)[1:7]
    return mtx1, mtx2, dist1, dist2, R, T

def undistort(img, mtx, dist, newmtx=True):
    '''
    undistort images and generate new camera matrix
    '''
    size = img.shape[:2]
    size = size[::-1]
    if newmtx:
        newMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, size,1,size)
        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newMtx)
        x,y,w,h = roi
        dst = dst[y:y+h,x:x+w]
        mtx = newMtx
    else:
        dst = cv2.undistort(img, mtx, dist, None, None)
    return dst, mtx

def drawlines(img1, img2, lines, pts1, pts2):
    '''
    draw epipolar lines
    '''
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv2.line(img1, (x0,y0),(x1,y1),color,2)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2

def calLines(iml, imr, numMatches=300, plot=False):
    '''
    draw epipolar lines
    '''
    iml = cv2.cvtColor(iml,cv2.COLOR_RGB2GRAY)
    imr = cv2.cvtColor(imr, cv2.COLOR_RGB2GRAY)
    sift = cv2.xfeatures2d_SIFT.create()
    kp1, des1 = sift.detectAndCompute(iml, None)
    kp2, des2 = sift.detectAndCompute(imr, None)
    matcher = cv2.FlannBasedMatcher()
    match = matcher.match(des1, des2)
    # select the most relevant matches
    match = sorted(match, key=lambda x:x.distance)
    pts1 = []
    pts2 = []
    for m in match[:numMatches]:
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    # select inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, F)
    lines1 = lines1.reshape(-1,3)
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, F)
    lines2 = lines2.reshape(-1,3)
    iml1, iml2 = drawlines(iml, imr, lines1, pts1, pts2)
    imr1, imr2 = drawlines(imr, iml, lines2, pts2, pts1)
    if plot:
        plt.subplot(1,2,1)
        plt.imshow(iml1)
        plt.subplot(1,2,2)
        plt.imshow(imr1)
        plt.show()
    return iml1, imr1

def stereoRectify(iml, imr, mtx1, dist1, mtx2, dist2, R, T):
    size = iml.shape[:2]
    size = size[::-1]
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtx1, dist1, mtx2, dist2, size, R, T)
    mapl1, mapl2 = cv2.initUndistortRectifyMap(mtx1, dist1, R1, P1, (iml.shape[:2])[::-1], cv2.CV_32FC1)
    mapr1, mapr2 = cv2.initUndistortRectifyMap(mtx2, dist2, R2, P2,(imr.shape[:2])[::-1], cv2.CV_32FC1)
    iml = cv2.remap(iml, mapl1, mapl2, cv2.INTER_LINEAR)
    imr = cv2.remap(imr, mapr1, mapr2, cv2.INTER_LINEAR)
    gray1 = cv2.cvtColor(iml, cv2.COLOR_BGR2GRAY)
    for i in range(gray1.shape[0]):
        if not gray1[i,:].min():
            break
    iml = iml[:i, :,:]
    gray2 = cv2.cvtColor(imr, cv2.COLOR_BGR2GRAY)
    for i in range(gray2.shape[0]):
        if not gray2[i,:].min():
            break
    imr = imr[:i, :,:]
    return iml, imr

def save(mtx1, dist1, mtx2, dist2, R, T):
    cameraInfo = {'mtx1':mtx1, 'dist1':dist1, 'mtx2':mtx2, 'dist2':dist2, 'R':R, 'T':T}
    with open('raw_data/cameraInfo.txt','w') as f:
        f.write(str(cameraInfo))
