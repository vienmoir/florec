# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 23:30:39 2017
@author: Daria
"""
#from __future__ import division
import numpy as np
import cv2
import math
from skimage.measure import regionprops
from scipy import stats as stts
from leafCheck import leafCheck
#from countHWC import CountHeightWidthCoord
#import pandas as pd

def process(checkedImage,cnt,coord):
    # Центроид
    props = regionprops(checkedImage)
    x1 = props[0].centroid[1]
    y1 = props[0].centroid[0]

    # Координаты точек контура
    _, contours, _ = cv2.findContours(checkedImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]

    N = len(cnt)
    dist = []
    for i in range(N):
        p = cnt[i]
        x2 = p.item(0)
        y2 = p.item(1)
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (.5)
        dist.append(distance)

    start = 0
    point = np.where(np.flip(cnt, 2) == coord)

    for i in range(len(np.unique(point[0]))):
        if (np.flip(cnt[np.unique(point[0])[i]], 1) == coord)[0][0] == True & \
                (np.flip(cnt[np.unique(point[0])[i]], 1) == coord)[0][1] == True:
            start = np.unique(point[0])[i]

    newdist = dist[start:N] + dist[0:start]
    arr = np.asarray(newdist)
    return arr