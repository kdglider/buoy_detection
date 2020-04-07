'''
Copyright (c) 2020 Hao Da (Kevin) Dong, Anshuman Singh
@file       opencvGMMTest2.py
@date       2020/04/02
@brief      TBD
@license    This project is released under the BSD-3-Clause license.
'''

import numpy as np
import cv2
from scipy.stats import multivariate_normal
from GMM import GMM


#image = cv2.imread('training_set/yellowTrainingSet.png')
image = cv2.imread('test_set/buoys.png')

height = image.shape[0]
width = image.shape[1]

yellowGMM = GMM()
yellowGMM.load('yellowGMM.npz')

orangeGMM = GMM()
orangeGMM.load('orangeGMM.npz')

greenGMM = GMM()
greenGMM.load('greenGMM.npz')

'''
yellowErrors = np.zeros((height, width))
orangeErrors = np.zeros((height, width))
greenErrors = np.zeros((height, width))

for i in range(height):
    for j in range(width):
        yellowErros[i] = yellowGMM.getLogLikelihoodError(image[i,j])
        orangeErros[i] = orangeGMM.getLogLikelihoodError(image[i,j])
        greenErros[i] = greenGMM.getLogLikelihoodError(image[i,j])
'''

yellowMask = np.zeros((height, width)).astype('uint8')
orangeMask = np.zeros((height, width)).astype('uint8')
greenMask = np.zeros((height, width)).astype('uint8')

for i in range(height):
    for j in range(width):
        yellowError = yellowGMM.getLogLikelihoodError(image[i,j])
        orangeError = orangeGMM.getLogLikelihoodError(image[i,j])
        greenError = greenGMM.getLogLikelihoodError(image[i,j])

        if (yellowError > 12 and orangeError > 15 and greenError > 12):
            continue
        elif (yellowError == min(yellowError, orangeError, greenError)):
            yellowMask[i,j] = 255
        elif (orangeError == min(yellowError, orangeError, greenError)):
            orangeMask[i,j] = 255
        elif (greenError == min(yellowError, orangeError, greenError)):
            greenMask[i,j] = 255

yellowMask = cv2.erode(yellowMask, None, iterations=1)
yellowMask = cv2.dilate(yellowMask, None, iterations=2)

orangeMask = cv2.erode(orangeMask, None, iterations=1)
orangeMask = cv2.dilate(orangeMask, None, iterations=2)

greenMask = cv2.erode(greenMask, None, iterations=1)
greenMask = cv2.dilate(greenMask, None, iterations=2)

yellowContours, hierarchy = cv2.findContours(yellowMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
orangeContours, hierarchy = cv2.findContours(orangeMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
greenContours, hierarchy = cv2.findContours(greenMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

if (len(yellowContours) != 0):
    maxContour = max(yellowContours, key = cv2.contourArea)
    cv2.drawContours(image, [maxContour], contourIdx=-1, color=(0, 255, 255), thickness=2)

if (len(orangeContours) != 0):
    maxContour = max(orangeContours, key = cv2.contourArea)
    cv2.drawContours(image, [maxContour], contourIdx=-1, color=(0, 125, 255), thickness=2)

if (len(greenContours) != 0):
    maxContour = max(greenContours, key = cv2.contourArea)
    cv2.drawContours(image, [maxContour], contourIdx=-1, color=(0, 255, 0), thickness=2)

cv2.imshow('Image', image)
cv2.imshow('yellowMask', yellowMask)
cv2.imshow('orangeMask', orangeMask)
cv2.imshow('greenMask', greenMask)
cv2.waitKey(0)

# Yellow threshold = 15
# Green threshold = 12
# Orange threshold = 14

