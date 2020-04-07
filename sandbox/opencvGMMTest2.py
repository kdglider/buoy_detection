'''
Copyright (c) 2020 Hao Da (Kevin) Dong, Girish Ethirajan, Anshuman Singh
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

samples = np.reshape(image, (height*width, 3))
samples = samples.astype('float')

yellowGMM = GMM()
yellowGMM.load('yellowGMM.npz')

greenGMM = GMM()
greenGMM.load('greenGMM.npz')

orangeGMM = GMM()
orangeGMM.load('orangeGMM.npz')

totalProbs = np.zeros(samples.shape[0])

for i in range(samples.shape[0]):
    totalProbs[i] = orangeGMM.getLogLikelihoodError(samples[i])

maxProb = max(totalProbs)
minProb = min(totalProbs)

print(maxProb)
print(minProb)

output = image.copy()
output = np.reshape(output, (height*width, 3))

for i in range(len(output)):
    if (totalProbs[i] > 14):
        output[i] = np.array([0,0,0])

output = np.reshape(output, (height, width, 3))

cv2.imshow('Image', image)
cv2.imshow('Output', output)
cv2.waitKey(0)

# Yellow threshold = 15
# Green threshold = 12
# Orange threshold = 14

