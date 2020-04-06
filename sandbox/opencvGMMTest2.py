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


#image = cv2.imread('training_set/yellowTrainingSet.png')
image = cv2.imread('test_set/buoys.png')

height = image.shape[0]
width = image.shape[1]

samples = np.reshape(image, (height*width, 3))

samples = samples.astype('float')

'''

trainingSet = np.load('training_set/yellowTrainingSet.npy')

gmm = cv2.ml.EM_create()
gmm.setClustersNumber(2)
gmm.trainEM(trainingSet)
gmm.save('a.txt')

'''

gmm = cv2.ml.EM_load('yellowGMM.txt')

means = gmm.getMeans()
covs = gmm.getCovs()
weights = gmm.getWeights()
gaussianList = []

for i in range(len(means)):
    gaussianList.append(multivariate_normal(means[i], covs[i]))

def predict(sample, weights, gaussianList):
    probs = np.zeros((len(gaussianList),1))

    for i in range(len(means)):
        probs[i,0] = gaussianList[i].pdf(sample)
    
    #print(np.matmul(weights, probs))
    return np.matmul(weights, probs)

totalProbs = np.zeros(samples.shape[0])

for i in range(samples.shape[0]):
    totalProbs[i] = np.log(predict(samples[i], weights, gaussianList))

maxProb = max(totalProbs)
minProb = min(totalProbs)

print(maxProb)
print(minProb)

output = image.copy()
output = np.reshape(output, (height*width, 3))

for i in range(len(output)):
    if (totalProbs[i] < -15):
        output[i] = np.array([0,0,0])

output = np.reshape(output, (height, width, 3))

cv2.imshow('Image', image)
cv2.imshow('Output', output)
cv2.waitKey(0)

