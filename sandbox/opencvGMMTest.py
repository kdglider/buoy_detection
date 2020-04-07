'''
Copyright (c) 2020 Hao Da (Kevin) Dong, Anshuman Singh
@file       opencvGMMTest.py
@date       2020/04/02
@brief      TBD
@license    This project is released under the BSD-3-Clause license.
'''

import numpy as np
import cv2


image = cv2.imread('training_set/yellowTrainingSet.png')

height = image.shape[0]
width = image.shape[1]

'''

trainingSet = np.load('training_set/yellowTrainingSet.npy')

gmm = cv2.ml.EM_create()
gmm.setClustersNumber(2)
gmm.trainEM(trainingSet)
gmm.save('a.txt')

'''

gmm = cv2.ml.EM_load('yellowGMM.txt')

means = gmm.getMeans()

#output = np.zeros((height, width, 3))
output = image.copy()

for i in range(height):
    for j in range(width):
        result = gmm.predict2(image[i,j,:])
        cluster = result[0][1]
        output[i,j,:] = means[int(cluster), :]

output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)

cv2.imshow('Image', image)
cv2.imshow('Output', output)
cv2.waitKey(0)

