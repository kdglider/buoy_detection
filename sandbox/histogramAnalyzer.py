'''
Copyright (c) 2020 Hao Da (Kevin) Dong, Girish Ethirajan, Anshuman Singh
@file       histogramAnalyzer.py
@date       2020/04/02
@brief      TBD
@license    This project is released under the BSD-3-Clause license.
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('buoys.png')

height = image.shape[0]
width = image.shape[1]

imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

'''
dataset = np.reshape(imageHSV, (height*width, 3))

gmm = cv2.ml.EM_create()
gmm.setClustersNumber(5)
gmm.trainEM(dataset)
gmm.save('a.txt')
'''

gmm = cv2.ml.EM_load('hsvParams.txt')

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
cv2.imshow('HSV Image', imageHSV)
cv2.imshow('Output', output)
cv2.waitKey(0)
