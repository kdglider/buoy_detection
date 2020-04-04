'''
Copyright (c) 2020 Hao Da (Kevin) Dong, Girish Ethirajan, Anshuman Singh
@file       LaneDetection.py
@date       2020/03/08
@brief      Lane detection application
@license    This project is released under the BSD-3-Clause license.
'''

import numpy as np
import cv2

image = cv2.imread('green.png')

height = image.shape[0]
width = image.shape[1]

'''
dataset = np.reshape(image, (height*width, 3))

gmm = cv2.ml.EM_create()
gmm.setClustersNumber(2)
gmm.trainEM(dataset)
gmm.save('a.txt')
'''

gmm = cv2.ml.EM_load('orangeParams.txt')

means = gmm.getMeans()

#output = np.zeros((height, width, 3))
output = image.copy()

for i in range(height):
    for j in range(width):
        result = gmm.predict2(image[i,j,:])
        cluster = result[0][1]
        output[i,j,:] = means[int(cluster), :]

print(image)
print(output)

cv2.imshow('Image', image)
cv2.imshow('Output', output)
cv2.waitKey(0)
