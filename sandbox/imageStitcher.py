'''
Copyright (c) 2020 Hao Da (Kevin) Dong
@file       imageStitcher.py
@date       2020/04/02
@brief      Horizontally stacks images of different sizes into one image by adding padding
@license    This project is released under the BSD-3-Clause license.
'''

import numpy as np
import cv2
import glob

# Lists to store images and their heights
imageList = []
imageHeights = []

# Read in each image and its height
for filename in glob.glob('training_set/green*.png'):
    image = cv2.imread(filename)
    height, width, layers = image.shape
    imageList.append(image)
    imageHeights.append(height)

# Determine max height of the images
maxHeight = max(imageHeights)

# Add black padding underneath images to make then the same height
for i in range(len(imageList)):
    image = imageList[i]
    height, width, layers = image.shape

    paddingThickness = maxHeight - height
    imageList[i] = cv2.copyMakeBorder(image, top=0, bottom=paddingThickness, \
        left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))

# Horizontally stack images and save result
output = np.hstack(tuple(imageList))
cv2.imwrite('output.png', output)

