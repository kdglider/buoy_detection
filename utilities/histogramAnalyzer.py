'''
Copyright (c) 2020 Hao Da (Kevin) Dong
@file       histogramAnalyzer.py
@date       2020/04/02
@brief      Creates a histogram for each colour channel using the training images
@license    This project is released under the BSD-3-Clause license.
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt


trainingImage = cv2.imread('training_set/greenTrainingSet.png')
trainingSet = []

# Append pixel to trainingSet if it is not black
for row in trainingImage:
    for pixel in row:
        if (np.all(pixel == 0) == False):
            trainingSet.append(pixel)

# Convert to NumPy array
trainingSet = np.array(trainingSet)

# Plot histograms
fig = plt.figure()
sp1 = fig.add_subplot(3,1,1)
sp1.hist(trainingSet[:,0], bins=255)
sp1.set_title('Blue Channel Histogram')
sp1.set_xlabel('Blue Channel Value')
sp1.set_ylabel('Number of Pixels')

sp2 = fig.add_subplot(3,1,2)
sp2.hist(trainingSet[:,1], bins=255)
sp2.set_title('Green Channel Histogram')
sp2.set_xlabel('Green Channel Value')
sp2.set_ylabel('Number of Pixels')

sp3 = fig.add_subplot(3,1,3)
sp3.hist(trainingSet[:,2], bins=255)
sp3.set_title('Red Channel Histogram')
sp3.set_xlabel('Red Channel Value')
sp3.set_ylabel('Number of Pixels')

# Display all plots
plt.tight_layout()
plt.show()

# Save the training set
np.save('TrainingSet', trainingSet)
