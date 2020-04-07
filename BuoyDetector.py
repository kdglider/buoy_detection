'''
Copyright (c) 2020 Hao Da (Kevin) Dong, Anshuman Singh
@file       TBD
@date       2020/04/02
@brief      TBD
@license    This project is released under the BSD-3-Clause license.
'''

import numpy as np
import cv2
from GMM import GMM

'''
@brief      
'''
class BuoyDetector:
    yellowGMM = GMM()
    orangeGMM = GMM()
    greenGMM = GMM()
    

    def __init__(self, yellowGMMParams, orangeGMMParams, greenGMMParams):
        self.yellowGMM.load(yellowGMMParams)
        self.orangeGMM.load(orangeGMMParams)
        self.greenGMM.load(greenGMMParams)
    

    def detectBuoys(self, frame):
        image = frame.copy()

        height = image.shape[0]
        width = image.shape[1]

        #print('BP1')

        yellowMask = np.zeros((height, width)).astype('uint8')
        orangeMask = np.zeros((height, width)).astype('uint8')
        greenMask = np.zeros((height, width)).astype('uint8')
        #print('BP2')

        yellowErrors = self.yellowGMM.getLogLikelihoodError(np.reshape(image, (height*width, 3)))
        orangeErrors = self.orangeGMM.getLogLikelihoodError(np.reshape(image, (height*width, 3)))
        greenErrors = self.greenGMM.getLogLikelihoodError(np.reshape(image, (height*width, 3)))

        yellowErrors = np.reshape(yellowErrors, (height, width))
        orangeErrors = np.reshape(orangeErrors, (height, width))
        greenErrors = np.reshape(greenErrors, (height, width))

        for i in range(height):
            for j in range(width):
                yellowError = yellowErrors[i,j]
                orangeError = orangeErrors[i,j]
                greenError = greenErrors[i,j]

                if (yellowError > 11 and orangeError > 14 and greenError > 12.5):
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
            center, radius = cv2.minEnclosingCircle(maxContour)
            cv2.circle(image, (int(center[0]), int(center[1])), int(radius), \
                color=(0, 255, 255), thickness=2)
            cv2.circle(image, (int(center[0]), int(center[1])), 1, \
                color=(0, 0, 255), thickness=1)
            #cv2.drawContours(image, [maxContour], contourIdx=-1, color=(0, 255, 255), thickness=2)

        if (len(orangeContours) != 0):
            maxContour = max(orangeContours, key = cv2.contourArea)
            center, radius = cv2.minEnclosingCircle(maxContour)
            cv2.circle(image, (int(center[0]), int(center[1])), int(radius), \
                color=(0, 125, 255), thickness=2)
            cv2.circle(image, (int(center[0]), int(center[1])), 1, \
                color=(0, 0, 255), thickness=1)
            #cv2.drawContours(image, [maxContour], contourIdx=-1, color=(0, 125, 255), thickness=2)

        if (len(greenContours) != 0):
            maxContour = max(greenContours, key = cv2.contourArea)
            center, radius = cv2.minEnclosingCircle(maxContour)
            cv2.circle(image, (int(center[0]), int(center[1])), int(radius), \
                color=(0, 255, 0), thickness=2)
            cv2.circle(image, (int(center[0]), int(center[1])), 1, \
                color=(0, 0, 255), thickness=1)
            #cv2.drawContours(image, [maxContour], contourIdx=-1, color=(0, 255, 0), thickness=2)
        
        return image
    

    def runApplication(self, videoFile, saveVideo=False):
        # Create video stream object
        videoCapture = cv2.VideoCapture(videoFile)
        
        # Define video codec and output file if video needs to be saved
        if (saveVideo == True):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # 720p 30fps video
            out = cv2.VideoWriter('BuoyDetection.mp4', fourcc, 30, (1280, 720))

        # Continue to process frames if the video stream object is open
        while(videoCapture.isOpened()):
            ret, frame = videoCapture.read()

            # Continue processing if a valid frame is received
            if ret == True:
                newFrame = self.detectBuoys(frame)

                # Save video if desired, resizing frame to 720p
                if (saveVideo == True):
                    out.write(cv2.resize(newFrame, (1280, 720)))
                
                # Display frame to the screen in a video preview
                cv2.imshow("Frame", cv2.resize(newFrame, (1280, 720)))

                # Exit if the user presses 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # If the end of the video is reached, wait for final user keypress and exit
            else:
                cv2.waitKey(0)
                break
        
        # Release video and file object handles
        videoCapture.release()
        if (saveVideo == True):
            out.release()
        
        print('Video and file handles closed')



if __name__ == '__main__':
    yellowGMMParams = 'trained_parameters/yellowGMM.npz'
    orangeGMMParams = 'trained_parameters/orangeGMM.npz'
    greenGMMParams = 'trained_parameters/greenGMM.npz'

    # Select video file and ID of the desired tag to overlay cube on
    videoFile = 'test_set/testVideo.avi'

    # Choose whether or not to save the output video
    saveVideo = False

    # Run application
    buoyDetector = BuoyDetector(yellowGMMParams, orangeGMMParams, greenGMMParams)
    buoyDetector.runApplication(videoFile, saveVideo)

    '''
    image = cv2.imread('test_set/buoys.png')
    output = buoyDetector.detectBuoys(image)
    cv2.imshow('image', output)
    cv2.waitKey(0)
    '''
