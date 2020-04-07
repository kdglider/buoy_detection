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

        yellowMask = np.zeros((height, width)).astype('uint8')
        orangeMask = np.zeros((height, width)).astype('uint8')
        greenMask = np.zeros((height, width)).astype('uint8')

        for i in range(height):
            for j in range(width):
                yellowError = self.yellowGMM.getLogLikelihoodError(image[i,j])
                orangeError = self.orangeGMM.getLogLikelihoodError(image[i,j])
                greenError = self.greenGMM.getLogLikelihoodError(image[i,j])

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
                print(frame)
                #cv2.imshow("Old Frame", cv2.resize(frame, (1280, 720)))
                # Overlay cube on tag
                newFrame = self.detectBuoys(frame)
                print(newFrame)

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
    yellowGMMParams = 'yellowGMM.npz'
    orangeGMMParams = 'orangeGMM.npz'
    greenGMMParams = 'greenGMM.npz'

    # Select video file and ID of the desired tag to overlay cube on
    videoFile = '../testVideo.avi'

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
