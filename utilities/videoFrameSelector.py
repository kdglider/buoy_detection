'''
Copyright (c) 2020 Hao Da (Kevin) Dong
@file       videoFrameSelector.py
@date       2020/04/02
@brief      Plays through a video stream and enables user to save selected frames
@license    This project is released under the BSD-3-Clause license.
'''

import cv2
 
# Create video capture object
videoCapture = cv2.VideoCapture('../testVideo.avi')

# Index to keep track of saved frames
i = 1

while(videoCapture.isOpened()):
    ret, frame = videoCapture.read()

    if ret == False:
        break
    
    # Display frame
    cv2.imshow('Frame', frame)

    # Wait for user keypress
    key = cv2.waitKey(0)

    # Exit if the user presses 'q'
    if key & 0xFF == ord('q'):
        break

    # Save the frame if user presses 's'
    elif key & 0xFF == ord('s'):
        cv2.imwrite('frame'+str(i)+'.png', frame)
        i += 1
    
    # Else, display next frame
    else:
        continue

# Release video handle
videoCapture.release()

