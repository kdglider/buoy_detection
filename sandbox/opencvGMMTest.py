'''
Copyright (c) 2020 Hao Da (Kevin) Dong, Girish Ethirajan, Anshuman Singh
@file       LaneDetection.py
@date       2020/03/08
@brief      Lane detection application
@license    This project is released under the BSD-3-Clause license.
'''

import numpy as np
import cv2

gmmObject = cv2.ml.EM_create()