# -*- encoding: utf-8 -*-
'''
@File    :   VISPEstimator.py
@Time    :   2020/05/19 09:44:49
@Author  :   Haokun Wang 
@Version :   1.0
@Contact :   wanghk@mail.sustech.edu.cn
@License :   (C)Copyright 2020
@Desc    :   None
'''
import os
import cv2
import numpy as np
from .Detector import AprilTagDetector




class VISPEstimator(object):
    def __init__(self, length=0.02):
        self.detector = AprilTagDetector(length, 1.0)

    def loadYAML(self, path: str):
        self.detector.LoadCameraParametersYAML(path)
    
    def saveYAML(self, path: str):
        self.detector.SaveCameraParametersYAML(path)

    def getDetectorOutput(self, image: np.array):
        # print(type(cv2.fromarray(image)))
        return self.detector.GetOutPut(image)

    def getDetectorOutputFromDepth(self, depth_image, color_image):
        return self.detector.GetOutPutFromDepth(depth_image, color_image)