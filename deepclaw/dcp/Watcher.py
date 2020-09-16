# Copyright (c) 2020 by BionicLab. All Rights Reserved.
# !/usr/bin/python
# -*- coding:utf-8 -*-
"""
@File: Master
@Author: Haokun Wang
@Date: 2020/4/3 14:29
@Description: 
"""
class Watcher:
    FLAG = True

    def __init__(self, master, camera):
        self.master = master
        self.camera = camera

    def run(self):
        self.FLAG = True
        while self.FLAG:
            frame = self.camera.get_frame()
            color_image = frame.color_image[0]
            depth_image = frame.depth_image[0]
            point_cloud = frame.point_cloud[0]
            self.master.update({'RGB': color_image, 'DepthInfo': depth_image, 'PointCloud': point_cloud})

    def notify(self, original_data):
        self.master.update(original_data)

    def stop(self):
        self.FLAG = False
