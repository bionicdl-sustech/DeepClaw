# Copyright (c) 2019 by Hank. All Rights Reserved.
# !/usr/bin/python
# coding=utf-8
import os
import sys
import numpy as np
import pyrealsense2 as rs

_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_root_path)

from driver.sensors.camera.CameraController import CameraController


class RealsenseController(CameraController):
    def __init__(self, configuration):
        super(RealsenseController, self).__init__()
        self.width = configuration["FRAME_ARGS"]["width"]
        self.height = configuration["FRAME_ARGS"]["height"]
        self.fps = configuration["FRAME_ARGS"]["fps"]
        self.serial_id = configuration["DEVICE_CONFIGURATION"]["serial_id"]

        self.points = rs.points()
        self.pipeline = rs.pipeline()
        config = rs.config()
        if self.serial_id != '':
            config.enable_device(serial=self.serial_id)
        config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, self.fps)
        config.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, self.fps)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        self.profile = self.pipeline.start(config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        for i in range(45):
            self.getImage()

    def getImage(self):
        frames = self.pipeline.wait_for_frames()
        irL_frame = frames.get_infrared_frame(1)
        irR_frame = frames.get_infrared_frame(2)
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not aligned_depth_frame or not color_frame:
            return None, None, None, None
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        image_L = np.asanyarray(irL_frame.get_data())
        image_R = np.asanyarray(irR_frame.get_data())
        return color_image, [depth_image, image_L, image_R]

    def get_device(self):
        return self.profile.get_device()

    def get_depth_scale(self):
        depth_sensor = self.profile.get_device().first_depth_sensor()
        return depth_sensor.get_depth_scale()

    def get_serial_number(self):
        device = self.get_device()
        return str(device.get_info(rs.camera_info.serial_number))

    def get_intrinsics(self):
        color_stream = self.profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        return intrinsics
