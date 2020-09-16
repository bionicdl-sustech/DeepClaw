# -*- encoding: utf-8 -*-
'''
@File    :   demo.py
@Time    :   2020/06/02 15:01:31
@Author  :   Haokun Wang 
@Version :   1.0
@Contact :   wanghk@mail.sustech.edu.cn
@License :   (C)Copyright 2020
@Desc    :   None
'''
import os
import sys
import cv2
import png
import time
import json
import copy
import datetime
import threading
import numpy as np
from queue import Queue
from cv2 import aruco
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QStringListModel
from PyQt5.QtWidgets import QApplication, QWidget

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from deepclaw.driver.sensors.camera.Realsense import Realsense

from dcp.Watcher import Watcher
from dcp.Master import Master
from dcp.GUI import Ui_Form
from dcp.ManipulationTasks import Experiment
from dcp.functions import *
from dcp.utils.VISPEstimator import VISPEstimator


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class DataCollectionPlatform(object):
    FLAG = True
    recording_flag = False
    data_buffer = Queue()
    recording_buffer = Queue()
    parameters = {}
    
    def __init__(self):
        self.master = Master()

    def init_devices(self):
        self.estimator = VISPEstimator(0.023)
        self.estimator.saveYAML(ROOT+'/configs/sim-none-rectangle-d435/estimator.yaml')
        self.estimator.loadYAML(ROOT+'/configs/sim-none-rectangle-d435/estimator.yaml')
        self.camera = Realsense(ROOT+'/configs/sim-none-rectangle-d435/d435.yaml')
        fx, fy, ppx, ppy, distortion = self.camera.get_intrinsics()
        self.camera_matrix = np.array([[fx, 0, ppx],
                                       [0, fy, ppy],
                                       [0, 0, 1]])
        self.distortion = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.parameters['camera_matrix'] = self.camera_matrix
        self.parameters['distortion'] = self.distortion
        self.parameters['estimator'] = self.estimator

    def init_environment(self):
        self.board_size = (7, 4)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.board_points = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
        self.board_points[:, 0] = 0.0125*np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)[:, 1]
        self.board_points[:, 1] = 0.0125*np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)[:, 0]
        # calibration
        base_r, base_t = self.calibrate()
        r_matrix = cv2.Rodrigues(base_r)[0]
        base_camera_h = np.vstack((np.hstack((r_matrix, base_t)), np.array([[0, 0, 0, 1]])))
        inv_base_camera_h = np.linalg.inv(base_camera_h)  # [:3, :]
        # print(inv_base_camera_h.dot(base_camera_h))
        self.parameters['inv_base_camera_h'] = inv_base_camera_h

    def init_task(self):
        self.task = Experiment()

    def start_recording(self):
        self.count = 0
        # time_s = time.localtime(int(time.time()))
        # self.experiment_name = ROOT+"/data/experiment_" + str(time_s.tm_mon) + str(time_s.tm_mday) + \
        #                   str(time_s.tm_hour) + str(time_s.tm_min) + str(time_s.tm_sec)
        time_stamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.experiment_name = ROOT+"/data/experiment_" + time_stamp

        if not os.path.exists(self.experiment_name):
            os.makedirs(self.experiment_name)
            os.makedirs(self.experiment_name+'/rgb')
            os.makedirs(self.experiment_name+'/depth')
            os.makedirs(self.experiment_name+'/point_cloud')
        self.json_file = open(self.experiment_name+"/data.json", 'w')
        self.recording_flag = True
        recording_thread = threading.Thread(target=self.save_frame, args=())
        recording_thread.start()
    
    def calibrate(self):
        last_rvecs = np.zeros((3, 1))
        last_tvecs = np.zeros((3, 1))
        count = 0
        while True:
            frame = self.camera.get_frame()
            color_image = frame.color_image[0]

            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.board_size, None)

            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), self.criteria)
                _, rvecs, tvecs, inliers = cv2.solvePnPRansac(self.board_points, corners2,
                                                              self.camera_matrix, self.distortion)
                if sum(abs(last_tvecs-tvecs)) <= 0.1 and sum(abs(last_rvecs-rvecs)) <= 0.05:
                    count += 1
                    if count >= 5:
                        return rvecs, tvecs
                last_rvecs = rvecs
                last_tvecs = tvecs
    
    def run(self):
        self.FLAG = True
        self.init_devices()
        self.init_environment()
        self.init_task()
        self.watcher = Watcher(self.master, self.camera)
        # detection thread starts
        watcher_thread = threading.Thread(target=self.watcher.run, args=())
        watcher_thread.start()

        while self.FLAG:
            # get original data from master, data update in 30 FPS
            original_data = self.master.get_data()
            data_frame = self.task.obtain_data(copy.deepcopy(original_data), self.parameters)
            if self.recording_flag:
                self.recording_buffer.put([copy.deepcopy(original_data), copy.deepcopy(data_frame)])
            self.data_buffer.put([copy.deepcopy(original_data), copy.deepcopy(data_frame)])

        self.watcher.stop()
        self.camera.pipeline.stop()

    def save_frame(self):
        while self.recording_flag:
            data = self.recording_buffer.get()
            original_data, frame = data[0], data[1]
            idx, data_frame = frame['idx'], frame['data_frame']
            color_image, depth_image, point_cloud = original_data['RGB'], original_data['DepthInfo'], original_data['PointCloud']
            json_data = json.dumps(data_frame, cls=JsonEncoder)
            self.json_file.write(json_data)
            self.json_file.write('\n')
            if color_image is not None:
                cv2.imwrite(self.experiment_name+'/rgb/'+str(idx)+'.png', color_image)
            if depth_image is not None:
                with open(self.experiment_name+'/depth/'+str(idx)+'.png') as depth_file:
                    writer = png.Writer(width=depth_image.shape[1], height=depth_image.shape[0], bitdepth=16, greyscale=True)
                    zgray2list = depth_image.tolist()
                    writer.write(depth_file, zgray2list)
        self.json_file.close()
    
    def stop_recording(self):
        self.recording_flag = False
    
    def stop(self):
        self.FLAG = False


class GUI(QWidget, Ui_Form):
    def __init__(self):
        super(GUI, self).__init__()
        self.setupUi(self)
        self.feature_model = QStringListModel(self)
        self.object_model = QStringListModel(self)
        self.meaning_model = QStringListModel(self)
        self.state_model = QStringListModel(self)
        self.action_model = QStringListModel(self)

        self.camera.toggled.connect(self.camera_state_func)
        self.record.toggled.connect(self.record_state_func)

        self.dcp = DataCollectionPlatform()

    def camera_state_func(self):
        if self.camera.isChecked():
            self.camera.setIcon(QtGui.QIcon('./gui/icons/no-camera.png'))
            self.camera.setIconSize(QtCore.QSize(25, 25))
            main_thread = threading.Thread(target=self.dcp.run, args=())
            main_thread.start()

            self.spy_thread = SpyThread(self.dcp)
            self.spy_thread.rgb_message_signal.connect(self.set_rgb_msg)
            self.spy_thread.features_message_signal.connect(self.update_features)
            self.spy_thread.objects_message_signal.connect(self.update_objects)
            self.spy_thread.meanings_message_signal.connect(self.update_meanings)
            self.spy_thread.sa_message_signal.connect(self.update_sa)
            self.spy_thread.start()
        else:
            self.camera.setIcon(QtGui.QIcon('./gui/icons/camera.png'))
            self.camera.setIconSize(QtCore.QSize(25, 25))
            self.spy_thread.stop()
            self.dcp.stop()
            self.clear_state()

    def record_state_func(self):
        if self.camera.isChecked():
            if self.record.isChecked():
                self.record.setIcon(QtGui.QIcon("./gui/icons/stop.png"))
                self.camera.setIconSize(QtCore.QSize(25, 25))
                self.dcp.start_recording()
            else:
                self.record.setIcon(QtGui.QIcon("./gui/icons/record.png"))
                self.camera.setIconSize(QtCore.QSize(25, 25))
                self.dcp.stop_recording()
        else:
            print('Please open your camera before recording.')


    def clear_state(self):
        self.rgb_displayer.setPixmap(QPixmap(""))
    
    def set_rgb_msg(self, rgb_msg: QImage):
        self.rgb_displayer.setPixmap(QPixmap.fromImage(rgb_msg))

    def update_features(self, features_msg: list):
        self.feature_model.setStringList(features_msg)
        self.listView.setModel(self.feature_model)

    def update_objects(self, objects_msg: list):
        self.object_model.setStringList(objects_msg)
        self.listView_2.setModel(self.object_model)

    def update_meanings(self, meanings_msg: list):
        self.meaning_model.setStringList(meanings_msg)
        self.listView_3.setModel(self.meaning_model)

    def update_sa(self, sa_msg: list):
        state_msg = sa_msg[0]
        action_msg = sa_msg[1]
        self.state_model.setStringList(state_msg)
        self.listView_4.setModel(self.state_model)
        self.action_model.setStringList(action_msg)
        self.listView_5.setModel(self.action_model)


class SpyThread(QThread):
    FLAG = True
    rgb_message_signal = pyqtSignal(QImage)
    features_message_signal = pyqtSignal(list)
    objects_message_signal = pyqtSignal(list)
    meanings_message_signal = pyqtSignal(list)
    sa_message_signal = pyqtSignal(list)

    def __init__(self, data_handle_bar):
        super(SpyThread, self).__init__()
        self.handle_bar = data_handle_bar

    def run(self):
        while self.FLAG:
            data = self.handle_bar.data_buffer.get()
            original_data, frame = data[0], data[1]['data_frame']
            color_image = original_data['RGB']
            # color_image = self.level2_process(frame, color_image)
            color_image = self.level3_process(frame, color_image)
            self.rgb_emit(color_image)
            self.features_emit(frame)
            self.objects_emit(frame)
            self.meanings_emit(frame)
            self.sa_emit(frame)
        self.clear_state()
            
    
    def rgb_emit(self, color_image):
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        convertToQtFormat = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], QImage.Format_RGB888)
        p = convertToQtFormat.scaled(800, 450, Qt.KeepAspectRatio)
        self.rgb_message_signal.emit(p)

    def level2_process(self, frame, color_image):
        if frame['level_2']:
            ids, corners = [], []
            for key in frame['level_2']:
                ids.append([int(key)])
                corners.append(np.array([list(frame['level_2'][key])]))
            color_image = aruco.drawDetectedMarkers(color_image, corners, np.array(ids))
        return color_image

    def level3_process(self, frame, color_image):
        if frame['level_3']:
            for key in frame['level_3']:
                rvec, tvec = frame['level_3'][key][1], frame['level_3'][key][2]
                color_image = aruco.drawAxis(color_image, self.handle_bar.camera_matrix, self.handle_bar.distortion, rvec, tvec, 0.05)
        return color_image
    
    def features_emit(self, frame):
        msg = []
        if frame['level_2']:
            for key in frame['level_2']:
                value = frame['level_2'][key]
                msg.append(key+": "+",".join([str(i) for i in value]))
        self.features_message_signal.emit(msg)

    def objects_emit(self, frame):
        msg = []
        if frame['level_3']:
            for key in frame['level_3']:
                rpyt = h2rpyt(frame['level_3'][key][0])
                msg.append(key+": "+",".join([str(i) for i in rpyt]))
        self.objects_message_signal.emit(msg)

    def meanings_emit(self, frame):
        msg = []
        if frame['level_4']:
            for key in frame['level_4']:
                if key == 'tool_center_point_pose':
                    msg.append(key+": "+",".join([str(i) for i in frame['level_4'][key]]))
                else:
                    msg.append(key+": "+str(frame['level_4'][key]))
        self.meanings_message_signal.emit(msg)

    def sa_emit(self, frame):
        msg = []
        state_msg = []
        action_msg = []
        if frame['level_5']:
            for key in frame['level_5']['current_state']:
                state_msg.append(key+": "+str(frame['level_5']['current_state'][key]))
            for key in frame['level_5']['current_action']:
                action_msg.append(key+": "+str(frame['level_5']['current_action'][key]))
        msg.append(state_msg)
        msg.append(action_msg)
        self.sa_message_signal.emit(msg)

    def clear_state(self):
        msg = []
        self.features_message_signal.emit(msg)
        self.objects_message_signal.emit(msg)
        self.meanings_message_signal.emit(msg)
        self.sa_message_signal.emit([[],[]])

    def stop(self):
        self.FLAG = False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    demo = GUI()
    demo.show()
    sys.exit(app.exec_())
