# -*- encoding: utf-8 -*-
'''
@File    :   ManipulationTasks.py
@Time    :   2020/06/05 14:34:47
@Author  :   Haokun Wang 
@Version :   1.0
@Contact :   wanghk@mail.sustech.edu.cn
@License :   (C)Copyright 2020
@Desc    :   None
'''
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from dcp.functions import *


def get_a(q):
    if len(q)<2:
        return q, 0
    elif len(q)==2:
        value_0 = q.pop(0)
        return q, (q[0]-value_0)/30.0

def get_v(q):
    if len(q)<2:
        return q, 0
    elif len(q)==2:
        pose_0 = q.pop(0)
        # print(q[0][:3], pose_0[:3])
        # print(q[0][:3] - pose_0[:3])
        v = np.linalg.norm(q[0][:3] - pose_0[:3])
        return q, v
        

class Task(object):
    FRAME = {}
    
    def __init__(self):
        pass

    def reset_frame(self):
        self.FRAME['header'] = {}
        self.FRAME['level_1'] = {}
        self.FRAME['level_2'] = {}
        self.FRAME['level_3'] = {}
        self.FRAME['level_4'] = {}
        self.FRAME['level_5'] = {}


class Experiment(Task):
    count = 0
    pose_queue = []
    velocity_queue = []

    def __init__(self):
        super(Experiment, self).__init__()

    def obtain_data(self, original_data, parameters):
        # time stamp
        time_s = time.localtime(int(time.time()))
        self.count += 1
        
        # decode original data
        color_image, depth_image, point_cloud = original_data['RGB'], original_data['DepthInfo'], original_data['PointCloud']

        # markers detection
        results = parameters['estimator'].getDetectorOutputFromDepth(depth_image, color_image)
        markers = get_pose_from_estimator(results)
        
        
        # obtain gripper information
        rvec_r, tvec_r, rvec_l, tvec_l = obtain_boards_from_markers(markers)
        r_world_h = present_in_world(rvec_r, tvec_r, parameters['inv_base_camera_h'])
        l_world_h = present_in_world(rvec_l, tvec_l, parameters['inv_base_camera_h'])
        # r_rpyt = h2rpyt(r_world_h)  # level 3 info
        # l_rpyt = h2rpyt(l_world_h)  # level 3 info

        tcp = obtain_tcp(r_world_h, l_world_h)  # level 4 info

        # obtain finger point
        finger_markers = screen_markers(markers, 5, 8)
        right_finger, left_finger = [], []

        # object information
        object_markers = screen_markers(markers, 10, 19)

        # pack data
        self.reset_frame()

        self.FRAME['header'] = {'timestamp': str(time_s.tm_mon) + str(time_s.tm_mday) + str(time_s.tm_hour) + \
                                             str(time_s.tm_min) + str(time_s.tm_sec) + str(self.count)}
        self.FRAME['level_1'] = {'RGB': './rgb/'+str(self.count)+'.png', 'Depth_Information': './depth/'+str(self.count)+'.png'}
        self.FRAME['level_5']['current_state'] = {}
        self.FRAME['level_5']['current_action'] = {}
        if len(markers)!=0:
            for marker in markers:
                self.FRAME['level_2']['tag_'+str(marker[0])] = marker[3:]
        if len(object_markers) != 0:
            for obj_marker in object_markers:
                # self.FRAME['level_2']['object_'+str(obj_marker[0])] = obj_marker[3:]
                obj_world_h = present_in_world(obj_marker[1], obj_marker[2], parameters['inv_base_camera_h'])
                # obj_rpyt = h2rpyt(obj_world_h)  # level 3 info
                self.FRAME['level_3']['object_'+str(obj_marker[0])] = [obj_world_h, obj_marker[1], obj_marker[2]]
                self.FRAME['level_5']['current_state']['object_'+str(obj_marker[0])] = [obj_world_h]
        if len(finger_markers)!=0:
            for finger_marker in finger_markers:
                if 5<=finger_marker[0]<=6:
                    left_finger.append(finger_marker)
                    finger_world_h = present_in_world(finger_marker[1], finger_marker[2], parameters['inv_base_camera_h'])
                    # finger_rpyt = h2rpyt(finger_world_h)
                    self.FRAME['level_3']['left_finger_'+str(finger_marker[0])] = [finger_world_h, finger_marker[1], finger_marker[2]]
                if 7<=finger_marker[0]<=8:
                    right_finger.append(finger_marker)
                    finger_world_h = present_in_world(finger_marker[1], finger_marker[2], parameters['inv_base_camera_h'])
                    # finger_rpyt = h2rpyt(finger_world_h)
                    self.FRAME['level_3']['right_finger_'+str(finger_marker[0])] = [finger_world_h, finger_marker[1], finger_marker[2]]
        if r_world_h is not None:
            self.FRAME['level_3']['right_board_pose'] = [r_world_h, rvec_r, tvec_r]
        if l_world_h is not None:
            self.FRAME['level_3']['left_board_pose'] = [l_world_h, rvec_l, tvec_l]  # Object: [array(x,y,z), array(row, pitch, yaw)]
        if tcp is not None:
            self.FRAME['level_4']['tool_center_point_pose'] = tcp  # ToolCenterPoint: array(x,y,z,row,pitch,yaw)
            self.FRAME['level_5']['current_state']['tool_pose'] = tcp
            self.pose_queue.append(np.array(list(tcp[0])+list(tcp[1][0])))
            self.pose_queue, v = get_v(self.pose_queue)
            self.velocity_queue.append(v)
            self.velocity_queue, a = get_a(self.velocity_queue)
            self.FRAME['level_5']['current_action']['tool_acceleration'] = a
        if len(left_finger)==2 and len(right_finger)==2:
            self.FRAME['level_4']['finger_distance_1'] = obtain_distance(left_finger[0][2], right_finger[0][2])
            self.FRAME['level_4']['finger_distance_2'] = obtain_distance(left_finger[1][2], right_finger[1][2])
            # self.FRAME['level_5']['current_state']['distance_1'] = obtain_distance(left_finger[0][2], right_finger[0][2])
            # self.FRAME['level_5']['current_state']['distance_2'] = obtain_distance(left_finger[1][2], right_finger[1][2])
        if tvec_r is not None and tvec_l is not None:
            self.FRAME['level_4']['finger_distance_3'] = obtain_distance(tvec_r, tvec_l)
            self.FRAME['level_5']['current_state']['distance_3'] = obtain_distance(tvec_r, tvec_l)
        if len(left_finger)==2 and len(right_finger)==2 and tvec_r is not None and tvec_l is not None:
            self.FRAME['level_5']['current_action']['d13'] = obtain_distance(left_finger[0][2], right_finger[0][2]) - obtain_distance(tvec_r, tvec_l)
            self.FRAME['level_5']['current_action']['d23'] = obtain_distance(left_finger[1][2], right_finger[1][2]) - obtain_distance(tvec_r, tvec_l)

        return {'idx': self.count, 'data_frame': self.FRAME}

