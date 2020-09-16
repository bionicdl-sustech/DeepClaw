# -*- encoding: utf-8 -*-
'''
@File    :   functions.py
@Time    :   2020/06/05 15:03:03
@Author  :   Haokun Wang 
@Version :   1.0
@Contact :   wanghk@mail.sustech.edu.cn
@License :   (C)Copyright 2020
@Desc    :   None
'''
import os
import sys
import cv2
import math
import cv2.aruco as aruco
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from dcp.utils.apriltag import apriltag
from dcp.apriltags_rgbd.transform_fuser import fuse_transform


markerLength = 0.01 # 0.0135
markerSeparation = 0.005
object_points = np.array([[0, 0, 0],   
                          [0, markerLength, 0],
                          [markerLength, markerLength, 0],
                          [markerLength, 0, 0],
                          
                          [0, markerSeparation + markerLength, 0],
                          [0, markerSeparation + 2 * markerLength, 0],
                          [markerLength, markerSeparation + 2 * markerLength, 0],
                          [markerLength, markerSeparation + markerLength, 0],
                          
                          [markerSeparation + markerLength, 0, 0],
                          [markerSeparation + markerLength, markerLength, 0],
                          [markerSeparation + 2 * markerLength, markerLength, 0],
                          [markerSeparation + 2 * markerLength, 0, 0],
                          
                          [markerSeparation + markerLength, markerSeparation + markerLength, 0],
                          [markerSeparation + markerLength, markerSeparation + 2 * markerLength, 0],
                          [markerSeparation + 2 * markerLength, markerSeparation + 2 * markerLength, 0],
                          [markerSeparation + 2 * markerLength, markerSeparation + markerLength, 0]], dtype=np.float64)

# object_points = np.array([[0, 0, 0],   
#                           [0, markerLength, 0],
#                           [markerLength, markerLength, 0],
#                           [markerLength, 0, 0],
#                           ], dtype=np.float64)

at_detector = apriltag("tagStandard41h12")


def detect_apriltag(color_image):
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    tags = at_detector.detect(gray)
    return tags


def detect_aruco(color_image):
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco.Dictionary_get(aruco.DICT_7X7_1000), 
                                            parameters=aruco.DetectorParameters_create())
    return corners, ids

# def obtain_tcp(r_world_h, l_world_h):
#     if r_world_h is not None and l_world_h is not None:
#         rrpyt = h2rpyt(r_world_h)
#         lrpyt = h2rpyt(l_world_h)
#         rt, re = np.array(rrpyt[:3]), np.array(rrpyt[3:])
#         lt, le = np.array(lrpyt[:3]), np.array(lrpyt[3:])
#         e, t = (re+le)/2.0, (rt+lt)/2.0
#         return np.array(list(t)+list(e))
#     else:
#         return None


def obtain_tcp(r_world_h, l_world_h):
    if r_world_h is not None and l_world_h is not None:
        rr = R.from_matrix(r_world_h[:3, :3])
        rt = r_world_h[:3, 3]
        lr = R.from_matrix(l_world_h[:3, :3])
        lt = l_world_h[:3, 3]

        rq = rr.as_quat()
        lq = lr.as_quat()

        q, t = (rq+lq)/2.0, (rt+lt)/2.0
        tcp_r = R.from_quat(list(q))
        tcp_rvec = tcp_r.as_rotvec().reshape(1,3)
        return np.array([t, tcp_rvec])
    else:
        return None


def h2rpyt(h_matrix):
    if h_matrix is None:
        return None
    r = R.from_matrix(h_matrix[:3, :3])
    t = list(h_matrix[:3, 3])
    e = list(r.as_euler('xyz', degrees=False))
    return t+e


def present_in_world(rvec, tvec, inv_h):
    base_object_h = None
    if rvec is not None and tvec is not None:
        matrix = cv2.Rodrigues(rvec)[0]
        t = tvec.reshape(3, 1)
        object_camera_h = np.vstack((np.hstack((matrix, t)), np.array([[0, 0, 0, 1]])))
        base_object_h = inv_h.dot(object_camera_h)
    return base_object_h


def obtain_boards(corners, ids, camera_matrix, distortion):
    rvec_r, tvec_r, rvec_l, tvec_l = None, None, None, None
    # rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, markerLength, camera_matrix, distortion)
    if len(corners) >= 4:
        right_corners, right_ids = id_filter(corners, ids, 1, 4)
        if len(right_corners) == 4:
            right_corners, right_ids = sort_ids(right_corners, right_ids)
            right_corners = corners_conversion(right_corners)
            _, rvec_r, tvec_r = cv2.solvePnP(object_points, right_corners,
                                             camera_matrix, distortion)
        
        left_corners, left_ids = id_filter(corners, ids, 5, 8)
        if len(left_corners) == 4:
            left_corners, left_ids = sort_ids(left_corners, left_ids)
            left_corners = corners_conversion(left_corners)
            _, rvec_l, tvec_l = cv2.solvePnP(object_points, left_corners,
                                             camera_matrix, distortion)
    return rvec_r, tvec_r, rvec_l, tvec_l


def obtain_corners_ids(tags):
    corners, ids = [], []
    for tag in tags:
        # print(type(tag['lb-rb-rt-lt']))
        corners.append(np.array([list(np.flipud(tag['lb-rb-rt-lt']))]))
        ids.append([tag['id']])
    return corners, ids


def obtain_boards_from_apriltag(tags: dict, camera_matrix, color_image, depth_image):
    rvec_r, tvec_r, rvec_l, tvec_l = None, None, None, None
    if len(tags) >= 1:
        left_tag = apriltag_filter(tags, 1, 1)
        if len(left_tag) == 1:
            rvec_l, tvec_l = fuse_transform(left_tag[0], color_image, depth_image, camera_matrix)
        right_tag = apriltag_filter(tags, 5, 5)
        if len(right_tag) == 1:
            rvec_r, tvec_r = fuse_transform(right_tag[0], color_image, depth_image, camera_matrix)
    return rvec_r, tvec_r, rvec_l, tvec_l


def apriltag_filter(tags, min_id, max_id):
    # corners, ids = [], []
    t = []
    for tag in tags:
        if min_id <= tag['id'] <= max_id:
            # corners.append(np.flipud(tag['lb-rb-rt-lt']))
            # ids.append([tag['id']])
            t.append(tag)
    # return corners, ids
    return t


def id_filter(corners, ids, min_id, max_id):
    if ids is None:
        return
    new_corners, new_ids = [], []
    for c, i in zip(corners, ids):
        if min_id <= i[0] <= max_id:
            new_corners.append(c)
            new_ids.append(i)
    return new_corners, np.array(new_ids)


def sort_ids(corners, ids):
    n = len(ids)
    for i in range(n-1):
        for j in range(1,n-i):
            if int(ids[j-1][0]) > int(ids[j][0]):
                tmp = ids[j][0]
                ids[j][0] = ids[j-1][0]
                ids[j-1][0] = tmp
                tmp = corners[j]
                corners[j] = corners[j-1]
                corners[j-1] = tmp
    return corners, ids


def corners_conversion(corners):
    new_corners = []
    for corner in corners:
        for i in range(4):
            new_corners.append(corner[0][i])
    return np.array(new_corners, dtype=np.float64)

area_threshold = (1200, 2600)
def find_cube(color_image):
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 110, 255, 0)

    raw_contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    centers = []
    for contour in raw_contours:
        area = cv2.contourArea(contour)
        if area_threshold[0] < area < area_threshold[1]:
            centers.append(find_center(contour))
    centers = color_filter(centers, color_image)
    return centers


def find_center(contour):
    M = cv2.moments(contour)
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])
    rect = cv2.minAreaRect(contour)
    return np.array([center_x, center_y, rect[2]*3.14159/180])


size=5
low_hsv, high_hsv = (0, 20, 20), (80, 255, 255)
def color_filter(centers, color_image):
    new_centers = []
    for center in centers:
        piece_image = color_image[int(center[1])-size:int(center[1])+size, int(center[0])-size:int(center[0])+size, :]
        piece_hsv = cv2.cvtColor(piece_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(piece_hsv, lowerb=low_hsv, upperb=high_hsv)
        mask[mask==255] = 1
        if np.sum(mask) >= 0.2*(2*size)**2:
            new_centers.append(center)
    return new_centers


def obtain_object_pose(centers, color_image, camera_parameters, inv_base_camera_h):
    cx, cy, fx, fy = camera_parameters
    aa = inv_base_camera_h[2]

    pose = []
    for center in centers:
        zc = (-aa[3]*fx*fy) / (aa[0]*fy*(center[0] - cx) + aa[1]*fx*(center[1] - cy) + aa[2]*fx*fy)
        
        xc = zc*(center[0] - cx) / fx
        yc = zc*(center[1] - cy) / fy
        point_base = inv_base_camera_h.dot(np.array([xc, yc, zc, 1]).T)
        pose.append(list(point_base[:2])+[center[2]])
    return pose


def get_pose_from_estimator(estimator_output):
    '''
    return: [[id: int, rvec: arrray, tvec: array, u: double, v: double], [], ...]
    '''
    markers = []
    if len(estimator_output)!=0:
        for i in range(int(len(estimator_output)/15)):
            u, v = estimator_output[0+i*15], estimator_output[1+i*15]
            r = R.from_matrix([estimator_output[2+i*15:5+i*15],
                               estimator_output[6+i*15:9+i*15],
                               estimator_output[10+i*15:13+i*15]])
            rvec = r.as_rotvec()
            tvec = np.array([estimator_output[5+i*15], estimator_output[9+i*15], estimator_output[13+i*15]])
            markers.append([int(estimator_output[14+i*15]), rvec, tvec, u, v])
    return markers


def obtain_boards_from_markers(markers):
    rvec_r, tvec_r, rvec_l, tvec_l = None, None, None, None
    for marker in markers:
        if marker[0] == 1:
            rvec_l = marker[1]
            tvec_l = marker[2]
        if marker[0] == 2:
            rvec_r = marker[1]
            tvec_r = marker[2]
    return rvec_r, tvec_r, rvec_l, tvec_l


def screen_markers(markers, low_bound, up_bound):
    new_markers = []
    for marker in markers:
        if low_bound <= marker[0] <= up_bound:
            new_markers.append(marker)
    return new_markers


def obtain_distance(point1, point2):
    return math.sqrt(pow(point1[0]-point2[0], 2)+pow(point1[1]-point2[1], 2)+pow(point1[2]-point2[2], 2))
