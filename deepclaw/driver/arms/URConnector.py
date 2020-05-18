# Copyright (c) 2020 by BionicDL Lab. All Rights Reserved.
# -*- coding:utf-8 -*-
"""
@File: URConnector
@Author: Haokun Wang
@Modified: Liu Xiaobo
@Date: 2020/3/16 15:35
@Description:
connect to a UR robot using socket
For e-Series, the frequency of port 30003 is 500Hz, and for CB-Series it is 125Hz.
The details of TCP/IP connection is here, https://www.universal-robots.com/articles/ur-articles/remote-control-via-tcpip/
"""
import socket
import struct
import time

class URConnector:
    def __init__(self, ip, port):
        # create socket
        self._ip = ip
        self._port = port
        self._recv_len = {'UR5': 1116, 'UR10e': 1108}

    def start(self):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(1)
        try:
            self._socket.connect((self._ip, self._port))
        except socket.gaierror as socketerror:
            print('\033[1;31;40m',socketerror,'\033[0m')
        except socket.error as socketerror:
            print('\033[1;31;40m',socketerror,'\033[0m')


    def send(self, message: str):
        # In Python 3.x, a bytes-like object is required
        self._socket.send(message.encode())

    def recv(self, message_len: int):
        # receive bytes
        try:
            msg = self._socket.recv(message_len,socket.MSG_WAITALL)
        except socket.error as socketerror:
            print('\033[1;31;40m',socketerror,'\033[0m')
            self.close()
            self.start()
            msg = self._socket.recv(message_len,socket.MSG_WAITALL)
        return msg

    def close(self):
        self._socket.close()

    def msg_unpack(self, ur_msg, start_mark: int, size_length: int, number_of_data: int):
        unpacked_msg = []
        for i in range(number_of_data):
            start = start_mark+i*size_length
            end = start_mark+(i+1)*size_length
            unpacked_msg.append(struct.unpack('!d', ur_msg[start:end])[0])
        return unpacked_msg

        # the state description is showed in ATTACHED FILES of https://www.universal-robots.com/articles/ur-articles/remote-control-via-tcpip/
        # go the this website, download the ATTACHED FILES 'Client_Interface_V3.12andV5.6.xlsx', click sheet 'RealTime5.1->5.3'
    def ur_get_state(self, ur='UR5'):
        ur_msg = self.recv(self._recv_len[ur])
        # check the received data
        cnt = 0
        while len(ur_msg) != self._recv_len[ur] and cnt <100:
            ur_msg = self.recv(self._recv_len[ur])
            cnt = cnt + 1

        msg = {'message_size': struct.unpack('!i', ur_msg[0:4])[0],
               'time': struct.unpack('!d', ur_msg[4:12])[0],
               'q_target': self.msg_unpack(ur_msg, 12, 8, 6),
               'qd_target': self.msg_unpack(ur_msg, 60, 8, 6),
               'qdd_target': self.msg_unpack(ur_msg, 108, 8, 6),
               'i_target': self.msg_unpack(ur_msg, 156, 8, 6),
               'm_target': self.msg_unpack(ur_msg, 204, 8, 6),
               'q_actual': self.msg_unpack(ur_msg, 252, 8, 6),
               'qd_actual': self.msg_unpack(ur_msg, 300, 8, 6),
               'i_actual': self.msg_unpack(ur_msg, 348, 8, 6),
               'i_control': self.msg_unpack(ur_msg, 396, 8, 6),
               'tool_vector_actual': self.msg_unpack(ur_msg, 444, 8, 6),
               'tcp_speed_actual': self.msg_unpack(ur_msg, 492, 8, 6),
               'tcp_force': self.msg_unpack(ur_msg, 540, 8, 6),
               'tool_vector_target': self.msg_unpack(ur_msg, 588, 8, 6),
               'tcp_speed_target': self.msg_unpack(ur_msg, 636, 8, 6),
               'digital_input_bits': self.msg_unpack(ur_msg, 684, 8, 1),
               'motor_temperatures': self.msg_unpack(ur_msg, 692, 8, 6),
               'controller_timer': self.msg_unpack(ur_msg, 740, 8, 1),
               'test_value': self.msg_unpack(ur_msg, 748, 8, 1),
               'robot_mode': self.msg_unpack(ur_msg, 756, 8, 1),
               'joint_mode': self.msg_unpack(ur_msg, 764, 8, 6),
               'safety_mode': self.msg_unpack(ur_msg, 812, 8, 1),
               'none_value_0': self.msg_unpack(ur_msg, 820, 8, 6),
               'tool_acelerometer_values': self.msg_unpack(ur_msg, 868, 8, 3),
               'none_value_1': self.msg_unpack(ur_msg, 892, 8, 6),
               'speed_scaling': self.msg_unpack(ur_msg, 940, 8, 1),
               'linear_momentum_norm': self.msg_unpack(ur_msg, 948, 8, 1),
               'none_value_2': self.msg_unpack(ur_msg, 956, 8, 1),
               'none_value_3': self.msg_unpack(ur_msg, 964, 8, 1),
               'v_main': self.msg_unpack(ur_msg, 972, 8, 1),
               'v_robot': self.msg_unpack(ur_msg, 980, 8, 1),
               'i_robot': self.msg_unpack(ur_msg, 988, 8, 1),
               'v_actual': self.msg_unpack(ur_msg, 996, 8, 6),
               'digital_outputs': self.msg_unpack(ur_msg, 1044, 8, 1),
               'program_state': self.msg_unpack(ur_msg, 1052, 8, 1),
               'elbow_position': self.msg_unpack(ur_msg, 1060, 8, 3),
               'elbow_velocity': self.msg_unpack(ur_msg, 1084, 8, 3)}
        return msg

if __name__ == '__main__':


    t1 = time.time()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.settimeout(10)
    s.connect(('192.168.1.10', 30004))
    for i in range(20):
        # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # s.settimeout(5)
        # s.connect(('192.168.1.10', 30003))
        tt1 = time.time()
        s.recv(4096)
        print('len:', len(kk), 'time:',time.time()-tt1)
    s.close()

    t2 = time.time()
    print((t2-t1)/10)


    # robot  = URConnector('192.168.1.10',30003)
    # robot.start()
    # import numpy as np
    #
    # joints_angle = np.array([-1.26675971, -1.50360084, -2.01986912, -1.18507832,  1.55369178,-0.8])
    # velocity=0.5
    # accelerate=0.6
    # move_command = ""
    # move_command = (f"movej([{joints_angle[0]},{joints_angle[1]},{joints_angle[2]},"
    #                             f"{joints_angle[3]},{joints_angle[4]},{joints_angle[5]}],"
    #                             f"a={accelerate},v={velocity})\n")
    # robot.send(move_command)
    #
    # t1 = time.time()
    # max_try = 2000
    # for i in range(max_try):
    #     time.sleep(0.008)
    #     # robot.send(move_command)
    #     # robot.send('get_state()\n')
    #     m = robot.ur_get_state('UR10e')['q_actual']
    #     print(i)
    #
    # t2 = time.time()
    # print((t2-t1))
    #
    # robot.close()
