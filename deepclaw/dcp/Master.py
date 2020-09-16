# -*- encoding: utf-8 -*-
'''
@File    :   Master.py
@Time    :   2020/05/26 16:21:07
@Author  :   Haokun Wang 
@Version :   1.0
@Contact :   wanghk@mail.sustech.edu.cn
@License :   (C)Copyright 2020
@Desc    :   None
'''
class Master:
    data_buffer = None
    SIGNAL = 0

    def __init__(self):
        pass

    def update(self, original_data):
        self.data_buffer = original_data
        self.SIGNAL = 1
    
    def get_data(self):
        data = None
        while self.SIGNAL == 0:
            continue
        data = self.data_buffer
        self.SIGNAL = 0
        return data
