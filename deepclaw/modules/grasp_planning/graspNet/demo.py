# Copyright (c) BioniDL@SUSTECH. All Rights Reserved
"""
This is a demo to run graspNet trained on soft finger grasping dataset on a test image paper.png
Please download the pretrained weights and put it under ./checkpoint folder before run the code
"""

from deepclaw.modules.grasp_planning.graspNet.fc_predictor import FCPredictor
import numpy as np
from PIL import Image, ImageDraw

NUM_THETAS = 9
p = FCPredictor(NUM_THETAS*2, './checkpoint/Network9-1000-100')

img = Image.open('test.jpg')
img_arr = np.array(img)
y_, p_best, grasp_pose = p.run(img_arr)
draw = ImageDraw.Draw(img, 'RGBA')

# Visualiza the prediction on the image
for i in range(15):
    for j in range(28):
        x =  114 + j*32
        y =  114 + i*32
        r = p_best[i][j] * 16
        draw.ellipse((x-r, y-r, x+r, y+r), (0, 0, 255, 125))
        
        # draw the grasp orientation if the model predict it
        if NUM_THETAS > 1:
            local_best_theta = np.argmax(y_[0, i, j, :, 1])
            local_best_theta = - 1.57 + (local_best_theta + 0.5) * (1.57 / NUM_THETAS)
            draw.line([(x - r*np.cos(local_best_theta), y + r*np.sin(local_best_theta)),
                        (x + r*np.cos(local_best_theta), y - r*np.sin(local_best_theta))],
                        fill=(255,255,255,125), width=10)
img.show()
