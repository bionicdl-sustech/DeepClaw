# Copyright (c) BioniDL@SUSTECH. All Rights Reserved
"""
This is a demo to run effcientnet trained on waste sorting dataset on a test image paper.png
Please download the pretrained weights and put it under ./weight folder before run the code
"""

from efficientnet_predictor import efficientnet
import cv2
import numpy as np
from deepclaw.driver.sensors.camera.Realsense import Realsense

camera = Realsense('../../../../configs/robcell-ur10e-hande-d435/d435.yaml')

img_size = 300

model = efficientnet(0, 'weights/Recyclable-bs32-weights.08-1.000-DenseNet169.hdf5')
print("Press q to quite the real-time detection")

while True:
    frame = camera.get_frame()
    image = frame.color_image[0]
    crop = image[350:650,850:1150,:]
    
    preds = model.run(crop[:,:,::-1])
    idx = np.argmax(preds[0])
    name = ['glass', 'metal', 'paper','plastic'][idx]
    ret = cv2.putText(crop, '{}, {:.3f}'.format(name, preds[0][idx]),
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 0, 255), 1)
    cv2.imshow('Prediction', crop)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cv2.destroyAllWindows()





