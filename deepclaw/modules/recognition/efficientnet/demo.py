# Copyright (c) BioniDL@SUSTECH. All Rights Reserved
"""
This is a demo to run effcientnet trained on waste sorting dataset on a test image paper.png
Please download the pretrained weights and put it under ./weight folder before run the code
"""

from efficientnet_predictor import efficientnet
import cv2
import numpy as np

img_size = 300

model = efficientnet(0, 'weights/Recyclable-bs32-weights.08-1.000-DenseNet169.hdf5')

image = cv2.resize(cv2.imread('paper.png'),(img_size,img_size))

# Feed the image in RGB order to the model.
# The input can be of shape [height, width, channels] or [number of images, height, width, channels]
preds = model.run(image[:,:,::-1])[0]

# The model pretrained is to classify four recycable waste type ['glass', 'metal', 'paper', 'plastic']
obj = ['glass', 'metal', 'paper', 'plastic'][np.argmax(preds)]

print("Recognize %s"%obj)

