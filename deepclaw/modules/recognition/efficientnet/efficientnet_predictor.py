# Copyright (c) BioniDL@SUSTECH. All Rights Reserved
"""
This efficientnet class is based on the efficientnet class in keras_application.
Please download the pretrained weights and put it under ./weight folder before run the code
"""

import keras, cv2
import numpy as np

import tensorflow as tf
from keras.models import load_model
from keras.utils import get_custom_objects
from keras_applications.efficientnet import preprocess_input
from keras import backend as K
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

def preprocessing_f(x):
    return preprocess_input(x, data_format=None, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)

def swish(x):
    return K.sigmoid(x) * x

get_custom_objects().update({'swish': keras.layers.Activation(tf.nn.swish)})

img_size = 300

class efficientnet(object):
    def __init__(self, compound_coef = 0, weight_path = None):
        self.compound_coef = compound_coef
        self.weight_path = weight_path

        self.obj_list = ['glass', 'paper', 'metal', 'plastic']
        self.model = load_model(self.weight_path, compile=False)

    def run(self, image):
        if len(image.shape)==3:
            x = np.expand_dims(image, axis=0)
        x = preprocessing_f(x)
        return self.model.predict(x)
        
if __name__ == '__main__':
    image = cv2.resize(cv2.imread('paper.png'),(300,300))
    model = efficientnet(0, 'weights/Recyclable-bs32-weights.08-1.000-DenseNet169.hdf5')
    preds = model.run(image)

