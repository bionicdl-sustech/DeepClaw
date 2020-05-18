# Efficientnet

This recognition model efficientnet is based on the implementation in keras-application.

## Installation
In addition to the dependency you have installed with deepclaw, you have to install the following extras to run the recognition model efficientnet.

```bash
pip install tensorflow-gpu==1.15 keras==2.2.5 keras-applications==1.0.8 keras-preprocessing 1.1.0
```

Install dependecy keras-application
```bash
git clone https://github.com/keras-team/keras-applications.git && cd keras-applications
git checkout tags/1.0.8
pip install -e .
```

## Inference
We have trained the efficientnet-B0 on the simple data of [Haihua waste sorting dataset](../../../../data/Haihua-Waste-Sorting/README.md)

The pretrained weights can be download [here](https://pan.baidu.com/s/1M7VXLzkBrFIbmgz9J8ic7g) with extract code ph2e.

Inference on a single image
```bash
python demo.py
```

Inference with realsense D435 in real-time
```bash
python demo_realsense.py
```

## Training on custom dataset
We have provided a train tutorial in train.ipynb, please open it with jupyter notebook.
