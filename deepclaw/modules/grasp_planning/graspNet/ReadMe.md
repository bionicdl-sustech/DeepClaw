# GraspNet Modified from AlexNet
This work has been published in our paper ["Reconfigurable Design for Omni-adaptive Grasp Learning"](https://arxiv.org/abs/2003.01582) and ["Rigid-Soft Interactive Learning for Robust Grasping"](https://arxiv.org/abs/2003.01584)

Since CNN performs better in classification tasks rather than regression, we divide grasp angle in [0, pi) into 18 angular bins, and a CNN model predicts the successful grasp probabilities independently for 0, 10, ..., 170 degrees. Therefore, our problem can be thought of as an 18-way binary classification problem.

We build a fully convolutional neural network (FCN) converting from AlexNet with the following architecture: the first five convolutional layers are taken from AlexNet, followed by a 6x6x4096 (kernel size times number of filters) convolutional layer, a 1x1x1024 and a 1x1x36 fully convolutional layers in sequence. The first five convolutional layers are initiated with weights pre-trained on ImageNet and are not trained with our dataset.

During training time, since each training data entry only has the label corresponding to one active binary classification among the 18 angular classes, the loss function is defined to compute cross-entropy of the active angular class. This is achieved by defining a mask from the grasp angles to filter out the non-active outputs of the last FCN layer, resulting in a FCN output of batch sizex2.

During the testing time, though the FCN is trained on the cropped patch with a single grasp, it can be applied to inference the entire image of any size and give relatively dense predictions pixel-wise at one time. The stride of the dense predictions equals to the multiplication of all the strides in the convolutional and max-pooling layers, which is 32 in our network architecture.

## Installation
In addition to the dependency you have installed with deepclaw, you have to install the following extras to run the recognition model efficientnet.

```bash
pip install tensorflow-gpu==1.15 pillow
```

## Inference
We have trained the graspNet on a grasping dataset using out soft fingers. Please download the checkpoint [here](https://pan.baidu.com/s/1SDoOEURdh3VJgk5DLRDyrQ) with extract code: 7hy5.

```bash
python demo.py
```

## Training on custom dataset
We have provided a training script in train_softgripper.py.


### Reference:
[1] Pinto L, Gandhi D, Han Y, et al. The curious robot: Learning visual representations via physical interactions[C]//European Conference on Computer Vision. Springer, Cham, 2016: 3-18.
[2] Iandola F N, Han S, Moskewicz M W, et al. SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 0.5 MB model size[J]. arXiv preprint arXiv:1602.07360, 2016.
