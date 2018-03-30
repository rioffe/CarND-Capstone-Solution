# Traffic Light Classifier Models 
This document provides an overview of all the models Our team used to classify traffic lights.

[//]: # (Image References)
[image_ssd_sim]: ./imgs/ssd_inception_simul.JPG "Classification using SSD Inception V2 over Simulation data"
[image_ssd_reallife]: ./imgs/ssd_inception_reallife.JPG "Classification using SSD Inception V2 over Reallife data"



[//]: # (Image References)
[image1]: ./imgs/real1.png
[image2]: ./imgs/real2.png
[image3]: ./imgs/real3.png
[image4]: ./imgs/sim1.png
[image5]: ./imgs/Loss_real.png
[image6]: ./imgs/loss_sim.png

## List of Models
1. faster_rcnn_sim
2. faster_rcnn_reallife
3. ssd_inception_v2_coco_sim
4. ssd_inception_v2_coco_reallife

## Model Details
### faster_rcnn_sim
This model is optimized to classify simulator images. This model is based on [Faster R-CNN Resnet 101](https://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf) model architecture and created using TensorFlow's Object Detection API. Model is first trained using [Bosch Small traffic Light Data set](https://hci.iwr.uni-heidelberg.de/node/6132) and then fine tuned for a hand annotated set of images from the Udacity Simulator.
#### Model Params:
epochs : 10000
learning rate : 0.0003

#### Loss Graph
![alt text][image6] 

#### Few Inference Images

![alt text][image4]


### faster_rcnn_reallife
This model is optimized to classify reallife traffic light images. This model is based on [Faster R-CNN Resnet 101](https://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf) model architecture and created using TensorFlow's Object Detection API. Model is first trained using [Bosch Small traffic Light Data set](https://hci.iwr.uni-heidelberg.de/node/6132) and then fine tuned for a hand annotated set of images from the Udacity test track.
#### Model Params:
epochs : 10000
learning rate : 0.0003

#### Loss Graph
![alt text][image5] 

#### Few Inference Images

![alt text][image1] | ![alt text][image2] | ![alt text][image3]

### ssd_inception_v2_coco_sim
This model is designed to classify simulator images. This model is based on [SSD inception v2](https://arxiv.org/pdf/1502.03167) model architecture and created using TensorFlow's object detection API. It is trained using a hand annotated set of images from the Udacity Simulator.
#### Model params:
epochs : 5000
learning rate : 0.004 
![alt text][image_ssd_sim]

### ssd_inception_v2_coco_reallife
This model is designed to classify reallife traffic light images. This model is based on [SSD inception v2](https://arxiv.org/pdf/1502.03167) model architecture and created using TensorFlow's object detection API. It is trained using a hand annotated set of images from the Udacity test track.
#### Model params:
epochs : 5000
learning rate : 0.004
![alt text][image_ssd_reallife]

