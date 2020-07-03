# Tesloyta

## Introduction
The focus of the project is to do real-time object detection by implementing newest techniques being researched in the field of Computer Vision. In practice, the project encompasses many different tasks like researching better model architectures, creating data pipeline, model training, deployment, and inference. But first focus of the project was to deploy a model trained on object detection datasets like COCO for real-time object detection. Since deployment on edge would be ideal for our project, we chose RaspberryPi 4 (4GB RAM) as the device of choice. Initially, to experiment with data collection, simple functions like lane detection, road markings etc. were used for basic image processing on live videos from the road while using multiple OpenCV techniques. Recently, we were able to deploy a MobileNet model with Single Shot MultiBox Detector for object-detection as is shown in the video above. The project is, however, still undergoing major changes to the architecture as the focus now is to do 3D object tracking, preferably on an edge device of similar kind.

## Model
A MobileNet model pre-trained on COCO dataset with an SSD detector was used for inference using OpenCV and Caffe libraries. Even though selective search is used by multiple SOTA models like RCNN, Fast RCNN, and Faster RCNN, it is not ideal for real-time object detection because of the way it works. While it is not an ideal detector because it only makes use of feature maps produced by upper convolution layers for detection, better detectors like FPN exist, which constructs higher resolution layers for detection. For this purpose, another model is still under production the code and research to which can be found in Jupyter Notebooks in the Models folder.

The above shown video was produced by a CaffeNet model as the speed was slightly better than the PyTorch MobileNet_v1 model. To re-produce this work, simply clone the repo and run:
<!-- Italics -->
*python RaspberryPi/PyScripts/MobileNet_Detection/real_time_object_detection.py --output test.avi --prototxt MobileNetSSD_deploy.prototxt --model MobileNetSSD_deploy.caffemodel*

## Research
One major blocker in getting the project completed and functioning was the inference time. Because of the compute limitations, a smaller model with higher mAP (mean Average Percision) was needed. Both SqueezeNet and MobileNet are ideal for Computer Vision applications while using devices like RaspberryPi. However, not only do they models need to be trained on datasets specific to objects on the road but they also need to use better and more accurate object detectors so that even the smaller objects get noticed. There is a need to take care of all the edge cases as well when it comes to real-time inference by autonomous cars on the road. Multiple different models with newer techniques like DETR are being researched and the current work being done can be found in Models folder.

### Next Steps:
The deployed MobileNet model was trained on COCO dataset but currently, a model that would be trained on dataset specific to objects on the road is under production. We are using Argoverse 3D object tracking dataset to train a model using latest detection techniques. The repository will be updated on a regular basis with latest work.
