""" Running faster RCNN in vid """

import numpy as np
import os
import time
import argparse
import imutils
import cv2
import torchvision
import torchvision.transforms as transforms

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input", required=True, help="input vid")
ap.add_argument("-o", "--output", required=True, help="output vid")

args = vars(ap.parse_args())

classes = ['car', 'motorcycle', 'bus', 'truck', 'street sign', 'stop sign',
        'traffic light', 'bicycle', 'person', 'background']


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
