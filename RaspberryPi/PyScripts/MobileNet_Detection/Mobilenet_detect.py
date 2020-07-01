""" Pytorch MobileNet model for Object Detection """

import cv2
import time
import numpy as np
import argeparse
import torchvision
import torchvision.transforms as transforms
import torch

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help = "input video / camera index")
ap.add_argument("-o", "--output". required=True,
    help = "link to output video")
ap.add_argument("-l", "--labels", required=True,
    help = "link to classes file")


args = vars(ap.parse_args())
