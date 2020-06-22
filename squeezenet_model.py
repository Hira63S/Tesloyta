import argparse
import numpy as np
import time
import cv2
import torchvision
import torch

# arguments for the input, for the output
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help = 'path to input image')
ap.add_argument("-o", "--output", required=True, help = 'path to output image')
ap.add_argument("-m", "--model", required=False, help = 'path to the model')   # might not even need this because PyTorch
