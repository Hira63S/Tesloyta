from flask import Flask, redirect, url_for, flash, request, render_template
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torch import optim
import matplotlib.image as mpimg
from IPython.display import Image
import os
import pickle
from random import shuffle
import operator
import urllib.request
from skimage import io
from .preprocessing import Preprocess
from .model import Predict, ConvolutionalNetwork
import argparse

#parser = argparse.ArgumentParser()

#parser.add_argument("path_image", help="path to input image to be displayed")
#args = parser.parse_args()

def pipeline(imgpath):
    img = io.imread(imgpath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed = Preprocess.pre_process_image(gray)
    corners = Preprocess.find_corners_of_largest_polygon(processed)
    cropped = Preprocess.crop_and_warp(img, corners)
    resized = Preprocess.resize(cropped)
    inverted = Preprocess.invert(resized)
    #cv2.imshow('Inverted', inverted)

    # Press q on keyboard to  exit
    #cv2.waitKey(25) & 0xFF == ord('q')
    
    
    cells = Preprocess.boxes(inverted)

    return cells

def predict(cells):
    model = ConvolutionalNetwork()
    model.load_state_dict(torch.load('Omega2020/model1.pth'))
    model.eval()

    transform = transforms.ToTensor()
    tensors = []
    for cell in cells:
            tensors.append((transform(cell).view(1,28,28).type(torch.FloatTensor)))
    
    grid = []
    for i in range(len(tensors)):
            if tensors[i].mean().item() <= .22:
                grid.append(".")
            else:
                grid.append(str(model(tensors[i].view(1,1,28,28)).argmax().item()))
    return grid



if __name__ == '__main__':
    pipeline(img)