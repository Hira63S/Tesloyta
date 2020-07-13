"""
Script for training the model
"""

import os
import operator

import torch
import torch.utils.data
from torch.optim.lr_scheduler import StepLR

# from engine.trainer import Trainer
from sqdetect import SqueezeDetWithLoss
from utils.config import Args

# load dataset
def load_dataset(dataset_name):
    if dataset_name.lower() == 'kitti':
        from datasets.kitti import KITTI as Dataset
    return Dataset

load_dataset('kitti')

def train(args):
    Dataset = load_dataset(args.dataset)


print('Hello Tesla!')
