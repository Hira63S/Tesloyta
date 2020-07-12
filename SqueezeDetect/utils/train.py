"""
Script for training the model
"""

import os
import operator

import torch
import torch.utils.data
from torch.optim.lr_scheduler import StepLR

# from engine.trainer import Trainer
from SqueezeNet_detect import SqueezeDetWithLoss
from config import Args

# load dataset
def load_dataset(dataset_name):
    if dataset_name.lower() == 'kitti':
        from kitti_class import KITTI as Dataset
    return Dataset

load_dataset('kitti')

def train(args):
    Dataset = load_dataset(args.dataset)
    training_data = Dataset('train', args)  # dataset takes in train, val, or trainval as params
    val_data = Dataset('val', args)
    args = Config().update_dataset_info(args, training_data)   # takes care of params in kitti class like mean, std
    Config().print(args)
    logger = Logger(args)


print('Hello Tesla!')
