import os
import glob
import tqdm

import numpy as np
import skimage.io
import torch
import torch.utils.data

from kitti_class import KITTI
from detector import detector
from SqueezeNet_detect import SqueezeDet
from config import Args
from load_model import load_model


print('Hi Tesla! I am demo ')
