import os

import numpy as np

import torch

EPSILON=1E-10

def init_env(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = not args.not_cuda_benchmark
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus_str
    args.device = torch.device('cuda' if args.gpus[0] >= 0 else 'cpu')

    return args

def load_dataset(dataset_name):
    if dataset_name.lower() == 'kitti':
        from datasets.kitti import KITTI as Dataset
    return dataset

class MetricLogger(object):
    def __int__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count + EPSILON)
