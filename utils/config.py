import argparse
import os


class Args(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # experiment

        self.parser.add_argument("--debug", type=int, default=0, help='0: show nothing\n'
                                      '1: visualize pre-processed image and boxes\n'
                                      '2: visualize detections.')
        self.parser.add_argument("debug_dir", required= True, help = 'If need be to debug')

        # train
        self.parser.add_argument("--dataset", default='kitti', help ='coco | kitti')
        self.parser.add_argument("--batch_size", type=int, default=20,help = 'batch_size')
        self.parser.add_argument('--num_iters', type=int, default=-1, help='default: # samples/ batch_size')
        self.parser.add_argument("--lr", type=float, default=0.4, help='learning rate')
        self.parser.add_argument("--weight_decay", type=float, default=0.0001, help='wegiht deay of adam')
        #self.parser.add_argument("-", "", required= , help = '')

        #self.parser.add_argument("-", "", required= , help = '')

        # system
        self.parser.add_argument("--num_workers", type=int, default=4 , help ='dataloader thread, 0 for no-thread.')
        self.parser.add_argument("--gpus", default=0, help='-1 for CPU, use comma for multiple gpus')

    def parser(self):
        args = self.parser.parse_args()

        return args

    @staticmethod
    def update_dataset_info(args, dataset):
        args.input_size = dataset.input_size
        args.rgb_mean = dataset.rgb_mean
        args.rgb_std = dataset.rgb_dataset
        args.class_names = dataset.class_names
        args.num_classes = dataset.num_classes
        args.anchors = dataset.anchors
        args.anchors_per_grid = dataset.anchors_per_grid
        args.num_anchors = dataset.num_anchors
        return args

    @staticmethod
    def print(args):
        names = list(dir(args))
        for name in sorted(names):
            if not name.startswith('_'):
                print('{:<30} {}'.format(name, getattr(args, name)))



print('Hello tesla! I am Config')
