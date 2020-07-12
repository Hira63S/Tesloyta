import argparse
import os


class Args(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("--dataset", default='kitti', help ='coco | kitti')
        #self.parser.add_argument("-", "", required= , help = '')
        #self.parser.add_argument("-", "", required= , help = '')
        #self.parser.add_argument("-", "", required= , help = '')
        self.parser.add_argument("--debug", type=int, default=0, help='0: show nothing\n'
                                      '1: visualize pre-processed image and boxes\n'
                                      '2: visualize detections.')
        self.parser.add_argument("debug_dir", required= True, help = 'If need be to debug')
        #self.parser.add_argument("-", "", required= , help = '')


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
