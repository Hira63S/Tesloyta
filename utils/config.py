import argparse
import os


class Args(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # video evaluation
        self.parser.add_argument("--model", required=False, help='path to pre-trained model.')
        self.parser.add_argument("--input", default=0, help='input source')
        self.parser.add_argument("--output", help='where to write the processed file at')
        # experiment
        self.parser.add_argument("--mode", help='train | eval | demo')
        self.parser.add_argument("--dataset", default='kitti',
                                help ='coco | kitti')
        self.parser.add_argument("--load_model", default='', help='path to pre-trained model')
        self.parser.add_argument("--debug", type=int, default=0, help='0: show nothing\n'
                                      '1: visualize pre-processed image and boxes\n'
                                      '2: visualize detections.')
        self.parser.add_argument("--debug_dir",
                                help = 'If need be to debug')
        self.parser.add_argument("--exp_id", default='default')

        # model
        self.parser.add_argument("--arch", default="squeezedetect",
                                 help="model architecture: squeezedetect | squeezedetectPlus")
        self.parser.add_argument("--dropout_prob", type=float, default=0.5, help='prob of dropout')

        # train
        self.parser.add_argument("--input_size", type=tuple, default=(384, 1248), help='input images size')
        self.parser.add_argument("--num_classes", default=5, help="num of classes to train")
        self.parser.add_argument("--momentum", type=float, default=0.9, help='momentum of SGD.')
        self.parser.add_argument("--grad_norm", type=float, default=5., help='max norm of the gradients.' )
        self.parser.add_argument("--num_epochs", type=int, default=30, help='total training epochs.')
        self.parser.add_argument("--num_iters", type=int, default=-1, help='default: # samples/batch_size')
        self.parser.add_argument("--batch_size", type=int, default=20, help = 'batch_size')
        self.parser.add_argument('--num_anchors', type=int, default=9, help='default: # samples/ batch_size')
        self.parser.add_argument("--lr", type=float, default=0.01, help='learning rate')
        self.parser.add_argument("--weight_decay", type=float, default=0.0001, help='wegiht deay of adam')
        self.parser.add_argument("--master_batch_size", type=int, default=-1, help = 'batch_size on the master GPU')
        self.parser.add_argument('--save_intervals', type=int, default=1, help='number of epochs to save model.')
        self.parser.add_argument("--val_intervals", type=int, default=5, help = 'num of epochs to run validations.')
        self.parser.add_argument("--no_eval", action='store_true',help = 'bypass mAP eval during training.')
        self.parser.add_argument("--print_interval", type=int, default=10, help = 'disable progress bar and print to screen.')
        self.parser.add_argument("--flip_prob", type=float, default=0.5,help = 'prob of horizontal flip during training')
        self.parser.add_argument("--drift_prob", type=float, default=1., help = 'prob of drifting image during training')
        self.parser.add_argument("--forbid_resize", action='store_true', help='disable image resizing during training, instead we use crop/pad')
        self.parser.add_argument("--class_loss_weight", type=float, default=1.,help = 'weight of classification loss')
        self.parser.add_argument("--positive_score_loss_weight", type = float, default=3.75, help = 'positive weight of score pred loss')
        self.parser.add_argument("--negative_score_loss_weight",type=float, default=100., help = 'negative weight of score prediction loss.')
        self.parser.add_argument("--bbox_loss_weight",type=float, default=6., help = 'weight of boxes reg loss')
        self.parser.add_argument("--anchors_per_grid", type=int, default=9, help ='anchors per grid')
        # inference

        self.parser.add_argument("--nms_thresh", type=float, default=0.4, help='discards all overlapping boxes with IoU < nms_thresh.')
        self.parser.add_argument("--score_thresh", type=float, default=0.3, help='discards all boxes with scores smalller than score_thresh.')
        self.parser.add_argument("--keep_top_k", type=int, default=64, help='keep top k detections before nms.')
        # system
        # system
        self.parser.add_argument("--num_workers", type=int, default=4,
                                help ='dataloader thread, 0 for no-thread.')
        self.parser.add_argument("--gpus", default='0',
                                help='-1 for CPU, use comma for multiple gpus')
        self.parser.add_argument("--seed", type=int, default=42,
                                help="random seed")
        self.parser.add_argument("--not_cuda_benchmark", action='store_true',
                                help="disable when the input size is not fixed.")

    def parse(self, cfg = ''):
        if cfg == '':
            args = self.parser.parse_args()
        else:
            args = self.parser.parse_args(args)

        args.gpus_str = args.gpus
        args.gpus = [int(gpu) for gpu in args.gpus.split(',')]
        args.gpus = [i for i in range(len(args.gpus))] if args.gpus[0] >= 0 else [-1]

        if args.mode != 'train' and len(args.gpus) > 1:
            print('Only single GPU is supported in {} mode'.format(args.mode))
            args.gpus = [args.gpus[0]]
            args.master_batch_size = -1

        if args.master_batch_size == -1:
            args.master_batch_size = args.batch_size // len(args.gpus)

        rest_batch_size = (args.batch_size - args.master_batch_size)
        args.chunk_sizes = [args.master_batch_size]
        for i in range(len(args.gpus) - 1):
            captain_chunk_size = rest_batch_size // (len(args.gpus) - 1)
            if i < rest_batch_size % (len(args.gpus)-1):
                captain_chunk_size +=1
            args.chunk_sizes.append(captain_chunk_size)
        print('trainig chunk_size:', args.chunk_sizes)

        args.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        args.data_dir = os.path.join(args.root_dir, 'data')
        args.exp_dir = os.path.join(args.root_dir, 'exp')
        args.save_dir = os.path.join(args.exp_dir, args.exp_id)
        args.debug_dir = os.path.join(args.save_dir, 'debug')
        print('The result will be saved to', args.save_dir)

        return args

    @staticmethod
    def update_dataset_info(args, dataset):
        args.input_size = dataset.input_size
        args.rgb_mean = dataset.rgb_mean
        args.rgb_std = dataset.rgb_std
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
