import os
import glob
import tqdm

import numpy as np
import skimage.io
import torch
import torch.utils.data

from kitti_class import KITTI
from detector import Detector
from SqueezeNet_detect import SqueezeDet
from config import Args
from load_model import load_model

def demo(args):
    """
    demo for the model
    """

    args.load_model = 'squeezedet_kitti_epoch280.pth'
    args.gpus = [-1]
    args.debug = 2    # visualize detection boxes
    dataset = KITTI('val', args)
    args = Config().update_dataset_info(args, dataset)

    preprocess_func = dataset.preprocess
    del dataset

    # prepare the model and detector
    model = SqueezeDet(args)
    model = load_model(model, args.load_model)
    detector = Detector(model.to(args.device), args)

    # prepare images
    sample_images_dir = '../data/kitti/testing/samples'
    sample_image_paths = glob.glob(os.path.join(sample_images_dir, '*.png'))

    # detection
    for path in tqdm.tqdm(sample_image_paths):
        image = skimage.io.imread(path).astype(np.float32)
        image_meta = {'image_id': os.path.basename(path)[:-4],
                      'orig_size': np.array(image.shape, dtype=np.int32)}

        inp = {'image': image,
               'image_meta': image_meta}

        _ = detector.detect(inp)



print('Hi Tesla! I am demo ')
