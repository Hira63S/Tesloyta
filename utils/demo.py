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
from imutils.video import VideoStream

def demo(args):
    """
    demo for the model
    """

    args.load_model = 'squeezedet_kitti_epoch280.pth'
    args.gpus = [-1]
    args.debug = 2    # visualize detection boxes
    # vs = VideoStream(src=0).start()
    # frame = vs.read()
    dataset = KITTI('val', args)
    args = Args().update_dataset_info(args, dataset)

    preprocess_func = dataset.preprocess
#    del frame

    # prepare the model and detector
    model = SqueezeDet(args)
    model = load_model(model, args.load_model)
    detector = Detector(model.to(args.device), args)

    # prepare images
    sample_images_dir = '../data/kitti/samples'
    sample_image_paths = glob.glob(os.path.join(sample_images_dir, '*.png'))

    # detection
    for path in tqdm.tqdm(sample_image_paths):
        image = skimage.io.imread(path).astype(np.float32)
        image_meta = {'image_id': os.path.basename(path)[:-4],
                      'orig_size': np.array(image.shape, dtype=np.int32)}

        image, image_meta, _ = preprocess_func(image, image_meta)
        image = torch.from_numpy(image.transpose(2,0,1)).unsqueeze(0).to(args.device)
        image_meta = {k: torch.from_numpy(v).unsqueeze(0).to(args.device) if isinstance(v, np.ndarray)
                     else [v] for k, v in image_meta.items()}

        inp = {'image': image,
               'image_meta': image_meta}

        _ = detector.detect(inp)



print('Hi Tesla! I am demo ')
