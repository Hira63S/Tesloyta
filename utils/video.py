import torch
import torch.utils.data
from images import whiten, drift, flip, resize, crop_or_pad
from boxes import compute_deltas, visualize_boxes
import numpy as np
from config import Args

args = Args().parse()

def preprocess_func(image, image_meta, boxes=None):
    """
    Performs preprocessing on images
    """
    rgb_mean = (np.array([93.877, 98.801, 95.923], dtype=np.float32).reshape(1, 1, 3))
    rgb_std = (np.array([78.782, 80.130, 81.200], dtype=np.float32).reshape(1, 1, 3))

    if boxes is not None:
        # [x, y, w, h] -> 0, 2 -> x and width
        # a, min max -> boxes x1 and x2 coordinations, min is 0 and max is the orig_size of input
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0., image_meta['orig_size'][1] - 1.)  # min max out -> deals with the width
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0., image_meta['orig_size'][0] - 1.)   # deals with the height

    drift_prob = 0.
    flip_prob = 0.

    #image, image_meta = whiten(image, image_meta, mean=rgb_mean, std=rgb_std)
    image, image_meta, boxes = drift(image, image_meta, prob=drift_prob, boxes=boxes)
    image,image_meta, boxes = flip(image, image_meta, prob=flip_prob, boxes=boxes)

#     if args.forbid_resize:
#         image, image_meta, boxes = crop_or_pad(image, image_meta, args.input_size, boxes=boxes)
#     else:
#    image, image_meta, boxes = resize(image, image_meta, (384, 1248), boxes=boxes)

    return image, image_meta, boxes
