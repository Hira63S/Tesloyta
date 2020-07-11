import os
import numpy as np
import matplotlib.pyplot as plt


def whiten(image, image_met, mean=0., std=1.):
    """
    Whitens the image by subtracting the mean and dividing by the std

    :param image: input image
    :param image_meta: identifications of image
    :param mean: calculated based on the dataset, float
    :param std: calculated from the dataset, float
    """

    image = (image - mean) / std
    image_meta.update({'rgb_mean': mean, 'rgb_std': std})
    return image, image_meta


def drift(image, image_meta, prob=0., boxes=None):
    """
    :param image: image to be transformed
    :param image_meta: dictionary object containing id, obj_id, image shape etc.
    :param prob: probability of performing this function
    :param boxes: ground-truth boxes for the image
    """
    drifts = np.array([0,0], dtype=np.int32)  # [0,0]
    drifted_size = np.array(image.shape, dtype=np.int32)   # [375, 1242, 3]

    # generates a random number, prob is passed in through args
    if np.random.uniform() < prob:
        max_drift_y = image_meta['orig_size'][0] //4   # divides the height by 4 i.e. 93
        max_drift_x = image_meta['orig_size'][1] // 8  # divides the width by 8 i.e. 155
        max_boxes_y = min(boxes[:, 1]) if boxes is not None else max_drift_y  # finds the min y value from boxes
        max_boxes_x = min(boxes[:, 0]) if boxes is not None else max_drift_x  # finds the min x value from boxes

        dy = np.random.randint(-max_drift_y, min(max_drift_y, max_boxes_y))   # find random integer b/w max drift and minimum number between max drift and minimum y value
        dx = np.random.randint(-max_drift_x, min(max_drift_x, max_boxes_x))
        drifts = np.array([dy, dx], dtype=np.int32)

        image_height = image_meta['orig_size'][1] - dy
        image_width = image_meta['orig_size'][0] - dx

        orig_x, orig_y = max(dx, 0), max(dy, 0)
        drift_x, drift_y = max(-dx, 0), max(-dy, 0)

        drifted_image = np.zeros((image_height, image_width, 3)).astype(np.float32)
        drifted_image[drift_y:, drift_x:, :] = image[orig_y:, orig_x:, :]
        image = drifted_image
        drifted_size = np.array(image.shape, dtype=np.int32)

        if boxes is not None:
            boxes[:, [0, 2]] -= dx
            boxes[:, [1, 3]] -= dy

    image_meta.update({'drifts': drifts, 'drifted_size': drifted_size})

    return image, image_meta, boxes


def flip(image, image_meta, prob=0., boxes=None):
    """
    :param image
    :param image_meta: dict
    :param prob: probability of flipping
    :param boxes: boxes data
    """

    flipped = False
    if np.random.uniform() < prob:
        flipped = True
        image = image[:, ::-1, :].copy()

    if flipped and boxes is not None:
        image_width = image.shape[1]
        boxes_widths = boxes[:, 2] - boxes[:, 0]   # take the right x coordinate and subtract from the left
        boxes[:, 0] = image_width - 1 - boxes[:, 2]
        boxes[:, 2] = boxes[:, 0] + boxes_widths

    image_meta.updates({'flipped': flipped})

    return image, image_meta, boxes






























print('Hello Tesla!')
