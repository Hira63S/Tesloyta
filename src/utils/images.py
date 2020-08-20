import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def whiten(image, image_meta, mean=0., std=1.):
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

        image_height = image_meta['orig_size'][0] - dy
        image_width = image_meta['orig_size'][1] - dx

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

    image_meta.update({'flipped': flipped})

    return image, image_meta, boxes

def resize(image, image_meta, target_size, boxes=None):
    """
    Can call on this function when your model would work with a different sized image

    :param image: image
    :param image_meta: dictionary
    :param target_size: target size you want to pass into the model
    :param boxes: gt boxes
    """
    height, width = image.shape[:2]
    scales = np.array([target_size[0] / height, target_size[1] / width], dtype=np.float32)
    image = cv2.resize(image, (target_size[1], target_size[0]))

    if boxes is not None:
        boxes[:, [0, 2]] *= scales[1]
        boxes[:, [1, 3]] *= scales[0]

    image_meta.update({'scales': scales})

    return image, image_meta, boxes

def crop_or_pad(image, image_meta, target_size, boxes=None):
    """
    :param image: image passed in
    :param image_meta: dictionary
    :param target_size (height, width)
    :param boxes: xyxy format
    """
    padding, crops = np.zeros(4, dtype=np.int16), np.zeros(4, dtype=np.int16)

    height, width = image.shape[:2]
    target_height, target_width = target_size

    if height < target_height:
        padding[0] = (target_height - height) // 2
        padding[1] = (target_height - height) - padding[0]

    elif height > target_height:
        crops[0] = (height - target_height) // 2
        crops[1] = (height - target_height) - crops[0]

    if width < target_width:
        padding[2] = (target_width - width) // 2
        padding[3] = (target_width - width) - padding[2]

    elif width > target_width:
        crops[2] = (width - target_width) // 2
        crops[3] = (width - target_width) - crops[2]

    image, boxes = pad(image, padding, boxes=boxes)
    image, boxes = crop(image, crops, boxes=boxes)

    image_meta.update({'padding': padding, 'crops': crops})

    return image, image_meta, boxes

def pad(image, padding, boxes = None):
    """
    adds padding to the images

    :param image: image passed in
    :param padding: what padding to add
    :param boxes: xyxy format
    :return:
    """
    if not np.all(padding==0):
        padding = (padding[:2], padding[2:], [0,0])
        image = np.pad(image, padding, mode='constant')

        if boxes is not None:
            boxes[:, [0, 2]] += padding[2]
            boxes[:, [1, 3]] += padding[0]

    return image, boxes

def crop(image, crops, boxes=None):
    """
    adds padding to the images

    :param image: image passed in
    :param padding: what padding to add
    :param boxes: xyxy format
    :return:
    """
    if not np.all(crops == 0):
        image = image[crops[0]:-crops[1], :, :] if crops[1] > 0 else image[crops[0]:, :, :]
        image = image[:, crops[2]:-crops[3], :] if crops[3] > 0 else image[:, crops[2]:, :]

        if boxes is not None:
            boxes[:, [0, 2]] -= crops[2]
            boxes[:, [1, 3]] -= crops[0]
            boxes = np.maximum(boxes, 0.)

    return image, boxes


def image_postprocess(image, image_meta):
    if 'scales' in image_meta:
        image = cv2.resize(image, tuple(image_meta['orig_size'][1::-1]))

    if 'padding' in image_meta:
        image = crop(image, image_meta['padding'])

    if 'crops' in image_meta:
        image = pad(image, image_meta['crops'])

    if 'flipped' in image_meta and image_meta['flipped']:
        image = image[:, ::-1, :]

    if 'drifts' in image_meta:
        padding = [image_meta['drifts'][0], 0, image_meta['drifts'][1], 0]
        image = pad(image, padding)[0]

    if 'rgb_mean' in image_meta and 'rgb_std' in image_meta:
        image = image * image_meta['rgb_std'] + image_meta['rgb_mean']

    return image




print('Hello Tesla! I am images')
