"""
Contains basic functions that help in resolving the predictions by the model
"""

import numpy as np
import os
import torch
import cv2

EPSILON=1E-10


def xyxy_to_xywh(boxes_xyxy):
    assert np.ndim(boxes_xyxy) == 2
    assert np.all(boxes_xyxy[:, 0] < boxes_xyxy[:, 2])
    assert np.all(boxes_xyxy[:, 1] < boxes_xyxy[:, 3])

    return np.concatenate([
        (boxes_xyxy[:, [0]] + boxes_xyxy[:, [2]]) / 2.,
        (boxes_xyxy[:, [1]] + boxes_xyxy[:, [3]]) / 2.,
        boxes_xyxy[:, [2]] - boxes_xyxy[:, [0]] + 1.,
        boxes_xyxy[:, [3]] - boxes_xyxy[:, [1]] + 1.
    ], axis=1)

def xywh_to_xyxy(boxes_xywh):
    assert np.ndim(boxes_xywh) == 2
    assert np.all(boxes_xywh[:, [2, 3]] > 0)

    return np.concatenate([
        boxes_xywh[:, [0]] - 0.5 * (boxes_xywh[:, [2]] -1),
        boxes_xywh[:, [1]] - 0.5 * (boxes_xywh[:, [3]] -1),
        boxes_xywh[:, [0]] + 0.5 * (boxes_xywh[:, [2]] -1),
        boxes_xywh[:, [1]] + 0.5 * (boxes_xywh[:, [3]] -1)
    ], axis=1)

def deltas_to_boxes(deltas, anchors, input_size):
    """
    Returns the boxes in xyxy format after prediction
    :param deltas: dxdydwdh format
    :param anchors: xywh format
    :param inputIsze: input image in (h, w) format
    :return: boxes in xyxy format
    """
    boxes_xywh = torch.cat([
        anchors[..., [0]] + anchors[..., [2]] * deltas[..., [0]],
        anchors[..., [1]] + anchors[..., [3]] * deltas[..., [1]],
        anchors[..., [2]] * torch.exp(deltas[..., [2]]),
        anchors[..., [3]] * torch.exp(deltas[..., [3]])
    ], dim=2)

    boxes_xyxy = xywh_to_xyxy(boxes_xywh)
    boxes_xyxy[..., [0, 2]] = torch.clamp(boxes_xyxy[..., [0, 2]], 0, input_size[1] - 1)
    boxes_xyxy[..., [1, 3]] = torch.clamp(boxes_xyxy[..., [1, 3]], 0, input_size[0] - 1)

    return boxes_xyxy

def compute_overlaps(boxes1, boxes2):
    """
    Compute IoUs between two sets of boxes
    boxes1 and boxes2 must have the same shape.
    :param boxes1: xyxy format
    :param boxes2: xyxy format
    :return:
    """
    # calculate the dimensions of the two boxes to calculate the intersection
    # area
    lr = torch.clamp_min(torch.min(boxes1[..., [2]], boxes2[..., [2]]) -
                         torch.max(boxes1[..., [0]], boxes2[..., [0]]), 0)
    tb = torch.clamp_min(torch.min(boxes1[..., [3]], boxes2[..., [3]]) -
                         torch.max(boxes1[..., [1]], boxes2[..., [1]]), 0)
    inter = lr * tb

    # THEN calculate the area of both boxes separately and add them together

    union = (boxes1[..., [2]] - boxes1[..., [0]]) * (boxes1[..., [3]] - boxes1[..., [1]]) + \
            (boxes2[..., [2]] - boxes2[..., [0]]) * (boxes2[..., [3]] - boxes2[..., [1]]) - inter

    # Intersection / union is the answer

    return inter / (union + EPSILON)


def safe_softmax(probs, dim=None):
    exp = torch.exp(probs - torch.max(probs, dim=dim, keepdim=True)[0])
    return exp / torch.sum(exp, dim=dim, keepdim=True)



print('Hi Tesla! I am Prediction Resolver')
