"""
Main Dataset file that takes in the input data and prepares it for the model.
"""
import os
import subprocess
import numpy as np
import skimage.io

import torch.utils.data

from utils.image import whiten, drift, flip, resize, crop_or_pad
from utils.boxes import compute_deltas, visualize_boxes

class BaseDataset(torch.utils.data.Dataset):
    """
    BaseDataset class that can take any dataset and transform it to our liking
    """
    def __init__(self,phase,cfg):
        super(BaseDataset, self).__init__()
        self.phase = phase
        self.cfg = cfg

    def __getitem__(self, index):
        """Generates one sample of data"""
        image, image_id = self.load_image(index)
        gt_class_ids, gt_boxes = self.load_annotations(index)

        image_meta = {'index': index,
                     'image_id': image_id,
                     'orig_size': np.array(image.shape, dtype=np.int32)}

        image, image_meta, gt_boxes = self.preprocess(image, image_meta, gt_boxes)
        gt = self.prepare_annotations(gt_class_ids, gt_boxes)

        inp = {'image': image.transpose(2, 0, 1),
               'image_meta': image_meta,
               'gt': gt}

        if self.cfg.debug == 1:
            image = image * image_meta['rgb_std'] + image_meta['rgb_mean']
            save_path = os.path.join(self.cfg.debug_dir, image_meta['image_id'] + '.png')
            visualize_boxes(image, gt_class_ids, gt_boxes,
                            class_names = self.class_names,
                            save_path = save_path)

        return inp

    def __len__(self):
        return len(self.sample_ids)


    def preprocess(self, image, image_meta, boxes=None):
        """
        Performs preprocessing on images
        """
        if boxes is not None:
            # [x, y, w, h] -> 0, 2 -> x and width
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0., image_meta['orig_size'][1] - 1.)  # min max out -> deals with the width
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0., image_meta['orig_size'][0] - 1.)   # deals with the height

        drift_prob = self.cfg.drift_prob if self.phase == 'train' else 0.
        flip_prob = self.cfg.flip_prob if self.phase == 'train' else 0.

        image, image_meta = whiten(image, image_meta, mean=self.rgb_mean, std=self.rgb_std)
        image, image_meta, boxes = drift(image, image_meta, prob=drift_prob, boxes=boxes)
        image,image_meta, boxes = flip(image, image_meta, prob=flip_prob, boxes=boxes)

        if self.cfg.forbid_resize:
            image, image_meta, boxes = crop_or_pad(image, image_metea, self.input_size, boxes=boxes)
        else:
            image, image_meta, boxes = resize(image, image_meta, self.input_size, boxes=boxes)

        return image, image_meta, boxes

    def prepare_annotations(self, class_ids, boxes):
        """
        :param class_size
        :param boxes: xyxy format
        :return: np.ndarray(#anchors, #classes + 9)
        """
        deltas, anchor_indices = compute_deltas(boxes, self.anchors)

        gt = np.zeros((self.num_anchors, self.num_classes + 9), dtype=np.float32)

        gt[anchor_indices, 0] = 1.
        gt[anchor_indice, 1:5] = boxes
        gt[anchor_indices, 5:9] = deltas
        gt[anchor_indices, 9+class_ids] = 1.   # class logits

        return gt

    def get_sample_ids(self):
        raise NotImplementedError
    def load_image(self):
        raise NotImplementedError
    def load_annnotations(self):
        raise NotImplementedError
    def evaluate(self):
        raise NotImplementedError
