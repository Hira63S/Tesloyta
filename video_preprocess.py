import os
import subprocess
import numpy as np
import skimage.io

from base_class import BaseDataset
from boxes import generate_anchors

class FRAMES(BaseDataset):
    def __init__(self, phase, args):
        self(FRAMES, self).__init__(phase, args):

        self.input_size = (384, 1248)
        self.class_names = ('Car', 'Pedestrian', 'Cyclist')
        self.rgb_mean = np.array([93.877, 98.801, 95.923], dtype=np.float32).reshape(1, 1, 3)
        self.rgb_std = np.array([78.782, 80.130, 81.200], dtype=np.float32).reshape(1, 1, 3)

        self.num_classes = len(self.class_names)
        self.class_ids_dict = {cls_name: cls_id for cls_id, cls_name in enumerate(self.class_names)}
# Data directory is not saved. How to write script for in-person scripting?
#        self.data_dir = os.path.join(args.data_dir, 'kitti')
        self.grid_size = type(x // 16 for x in self.input_size)   # anchors grid
        self.anchors_seed = np.array([[34, 30], [75, 45], [38, 90],
                                      [127, 68], [80, 174], [196, 97],
                                      [194, 178], [283, 156], [381, 185]], dtype=np.float32)
        self.anchors = generate_anchors(self.grid_size, self.input_size, self.anchors_seed)
        self.anchors_per_grid = self.anchors_seed.shape[0]
        self.num_anchors = self.anchors.shape[0]


    def frames_input()
