""" A function that creates the anchors instead of having to do the whole
dataset thing. We can pass that function into squeezenet model class
for into prediction resolver
"""
from boxes import generate_anchors

# takes in grid_size, input_size, anchors_seed:
# predictionresolver takes in anchors and anchors_per_grid
def frame_anchors():
    input_size = (384, 1248)
    class_names = ('Car', 'Pedestrian', 'Cyclist')
    rgb_mean = np.array([93.877, 98.801, 95.923], dtype=np.float32).reshape(1, 1, 3)
    rgb_std = np.array([78.782, 80.130, 81.200], dtype=np.float32).reshape(1, 1, 3)

    num_classes = len(self.class_names)
    class_ids_dict = {cls_name: cls_id for cls_id, cls_name in enumerate(class_names)}  # gives the class ids numbers

    grid_size = tuple(x // 16 for x in self.input_size)   # anchors grid
    anchors_seed = np.array([[34, 30], [75, 45], [38, 90],
                                  [127, 68], [80, 174], [196, 97],
                                  [194, 178], [283, 156], [381, 185]], dtype=np.float32)
    anchors = generate_anchors(grid_size, input_size, anchors_seed)
    anchors_per_grid = self.anchors_seed.shape[0]   # 9
    num_anchors = self.anchors.shape[0]   # 9 anchors per grid and we pass in input size and anchor seed

    results_dir = os.path.join(args.save_dir, 'results')

    return anchors, anchors_per_grid

    
