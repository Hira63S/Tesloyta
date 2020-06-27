""" Passing images to be predicted vs. videos"""

import cv2
import os
import argparse
import torchvision
import torchvision.transforms as transforms
import numpy as np


ap = argparse.ArgumentParser()

ap.add_argument("path_image", help="image to be predicted")
ap.add_argument("output", help="image with predictions")

args = ap.parse_args()

# load the input image from disk
img = cv2.imread(args.path_image)

# parse the argument and store it in a dictionary:
# args = vars(parser.parse_args())
# to load the input image from disk using args:
# image2 = cv2.imread(args["path_image"])

classes = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack',
    'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
    'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush']

# load the model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
# model = torchvision.models.resnet50(pretrained=True)
# set the model in evaluation model
model.eval()


transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
nn_input = transform(img)
output = model([nn_input])

colors = np.random.uniform(0, 255, size=(len(classes), 3))

# iterate over the network output for all boxes

for box, box_class, score in zip(output[0]['boxes'].detach().numpy(),
                                 output[0]['labels'].detach().numpy(),
                                 output[0]['scores'].detach().numpy()):

    # filter the boxes by score
    if score > 0.5:
        # transform bounding box format
        box = [(box[0], box[1]), (box[2], box[3])]

        # select class color
        color = colors[box_class]

        # extract class name
        class_name = classes[box_class]

        # draw the bounding box
        cv2.rectangle(img=img,
                      pt1=box[0],
                      pt2=box[1],
                      color=color,
                      thickness=2)

        # display the box class label
        cv2.putText(img=img,
                    text=class_name,
                    org=box[0],
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=color,
                    thickness=2)


cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
