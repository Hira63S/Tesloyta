""" Running the Mask RCNN in videos """

import numpy as np
import argparse
import imutils
import os
import cv2
import time
import torchvision
import torchvision.transforms as transforms


ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input", required=True, help="path to input video file")
ap.add_argument("-o", "--output", required=True, help="path to the output file")
# ap.add_argument("-m", "--mask-rcnn", required=True, help="base path to mask-rcnn directory")
# ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
# ap.add_argument("-t", "--threshold", type=float, default=0.3, help="minimum threshold for pixel-wise mask segmentation")

args = vars(ap.parse_args())


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


#model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
#model.eval()
model = torchvision.models.resnet50(pretrained=True)

transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

print('model loaded')

# initialize the video stream and pointer to output video file
vs = cv2.VideoCapture(args["input"])
writer=None
fps = vs.get(cv2.CAP_PROP_FPS)
print("CAP_PROP_FPS: '{}'".format(fps))
# try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))

except:
    print("[INFO] could not determine # of frames in the video")
    total = -1

# let's loop over the video frames from the video filter

while True:

    (grabbed, frame) = vs.read()

    # if the frame is not grabbed, we have reached the blender
    if not grabbed:
        break

    # construct a blob from the input frame and then, do a forward pass:

#    blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
    fpsLimit =1
    startTime=time.time()
#    cv = cv2.VideoCapture(args.input)
    nowTime = time.time()
    if (int(nowTime-startTime)) > fpsLimit:
        startTime = time.time()

    nn_input = transform(frame)
    output = model([nn_input])

# let's iterate over the network output for all boxes
    processing_time =  time.time()

    for mask, box, score in zip(output[0]['masks'].detach().numpy(),
                                output[0]['boxes'].detach().numpy(),
                                output[0]['scores'].detach().numpy()):

        if score > 0.5:
            box = [(box[0], box[1]), (box[2], box[3])]

        # overlay the segmentation mask on the image with random color

            frame[(mask > 0.5).squeeze(), :] = np.random.uniform(0, 255, size=3)

        # draw the bounding boxes
            cv2.rectangle(img=frame,
                        pt1=box[0],
                        pt2=box[1],
                        color=(255, 255, 255),
                        thickness=2)


cv2.imshow("object detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
