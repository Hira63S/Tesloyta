
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import torchvision
import torchvision.transforms as transforms

ap = argparse.ArgumentParser()
ap.add_argument("input", help="path to the input file")
ap.add_argument("output", help="path to output file")
# ap.add_argument("-o", "--output", required=True, help="path to the output video file")
# ap.add_argument("-m", "--mask-rcnn", required=True, help="path to the mask-rcnn directory")
# ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
# ap.add_argument("-t", "--threshold", type=float, default=0.3, help="minimum threshold for pixel-wise mask segmentation")
args = ap.parse_args()
capture = cv2.VideoCapture(args.input)
writer = None
capture.set(cv2.CAP_PROP_FPS, 1)

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

# set the model in evaluation model
model.eval()

# read the file:


# trying to determine the total number of frames in the video:
#try:
#    prop = cv2.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
#        else cv2.CV_CAP_PROP_FRAME_COUNT
#    total = int(capture.get(prop))
#    print("[INFO] {} total frames in video".format(total))

#except:
#    print("[INFO] could not determine # of frames in the video")
#    total = -1

# frame processing loop:
while True:
    # read the tnext frame from the filter
    (grabbed, frame) = capture.read()

    fpsLimit =1
    startTime=time.time()
    cv = cv2.VideoCapture(args.input)
    nowTime = time.time()
    if (int(nowTime-startTime)) > fpsLimit:
        startTime = time.time()

    # if the frame was not grabbed, we have reached the end of the stream:
    if not grabbed:
        break

    transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
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
