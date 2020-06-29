""" Object Detection in w/ PyTorch and Pytorch pre-trained model: MobileNet """

import cv2
import time
import imutils
import argparse
import numpy as np
from imutils.video import FPS
from imutils.video import VideoStream
import torchvision
import torch
import torchvision.transforms as transforms
from PIL import Image


ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input", required=True, help="Input file")
ap.add_argument("-o", "--output", required=True, help="output location/name")
# ap.add_argument("-p", "--picamera", type=int, default=-1, help="whether to use PiCamera")
ap.add_argument("-f", "--fps", type=int, default=20, help="FPS of output video")
# ap.add_argument("-c", "--codec", type=str, default="MJPG", help="codec of output video")
# ap.add_argument("-p", "--prototxt", required=True,
#	help="path to Caffe 'deploy' prototxt file")
# ap.add_argument("-m", "--model", required=True,
#	help="path to Caffe pre-trained model")

ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")


args = vars(ap.parse_args())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
model = torchvision.models.mobilenet_v2(pretrained=True)
# set the model mode:
model.eval()

print("[INFO] warming up the camera ... ")
vs = VideoStream(src=0).start()

time.sleep(2.0)
fps = FPS().start()

while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # net.setInput(blob)
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # frame = torch.from_numpy(frame).long()
    # print(type(frame))
    frame = Image.fromarray(frame)
    input_tensor = preprocess(frame)
    input_batch = input_tensor.unsqueeze(0)


    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    #    detections = model(blob)

    for i in np.arange(0, output.shape[2]):
        confidence = output[0, 0, i, 2]

        if confidence > args["confidence"]:
            idx = int(output[0, 0, i, 1])
            box = output[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    out = cv2.VideoWriter(args["output"], cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
            6, (h, w), True)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    fps.update()

out.write(frame)
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyALLWindows()
vs.stop()
out.release()

# read the frame
# time.sleep(2.0)
# vs = cv2.VideoCapture(args["input"])
