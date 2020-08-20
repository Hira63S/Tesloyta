import torch
import argparse
import cv2
import numpy as np
from load_model import load_model, load_official_model
from SqueezeNet_detect import SqueezeDet
from detector import Detector
from config import Args
print('Hello Tesla! I am Raspberry Pi script')

ap = argparse.ArgumentParser()

ap.add_argument("--model", required=False, help='path to pre-trained model.')
ap.add_argument("--input", default=0, help='input source')
ap.add_argument("--output", help='where to write the processed file at')
# ap.add_argument("--num_classes", default=3, help='num of classes to predict')
# ap.add_argument("--num_anchors", default=9, help="Num of anchors to create")
# ap.add_argument("--arch", default='squeezedet', help="Either squeezedet | or SqueezeDetWithLoss")
# ap.add_argument("--dropout_prob", default=0.2, help="drop out proabbility")
# ap.add_argument("--anchors_per_grid",  default=9, help="num of anchors per grid")
# args = ap.parse_args()

CLASSES = ['Car', 'Pedestrian', 'Cyclist']
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load the model:

print("[INFO] loading model ... ")

model = SqueezeDet(args)
model = load_model(model, args.model)

print("[INFO] starting video stream ...")
vs = VideoStream(src=0).start()

# loop over the fraamesfrom the threaded video stream:

while True:

    frame = vs.read()
    frame = cv2.resize(frame, (384, 1248))

    # pass through the model:
    model.eval()
    dets = model(frame)
    print(dets)
    #
    # num_classes = len(CLASSES)
    # class_ids_dict = {cls_name: cls_id for cls_id, cls_name in enumerate(CLASSES)}
    #
    # (h, w) = frame.shape[:2]

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF:
        cv2.destroyAllWindows()
        vs.stop()
# preprocessing:
