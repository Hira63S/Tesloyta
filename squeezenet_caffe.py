import argparse
import numpy as np
import time
import cv2

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image", required=True, help="input image link")
# ap.add_argument("-o", "--output", required=True, help="output image file")
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to caffe pretrained model")
ap.add_argument("-l", "--labels", required=True, help="labels file")

args = vars(ap.parse_args())

# loas the class labels from disk
rows = open(args["labels"]).read().strip().split("\n")
classes = [r[r.find(" ") +1:].split(",")[0] for r in rows]

image = cv2.imread(args["image"])

blob = cv2.dnn.blobFromImage(image, 1, (227, 227), (104, 117, 123))

print("[INFO] loading model ... ")

net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# set the blobe equal to the input to the network
net.setInput(blob)
# start the timer
start = time.time()
preds = net.forward()
end = time.time()

print("[INFO] classification took {:.5} seconds".format(end - start))

# predictions and idxs
preds = preds.reshape((1, len(classes)))
idxs = np.argsort(preds[0])[::-1][:5]

for (i, idx) in enumerate(idxs):
    if i == 0:
        text = 'Label: {}, {:.2f}%'.format(classes[idx],
             preds[0][idx] * 100)
        cv2.putText(image, text, (5,25), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2)

    print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1,
        classes[idx], preds[0][idx]))

cv2.imshow("Image", image)
cv2.waitKey(0)
