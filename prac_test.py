""" Practice file to see capturing frame working """


import argparse
import numpy as np
import cv2


ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--input", required=True,
#    help = "input file")
ap.add_argument("-o", "--output", required=True,
    help="output file")
ap.add_argument("-p", "--prototxt", required=True,
    help="layers structure")
ap.add_argument("-m", "--model", required=True,
    help="weights file?")
ap.add_argument("-c", "--confidence", required=True,
    help="confidence level")

args = vars(ap.parse_args())

# CLASSES and Model:
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load the model:
print("[INFO] loading model ...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Start the capturing of the video
print("Starting video stream ... ")
capture = cv2.VideoCapture(0)

w = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
h = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = capture.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'XVID')

writer = cv2.VideoWriter(args["output"], fourcc, int(fps), (int(w), int(h)), False)


while capture.isOpened():

    ret, frame = capture.read()
    if ret is True:
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
            0.007843, (300, 300), 127.5)

        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence > args["confidence"]:

                idx = int(detections[0, 0, i, 1])
                # i is the confidence score
                # the 3:7 get the 4 coordinates of the box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                label = "{}: {:.2f}%".format(CLASSES[idx],
                    confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        writer.write(gray_frame)

        cv2.imshow('gray', gray_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

capture.release()
writer.release()
cv2.destroyAllWindows()
