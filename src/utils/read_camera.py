"""reading camera and starting videostream"""

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=1248, height=384)

    cv2.imshow("frame", frame)
    cv2.waitKey(0)

cv2.destroyWindow()
