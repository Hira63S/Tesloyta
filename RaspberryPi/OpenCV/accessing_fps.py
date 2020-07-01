""" Reading videos' number of frames """

import cv2
import argparse
import time

# Let's create the object:
parser = argparse.ArgumentParser()

parser.add_argument("input", help="input video")
parser.add_argument("output", help="output video")
args = parser.parse_args()

# create a video capture object...?
capture = cv2.VideoCapture(args.input)
writer = None

# try:
#    prop = cv2.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
#        else cv2.CAP_CROP_FRAME_COUNT
#    total = int(capture.get(prop))
#    print("[INFO] {} total frames in video".format(total))

#except:
#    print("Could not print get the total frame count")

frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = capture.get(cv2.CAP_PROP_FPS)

print("Frame_width: '{}'".format(frame_width))
print("frame_height: '{}'".format(frame_height))
print("fps: '{}'".format(fps))

while capture.isOpened():
    # capture frame-by-frame from the source
    ret, frame = capture.read()

    if ret is True:

        processing_start = time.time()

        # display the captured frame:
        cv2.imshow('Input frame from the source', frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

        processing_end = time.time()

        processing_time_frame = processing_end - processing_start

        print('fps: {}'. format(1.0 / processing_time_frame))

    else:
        break

capture.release()
cv2.destroyAllWindows()
