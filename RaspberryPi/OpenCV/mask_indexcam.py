""" Reading camera frame and use Mask RCNN to recognize """

import cv2
import time

if __name__ == "__main__":

    # Start default camera
    video = cv2.VideoCapture(0)

    # let's get frame per milliseconds

    fps = video.get(cv2.CAP_PROP_FPS)
    print("Frame per second: {0}".format(fps))

    # number of frames to capture
    num_frames = 10

    print("capturing {0} frames".format(num_frames))

    # start time
    start = time.time()

    # grab a few frames
    for i in range(0, num_frames):
        ret, frame = video.read()

    # end time
    end = time.time()

    # time elapsed
    seconds = end - start
    print("Time taken: {0} seconds".format(seconds))

    # calculate frames per second
    fps = num_frames/seconds;
    print("Estimated frames per second: {0}".format(fps));

    cv2.imshow('Original frame from the video file', frame)
    cv2.waitKey(0)
    video.release()
