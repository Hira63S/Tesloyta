""" Read_camera module to read videos from raspberry pi camera """

from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import argparse

# initialize the camera and grab a refernce to the raw camera capture
# interface with Pi camera
camera = PiCamera()
# Resolution of camera
camera.resolution = (640, 480)
# frame rate i.e. frames per second
camera.framerate = 32
# changes the array to RGB and size?
rawCapture = PiRGBArray(camera, size=(640,480))

# test w/ index_camera
# parser = argparse.ArgumentParser()
# parser.add_argument("index_camera", help="index of the camera to read from", type=int)
# args = parser.parse_args()

# rawCapture = cv2.VideoCapture(args.index_camera)

# allow the camera to warup
time.sleep(0.1)

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw numpy array representing the image,
    # then, initialize the timestamp
    # and occupied/unoccpied text
    # capture frame-by-frame from the video file
    ret, frame = capture.read()

    image = frame.PiRGBArray

    # show the frame
    cv2.imshow("video_frames", image)
    key = cv2.waitKey(1) & 0xFF

    # show the grayscale frames
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale input camera", gray_frame)
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # if the 'q' key is pressed, break the loop

    if key == ord("q"):
        break


rawCapture.release()
##  grab an image from the camera
# camera.capture(rawCapture, format="bgr")
# image = rawCapture.array

# cv2.imshow("Image", rawCapture)
# cv2.waitKey(0)
