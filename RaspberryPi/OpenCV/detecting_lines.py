""" Class code to detect lane lines"""

import cv2
import argparse
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
# create an argument parser object

parser = argparse.ArgumentParser()

# add the video_path argument
parser.add_argument("video_path", help="path to the video file")
args = parser.parse_args()

# using openCV to  capture the camera feed
# capture = cv2.VideoCapture(args.index_camera)

# create a video capture object to read from pi camera.
camera = PiCamera()
# Resolution of camera
camera.resolution = (640, 480)
# frame rate i.e. frames per second
camera.framerate = 32
# changes the array to RGB and size?
rawCapture = PiRGBArray(camera, size=(640,480))

# allow the camera to warm up
time.sleep(0.1)

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw numpy array representing the image,
    # then, initialize the timestamp
    # and occupied/unoccpied text
    # capture frame-by-frame from the video file
    # ret, frame = capture.read()
    image = frame.array
#    image = frame.PiRGBArray

    # show the frame
    cv2.imshow("video_frames", image)
    key = cv2.waitKey(1) & 0xFF

    # show the grayscale frames
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale input camera", gray_frame)
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # if the 'q' key is pressed, break the loop
    blur = cv2.GaussianBlur(gray_frame, (7,7), 0)
    cv2.imshow('blurred', blur)

    #use canny to get the detection
    threshold_low = 10
    threshold_high = 200
    canny = cv2.Canny(blur, threshold_low, threshold_high)
    cv2.imshow('canny', canny)
    vertices = np.array([[(250,1080), (1000, 600), (1300,600), (1920,1080)]])
    mask = np.zeros_like(canny) # gray_frame nstead of canny
    cv2.fillPoly(mask, vertices, 255)
    masked_frame = cv2.bitwise_and(canny, mask)
    cv2.imshow('masked', masked_frame)

    # Hough lines

    rho = 2
    theta = np.pi/108
    threshold=40
    min_line_len=100
    max_line_gap=50
    lines = cv2.HoughLinesP(masked_frame, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

    #create an empty frame and draw all the lines in it:

    line_image = np.zeros((masked_frame.shape[0], masked_frame.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), [255, 0, 0], 20)

    α = 1
    β = 1
    γ = 0

# Resultant weighted image is calculated as follows: original_img * α + img * β + γ
    Image_with_lines = cv2.addWeighted(frame, α, line_image, β, γ)
    cv2.imshow('image with lines', Image_with_lines)

    # Press q on keyboard to exit the program
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
# Break the loop
    else:
        break

rawCapture.release()
cv2.destroyAllWindows()
