"""
Example to introduce how to read a video file
"""

# Import the required packages
import cv2
import argparse
import numpy as np

# We first create the ArgumentParser object
# The created object 'parser' will have the necessary information
# to parse the command-line arguments into data types.
parser = argparse.ArgumentParser()

# We add 'video_path' argument using add_argument() including a help.
parser.add_argument("video_path", help="path to the video file")
args = parser.parse_args()

# Create a VideoCapture object. In this case, the argument is the video file name:
capture = cv2.VideoCapture(args.video_path)

frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = capture.get(cv2.CAP_PROP_FPS)

print("CV_frame_width:'{}'".format(frame_width))
print("CV_frame_height:'{}'".format(frame_height))
print("CAP_PROP_FPS:'{}'".format(fps))

# Check if the video is opened successfully
if capture.isOpened()is False:
    print("Error opening the video file!")

# Read until video is completed, or 'q' is pressed
while capture.isOpened():
    # Capture frame-by-frame from the video file
    ret, frame = capture.read()

    if ret is True:
        # Display the resulting frame
        cv2.imshow('Original frame from the video file', frame)

        # Convert the frame from the video file to grayscale:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the grayscale frame
        #cv2.imshow('Grayscale frame', gray_frame)

        # now do gaussian blur:
        blur = cv2.GaussianBlur(gray_frame, (7,7), 0)
        # sharpened image
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(frame, -1, kernel)
        #cv2.imshow('blurred', blur)

        #use canny to get the detection
        threshold_low = 10
        threshold_high = 200
        canny = cv2.Canny(blur, threshold_low, threshold_high)
        cv2.imshow('canny', canny)
        cv2.imshow('sharpened', sharpened)
        # Press q on keyboard to exit the program
        frame_index = 0
        if cv2.waitKey(20) & 0xFF == ord('d'):
            frame_name = "camera_frame_{}.png".format(frame_index)
            gray_frame_name = "grayscale_camera_frame_{}.png".format(frame_index)
            cv2.imwrite(frame_name, frame)
            cv2.imwrite(gray_frame_name, gray_frame)
            frame_index += 1

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    # Break the loop
    else:
        break

# Release everything
capture.release()
cv2.destroyAllWindows()
