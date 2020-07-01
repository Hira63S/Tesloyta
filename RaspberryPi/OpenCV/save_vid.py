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
parser.add_argument("output_video_path", help="path to the vid file to write")
args = parser.parse_args()

# Create a VideoCapture object. In this case, the argument is the video file name:
capture = cv2.VideoCapture(args.video_path)

frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = capture.get(cv2.CAP_PROP_FPS)

print("CV_frame_width:'{}'".format(frame_width))
print("CV_frame_height:'{}'".format(frame_height))
print("CAP_PROP_FPS:'{}'".format(fps))

fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')

out_gray = cv2.VideoWriter(args.output_video_path, fourcc, int(fps), (int(frame_width), int(frame_height)), False)



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

        out_gray.write(gray_frame)
        # Display the grayscale frame
        cv2.imshow('gray', gray_frame)

        #cv2.imshow('Grayscale frame', gray_frame)

        # Define the region of interest


        #now do gaussian blur:
        blur = cv2.GaussianBlur(gray_frame, (7,7), 0)
        cv2.imshow('blurred', blur)

        #use canny to get the detection
        threshold_low = 10
        threshold_high = 200
        canny = cv2.Canny(blur, threshold_low, threshold_high)
        cv2.imshow('canny', canny)
        vertices = np.array([[(100,700),(1000, 500), (1250, 500), (1900,800)]], dtype=np.int32)
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

# Release everything
capture.release()
out_gray.release()
cv2.destroyAllWindows()
