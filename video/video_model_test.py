import torch
import torch.utils.data
from imutils.video import FPS
import argparse
import cv2
import numpy as np
from load_model import load_model, load_official_model
from SqueezeNet_detect_vid import SqueezeDet
from video_detector import Detector
from config import Args
from imutils.video import VideoStream
import os
import glob
import tqdm
import skimage.io
import imutils
import PIL
from video import preprocess_func

args = Args().parse()

def vid_demo(args):
    """
    demo for the model
    """

    args.load_model = 'squeezedet_kitti_epoch280.pth'
    args.gpus = [-1]
    args.debug = 2    # visualize detection boxes

    print('Detector Loaded')

    print("[INFO] starting video stream ...")

    capture = cv2.VideoCapture(0)
    frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_index = capture.get(cv2.CAP_PROP_FRAME_COUNT) - 1

    print("Frame_width is: {}, Frame_height is: {}, Total FPS are: {}".format(frame_width, frame_height, fps))

    # loop over the fraamesfrom the threaded video stream:

    while capture.isOpened():

        ret, frame = capture.read()
        for iter_id, frame in enumerate(frame):
            for k in frame:
                if 'image_meta'
        frame = cv2.resize(frame, (1248, 384))

        print(type(frame))
        print(frame.shape)
#        model.eval()
        model = SqueezeDet(args)
        model = load_model(model, args.load_model)
        detector = Detector(model.to(args.device), args)     # detector takes care of the model.eval()

        if ret is True:
            # detection
    #        for image in tqdm.tqdm(frame):

                # image = skimage.io.imread(img).astype(np.float32)
    #        image = torch.from_numpy(frame.transpose(2, 0, 1)).unsqueeze(0).to(args.device)
            image = frame
            image_meta = {'image_id': capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index),
            'orig_size': np.array(image.shape, dtype=np.int32)}

            print("Image shape before the preprocessing function is: {}".format(image.shape))
            image, image_meta, _ = preprocess_func(image, image_meta)
            print(type(image))
            print("Image shape is: {}".format(image.shape))
            image = torch.from_numpy(image.transpose(2,0,1)).unsqueeze(0).to(args.device)
            print(type(image))
            image_meta = {k: torch.from_numpy(v).unsqueeze(0).to(args.device) if isinstance(v, np.ndarray)
            else [v] for k, v in image_meta.items()}

            inp = {'image': image,
            'image_meta': image_meta}
            print(image.shape)
            # img = torch.from_numpy(img)
            _ = detector.detect(inp)


            cv2.imshow("Frame", _)
            if cv2.waitKey(1) & 0xFF == ord('1'):
                break

    # fps.update()
    # fps.stop()
    capture.release()
    cv2.destroyAllWindows()
    print('Hi Tesla! I am demo ')
