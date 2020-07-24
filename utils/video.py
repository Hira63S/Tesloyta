import torch
import torch.utils.data
from imutils.video import FPS
import argparse
import cv2
import numpy as np
from load_model import load_model, load_official_model
from SqueezeNet_detect_vid import SqueezeDet
from detector import Detector
from config import Args
from imutils.video import VideoStream
import os
import glob
import tqdm
import skimage.io
import imutils

args = Args().parse()

def vid_demo(args):
    """
    demo for the model
    """

    args.load_model = 'squeezedet_kitti_epoch280.pth'
    args.gpus = [-1]
    args.debug = 2    # visualize detection boxes

    # prepare the model and detector
    print("[INFO] loading model ... ")
    model = SqueezeDet(args)
    model = load_model(model, args.load_model)
    detector = Detector(model.to(args.device), args)

    print('Dector Loaded')
    # # prepare images
    # sample_images_dir = '../data/kitti/samples'
    # sample_image_imgs = glob.glob(os.img.join(sample_images_dir, '*.png'))

print("[INFO] starting video stream ...")
# vs = VideoStream(src=0).start()
# fps = FPS().start()
# loop over the fraamesfrom the threaded video stream:
capture = cv2.VideoCapture(-1)
frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = capture.get(cv2.CAP_PROP_FPS)

print("CV_Frame_width: '{}'")


while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=1248, height=384)


    cv2.imshow("frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    fps.update()
fps.stop()

cv2.destroyAllWindows()
    # pass through the model:
#         model.eval()
#         dets = model(frame)
#         print(dets)
#         #
#         # detection
#         for img in tqdm.tqdm(frame):
#             image = skimage.io.imread(img).astype(np.float32)
#             image_meta = {'image_id': os.img.basename(img)[:-4],
#                           'orig_size': np.array(image.shape, dtype=np.int32)}
#
#             image, image_meta, _ = preprocess_func(image, image_meta)
#             image = torch.from_numpy(image.transpose(2,0,1)).unsqueeze(0).to(args.device)
#             image_meta = {k: torch.from_numpy(v).unsqueeze(0).to(args.device) if isinstance(v, np.ndarray)
#                          else [v] for k, v in image_meta.items()}
#
#             inp = {'image': image,
#                    'image_meta': image_meta}
#
#             frm = detector.detect(inp)
#
#             cv2.imshow("Frame", frm)
#             if cv2.waitKey(1) & 0xFF:
#                 cv2.destroyAllWindows()
#                 vs.stop()
#
#
# print('Hi Tesla! I am demo ')
