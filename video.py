import torch
import torch.utils.data
# from .images import whiten, drift, flip, resize, crop_or_pad
# from .boxes import compute_deltas, visualize_boxes
import numpy as np
from config import Args
import cv2
from demo import demo
import os
import skimage
import time

from datasets.kitti_class import KITTI
# from engine.detector import Detector
from utils.detector import Detector
from model.SqueezeNet_detect import SqueezeDet
from utils.config import Args
from load_model import load_model
from utils.boxes import visualize_boxes, boxes_postprocess
args = Args().parse()

# can make use of OpenCV to do the video frames processing

# import the demo function from the demo.py file
# maybe can just import the demo func

# cfg would have the video link
def vid_demo(cfg):

    # model input:
    cfg.load_model = cfg.load_model
    cfg.gpus = [-1]  # -1 to use CPU
    cfg.debug = 2  # to visualize detection boxes
    dataset = KITTI('val', cfg)
    cfg = Args().update_dataset_info(cfg, dataset)

    # preprocess image to match model's input resolution
    preprocess_func = dataset.preprocess
    del dataset

    class_colors = (255. * np.array(
    [0.850, 0.325, 0.098,
     0.466, 0.674, 0.188,
     0.098, 0.325, 0.850,
     0.301, 0.745, 0.933,
     0.635, 0.078, 0.184,
     0.300, 0.300, 0.300,
     0.600, 0.600, 0.600,
     1.000, 0.000, 0.000,
     1.000, 0.500, 0.000,
     0.749, 0.749, 0.000,
     0.000, 1.000, 0.000,
     0.000, 0.000, 1.000,
     0.667, 0.000, 1.000,
     0.333, 0.333, 0.000,
     0.333, 0.667, 0.000,
     0.333, 1.000, 0.000,
     0.667, 0.333, 0.000,
     0.667, 0.667, 0.000,
     0.667, 1.000, 0.000,
     1.000, 0.333, 0.000,
     1.000, 0.667, 0.000,
     1.000, 1.000, 0.000,
     0.000, 0.333, 0.500,
     0.000, 0.667, 0.500,
     0.000, 1.000, 0.500]))
    # prepare model & detector
    model = SqueezeDet(cfg)
    model = load_model(model, cfg.load_model)
    detector = Detector(model.to(cfg.device), cfg)
    # video input:
    # file_dir = cfg.root_dir 
    # file_dir = 'C://users/cathx/repos/Tesloyta/data/CV_model2.mp4'
    file_dir = 'C://users/cathx/repos/PythonPractice/OpenCV/CV_model_30.avi'
    print('file_dir: ', file_dir)
    img_array = []

    vid  = cv2.VideoCapture(file_dir)

    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = vid.get(cv2.CAP_PROP_FPS)
    path = os.path.join(cfg.save_dir, 'test_vids.avi')
    fourcc = cv2.VideoWriter_fourcc(*'DVIX')
    video_writer = cv2.VideoWriter(filename=path, apiPreference=0, fourcc=fourcc, fps=fps, frameSize=(int(width), int(height)))

    if not vid.isOpened():
        print('cannot open vid')
        exit()

    while vid.isOpened():
        fps = vid.get(cv2.CAP_PROP_FPS)
        # print('fps', fps)

        ret, frame = vid.read()

        frame_count = 0
        started = time.time()
        last_logged = time.time()
        if ret == True:
            # print('fps', fps)
            image = frame
            # cv2.imshow('image', image)
            img_h, img_w = image.shape[:2]
            image = np.array(image).astype(np.float32)
            vis_image = image
            # image = skimage.io.imread(image).astype(np.float32)
            # image = cv2.imread(image).astype(np.float32)
            image_meta = {'image_id': str(vid.get(cv2.CAP_PROP_POS_FRAMES)),
                          'orig_size': np.array(image.shape, dtype=np.int32)}
            
            image, image_meta, _ = preprocess_func(image, image_meta)
            image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(cfg.device)
            image_meta = {k: torch.from_numpy(v).unsqueeze(0).to(cfg.device) if isinstance(v, np.ndarray)
                      else [v] for k, v in image_meta.items()}

            inp = {'image': image,
                'image_meta': image_meta}
            # print('inp: ', inp)

            output = detector.detect(inp)
            # print('output: ', output)
            # image = np.ascontiguousarray(image.cpu()).astype(np.uint8)
            # if output[0]['image_meta'] is not None:
            #     continue
            if output is None or output == []:
                continue
            else:
                num_boxes = output[0]['boxes'].shape[0]

                dets_ids = output[0]['class_ids']
                dets_boxes = output[0]['boxes']
                dets_scores = output[0]['scores']
                class_names = ('Car', 'Pedestrian', 'Cyclist', 'Truck')
                image = image[0].cpu().numpy().transpose(1, 2, 0) #.astype(np.uint8)
                # print('image_shape', image.shape)
                # img = np.array(image.cpu()).astype(np.uint8)
                # img = img.transpose((1,2,0))
                # print('img_shape', img.shape)
                # vis_image = image.cpu()
                # vis_image = np.array(vis_image).astype(np.uint8)
                # save_path = args.save_path
                # img = image.transpose((1, 2, 0)).astype(np.uint8)
                if num_boxes > 0:
                    for i in range(num_boxes):
                        class_id = dets_ids[i]
                        bbox = dets_boxes[i].astype(np.uint32).tolist()

                        # print('image_shape', image.shape)
                        image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                                        class_colors[class_id].tolist(), 2)
                        
                        class_name= class_names[class_id] if class_names is not None else 'class_{}'.format(class_id)
                        text = '{} {}:.2f'.format(class_name, dets_scores[i]) if dets_scores is not None else class_name
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text_size = cv2.getTextSize(text, font, fontScale=.5, thickness=1)[0]
                        image = cv2.rectangle(image,
                                            (bbox[0], bbox[1] - text_size[1] - 8),
                                            (bbox[0] + text_size[0] + 8, bbox[1]),
                                            class_colors[class_id].tolist(), -1)
                        image = cv2.putText(image, text, (bbox[0] + 4, bbox[1] - 4), font,
                                            fontScale=.5, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                        # print('image_shape', image.shape)
                
            print('do we get here: ')
            frame_count += 1
            now = time.time()
            print(f"{frame_count / (now-last_logged)} fps")
            if now - last_logged > 1:
                print(f"{frame_count / (now-last_logged)} fps")
                last_logged = now
                frame_count = 0
            video_writer.write(image)
            cv2.imshow('image', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            break
        # cv2.waitKey()
    vid.release()
    video_writer.release()
    cv2.destroyAllWindows()
