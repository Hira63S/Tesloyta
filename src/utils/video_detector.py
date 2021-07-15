import torch
from torchvision.ops import nms
from images import image_postprocess
from boxes import boxes_postprocess, visualize_boxes
from metriclogger import MetricLogger
import cv2


class Detector(object):
    def __init__(self, model, args):
        self.model = model.to(args.device)
        self.model.eval()
        self.args = args

    @torch.no_grad()
    def detect(self, frame):
        # We get the detections from the frame
        dets = self.model(frame)

        results = []
        batch_size = dets['class_ids'].shape[0]
        for b in range(batch_size):
            image_meta = {k: v[b].cpu().numpy() if not isinstance(v, list) else v[b]
                         for k, v in frame['image_meta'].items()}

            det = {k: v[b] for k, v in dets.items()}
            det = self.filter(det)

            if det is None:
                results.append({'image_meta': image_meta})
                continue

            det = {k: v.cpu().numpy() for k, v in det.items()}
            det['boxes'] = boxes_postprocess(det['boxes'], image_meta)
            det['image_meta'] = image_meta
            results.append(det)

            if self.args.debug==2:
                image = image_postprocess(frame['image'][b].cpu().numpy().transpose(1, 2, 0), image_meta)
                save_path = os.path.join(self.args.debug_dir, image_meta['image_id'] + '.png')
                visualize_boxes(image, det['class_ids'], det['boxes'], det['scores'],
                                 class_names = self.args.class_names,
                                 save_path = save_path,
                                 show = self.args.mode=='demo')
        return results
    #
    # def detect_frame(self, frame):
    #
    #     start_time = time.time()
    #
    #     capture = cv2.VideoCapture(0)
    #
    #     frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    #     frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    #     fps = capture.get(cv2.CAP_PROP_FPS)
    #     frame_index = capture.get(cv2.CAP_PROP_FRAME_COUNT) - 1
    #     # load the dataset:
    #
    #
    #     ret, frame = capture.read()

    def filter(self, det):
            orders = torch.argsort(det['scores'], descending=True)[:self.cfg.keep_top_k]
            class_ids = det['class_ids'][orders]
            scores = det['scores'][orders]
            boxes = det['boxes'][orders, :]

            # class-wise nms
            filtered_class_ids, filtered_scores, filtered_boxes = [], [], []
            for cls_id in range(self.cfg.num_classes):
                idx_cur_class = (class_ids == cls_id)
                if torch.sum(idx_cur_class) == 0:
                    continue

                class_ids_cur_class = class_ids[idx_cur_class]
                scores_cur_class = scores[idx_cur_class]
                boxes_cur_class = boxes[idx_cur_class, :]

                keeps = nms(boxes_cur_class, scores_cur_class, self.cfg.nms_thresh)

                filtered_class_ids.append(class_ids_cur_class[keeps])
                filtered_scores.append(scores_cur_class[keeps])
                filtered_boxes.append(boxes_cur_class[keeps, :])

            filtered_class_ids = torch.cat(filtered_class_ids)
            filtered_scores = torch.cat(filtered_scores)
            filtered_boxes = torch.cat(filtered_boxes, dim=0)

            keeps = filtered_scores > self.cfg.score_thresh
            if torch.sum(keeps) == 0:
                det = None
            else:
                det = {'class_ids': filtered_class_ids[keeps],
                       'scores': filtered_scores[keeps],
                       'boxes': filtered_boxes[keeps, :]}

            return det
