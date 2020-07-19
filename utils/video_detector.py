import torch
from torchvision.ops import nms
from images import image_postprocess
from boxes import boxes_postprocess, visualize_boxes
from metriclogger import MetricLogger


class Detector(object):
    def __init__(self, model, args):
        self.model = model.to(args.device)
        self.model.eval()
        self.args = args

    @torch.no_grad()
    def detect(self, frame):
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
