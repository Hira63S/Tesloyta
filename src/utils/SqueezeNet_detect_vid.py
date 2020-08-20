import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from resolver import deltas_to_boxes, compute_overlaps, safe_softmax
from boxes import generate_anchors
__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']

model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        x = torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], dim=1)
        return x


class SqueezenetDet(nn.Module):

    def __init__(self, args):
        super(SqueezenetDet, self).__init__()
        self.num_classes = args.num_classes   # we get the number of classes and anchors
        self.num_anchors = args.num_anchors   # from the configs file

        if args.arch == 'squeezedet':
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
                Fire(512, 96, 384, 384),
                Fire(768, 96, 384, 384)
            )
        elif args.arch == 'squeezedetplus':
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 96, 64, 64),
                Fire(128, 96, 64, 64),
                Fire(128, 192, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 192, 128, 128),
                Fire(256, 288, 192, 192),
                Fire(384, 288, 192, 192),
                Fire(384, 384, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 384, 256, 256),
                Fire(512, 384, 256, 256),
                Fire(512, 384, 256, 256),
            )

        else:
            raise ValueError("Unsupported SqueezeNet version")

        # adding a drop out layer, might get rid of it later on
        self.dropout = nn.Dropout(args.dropout_prob, inplace=True) if args.dropout_prob > 0 else None
        self.convdet = nn.Conv2d(768 if args.arch == 'squeezedet' else 512,                              # score of how likely it is that the object exists in the box
                                 args.anchors_per_grid * (args.num_classes + 5),  # K (n_classes + 5) from the SqueezeDet paper +1 is for confidence
                                 kernel_size=3, padding=1)
        self.init_weights()

    def forward(self, x):
        x = self.features(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.convdet(x)
        x = x.permute(0, 2, 3, 1).contiguous()          # already doing that with transform so let's compare
        return x.view(-1, self.num_anchors, self.num_classes + 5)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.convdet:
                    nn.init.normal_(m.weight, mean=0.0, std=0.002)
                else:
                    nn.init.normal_(m.weight, mean=0.0, std=0.005)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
# would probably have to do something here

# prediction resolver i.e. make them interpretable


class PredictionResolver(nn.Module):
    def __init__(self, args, log_softmax=False):
        super(PredictionResolver, self).__init__()

        self.log_softmax = log_softmax
        self.input_size = args.input_size
        self.num_classes = args.num_classes
        self.grid_size = tuple(x // 16 for x in self.input_size)
        self.anchors_seed = np.array([[34, 30], [75, 45], [38, 90],
                                      [127, 68], [80, 174], [196, 97],
                                      [194, 178], [283, 156], [381, 185]], dtype=np.float32)
        self.gen_anchors = generate_anchors(self.grid_size, self.input_size, self.anchors_seed)
        self.anchors = torch.from_numpy(self.gen_anchors).unsqueeze(0).float()
        self.anchors_per_grid = self.anchors_seed.shape[0]


    def forward(self, pred):
        pred_class_probs = safe_softmax(pred[..., :self.num_classes].contiguous(), dim=-1)
        pred_log_class_probs = None if not self.log_softmax else \
            torch.log_softmax(pred[..., :self.num_classes].contiguous(), dim=-1)   # this would not include the +1 for C so we will
        # be fine because we will only have the number of probablities for all the classes expected
        pred_scores = torch.sigmoid(pred[..., self.num_classes:self.num_classes + 1].contiguous())
        pred_deltas = pred[..., self.num_classes + 1:].contiguous()
        pred_boxes = deltas_to_boxes(pred_deltas, self.anchors.to(pred_deltas.device),
                                     input_size=self.input_size)
        return pred_class_probs, pred_log_class_probs, pred_scores, pred_deltas, pred_boxes



# Define the loss function next

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.resolver = PredictionResolver(args, log_softmax=True)
        self.num_anchors = args.num_anchors
        self.class_loss_weight = args.class_loss_weight
        self.positive_score_loss_weight = args.positive_score_loss_weight
        self.negative_score_loss_weight = args.negative_score_loss_weight
        self.bbox_loss_weight = args.bbox_loss_weight

    def forward(self, pred, gt):
        # slice the gt tensor ground truth tensor
        anchor_masks = gt[..., :1]   # for all the rows, get the first one
        gt_boxes = gt[..., 1:5]      # the next 4 are the boxes
        gt_deltas = gt[..., 5:9]    # the next four are deltas
        gt_class_logits = gt[..., 9:]    # classification of the classes

        # resolver predictions
        pred_class_probs, pred_log_class_probs, pred_scores, pred_deltas, pred_boxes = self.resolver(pred)

        num_objects = torch.sum(anchor_masks, dim=[1,2])
        overlaps = computer_overlaps(gt_boxes, pred_boxes) * anchor_masks

        class_loss = torch.sum(
            self.class_loss_weight * anchor_masks * gt_class_logits * (-pred_log_class_probs),
        dim=[1,2],
        ) / num_objects

        positive_score_loss = torch.sum(self.position_score_loss_weight * anchor_masks * (overlaps - pred_scores) ** 2,
                                       dim=[1,2]) / num_objects

        negative_score_loss = torch.sum(self.negative_score_loss_weight * (1 - anchor_masks) * (overlaps - pred_scores) ** 2,
                                       dim=[1,2]) / (self.num_anchors - num_objects)


        bbox_loss = torch.sum(self.bbox_loss_weight * anchor_masks * (pred_deltas - gt_deltas) ** 2,
                             dim=[1,2],) / num_objects

        loss = class_loss + positive_score_loss + negative_score_loss + bbox_loss

        loss_stat = {
            'loss': loss,
            'class_loss': class_loss,
            'score_loss': positive_score_loss + negative_score_loss,
            'bbox_loss': bbox_loss

        }
        return loss, loss_stat


class SqueezeDetWithLoss(nn.Module):
    def __init__(self, args):
        super(SqueezeDetWithLoss, self).__init__()
        self.base = SqueezenetDet(args)
        self.loss = Loss(args)

    def forward(self, batch):
        pred = self.base(batch['image'])
        loss, loss_stats = self.loss(pred, batch['gt'])
        return loss, loss_stats

class SqueezeDet(nn.Module):
    """ Inference Model"""
    def __init__(self, args):
        super(SqueezeDet, self).__init__()
        self.base = SqueezenetDet(args)
        self.resolver = PredictionResolver(args, log_softmax=False)

    def forward(self, batch):
        pred = self.base(batch['image'])
        pred_class_probs, _, pred_scores, _, pred_boxes = self.resolver(pred)
        pred_class_probs *= pred_scores
        pred_class_ids = torch.argmax(pred_class_probs, dim=2)
        pred_scores = torch.max(pred_class_probs, dim=2)[0]
        det = {'class_ids': pred_class_ids,
              'scores': pred_scores,
              'boxes': pred_boxes}

        return det
