def detlas_to_boxes(deltas, anchors, input_size):
    """
    :params deltas: xywh format
    :params anchorsL xywh format
    :params input_size: input image size in hw format
    :return: boxes in xyxy format
    """
    boxes_xywh = torch.cat([
        anchors[..., [0]] + anchors[..., [2]] * deltas[..., [0]],
        anchors[..., [1]] + anchors[..., [3]] * deltas[..., [1]],
        anchors[..., [2]] * torch.exp(deltas[..., [2]]),
        anchors[..., [3]] * torch.exp(deltas[..., [3]])
    ], dim=2)

    boxes_xyxy = xywh_to_xyxy(boxes_xywh)
    boxes_xyxy[..., [0, 2]] = torch.clamp(boxes_xyxy[..., [0,2]], 0, input_size[1] -1)
    boxes_xyxy[..., [1, 3]] = torch.clamp(boxes_xyxy[..., [1,3]], 0, input_size[0] -1)

    return boxes_xyxy

def xyxy_to_xywh(boxes_xyxy):
    assert torch.all(boxes_xyxy[..., 0] < boxes_xyxy[..., 2])
    assert torch.all(boxes_xyxy[..., 1] < boxes_xyxy[..., 3])
    return torch.cat([
        (boxes_xyxy[..., [0]] + boxes_xyxy[..., [2]]) / 2.,
        (boxes_xyxy[..., [1]] + boxes_xyxy[..., [3]]) / 2.,
        boxes_xyxy[..., [2]] - boxes_xyxy[..., [0]] + 1.,
        boxes_xyxy[..., [3]] - boxes_xyxy[..., [1]] + 1.
    ], dim=-1)


def xywh_to_xyxy(boxes_xywh):
    assert torch.all(boxes_xywh[..., [2, 3]] > 0)
    return torch.cat([
        boxes_xywh[..., [0]] - 0.5 * (boxes_xywh[..., [2]] - 1),
        boxes_xywh[..., [1]] - 0.5 * (boxes_xywh[..., [3]] - 1),
        boxes_xywh[..., [0]] + 0.5 * (boxes_xywh[..., [2]] - 1),
        boxes_xywh[..., [1]] + 0.5 * (boxes_xywh[..., [3]] - 1)
    ], dim=-1)
