import torch
import torch.utils.data

from detector import Detector
from SqueezeNet_detect import SqueezeDet
from config import Args
from load_model import load_model
from train import load_dataset


def eval(cfg):
    dataset = load_dataset(cfg.dataset)('val', cfg)
    cfg = Args().update_dataset_info(cfg, dataset)
    Args().print(cfg)

    aps = eval_dataset(dataset, cfg.load_model, cfg)
    for k, v in aps.items():
        print('{:<20} {:.3f}'.format(k, v))

    torch.cuda.empty_cache()


def eval_dataset(dataset, model_path, cfg):
    model = SqueezeDet(cfg)
    model = load_model(model, model_path)

    detector = Detector(model, cfg)

    results = detector.detect_dataset(dataset)
    dataset.save_results(results)
    aps = dataset.evaluate()

    return aps


def eval_video(frame, model_path, args):
    model = SqueezeDet(args)
    model = load_model(model, model_path)
    detector = Detector(model, args)
    results = detector.detect_dataset(frame)
    dataset.save_results(results)
    aps = dataset.evaluate()
    return aps
