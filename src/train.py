"""
Script for training the model
"""

import os
import operator

import torch
import torch.utils.data
from torch.optim.lr_scheduler import StepLR

from load_model import load_model, load_official_model, save_model
from engine.trainer import Trainer
from utils.logger import MetricLogger
from model.SqueezeNet_detect import SqueezeDetWithLoss
from config import Args

# load dataset
def load_dataset(dataset_name):
    if dataset_name.lower() == 'kitti':
        from datasets.kitti_class import KITTI as Dataset
    return Dataset

# load_dataset('kitti')

def train(args):
    Dataset = load_dataset(args.dataset)
    training_data = Dataset('train', args)  # dataset takes in train, val, or trainval as params
    val_data = Dataset('val', args)
    args = Args().update_dataset_info(args, training_data)   # takes care of params in kitti class like mean, std
    Args().print(args)
    logger = MetricLogger()

    model = SqueezeDetWithLoss(args)
    if args.load_model != '':
        if args.load_model.endswith('f364aa15.pth') or args.load_model.endswith('a815701f.pth'):
            model = load_official_model(model, args.load_model)
        else:
            model = load_model(model, args.load_model)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr= args.lr,
                                momentum=args.momentum,
                                weight_decay = args.weight_decay)
#   Adam does not use momentum  momentum = args.momentum,
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.5)

    # Trainer is the model training class
    trainer = Trainer(model, optimizer, lr_scheduler, args)

    train_loader = torch.utils.data.DataLoader(training_data,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               pin_memory=True,
                                               shuffle=True,
                                               )
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers,
                                             pin_memory=True)

    metrics = trainer.metrics if args.no_eval else trainer.metrics + ['mAP']
    best = 1E9 if args.no_eval else 0
    # FIX THIS
    better_than = operator.lt if args.no_eval else operator.gt

    for epoch in range(1, args.num_epochs+1):
        train_stats=trainer.train_epoch(epoch, train_loader)
        logger.update(train_stats, phase='train', epoch=epoch)

        # save the model weights
        save_path = os.path.join(args.save_dir, 'model_last.pth')
        save_model(model, save_path, epoch)

        if epoch % args.save_intervals == 0:
            save_path = os.path.join(args.save_dir, 'model_{}.pth'.format(epoch))
            save_model(model, save_path, epoch)

        if args.val_intervals > 0 and epoch % args.val_intervals == 0:
            val_stats = trainer.val_epoch(epoch, val_loader)
            logger.update(val_stats, phase='val', epoch=epoch)

            if not args.no_eval:
                aps = eval_dataset(val_dataset, save_path, args)
                logger.update(aps, phase='val', epoch=epoch)

            value = val_stats['loss'] if args.no_eval else aps['mAP']
            if better_than(value, best):
                best = value
                save_path = os.path.join(args.save_dir, 'model_best.pth')
                save_model(model, save_path, epoch)

        # logger.plot(metrics)
        logger.print_bests(metrics)

    torch.cuda.empty_cache()


print('Hello Tesla!')
