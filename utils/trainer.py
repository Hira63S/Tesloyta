import torch
import torchvision


class Trainer():
    def __init__(self, model, optimizzer, lr_scheduler, args):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = scheduler
        self.args = args
        self.set_device(args.gpus, args.chunk_sizes, args.device)
        self.metrics = ['loss', 'class_labels', 'score_loss', 'bbox_loss']

    def run_epoch(self, phase, epoch, data_loader):
        start_time = time.time()

        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
            torch.cuda.empty_cache()

        metric_logger = {m: MetricLogger() for m in self.metrics}
        data_timer, net_timer = MetricLogger(), MetricLogger()
        num_iters = len(data_laoder) if self.args.num_iters < 0 else self.args.num_iters
        end = end.time()

        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break

            for k in batch: 
