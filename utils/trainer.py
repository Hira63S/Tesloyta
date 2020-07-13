import torch
import torchvision

from metriclogger import MetricLogger

class Trainer():
    def __init__(self, model, optimizer, lr_scheduler, args):
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

        metric_loggers = {m: MetricLogger() for m in self.metrics}
        data_timer, net_timer = MetricLogger(), MetricLogger()
        num_iters = len(data_laoder) if self.args.num_iters < 0 else self.args.num_iters
        end = end.time()

        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:  # num_iters num samples/batch_size
                break

            for k in batch:
                if 'image_meta' not in k:
                    batch[k] = batch[k].to(device=self.args.device, non_blocking=True)
            data_timer.update(time.time() - end)
            end = time.time()

            loss, loss_stats = self.model(batch)
            loss = loss.mean()

            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, self.model.parameters()),
                                        self.args.grad_norm)
                self.optimizer.step()

            msg = 'epoch {0:<3s} {1:<5s} [{2}/{3}]'.format(str(epoch) + ':', phase, iter_id, num_iters)

            for m in metric_loggers:
                value = loss_stats[m].mean().item()
                metric_loggers[m].update(value, batch['image'].shape[0])
                msg += '| {} {:.3f}'.format(m, value)

            net_timer.update(time.time() - end)
            end = time.time()

            msg = '| data {:.1f}ms | net {:.1f}ms'.format(1000. * data_timer.val, 1000. * net_timer.val)
            if iter_id % self.args.print_interval == 0:
                print(msg)

            del loss, loss_stats

        if phase == 'train':
            self.lr_scheduler.step()

        stats = {k: v.avg for k, v in metric_loggers.items()}
        stats.update({'epoch_time': (time.time() - start_time) / 60.})

        return stats

    def train_epoch(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)

    @torch.no_grad()
    def val_epoch(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)

    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            self.model = DataParallel(self.model, device_ids=gpus,
                                      chunk_sizes=chunk_sizes).to(device)
        else:
            self.model = self.model.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)
