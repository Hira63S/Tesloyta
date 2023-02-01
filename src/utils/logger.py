import os
import time
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

class MetricLogger(object):
    def __init__(self, args):
        dargs = dict((name, getattr(args, name)) for name in dir(args)
                    if not name.startswith('_'))

        os.makedirs(args.save_dir, exist_ok=True)
        file_name = os.path.join(args.save_dir, 'config.txt')
        msg = 'torch veersion: {}\ncudnn version: {}\ncmd: {}\n\nconfig:\n'.format(
            torch.__version__, torch.backends.cudnn.version(), str(sys.argv)
        )
        with open(file_name, 'w') as fp:
            fp.write(msg)
            for k, v in sorted(dargs.items()):
                fp.write('%s: %s\n' % (str(k), str(v)))

        self.log_dir = os.path.join(args.save_dir, 'logs_{}'.format(time.strftime('%Y-%m-%d-%H-%M')))
        os.makedirs(self.log_dir, exist_ok=True)
        os.system('cp {}/config.txt {}/'.format(args.save_dir, self.log_dir))

        self.metrics_history = {}

    def write(self, txt):
        print(txt + '\n')
        time_str = time.strftime('%Y-%m-%d-%H-%M')
        with open(os.path.join(self.log_dir, 'log.txt'), 'a') as fp:
            fp.write('{}: {}\n'.format(time_str, txt))

    def update(self, metrics, phase, epoch):
        text = 'epoch {0:<3s} {1:<5s} '.format(str(epoch) + ':', phase)
        for metric, value in metrics.items():
            if epoch not in self.metrics_history:
                self.metrics_history[epoch] = {}
            if phase not in self.metrics_history:
                self.metrics_history[epoch][phase] = {}
            self.metrics_history[epoch][phase].update({metric: value})

            if 'time' in metric:
                text += '| {} {:.2f}min '.format(metric, value)
            else:
                text += '| {} {:.3f} '.format(metric, value)

        self.write(text)

    def plot(self, metrics):
        for metric in metrics:
            train_epochs, train_values = [], []
            val_epochs, val_values = [], []
            for ep in self.metrics_history:
                if 'train' in self.metrics_history[ep] and \
                        metric in self.metrics_history[ep]['train']:
                    train_epochs.append(ep)
                    train_values.append(self.metrics_history[ep]['train'][metric])
                if 'val' in self.metrics_history[ep] and \
                        metric in self.metrics_history[ep]['val']:
                    val_epochs.append(ep)
                    val_values.append(self.metrics_history[ep]['val'][metric])

            plt.figure(figsize=(9, 6), dpi=150)
            plt.gcp().clear()
            plt.plot(train_epochs, train_values, label='train')
            plt.plot(val_epochs, val_values, label='validation')
            plt.xlabel('epoch')
            plt.ylabel('metric')
            plt.grid()
            plt.legend()
            save_path = os.path.join(self.log_dir, metric + '.png')
            plt.savefig(save_path)
            plt.close()

    def print_bests(self, metrics):
        """ print best metrics on validation set """

        for metric in metrics:
            epochs, values = [], []
            for ep in self.metrics_history:
                if 'val' in self.metrics_history[ep]:
                    epochs.append(ep)
                    values.append(self.metrics_history[ep]['val'][metric])
            if len(values) == 0:
                continue

            f = np.argmin if 'loss' in metric else np.argmax
            best_idx = int(f(values))
            print('Best {}: {:.3f} (epoch {]})'.format(
                metric, values[best_idx], epochs[best_idx]))
        print('\n')
