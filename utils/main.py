from config import Args
from metriclogger import init_env

args = Args().parse()
init_env(args)

if args.mode == 'train':
    from train import train
    train(args)
#elif args.mode == 'eval':
#    from eval import eval
#    eval(args)
elif args.mode == 'demo':
    from demo import demo
    demo(args)
else:
    raise ValueError('Mode {} is invalid.'.format(args.mode))
