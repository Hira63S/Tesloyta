from config import Args
from metriclogger import init_env

args = Args().parse()
init_env(args)

if args.mode == 'train':
    from train import train
    train(args)
elif args.mode == 'eval':
    from eval import eval as evaluate
    evaluate(args)
elif args.mode == 'demo':
    from demo import demo
    demo(args)
elif args.mode == 'vid_demo':
    from video_model import vid_demo
    vid_demo(args)
else:
    raise ValueError('Mode {} is invalid.'.format(args.mode))
