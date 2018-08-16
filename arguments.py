import argparse
import torch

def get_args():

    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--batch_size', default=16,
                            help='Batch Size to be used')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--z_size', default=100)
    parser.add_argument('--env_names',
                        default=('Breakout-v0', 'AirRaid-v0', 'Pong-v0'))
    parser.add_argument('--lr',
                        default=0.0001)
    parser.add_argument('--log_iter',
                        default=100)
    parser.add_argument('--save_image_iter',
                        default=1000)
    parser.add_argument('--save_model_iter',
                        default =5000)
    parser.add_argument('--disc_filters',
                        default=64)
    parser.add_argument('--gener_filters',
                        default =64)
    parser.add_argument('--image_size', default=(64, 64))
    parser.add_argument('--save_dir', default='data')
    parser.add_argument('--image_to_generate', default=10000)
    parser.add_argument('--max_images_per_env_instance', default=5)
    parser.add_argument("--restore", type=str,
                        help="Restore already existing model from the path provided")
    parser.add_argument('--gen_iter', default=1)
    parser.add_argument('--dist_iter', default=5, type=int)
    parser.add_argument('--start_iter', default=0, type=int)
    parser.add_argument('--end_iter', default=int(10e4), type=int)


    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print("###########################################################################\n")
    print(args)
    print("###########################################################################\n")
    return args
