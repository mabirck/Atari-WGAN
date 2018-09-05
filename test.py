from arguments import get_args
import os
import random

import gym
import numpy as np
import torch
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import tools
import matplotlib.pyplot as plt

from models import WGAN
from wrappers import InputTransformation

MODEL_DIR = 'model'
DEFAULT_MODEL_NAME = 'model_{}.save'

log = gym.logger
log.set_level(gym.logger.INFO)

lamda=10.

args = get_args()


def iterate_batches(envs, batch_size=args.batch_size):
    batch = [e.reset() for e in envs]
    env_gen = iter(lambda: random.choice(envs), None)

    while True:
        e = next(env_gen)
        obs, reward, is_done, _ = e.step(e.action_space.sample())
        if np.mean(obs) > 0.01:
            batch.append(obs)
        if len(batch) == batch_size:
            yield torch.tensor(np.array(batch, dtype=np.float32))
            batch.clear()
        if is_done:
            e.reset()
def main():
    device = torch.device("cpu" if args.no_cuda else "cuda")
    envs = [InputTransformation(gym.make(name)) for name in args.env_names]
    input_shape = envs[0].observation_space.shape

    wgan = torch.load('./model/model_wgan.save')


    wgan.eval()

    z = wgan.sample_noise(args.batch_size)
    gen_output_v = wgan.generator(z).detach().numpy()
    tools.make_plots(gen_output_v, path='fake')
    batches_generator = iterate_batches(envs)
    x = next(batches_generator)

    tools.make_plots(x.detach().numpy(), path='real')

if __name__ == "__main__":
    main()
