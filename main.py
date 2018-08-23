from arguments import get_args
import os
import random

import gym
import numpy as np
import torch
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import tools

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


if __name__ == "__main__":

    START_ITER = args.start_iter
    END_ITER = args.end_iter


    device = torch.device("cpu" if args.no_cuda else "cuda")
    envs = [InputTransformation(gym.make(name)) for name in args.env_names]
    input_shape = envs[0].observation_space.shape

    if args.restore:
        net_gener = torch.load(
            os.path.join(args.restore, DEFAULT_MODEL_NAME.format("generator")))
        net_discr = net_gener = torch.load(
            os.path.join(args.restore,
                         DEFAULT_MODEL_NAME.format("discriminator")))
    else:
        # net_discr = Discriminator(input_shape=input_shape).to(device)
        # net_gener = Generator(output_shape=input_shape).to(device)
        wgan = WGAN(label=args.dataset, z_size=args.z_size,
                    image_size=args.image_size,
                    image_channel_size=args.channel_size,
                    c_channel_size=args.disc_filters,
                    g_channel_size=args.gener_filters)
        tools.gaussian_intiailize(wgan, 0.02)

    # objective = torch.nn.BCELoss()
    gen_optimizer = torch.optim.Adam(params=wgan.generator.parameters(),
                                     lr=args.lr,
                                     betas=(0.5, 0.999)
                                     )
    critic_optimizer = torch.optim.Adam(params=wgan.critic.parameters(),
                                     lr=args.lr,
                                     betas=(0.5, 0.999)
                                     )

    # prepare the model and statistics.
    wgan.train()

    writer = SummaryWriter()

    gen_losses = []
    c_losses = []
    iter_no = 0

    true_labels_v = torch.ones(args.batch_size, dtype=torch.float32, device=device)
    fake_labels_v = torch.zeros(args.batch_size, dtype=torch.float32, device=device)

    batches_generator = iterate_batches(envs)

    for iter_no in range(START_ITER, END_ITER):
        print("Iter: " + str(iter_no))

        for i in range(args.disc_iter):

            x = next(batches_generator)

            # train generator
            critic_optimizer.zero_grad()
            z = wgan.sample_noise(args.batch_size)
            c_loss, g = wgan.c_loss(x, z, return_g=True)
            c_loss_gp = c_loss + wgan.gradient_penalty(x, g, lamda=lamda)
            # print(c_loss_gp)
            c_losses.append(c_loss_gp.item())
            c_loss_gp.backward()
            critic_optimizer.step()

        for i in range(args.gen_iter):

            batch_v = next(batches_generator)

            # generate extra fake samples, input is 4D: (batch, filters, x, y)
            z = wgan.sample_noise(args.batch_size)
            batch_v = batch_v.to(device)
            gen_output_v = wgan.generator(z)

            # train discriminator
            gen_optimizer.zero_grad()
            z = wgan.sample_noise(args.batch_size)
            g_loss = wgan.g_loss(z)
            # print(g_loss)
            gen_losses.append(g_loss.item())
            g_loss.backward()
            gen_optimizer.step()

        if iter_no % args.log_iter == 0:
            log.info("Iter %d: gen_loss=%.3e, dis_loss=%.3e",
                     iter_no, np.mean(gen_losses),
                     np.mean(c_losses))
            writer.add_scalar("gen_loss", np.mean(gen_losses), iter_no)
            writer.add_scalar("c_loss", np.mean(c_losses), iter_no)
            gen_losses = []
            c_losses = []
        if iter_no % args.save_image_iter == 0:
            writer.add_image("fake",
                             vutils.make_grid(gen_output_v.data[:64]),
                             iter_no)
            writer.add_image("real",
                             vutils.make_grid(batch_v.data[:64]),
                             iter_no)
        if iter_no % args.save_model_iter == 0:
            if not os.path.exists(MODEL_DIR):
                os.makedirs(MODEL_DIR)

            torch.save(wgan,
                       os.path.join(MODEL_DIR,
                                    DEFAULT_MODEL_NAME.format("wgan")))
