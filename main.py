
from arguments import get_args
import os
import random

import gym
import numpy as np
import torch
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

from models import Discriminator, Generator
from wrappers import InputTransformation


MODEL_DIR = 'model'
DEFAULT_MODEL_NAME = 'model_{}.save'

log = gym.logger
log.set_level(gym.logger.INFO)

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
    device = torch.device("cuda" if args.gpu else "cpu")
    envs = [InputTransformation(gym.make(name)) for name in args.env_names]
    input_shape = envs[0].observation_space.shape

    if args.restore:
        net_gener = torch.load(
            os.path.join(args.restore, DEFAULT_MODEL_NAME.format("generator")))
        net_discr = net_gener = torch.load(
            os.path.join(args.restore,
                         DEFAULT_MODEL_NAME.format("discriminator")))
    else:
        net_discr = Discriminator(input_shape=input_shape).to(device)
        net_gener = Generator(output_shape=input_shape).to(device)

    objective = torch.nn.BCELoss()
    gen_optimizer = torch.optim.Adam(params=net_gener.parameters(),
                                     lr=args.lr,
                                     betas=(0.5, 0.999))
    dis_optimizer = torch.optim.Adam(params=net_discr.parameters(),
                                     lr=args.lr,
                                     betas=(0.5, 0.999))
    writer = SummaryWriter()

    gen_losses = []
    dis_losses = []
    iter_no = 0

    true_labels_v = torch.ones(args.batch_size, dtype=torch.float32, device=device)
    fake_labels_v = torch.zeros(args.batch_size, dtype=torch.float32, device=device)

    for batch_v in iterate_batches(envs):
        # generate extra fake samples, input is 4D: (batch, filters, x, y)
        gen_input_v = torch.FloatTensor(args.batch_size, args.z_size, 1, 1)
        gen_input_v = gen_input_v.normal_(0, 1).to(device)
        batch_v = batch_v.to(device)
        gen_output_v = net_gener(gen_input_v)

        # train discriminator
        dis_optimizer.zero_grad()
        dis_output_true_v = net_discr(batch_v)
        dis_output_fake_v = net_discr(gen_output_v.detach())
        dis_loss = objective(dis_output_true_v, true_labels_v) + \
            objective(dis_output_fake_v, fake_labels_v)
        dis_loss.backward()
        dis_optimizer.step()
        dis_losses.append(dis_loss.item())

        # train generator
        gen_optimizer.zero_grad()
        dis_output_v = net_discr(gen_output_v)
        gen_loss_v = objective(dis_output_v, true_labels_v)
        gen_loss_v.backward()
        gen_optimizer.step()
        gen_losses.append(gen_loss_v.item())

        iter_no += 1
        if iter_no % args.log_iter == 0:
            log.info("Iter %d: gen_loss=%.3e, dis_loss=%.3e",
                     iter_no, np.mean(gen_losses),
                     np.mean(dis_losses))
            writer.add_scalar("gen_loss", np.mean(gen_losses), iter_no)
            writer.add_scalar("dis_loss", np.mean(dis_losses), iter_no)
            gen_losses = []
            dis_losses = []
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

            torch.save(net_gener,
                       os.path.join(MODEL_DIR,
                                    DEFAULT_MODEL_NAME.format("generator")))
            torch.save(
                net_gener,
                os.path.join(MODEL_DIR,
                DEFAULT_MODEL_NAME.format("discriminator")))
