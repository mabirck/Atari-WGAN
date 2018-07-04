import argparse
import os
import random
from PIL import Image

import cv2
import gym
import numpy as np

def save_as_image(observation,
                  save_dir,
                  img_name,
                  prefix="img_"):
    # donwnscaling the image
    im_array = cv2.resize(observation, IMAGE_SIZE)
    im = Image.fromarray(im_array, 'RGB')
    imname = '{}{}.png'.format(prefix, img_name)
    im.save(os.path.join(save_dir, imname))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()

    # Adding the arguments
    arg_parser.add_argument("--save_dir", type=str, default=SAVE_DIR,
                            help="Relative path to the directory to store "
                                 "the data (default value is 'data/'")
    arg_parser.add_argument("--num_images", type=int,
                            default=IMAGES_TO_GENERATE,
                            help="Number of images to generate (default "
                                 "value is 10000)")

    args = arg_parser.parse_args()

    save_dir = args.save_dir
    num_images = args.num_images

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    envs = [(gym.make(name)) for name in ENV_NAMES]
    env = random.choice(envs)
    env.reset()
    i, current_env_images = 0, 0

    while i < num_images:
        obs, _, is_done, _ = env.step(env.action_space.sample())
        if np.mean(obs) > 0.01:
            save_as_image(obs, save_dir, str(i))
            current_env_images += 1
            i += 1
        else:
            continue
        if is_done or current_env_images % MAX_IMAGES_PER_ENV_INSTANCE == 0:
            current_env_images = 0
            env = random.choice(envs)
env.reset()
