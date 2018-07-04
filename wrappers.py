import cv2
import gym
import numpy as np

IMAGE_SIZE = (64, 64)


class InputTransformation(gym.ObservationWrapper):
    """
    A wrapper class that transforms the enivornments
    into images required for training of our GAN.
    """
    def __init__(self, *args):
        super(InputTransformation, self).__init__(*args)
        # Assert that the observation space is not discrete but
        # bounded.
        assert isinstance(self.observation_space, gym.spaces.Box)

        obs_space = self.observation_space
        self.observation_space = gym.spaces.Box(
            self.observation(obs_space.low),
            self.observation(obs_space.high),
            dtype=np.float32)

    def observation(self, observation):
        """
        1. Resize the image according to image_size specified.
        2. Transform the dimensions of observation from
            `(height, width, channels)` to
            `(channels, height, width)`.
        """
        result = cv2.resize(observation, IMAGE_SIZE)
        result = np.moveaxis(result, 2, 0)
        result = result.astype(np.float32) / 255.0
        return result
