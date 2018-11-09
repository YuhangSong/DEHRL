import logging
import math
import gym
from gym import error, spaces
from gym.utils import seeding
import numpy as np
import cv2
import random

logger = logging.getLogger(__name__)

class Explore2DContinuous(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, args=None):

        self.args = args

        '''config'''
        high = np.ones([2])
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float64)
        self.episode_length_limit = self.args.episode_length_limit
        high = np.inf * np.ones([2])
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float64)

    def seed(self, seed):
        np.random.seed(seed)

    def step(self, action):

        done = False
        self.eposide_length += 1
        reward = 0.0

        self.position += action

        if self.episode_length_limit > 0:
            if self.eposide_length >= self.episode_length_limit:
                done = True

        self.info = {}

        return self.obs(), reward, done, self.info

    def obs(self):
        return self.position

    def reset(self):
        self.eposide_length = 0
        self.position = np.array([0.0,0.0])
        return self.obs()
