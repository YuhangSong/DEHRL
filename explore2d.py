import logging
import math
import gym
from gym import error, spaces
from gym.utils import seeding
import numpy as np
import cv2
import random

logger = logging.getLogger(__name__)

class Explore2D(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, args=None):

        self.args = args

        '''config'''
        self.action_space = spaces.Discrete(5)
        self.episode_length_limit = self.args.episode_length_limit
        self.observation_space = spaces.Box(
            low   = -float(self.episode_length_limit),
            high  = +float(self.episode_length_limit),
            shape = (2, 2, 1),
            dtype = np.float64,
        )
        self.action_to_delta_position_map = {
            0: np.array([ 0.0, 0.0]),
            1: np.array([ 1.0, 0.0]),
            2: np.array([ 0.0, 1.0]),
            3: np.array([-1.0, 0.0]),
            4: np.array([ 0.0,-1.0]),
        }

    def seed(self, seed):
        np.random.seed(seed)

    def step(self, action):

        done = False
        self.eposide_length += 1
        reward = 0.0

        self.position += self.action_to_delta_position_map[action]

        if self.episode_length_limit > 0:
            if self.eposide_length >= self.episode_length_limit:
                done = True

        self.info = {}

        return self.obs(), reward, done, self.info

    def obs(self):
        return np.array(
            [[[self.position[0]],[self.position[1]]],
             [[0.0             ],[0.0             ]]]
        )

    def reset(self):
        self.eposide_length = 0
        self.position = np.array([3.0,2.0])
        return self.obs()
