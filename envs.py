import os

import gym
import numpy as np
from gym.spaces.box import Box

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

try:
    import dm_control2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass

class SleepAfterDone(gym.Wrapper):
    def __init__(self, env):
        """make the env sleep after returning done,
        keep sleeping untill be reset() is called
        """
        gym.Wrapper.__init__(self, env)
        self.sleeping = True

    def reset(self, **kwargs):
        self.sleeping = False
        self.obs = self.env.reset(**kwargs)
        return self.obs

    def step(self, ac):
        if not self.sleeping:
            self.obs, self.reward, done, self.info = self.env.step(ac)
            if done:
                self.sleeping = True
        else:
            self.reward, done, self.info = type(self.reward)(0), True, self.info
        return self.obs, self.reward, done, self.info

def make_env(rank, args):
    def _thunk():
        if args.env_name.startswith("dm"):
            _, domain, task = args.env_name.split('.')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        elif args.env_name in ['OverCooked']:
            import overcooked
            env = overcooked.OverCooked(
                args = args,
            )
        else:
            env = gym.make(args.env_name)

        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(args.env_name)
        env.seed(args.seed + rank)

        obs_shape = env.observation_space.shape
        if args.add_timestep and len(obs_shape) == 1 and str(env).find('TimeLimit') > -1:
            env = AddTimestep(env)

        if is_atari:
            env = wrap_deepmind(env, clip_rewards=False)

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = WrapPyTorch(env)

        env = SleepAfterDone(env)

        return env

    return _thunk


class AddTimestep(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AddTimestep, self).__init__(env)
        self.observation_space = Box(
            self.observation_space.low[0],
            self.observation_space.high[0],
            [self.observation_space.shape[0] + 1],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return np.concatenate((observation, [self.env._elapsed_steps]))


class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)
