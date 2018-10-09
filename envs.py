import os

import gym
import numpy as np
from gym.spaces.box import Box

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

    def get_sleeping(self, env_index=None):
        return self.sleeping

class SingleThread(gym.Wrapper):
    def __init__(self, env):
        """make the env return things in a multi-thread fashion
        """
        gym.Wrapper.__init__(self, env)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return np.stack([obs])

    def step(self, ac):
        obs, reward, done, info = self.env.step(ac[0])
        return np.stack([obs]), np.stack([reward]), np.stack([done]), np.stack([info])

class DelayDone(gym.Wrapper):
    def __init__(self, env):
        """make done delay for one step
        """
        gym.Wrapper.__init__(self, env)
        self.going_to_done = False

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, ac):

        if not self.going_to_done:
            self.obs, self.reward, self.done, self.info = self.env.step(ac)

            if self.done:
                self.done = False
                self.going_to_done = True

        else:
            self.done = True
            self.going_to_done = False

        return self.obs, self.reward, self.done, self.info

def make_env(rank, args):
    def _thunk():

        if args.env_name.startswith("dm"):
            '''deepmind control suite'''
            _, domain, task = args.env_name.split('.')
            env = dm_control2gym.make(domain_name=domain, task_name=task)

        elif args.env_name in ['OverCooked']:
            '''OverCooked game we wrote'''
            import overcooked
            env = overcooked.OverCooked(
                args = args,
            )

        elif args.env_name in ['MineCraft']:
            '''OverCooked game we wrote'''
            import minecraft
            env = minecraft.MineCraft(
                args = args,
            )
            env.set_render(False)

        else:
            '''envs from openai gym'''
            env = gym.make(args.env_name)

            is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)

            if is_atari:
                '''atari from openai gym'''
                '''we need the environment have no frame skip
                and no action repeat stochasticity'''
                assert 'NoFrameskip-v4' in env.spec.id
                from baselines.common.atari_wrappers import NoopResetEnv, EpisodicLifeEnv, FireResetEnv, WarpFrame
                env = NoopResetEnv(env, noop_max=30)
                env = EpisodicLifeEnv(env)
                if 'FIRE' in env.unwrapped.get_action_meanings():
                    env = FireResetEnv(env)
                env = WarpFrame(env)

        env.seed(args.seed + rank)

        obs_shape = env.observation_space.shape

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = WrapPyTorch(env)

        env = DelayDone(env)
        # env = SleepAfterDone(env)

        if args.num_processes in [1]:
            env=SingleThread(env)

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
