import os

import gym
import numpy as np
from gym.spaces.box import Box
from baselines.common.vec_env.vec_normalize import VecNormalize as VecNormalize_

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

class WrapperMontezumaRevenge(gym.Wrapper):
    def __init__(self, env):
        """MontezumaRevenge has some very weird actions, fix this
        """
        gym.Wrapper.__init__(self, env)
        '''
        0 is nope action
        1 is jump, but only function after take 0
        2 is nope
        3 is right ->
        4 is lift ->
        5 is down ->
        6 is duplicated right/lift,
        7 is duplicated right/lift,
        8 is duplicated right/lift,
        9 is duplicated right/lift,
        10 is up and jump but still need 0 before jump ->
        11 is right jump ->
        12 is duplicated lift
        13 is down
        14 is right
        15 is lift
        16 is right jump
        17 is lift jump ->
        '''
        from gym import error, spaces
        self.action_space = spaces.Discrete(5)
        self.action_map = {
            0: [3], # right
            1: [4], # lift
            2: [5], # down
            3: [0,10], # jump
            4: [0,11], # right jump
            5: [0,17], # lift jump
        }

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        for reset_times in range(10):
            self.obs, self.reward, done, self.info = self.env.step(0)
        return self.obs

    def step(self, ac):
        for action in self.action_map[ac]:
            self.obs, self.reward, done, self.info = self.env.step(action)
        return self.obs, self.reward, done, self.info

class SleepAfterDone(gym.Wrapper):
    def __init__(self, env):
        """make the env sleep after returning done,
        keep sleeping untill be reset() is called
        """
        gym.Wrapper.__init__(self, env)
        self.going_to_sleep = None
        self.sleeping = None

    def reset(self, **kwargs):

        self.going_to_sleep = False
        self.sleeping = False

        self.obs = self.env.reset(**kwargs)
        return self.obs

    def step(self, ac):
        if self.going_to_sleep:
            self.sleeping = True
            self.going_to_sleep = False
        if not self.sleeping:
            self.obs, self.reward, done, self.info = self.env.step(ac)
            if done:
                self.going_to_sleep = True
        else:
            self.reward, done, self.info = type(self.reward)(0), True, self.info
        return self.obs, self.reward, done, self.info

    def get_sleeping(self):
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
    def get_sleeping(self, env_index=0):
        return self.env.get_sleeping()

class DelayDone(gym.Wrapper):
    def __init__(self, env):
        """make the env sleep after returning done,
        keep sleeping untill be reset() is called
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
            '''this is an additional step, no reward is provided'''
            self.reward = type(self.reward)(0)
            self.done = True
            self.going_to_done = False

        return self.obs, self.reward, self.done, self.info

class ScaleActions(gym.ActionWrapper):
    def __init__(self, env):
        super(ScaleActions, self).__init__(env)

    def action(self, action):
        action = (np.tanh(action) + 1) / 2 * (self.action_space.high - self.action_space.low) + self.action_space.low
        return action

class VecNormalize(VecNormalize_):

    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs):
        if self.ob_rms:
            if self.training:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

def make_env(rank, args):
    def _thunk():

        if args.env_name.startswith("dm"):
            '''deepmind control suite'''
            _, domain, task = args.env_name.split('.')
            env = dm_control2gym.make(domain_name=domain, task_name=task)

        elif args.env_name.find('Bullet') > -1:
            import pybullet_envs
            env = pybullet_envs.make(args.env_name)
            env = ScaleActions(env)

        elif args.env_name in ['OverCooked']:
            '''OverCooked game we wrote'''
            import overcooked
            env = overcooked.OverCooked(
                args = args,
            )

        elif args.env_name in ['GridWorld']:
            '''OverCooked game we wrote'''
            import gridworld
            env = gridworld.GridWorld(
                args = args,
            )

        elif args.env_name in ['Explore2D']:
            '''OverCooked game we wrote'''
            import explore2d
            env = explore2d.Explore2D(
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
                assert 'NoFrameskip-v4' in env.spec.id # so that we make sure no action repeat stochasticity is introduced
                from baselines.common.atari_wrappers import NoopResetEnv, NoLifeEnv, FireResetEnv, WarpFrame, MaxAndSkipEnv
                # env = NoopResetEnv(env, noop_max=30)
                if args.env_name in ['MontezumaRevengeNoFrameskip-v4']:
                    frame_skip = 2
                else:
                    frame_skip = 4
                env = MaxAndSkipEnv(env, skip=frame_skip)
                env = NoLifeEnv(env)
                if 'FIRE' in env.unwrapped.get_action_meanings():
                    env = FireResetEnv(env)
                env = WrapperMontezumaRevenge(env)
                env = WarpFrame(env)

        env.seed(args.seed + rank)

        obs_shape = env.observation_space.shape

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = WrapPyTorch(env)

        env = DelayDone(env)
        env = SleepAfterDone(env)

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
