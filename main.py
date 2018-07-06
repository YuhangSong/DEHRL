import copy
import glob
import os
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from arguments import get_args
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from envs import make_env
from model import Policy
from storage import RolloutStorage
import tensorflow as tf

import algo

args = get_args()

assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

try:
    os.makedirs(args.save_dir)
except OSError:
    files = glob.glob(os.path.join(args.save_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

torch.set_num_threads(1)

if args.vis:
    summary_writer = tf.summary.FileWriter(args.save_dir)

envs = [make_env(i, args=args)
            for i in range(args.num_processes)]

if args.num_processes > 1:
    envs = SubprocVecEnv(envs)
else:
    envs = DummyVecEnv(envs)

if len(envs.observation_space.shape) == 1:
    if args.env_name in ['OverCooked']:
        raise Exception("I donot know why they have VecNormalize for ram observation")
    envs = VecNormalize(envs, gamma=args.gamma)

obs_shape = envs.observation_space.shape
obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

macro_action_space = gym.spaces.Discrete(args.num_subpolicy)

if envs.action_space.__class__.__name__ == "Discrete":
    action_shape = 1
else:
    action_shape = envs.action_space.shape[0]

class HierarchyLayer(object):
    """docstring for HierarchyLayer."""
    """
    HierarchyLayer is a learning system, containning actor_critic, agent, rollouts.
    In the meantime, it is a environment, which has step, reset functions, as well as action_space, observation_space, etc.
    """
    def __init__(self, envs, hierarchy_id):
        super(HierarchyLayer, self).__init__()

        print('================================================================')
        print('Building hierarchy layer: {}'.format(
            hierarchy_id
        ))

        self.envs = envs
        self.hierarchy_id = hierarchy_id

        '''as an env, it should have action_space and observation space'''
        self.action_space = macro_action_space
        self.observation_space = self.envs.observation_space

        self.actor_critic = Policy(
            obs_shape = obs_shape,
            input_action_space = macro_action_space,
            output_action_space = self.envs.action_space,
            recurrent_policy = args.recurrent_policy,
        )

        if args.cuda:
            self.actor_critic.cuda()

        if args.algo == 'a2c':
            self.agent = algo.A2C_ACKTR(
                self.actor_critic, args.value_loss_coef, args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                alpha=args.alpha,
                max_grad_norm=args.max_grad_norm,
            )
        elif args.algo == 'ppo':
            self.agent = algo.PPO(
                self.actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch, args.value_loss_coef, args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                max_grad_norm=args.max_grad_norm,
            )
        elif args.algo == 'acktr':
            self.agent = algo.A2C_ACKTR(
                self.actor_critic, args.value_loss_coef, args.entropy_coef,
                acktr=True,
            )

        self.rollouts = RolloutStorage(
            num_steps = args.num_steps,
            num_processes = args.num_processes,
            obs_shape = obs_shape,
            macro_action_space = macro_action_space,
            action_space = self.envs.action_space,
            state_size = self.actor_critic.state_size,
        )
        self.current_obs = torch.zeros(args.num_processes, *obs_shape)

        obs = self.envs.reset()
        self.update_current_obs(obs)

        self.rollouts.observations[0].copy_(self.current_obs)

        '''for summarizing reward'''
        self.episode_reward = 0.0
        self.final_reward = 0.0
        if self.hierarchy_id in [0]:
            '''for hierarchy_id=0, we need to summarize reward_raw'''
            self.episode_reward_raw = 0.0
            self.final_reward_raw = 0.0

        self.input_gpu_actions_onehot = torch.zeros(args.num_processes, macro_action_space.n)

        if args.cuda:
            self.current_obs = self.current_obs.cuda()
            self.rollouts.cuda()
            self.input_gpu_actions_onehot = self.input_gpu_actions_onehot.cuda()

        '''try to load checkpoint'''
        try:
            self.num_trained_frames = np.load(args.save_dir+'/hierarchy_{}_num_trained_frames.npy'.format(self.hierarchy_id))[0]
            try:
                self.actor_critic.load_state_dict(torch.load(args.save_dir+'/hierarchy_{}_trained_learner.pth'.format(self.hierarchy_id)))
                print('Load learner previous point: Successed')
            except Exception as e:
                print('Load learner previous point: Failed, due to {}'.format(e))
        except Exception as e:
            self.num_trained_frames = 0
        print('Learner has been trained to step: '+str(self.num_trained_frames))
        self.num_trained_frames_at_start = self.num_trained_frames

        self.start = time.time()
        self.j = 0
        self.step_i = 0


    def step(self, input_cpu_actions):
        '''as a environment, it has step method'''

        '''convert: input_cpu_actions >> self.input_gpu_actions_onehot'''
        self.input_gpu_actions_onehot *= 0.0
        for process_i in range(args.num_processes):
            self.input_gpu_actions_onehot[process_i,input_cpu_actions[process_i]] = 1.0

        '''macro step forward, when doing this, record every step returns.
        This is because we have to make done single pass all the way up to top hierarchy layer,
        we will do mask operation afterwards'''
        obs_macro = None
        reward_macro = None
        mask_macro = None
        for macro_step_i in range(args.hierarchy_interval):
            obs, reward, done, info = self.one_step()

            obs = np.expand_dims(obs, 0)
            if obs_macro is None:
                obs_macro = obs
            else:
                obs_macro = np.concatenate((obs_macro, obs),0)

            reward = reward.squeeze().unsqueeze(0)
            if reward_macro is None:
                reward_macro = reward
            else:
                reward_macro = torch.cat([reward_macro,reward],0)

            mask = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in self.done]).squeeze(1).unsqueeze(0)
            if mask_macro is None:
                mask_macro = mask
            else:
                mask_macro = torch.cat([mask_macro,mask],0)

        '''this is the mask operation'''

        '''if done, the following returns are done'''
        print(mask_macro)
        for macro_step_i in range(args.hierarchy_interval-1):
            for mask_i in range(macro_step_i+1, args.hierarchy_interval):
                mask_macro[mask_i] = mask_macro[mask_i]*mask_macro[macro_step_i]
        print(mask_macro)

        print(reward_macro)
        reward_macro = reward_macro*mask_macro
        print(reward_macro)
        reward = reward_macro.sum(dim=1,keepdim=False)
        print(reward)
        print(s)

        return obs, reward, done, info

    def one_step(self):
        '''as a environment, it has step method.
        But the step method step forward for args.hierarchy_interval times,
        as a macro action, this method is to step forward for a singel step'''

        '''for each one_step, interact with env for one step'''
        obs, reward, done, info = self.interact_one_step()

        self.step_i += 1
        if self.step_i==args.num_steps:
            '''if reach args.num_steps, update agent for one step with the experiences stored in rollouts'''
            self.update_agent_one_step()
            self.step_i = 0

        return obs, reward, done, info

    def interact_one_step(self):
        '''interact with self.envs for one step and store experience into self.rollouts'''

        self.rollouts.input_actions[self.step_i].copy_(self.input_gpu_actions_onehot)

        # Sample actions
        with torch.no_grad():
            self.value, self.action, self.action_log_prob, self.states = self.actor_critic.act(
                inputs = self.rollouts.observations[self.step_i],
                states = self.rollouts.states[self.step_i],
                masks = self.rollouts.masks[self.step_i],
                deterministic = False,
                input_action = self.rollouts.input_actions[self.step_i],
            )
        self.cpu_actions = self.action.squeeze(1).cpu().numpy()

        # Obser reward and next obs
        self.obs, self.reward_raw_OR_reward, self.done, self.info = envs.step(self.cpu_actions)

        if self.hierarchy_id in [0]:
            '''only when hierarchy_id is 0, the envs is returning reward_raw from the basic game emulator'''
            self.reward_raw = self.reward_raw_OR_reward
            self.reward = np.sign(self.reward_raw)
        else:
            '''otherwise, this is reward'''
            self.reward = self.reward_raw_OR_reward

        '''summarize reward'''
        self.episode_reward += self.reward[0]
        if self.hierarchy_id in [0]:
            '''for hierarchy_id=0, summarize reward_raw'''
            self.episode_reward_raw += self.reward_raw[0]

        if self.done[0]:
            self.final_reward = self.episode_reward
            self.episode_reward = 0.0
            if self.hierarchy_id in [0]:
                self.final_reward_raw = self.episode_reward_raw
                self.episode_reward_raw = 0.0

        self.reward = torch.from_numpy(np.expand_dims(np.stack(self.reward), 1)).float()

        # If done then clean the history of observations.
        self.masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in self.done])

        if args.cuda:
            self.masks = self.masks.cuda()

        if self.current_obs.dim() == 4:
            self.current_obs *= self.masks.unsqueeze(2).unsqueeze(2)
        else:
            self.current_obs *= self.masks

        self.update_current_obs(self.obs)
        self.rollouts.insert(self.current_obs, self.states, self.action, self.action_log_prob, self.value, self.reward, self.masks)

        return self.obs, self.reward, self.done, self.info

    def update_agent_one_step(self):
        '''update the self.actor_critic with self.agent,
        according to the experiences stored in self.rollouts'''

        with torch.no_grad():
            self.next_value = self.actor_critic.get_value(
                inputs=self.rollouts.observations[-1],
                states=self.rollouts.states[-1],
                masks=self.rollouts.masks[-1],
                input_action=self.rollouts.input_actions[-1],
            ).detach()

        self.rollouts.compute_returns(self.next_value, args.use_gae, args.gamma, args.tau)

        self.value_loss, self.action_loss, self.dist_entropy = self.agent.update(self.rollouts)

        self.rollouts.after_update()

        self.num_trained_frames += (args.num_steps*args.num_processes)
        self.j += 1

        '''save checkpoint'''
        if self.j % args.save_interval == 0 and args.save_dir != "":
            try:
                np.save(
                    args.save_dir+'/hierarchy_{}_num_trained_frames.npy'.format(self.hierarchy_id),
                    np.array([self.num_trained_frames]),
                )
                self.actor_critic.save_model(args.save_dir+'/hierarchy_{}_trained_learner.pth'.format(self.hierarchy_id))
            except Exception as e:
                print("Save checkpoint failed")

        '''print info'''
        if self.j % args.log_interval == 0:
            self.end = time.time()
            self.total_num_steps = (self.j + 1) * args.num_processes * args.num_steps
            print_string = "[H-{}][{}/{}], FPS {}, final_reward {:.2f}".format(
                self.hierarchy_id,
                self.num_trained_frames, args.num_frames,
                int(self.num_trained_frames / (self.end - self.start)),
                self.final_reward,
            )
            if self.hierarchy_id in [0]:
                print_string += ', remaining {} hours'.format(
                    (self.end - self.start)/(self.num_trained_frames-self.num_trained_frames_at_start)*(args.num_frames-self.num_trained_frames)/60.0/60.0,
                )
            print(print_string)


        '''visualize results'''
        if args.vis and self.j % args.vis_interval == 0:
            '''we use tensorboard since its better when comparing plots'''
            self.summary = tf.Summary()
            self.summary.value.add(
                tag = 'hierarchy_{}_final_reward_raw'.format(self.hierarchy_id),
                simple_value = self.final_reward_raw,
            )
            self.summary.value.add(
                tag = 'hierarchy_{}_value_loss'.format(self.hierarchy_id),
                simple_value = self.value_loss,
            )
            self.summary.value.add(
                tag = 'hierarchy_{}_action_loss'.format(self.hierarchy_id),
                simple_value = self.action_loss,
            )
            self.summary.value.add(
                tag = 'hierarchy_{}_dist_entropy'.format(self.hierarchy_id),
                simple_value = self.dist_entropy,
            )
            summary_writer.add_summary(self.summary, self.num_trained_frames)
            summary_writer.flush()

        if self.hierarchy_id in [0]:
            '''if hierarchy_id is 0, it is the basic env, then control the training
            progress by its num_trained_frames'''
            if self.num_trained_frames > args.num_frames:
                raise Exception('Done')

    def reset(self):
        '''as a environment, it has reset method'''
        return self.envs.reset()

    def update_current_obs(self, obs):
        '''update self.current_obs, which contains args.num_stack frames, with obs, which is current frame'''
        shape_dim0 = self.envs.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        if args.num_stack > 1:
            self.current_obs[:, :-shape_dim0] = self.current_obs[:, shape_dim0:]
        self.current_obs[:, -shape_dim0:] = obs

def main():

    hierarchy_layer = []
    hierarchy_layer += [HierarchyLayer(
        envs = envs,
        hierarchy_id = 0,
    )]
    for hierarchy_i in range(1, args.num_hierarchy):
        hierarchy_layer += [HierarchyLayer(
            envs = hierarchy_layer[hierarchy_i-1],
            hierarchy_id=hierarchy_i,
        )]

    empty_actions = np.zeros(args.num_processes, dtype=int)

    while True:

        '''as long as the top hierarchy layer is stepping forward, the downer layers is controlled and kept running'''
        hierarchy_layer[-1].step(empty_actions)

if __name__ == "__main__":
    main()
