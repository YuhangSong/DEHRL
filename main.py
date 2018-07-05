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

class HierarchyLayer(object):
    """docstring for HierarchyLayer."""
    def __init__(self, args, envs, summary_writer, obs_shape, macro_action_space, hierarchy_id):
        super(HierarchyLayer, self).__init__()

        print('================================================================')
        print('Building hierarchy layer: {}'.format(
            hierarchy_id
        ))

        self.args = args
        self.envs = envs
        self.summary_writer = summary_writer
        self.obs_shape = obs_shape
        self.macro_action_space = macro_action_space
        self.hierarchy_id = hierarchy_id

        if self.envs.__class__.__name__ in ['HierarchyLayer']:
            self.action_space = macro_action_space
        self.observation_space = self.envs.observation_space

        self.actor_critic = Policy(
            obs_shape = obs_shape,
            input_action_space = macro_action_space,
            output_action_space = self.envs.action_space,
            recurrent_policy = self.args.recurrent_policy,
        )

        if self.args.cuda:
            self.actor_critic.cuda()

        if self.args.algo == 'a2c':
            self.agent = algo.A2C_ACKTR(
                self.actor_critic, self.args.value_loss_coef, self.args.entropy_coef,
                lr=self.args.lr,
                eps=self.args.eps,
                alpha=self.args.alpha,
                max_grad_norm=self.args.max_grad_norm,
            )
        elif args.algo == 'ppo':
            self.agent = algo.PPO(
                self.actor_critic, self.args.clip_param, self.args.ppo_epoch, self.args.num_mini_batch, self.args.value_loss_coef, self.args.entropy_coef,
                lr=self.args.lr,
                eps=self.args.eps,
                max_grad_norm=self.args.max_grad_norm,
            )
        elif args.algo == 'acktr':
            self.agent = algo.A2C_ACKTR(
                self.actor_critic, self.args.value_loss_coef, self.args.entropy_coef,
                acktr=True,
            )

        self.rollouts = RolloutStorage(self.args.num_steps, self.args.num_processes, self.obs_shape, self.macro_action_space, self.envs.action_space, self.actor_critic.state_size)
        self.current_obs = torch.zeros(self.args.num_processes, *self.obs_shape)

        obs = self.envs.reset()
        self.update_current_obs(obs)

        self.rollouts.observations[0].copy_(self.current_obs)

        self.episode_reward_raw = 0.0
        self.final_reward_raw = 0.0

        if self.args.cuda:
            self.current_obs = self.current_obs.cuda()
            self.rollouts.cuda()

        # try to load checkpoint
        try:
            self.num_trained_frames = np.load(self.args.save_dir+'/hierarchy_{}_num_trained_frames.npy'.format(self.hierarchy_id))[0]
            try:
                self.actor_critic.load_state_dict(torch.load(self.args.save_dir+'/hierarchy_{}_trained_learner.pth'.format(self.hierarchy_id)))
                print('Load learner previous point: Successed')
            except Exception as e:
                print('Load learner previous point: Failed')
        except Exception as e:
            self.num_trained_frames = 0
        print('Learner has been trained to step: '+str(self.num_trained_frames))

        if self.hierarchy_id in [0]:
            self.start = time.time()
        self.j = 0
        self.step_i = 0

        self.input_gpu_actions_onehot = torch.zeros(self.args.num_processes, self.macro_action_space.n).cuda()

    def update_current_obs(self, obs):
        shape_dim0 = self.envs.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        if args.num_stack > 1:
            self.current_obs[:, :-shape_dim0] = self.current_obs[:, shape_dim0:]
        self.current_obs[:, -shape_dim0:] = obs

    def step(self, input_cpu_actions):

        '''convert: input_cpu_actions >> self.input_gpu_actions_onehot'''
        self.input_gpu_actions_onehot *= 0.0
        for process_i in range(self.args.num_processes):
            self.input_gpu_actions_onehot[process_i,input_cpu_actions[process_i]] = 1.0

        '''macro step forward'''
        for macro_step_i in range(self.args.hierarchy_interval):
            obs, reward_raw, done, info = self.one_step()

        # print('xxx: need mask here!!!!')

        return obs, reward_raw, done, info

    def reset(self):
        return self.envs.reset()

    def one_step(self):

        if self.hierarchy_id in [0]:
            if self.num_trained_frames > self.args.num_frames:
                raise Exception('Done')

        ##################################################################################################

        self.rollouts.input_actions[self.step_i].copy_(self.input_gpu_actions_onehot)

        with torch.no_grad():
            value, action, action_log_prob, states = self.actor_critic.act(
                self.rollouts.observations[self.step_i],
                self.rollouts.input_actions[self.step_i],
                self.rollouts.states[self.step_i],
                self.rollouts.masks[self.step_i],
            )
        cpu_actions = action.squeeze(1).cpu().numpy()

        # Obser reward and next obs
        obs, reward_raw, done, info = self.envs.step(cpu_actions)
        self.episode_reward_raw += reward_raw[0]
        if done[0]:
            self.final_reward_raw = self.episode_reward_raw
            self.episode_reward_raw = 0.0
        reward = np.sign(reward_raw)
        reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()

        # If done then clean the history of observations.
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

        if args.cuda:
            masks = masks.cuda()

        if self.current_obs.dim() == 4:
            self.current_obs *= masks.unsqueeze(2).unsqueeze(2)
        else:
            self.current_obs *= masks

        self.update_current_obs(obs)
        self.rollouts.insert(self.current_obs, states, action, action_log_prob, value, reward, masks)

        ##################################################################################################

        if self.step_i == (self.args.num_steps-1):
            self.update_agent()
            self.step_i = 0
        self.step_i += 1

        return obs, reward_raw, done, info

    def update_agent(self):

        with torch.no_grad():
            self.next_value = self.actor_critic.get_value(
                self.rollouts.observations[-1],
                self.rollouts.input_actions[-1],
                self.rollouts.states[-1],
                self.rollouts.masks[-1],
            ).detach()

        self.rollouts.compute_returns(self.next_value, self.args.use_gae, self.args.gamma, self.args.tau)

        self.value_loss, self.action_loss, self.dist_entropy = self.agent.update(self.rollouts)

        self.rollouts.after_update()

        self.num_trained_frames += (args.num_steps*args.num_processes)
        self.j += 1

        # save checkpoint
        if self.j % self.args.save_interval == 0 and self.args.save_dir != "":
            try:
                np.save(
                    args.save_dir+'/hierarchy_{}_num_trained_frames.npy'.format(
                        self.hierarchy_id,
                    ),
                    np.array([self.num_trained_frames]),
                )
                self.actor_critic.save_model(
                    save_path=self.args.save_dir+'/hierarchy_{}_trained_learner.pth'.format(
                        self.hierarchy_id,
                    ),
                )
            except Exception as e:
                print("Save checkpoint failed: {}".format(
                    e
                ))

        # print info
        if self.j % self.args.log_interval == 0:
            if self.hierarchy_id in [0]:
                print("[{}/{}], FPS {}, final_reward_raw {:.2f}, remaining {} hours".
                    format(
                        self.num_trained_frames, self.args.num_frames,
                        int(self.num_trained_frames / (time.time() - self.start)),
                        self.final_reward_raw,
                        (time.time() - self.start)/self.num_trained_frames*(self.args.num_frames-self.num_trained_frames)/60.0/60.0
                    )
                )

        # visualize results
        if self.args.vis and self.j % self.args.vis_interval == 0:
            '''we use tensorboard instead of visdom since its better when comparing plots'''
            summary = tf.Summary()
            summary.value.add(
                tag = 'hierarchy_{}_final_reward_raw'.format(
                    self.hierarchy_id
                ),
                simple_value = self.final_reward_raw,
            )
            summary.value.add(
                tag = 'hierarchy_{}_value_loss'.format(
                    self.hierarchy_id
                ),
                simple_value = self.value_loss,
            )
            summary.value.add(
                tag = 'hierarchy_{}_action_loss'.format(
                    self.hierarchy_id
                ),
                simple_value = self.action_loss,
            )
            summary.value.add(
                tag = 'hierarchy_{}_dist_entropy'.format(
                    self.hierarchy_id
                ),
                simple_value = self.dist_entropy,
            )
            self.summary_writer.add_summary(summary, self.num_trained_frames)
            self.summary_writer.flush()


def main():

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

    hierarchy_layer = []
    hierarchy_layer += [HierarchyLayer(args, envs, summary_writer, obs_shape, macro_action_space, hierarchy_id=0)]
    for hierarchy_i in range(1, args.num_hierarchy):
        hierarchy_layer += [HierarchyLayer(args, hierarchy_layer[hierarchy_i-1], summary_writer, obs_shape, macro_action_space, hierarchy_id=hierarchy_i)]

    # actor_critic = Policy(
    #     obs_shape = obs_shape,
    #     input_action_space = macro_action_space,
    #     output_action_space = envs.action_space,
    #     recurrent_policy = args.recurrent_policy,
    # )

    # if envs.action_space.__class__.__name__ == "Discrete":
    #     action_shape = 1
    # else:
    #     action_shape = envs.action_space.shape[0]

    # if args.cuda:
    #     actor_critic.cuda()
    #
    # if args.algo == 'a2c':
    #     agent = algo.A2C_ACKTR(
    #         actor_critic, args.value_loss_coef, args.entropy_coef,
    #         lr=args.lr,
    #         eps=args.eps,
    #         alpha=args.alpha,
    #         max_grad_norm=args.max_grad_norm,
    #     )
    # elif args.algo == 'ppo':
    #     agent = algo.PPO(
    #         actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch, args.value_loss_coef, args.entropy_coef,
    #         lr=args.lr,
    #         eps=args.eps,
    #         max_grad_norm=args.max_grad_norm,
    #     )
    # elif args.algo == 'acktr':
    #     agent = algo.A2C_ACKTR(
    #         actor_critic, args.value_loss_coef, args.entropy_coef,
    #         acktr=True,
    #     )
    #
    # rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space, actor_critic.state_size)
    # current_obs = torch.zeros(args.num_processes, *obs_shape)
    #
    # def update_current_obs(obs):
    #     shape_dim0 = envs.observation_space.shape[0]
    #     obs = torch.from_numpy(obs).float()
    #     if args.num_stack > 1:
    #         current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
    #     current_obs[:, -shape_dim0:] = obs
    #
    # obs = envs.reset()
    # update_current_obs(obs)
    #
    # rollouts.observations[0].copy_(current_obs)
    #
    # episode_reward_raw = 0.0
    # final_reward_raw = 0.0
    #
    # if args.cuda:
    #     current_obs = current_obs.cuda()
    #     rollouts.cuda()
    #
    # # try to load checkpoint
    # try:
    #     num_trained_frames = np.load(args.save_dir+'/num_trained_frames.npy')[0]
    #     try:
    #         actor_critic.load_state_dict(torch.load(args.save_dir+'/trained_learner.pth'))
    #         print('Load learner previous point: Successed')
    #     except Exception as e:
    #         print('Load learner previous point: Failed')
    # except Exception as e:
    #     num_trained_frames = 0
    # print('Learner has been trained to step: '+str(num_trained_frames))

    # start = time.time()
    # j = 0

    empty_actions = np.zeros(args.num_processes, dtype=int)

    while True:

        hierarchy_layer[-1].step(empty_actions)

        # if num_trained_frames > args.num_frames:
        #     break

        # for step in range(args.num_steps):
            # Sample actions
            # with torch.no_grad():
            #     value, action, action_log_prob, states = actor_critic.act(
            #         rollouts.observations[step],
            #         rollouts.states[step],
            #         rollouts.masks[step],
            #     )
            # cpu_actions = action.squeeze(1).cpu().numpy()
            #
            # # Obser reward and next obs
            # obs, reward_raw, done, info = envs.step(cpu_actions)
            # episode_reward_raw += reward_raw[0]
            # if done[0]:
            #     final_reward_raw = episode_reward_raw
            #     episode_reward_raw = 0.0
            # reward = np.sign(reward_raw)
            # reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            #
            # # If done then clean the history of observations.
            # masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            #
            # if args.cuda:
            #     masks = masks.cuda()
            #
            # if current_obs.dim() == 4:
            #     current_obs *= masks.unsqueeze(2).unsqueeze(2)
            # else:
            #     current_obs *= masks
            #
            # update_current_obs(obs)
            # rollouts.insert(current_obs, states, action, action_log_prob, value, reward, masks)

        # with torch.no_grad():
        #     next_value = actor_critic.get_value(
        #         rollouts.observations[-1],
        #         rollouts.states[-1],
        #         rollouts.masks[-1],
        #     ).detach()
        #
        # rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)
        #
        # value_loss, action_loss, dist_entropy = agent.update(rollouts)
        #
        # rollouts.after_update()
        #
        # num_trained_frames += (args.num_steps*args.num_processes)
        # j += 1
        #
        # # save checkpoint
        # if j % args.save_interval == 0 and args.save_dir != "":
        #     try:
        #         np.save(
        #             args.save_dir+'/num_trained_frames.npy',
        #             np.array([num_trained_frames]),
        #         )
        #         actor_critic.save_model(save_path=args.save_dir)
        #     except Exception as e:
        #         print("Save checkpoint failed")
        #
        # # print info
        # if j % args.log_interval == 0:
        #     end = time.time()
        #     total_num_steps = (j + 1) * args.num_processes * args.num_steps
        #     print("[{}/{}], FPS {}, final_reward_raw {:.2f}, remaining {} hours".
        #         format(
        #             num_trained_frames, args.num_frames,
        #             int(num_trained_frames / (end - start)),
        #             final_reward_raw,
        #             (end - start)/num_trained_frames*(args.num_frames-num_trained_frames)/60.0/60.0
        #         )
        #     )
        #
        # # visualize results
        # if args.vis and j % args.vis_interval == 0:
        #     '''we use tensorboard since its better when comparing plots'''
        #     summary = tf.Summary()
        #     summary.value.add(
        #         tag = 'final_reward_raw',
        #         simple_value = final_reward_raw,
        #     )
        #     summary.value.add(
        #         tag = 'value_loss',
        #         simple_value = value_loss,
        #     )
        #     summary.value.add(
        #         tag = 'action_loss',
        #         simple_value = action_loss,
        #     )
        #     summary.value.add(
        #         tag = 'dist_entropy',
        #         simple_value = dist_entropy,
        #     )
        #     summary_writer.add_summary(summary, num_trained_frames)
        #     summary_writer.flush()



if __name__ == "__main__":
    main()
