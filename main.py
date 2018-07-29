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
import cv2

import utils

import algo

args = get_args()

assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

try:
    print('Dir empty, making new log dir...')
    os.makedirs(args.save_dir)
except Exception as e:
    if e.__class__.__name__ in ['FileExistsError']:
        print('Dir exsit, checking checkpoint...')
    else:
        raise e

print('Log to {}'.format(args.save_dir))

torch.set_num_threads(1)

summary_writer = tf.summary.FileWriter(args.save_dir)

bottom_envs = [make_env(i, args=args)
            for i in range(args.num_processes)]

bottom_envs = SubprocVecEnv(bottom_envs)

if len(bottom_envs.observation_space.shape) == 1:
    if args.env_name in ['OverCooked']:
        raise Exception("I donot know why they have VecNormalize for ram observation")
    bottom_envs = VecNormalize(bottom_envs, gamma=args.gamma)

obs_shape = bottom_envs.observation_space.shape
obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

if len(args.num_subpolicy) != (args.num_hierarchy-1):
    print('# WARNING: Exlicity num_subpolicy is not matching args.num_hierarchy, use the first num_subpolicy for all layers')
    args.num_subpolicy = [args.num_subpolicy[0]]*(args.num_hierarchy-1)
'''for top hierarchy layer'''
args.num_subpolicy += [2]

if len(args.hierarchy_interval) != (args.num_hierarchy-1):
    print('# WARNING: Exlicity hierarchy_interval is not matching args.num_hierarchy, use the first hierarchy_interval for all layers')
    args.hierarchy_interval = [args.hierarchy_interval[0]]*(args.num_hierarchy-1)

if len(args.num_steps) != (args.num_hierarchy):
    print('# WARNING: Exlicity num_steps is not matching args.num_hierarchy, use the first num_steps for all layers')
    args.num_steps = [args.num_steps[0]]*(args.num_hierarchy-1)

if bottom_envs.action_space.__class__.__name__ == "Discrete":
    action_shape = 1
else:
    action_shape = bottom_envs.action_space.shape[0]

input_actions_onehot_global = []
for hierarchy_i in range(args.num_hierarchy):
    input_actions_onehot_global += [torch.zeros(args.num_processes, args.num_subpolicy[hierarchy_i]).cuda()]

input_actions_onehot_global[args.num_hierarchy-1][:,0] = 1.0

sess = tf.Session()

if args.act_deterministically:
    print('==========================================================================')
    print("================ Note that I am acting deterministically =================")
    print('==========================================================================')

class HierarchyLayer(object):
    """docstring for HierarchyLayer."""
    """
    HierarchyLayer is a learning system, containning actor_critic, agent, rollouts.
    In the meantime, it is a environment, which has step, reset functions, as well as action_space, observation_space, etc.
    """
    def __init__(self, envs, hierarchy_id):
        super(HierarchyLayer, self).__init__()

        self.envs = envs
        self.hierarchy_id = hierarchy_id
        self.args = args

        '''as an env, it should have action_space and observation space'''
        self.action_space = gym.spaces.Discrete((input_actions_onehot_global[self.hierarchy_id]).size()[1])
        self.observation_space = self.envs.observation_space
        if self.hierarchy_id not in [args.num_hierarchy-1]:
            self.hierarchy_interval = args.hierarchy_interval[self.hierarchy_id]
        else:
            self.hierarchy_interval = None

        print('[H-{:1}] Building hierarchy layer. Action space {}. Observation_space {}. Hierarchy interval {}'.format(
            self.hierarchy_id,
            self.action_space,
            self.observation_space,
            self.hierarchy_interval,
        ))

        self.actor_critic = Policy(
            obs_shape = obs_shape,
            input_action_space = self.action_space,
            output_action_space = self.envs.action_space,
            recurrent_policy = args.recurrent_policy,
        ).cuda()

        if args.reward_bounty > 0.0 and self.hierarchy_id not in [0]:
            from model import TransitionModel
            self.transition_model = TransitionModel(
                input_observation_shape = obs_shape,
                input_action_space = self.envs.action_space,
                output_observation_space = self.envs.observation_space,
            ).cuda()
            self.action_onehot_batch = torch.zeros(args.num_processes*self.envs.action_space.n,self.envs.action_space.n).cuda()
            batch_i = 0
            for action_i in range(self.envs.action_space.n):
                for process_i in range(args.num_processes):
                    self.action_onehot_batch[batch_i][action_i] = 1.0
                    batch_i += 1
        else:
            self.transition_model = None

        if args.algo == 'a2c':
            self.agent = algo.A2C_ACKTR(
                self.actor_critic, args.value_loss_coef, args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                alpha=args.alpha,
                max_grad_norm=args.max_grad_norm,
            )
        elif args.algo == 'ppo':
            self.agent = algo.PPO()
        elif args.algo == 'acktr':
            self.agent = algo.A2C_ACKTR(
                self.actor_critic, args.value_loss_coef, args.entropy_coef,
                acktr=True,
            )

        self.rollouts = RolloutStorage(
            num_steps = args.num_steps[self.hierarchy_id],
            num_processes = args.num_processes,
            obs_shape = obs_shape,
            input_actions = self.action_space,
            action_space = self.envs.action_space,
            state_size = self.actor_critic.state_size,
            observation_space = self.envs.observation_space,
        ).cuda()
        self.current_obs = torch.zeros(args.num_processes, *obs_shape).cuda()

        '''for summarizing reward'''
        self.episode_reward = {}
        self.final_reward = {}

        self.episode_reward['norm'] = 0.0
        self.episode_reward['bounty'] = 0.0
        self.episode_reward['len'] = 0.0

        if self.hierarchy_id in [0]:
            '''for hierarchy_id=0, we need to summarize reward_raw'''
            self.episode_reward['raw'] = 0.0
            self.final_reward['raw'] = 0.0

        '''try to load checkpoint'''
        try:
            self.num_trained_frames = np.load(args.save_dir+'/hierarchy_{}_num_trained_frames.npy'.format(self.hierarchy_id))[0]
            try:
                self.actor_critic.load_state_dict(torch.load(args.save_dir+'/hierarchy_{}_actor_critic.pth'.format(self.hierarchy_id)))
                print('[H-{:1}] Load actor_critic previous point: Successed'.format(self.hierarchy_id))
            except Exception as e:
                print('[H-{:1}] Load actor_critic previous point: Failed, due to {}'.format(self.hierarchy_id,e))
            if self.transition_model is not None:
                try:
                    self.transition_model.load_state_dict(torch.load(args.save_dir+'/hierarchy_{}_transition_model.pth'.format(self.hierarchy_id)))
                    print('[H-{:1}] Load transition_model previous point: Successed'.format(self.hierarchy_id))
                except Exception as e:
                    print('[H-{:1}] Load transition_model previous point: Failed, due to {}'.format(self.hierarchy_id,e))
        except Exception as e:
            self.num_trained_frames = 0
        print('[H-{:1}] Learner has been trained to step: {}'.format(self.hierarchy_id, self.num_trained_frames))
        self.num_trained_frames_at_start = self.num_trained_frames

        self.start = time.time()
        self.step_i = 0
        self.update_i = 0

        self.refresh_update_type()

        self.last_time_log_behavior = time.time()
        self.log_behavior = True
        self.episode_visilize_stack = {}

        self.predicted_next_observations_to_downer_layer = None

        self.agent.set_this_layer(self)

    def set_upper_layer(self, upper_layer):
        self.upper_layer = upper_layer
        self.agent.set_upper_layer(self.upper_layer)

    def step(self, inputs):
        '''as a environment, it has step method'''
        if args.reward_bounty > 0.0:
            input_cpu_actions = inputs[0]
            predicted_next_observations_by_upper_layer = inputs[1]
        else:
            input_cpu_actions = inputs
            predicted_next_observations_by_upper_layer = None

        '''convert: input_cpu_actions >> input_actions_onehot_global[self.hierarchy_id]'''
        input_actions_onehot_global[self.hierarchy_id].fill_(0.0)
        input_actions_onehot_global[self.hierarchy_id].scatter_(1,torch.from_numpy(input_cpu_actions).long().unsqueeze(1).cuda(),1.0)

        '''macro step forward'''
        reward_macro = None
        for macro_step_i in range(self.hierarchy_interval):

            obs, reward, done, info = self.one_step(
                predicted_next_observations_by_upper_layer = predicted_next_observations_by_upper_layer,
                is_final_step_by_upper_layer = (macro_step_i==(self.hierarchy_interval-1)),
            )

            if reward_macro is None:
                reward_macro = reward
            else:
                reward_macro += reward

        reward = reward_macro

        return obs, reward, done, info

    def one_step(self, predicted_next_observations_by_upper_layer, is_final_step_by_upper_layer):
        '''as a environment, it has step method.
        But the step method step forward for args.hierarchy_interval times,
        as a macro action, this method is to step forward for a singel step'''

        '''for each one_step, interact with env for one step'''
        obs, reward, done, info = self.interact_one_step(predicted_next_observations_by_upper_layer, is_final_step_by_upper_layer)

        self.step_i += 1
        if self.step_i==args.num_steps[self.hierarchy_id]:
            '''if reach args.num_steps[self.hierarchy_id], update agent for one step with the experiences stored in rollouts'''
            self.update_agent_one_step()
            self.step_i = 0

        return obs, reward, done, info

    def interact_one_step(self, predicted_next_observations_by_upper_layer, is_final_step_by_upper_layer):
        '''interact with self.envs for one step and store experience into self.rollouts'''

        self.rollouts.input_actions[self.step_i].copy_(input_actions_onehot_global[self.hierarchy_id])

        # Sample actions
        with torch.no_grad():
            self.value, self.action, self.action_log_prob, self.states = self.actor_critic.act(
                inputs = self.rollouts.observations[self.step_i],
                states = self.rollouts.states[self.step_i],
                masks = self.rollouts.masks[self.step_i],
                deterministic = self.deterministic,
                input_action = self.rollouts.input_actions[self.step_i],
            )
        self.cpu_actions = self.action.squeeze(1).cpu().numpy()

        if args.test and self.hierarchy_id in [0]:
            self.cpu_actions[0] = int(
                input(
                    '[Macro Action {}, actual action {}], Act: '.format(
                        utils.onehot_to_index(input_actions_onehot_global[0][0].cpu().numpy()),
                        self.cpu_actions[0],
                    )
                )
            )

        self.actions_to_step = self.cpu_actions

        env_0_sleeping = self.envs.get_sleeping(env_index=0)

        if args.use_fake_reward_bounty:

            if self.hierarchy_id in [0]:
                self.actions_to_step = [None]*args.num_processes
                for process_i in range(args.num_processes):
                    self.actions_to_step[process_i] = []
                    self.actions_to_step[process_i] += [self.cpu_actions[process_i]]
                    for hierarchy_i in range(args.num_hierarchy-1):
                        self.actions_to_step[process_i] += [utils.onehot_to_index(
                            input_actions_onehot_global[hierarchy_i][process_i].cpu().numpy()
                        )]

            elif self.hierarchy_id in [1]:
                self.actions_to_step = np.random.randint(low=0, high=self.envs.action_space.n, size=self.cpu_actions.shape, dtype=self.cpu_actions.dtype)

        else:

            if args.reward_bounty > 0.0 and self.hierarchy_id not in [0]:
                '''predict states'''
                self.transition_model.eval()
                with torch.no_grad():
                    self.predicted_next_observations_to_downer_layer, _ = self.transition_model(
                        inputs = self.rollouts.observations[self.step_i].repeat(self.envs.action_space.n,1,1,1),
                        input_action = self.action_onehot_batch,
                    )
                self.predicted_next_observations_to_downer_layer = self.predicted_next_observations_to_downer_layer.view(self.envs.action_space.n,args.num_processes,*self.predicted_next_observations_to_downer_layer.size()[1:])

                if self.hierarchy_id in [1]:
                    self.actions_to_step = np.random.randint(low=0, high=self.envs.action_space.n, size=self.cpu_actions.shape, dtype=self.cpu_actions.dtype)
                    if args.test:
                        self.actions_to_step[0] = int(
                            input(
                                'Macro Action: '
                            )
                        )

                self.actions_to_step = [self.actions_to_step, self.predicted_next_observations_to_downer_layer]

        # Obser reward and next obs
        self.obs, self.reward_raw_OR_reward, self.done, self.info = self.envs.step(self.actions_to_step)
        self.masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in self.done]).cuda()

        if self.hierarchy_id in [(args.num_hierarchy-1)]:
            '''top hierarchy layer is responsible for reseting env if all env has done'''
            if self.masks.sum() == 0.0:
                self.obs = self.reset()

        if self.hierarchy_id in [0]:
            '''only when hierarchy_id is 0, the envs is returning reward_raw from the basic game emulator'''
            self.reward_raw = self.reward_raw_OR_reward.astype(float)

            self.reward = np.sign(self.reward_raw)
            self.reward = self.reward_raw
        else:
            '''otherwise, this is reward'''
            self.reward = self.reward_raw_OR_reward

        if (predicted_next_observations_by_upper_layer is not None) and is_final_step_by_upper_layer:

            obs_rb = torch.from_numpy(self.obs).float().cuda()
            prediction_rb = predicted_next_observations_by_upper_layer
            obs_rb = obs_rb.view(obs_rb.size()[0],-1)/255.0
            prediction_rb = prediction_rb.view(*prediction_rb.size()[:2],-1)/255.0
            action_rb = self.rollouts.input_actions[self.step_i].nonzero()[:,1]

            self.reward_bounty = torch.zeros(args.num_processes).cuda()
            for process_i in range(args.num_processes):
                for action_i in range(prediction_rb.size()[0]):
                    if action_i==action_rb[process_i]:
                        continue
                    self.reward_bounty[process_i] += (obs_rb[process_i]-prediction_rb[action_i,process_i]).abs().mean()
            self.reward_bounty = self.reward_bounty/(prediction_rb.size()[0]-1)*args.reward_bounty

            '''mask reward bounty, since the final state is start state,
            and the estimation from transition model is not accurate'''
            self.reward_bounty *= self.masks.squeeze()
            '''convert to numpy'''
            self.reward_bounty = self.reward_bounty.cpu().numpy()

        else:
            self.reward_bounty = np.copy(self.reward)
            self.reward_bounty.fill(0.0)

        if is_final_step_by_upper_layer:
            '''mask it and stop reward function'''
            self.masks = self.masks * 0.0

        if args.test and self.hierarchy_id in [0]:
            if args.reward_bounty > 0.0:
                print('[reward {} ][reward_bounty {}][done {}][masks {}]'.format(
                    self.reward_raw_OR_reward[0],
                    self.reward_bounty[0],
                    self.done[0],
                    self.masks[0].item(),
                ))
            if args.use_fake_reward_bounty:
                print('[reward {} ][done {}][masks {}]'.format(
                    self.reward_raw_OR_reward[0],
                    self.done[0],
                    self.masks[0].item(),
                ))


        self.reward_final = self.reward + self.reward_bounty

        if not env_0_sleeping:
            self.step_summary_from_env_0()

        '''If done then clean the history of observations'''
        if self.current_obs.dim() == 4:
            self.current_obs *= self.masks.unsqueeze(2).unsqueeze(2)
        else:
            self.current_obs *= self.masks

        self.update_current_obs(self.obs)

        self.rollouts.insert(
            self.current_obs,
            self.states,
            self.action,
            self.action_log_prob,
            self.value,
            torch.from_numpy(
                np.expand_dims(
                    np.stack(
                        self.reward_final
                    ),
                    1,
                )
            ).float(),
            self.masks,
        )

        # return reward instead of reward_final to upper layer
        # since reward_bounty will stop upper layer from exploring
        return self.obs, self.reward, self.done, self.info

    def refresh_update_type(self):
        if args.reward_bounty > 0.0:

            if args.train_mode in ['together']:
                '''train_mode is together'''

                self.update_type = 'both'
                self.deterministic = False

            elif args.train_mode in ['switch']:
                '''train_mode is switch'''

                '''switch training between actor_critic and transition_model'''
                if self.update_i%2 == 1:
                    self.update_type = 'actor_critic'
                    self.deterministic = False
                else:
                    self.update_type = 'transition_model'
                    self.deterministic = True

            '''top layer do not have a transition_model'''
            if self.hierarchy_id in [args.num_hierarchy-1]:
                self.update_type = 'actor_critic'
                self.deterministic = False

        else:
            '''there is no transition_model'''

            self.update_type = 'actor_critic'
            self.deterministic = False

        '''overwrite if args.act_deterministically'''
        if args.act_deterministically:
            self.deterministic = True

    def update_agent_one_step(self):
        '''update the self.actor_critic with self.agent,
        according to the experiences stored in self.rollouts'''

        '''prepare rollouts for updating actor_critic'''
        if self.update_type in ['actor_critic','both']:
            with torch.no_grad():
                self.next_value = self.actor_critic.get_value(
                    inputs=self.rollouts.observations[-1],
                    states=self.rollouts.states[-1],
                    masks=self.rollouts.masks[-1],
                    input_action=self.rollouts.input_actions[-1],
                ).detach()
            self.rollouts.compute_returns(self.next_value, args.use_gae, args.gamma, args.tau)

        '''update, either actor_critic or transition_model'''
        epoch_loss = self.agent.update(self.update_type)
        self.num_trained_frames += (args.num_steps[self.hierarchy_id]*args.num_processes)
        self.update_i += 1

        '''prepare rollouts for new round of interaction'''
        self.rollouts.after_update()

        '''save checkpoint'''
        if self.update_i % args.save_interval == 0 and args.save_dir != "":
            try:
                np.save(
                    args.save_dir+'/hierarchy_{}_num_trained_frames.npy'.format(self.hierarchy_id),
                    np.array([self.num_trained_frames]),
                )
                self.actor_critic.save_model(args.save_dir+'/hierarchy_{}_actor_critic.pth'.format(self.hierarchy_id))
                if self.transition_model is not None:
                    self.transition_model.save_model(args.save_dir+'/hierarchy_{}_transition_model.pth'.format(self.hierarchy_id))
                print("[H-{:1}] Save checkpoint successed.".format(self.hierarchy_id))
            except Exception as e:
                print("[H-{:1}] Save checkpoint failed, due to {}.".format(self.hierarchy_id,e))

        '''print info'''
        if self.update_i % args.log_interval == 0:
            self.end = time.time()
            print_string = "[H-{:1}][{:9}/{}], FPS {:4}".format(
                self.hierarchy_id,
                self.num_trained_frames, args.num_frames,
                int((self.num_trained_frames-self.num_trained_frames_at_start) / (self.end - self.start)),
            )

            print_string += ', final_reward '
            for episode_reward_type in self.episode_reward.keys():
                print_string += '[{}:{:8.2f}]'.format(
                    episode_reward_type,
                    self.final_reward[episode_reward_type]
                )

            if self.hierarchy_id in [0]:
                print_string += ', remaining {:4.1f} hours'.format(
                    (self.end - self.start)/(self.num_trained_frames-self.num_trained_frames_at_start)*(args.num_frames-self.num_trained_frames)/60.0/60.0,
                )
            print(print_string)

        '''visualize results'''
        if self.update_i % args.vis_interval == 0:
            '''we use tensorboard since its better when comparing plots'''
            self.summary = tf.Summary()

            for episode_reward_type in self.episode_reward.keys():
                self.summary.value.add(
                    tag = 'hierarchy_{}/final_reward_{}'.format(
                        self.hierarchy_id,
                        episode_reward_type,
                    ),
                    simple_value = self.final_reward[episode_reward_type],
                )

            for epoch_loss_type in epoch_loss.keys():
                self.summary.value.add(
                    tag = 'hierarchy_{}/epoch_loss_{}'.format(
                        self.hierarchy_id,
                        epoch_loss_type,
                    ),
                    simple_value = epoch_loss[epoch_loss_type],
                )

            summary_writer.add_summary(self.summary, self.num_trained_frames)
            summary_writer.flush()

        '''update system status'''
        self.refresh_update_type()

        '''check end condition'''
        if self.hierarchy_id in [0]:
            '''if hierarchy_id is 0, it is the basic env, then control the training
            progress by its num_trained_frames'''
            if self.num_trained_frames > args.num_frames:
                raise Exception('Done')

    def reset(self):
        '''as a environment, it has reset method'''
        obs = self.envs.reset()
        self.update_current_obs(obs)
        self.rollouts.observations[0].copy_(self.current_obs)
        return obs

    def update_current_obs(self, obs):
        '''update self.current_obs, which contains args.num_stack frames, with obs, which is current frame'''
        shape_dim0 = self.envs.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        if args.num_stack > 1:
            self.current_obs[:, :-shape_dim0] = self.current_obs[:, shape_dim0:]
        self.current_obs[:, -shape_dim0:] = obs

    def step_summary_from_env_0(self):

        '''for log behavior'''
        if (time.time()-self.last_time_log_behavior)/60.0 > args.log_behavior_interval:
            '''log behavior every x minutes'''
            if self.episode_reward['len']==0:
                self.last_time_log_behavior = time.time()
                self.log_behavior = True

        if self.log_behavior:
            self.summary_behavior_at_step()

        '''summarize reward'''
        self.episode_reward['norm'] += self.reward[0]
        self.episode_reward['bounty'] += self.reward_bounty[0]
        if self.hierarchy_id in [0]:
            '''for hierarchy_id=0, summarize reward_raw'''
            self.episode_reward['raw'] += self.reward_raw[0]

        self.episode_reward['len'] += 1

        if self.done[0]:
            for episode_reward_type in self.episode_reward.keys():
                if 'sp' in episode_reward_type:
                    if int(episode_reward_type.split('_')[1]) != self.actions_to_step[0][1]:
                        continue
                self.final_reward[episode_reward_type] = self.episode_reward[episode_reward_type]
                self.episode_reward[episode_reward_type] = 0.0

            if self.log_behavior:
                self.summary_behavior_at_done()
                self.log_behavior = False

    def summary_behavior_at_step(self):

        '''Summary observation'''

        img = None

        for hierarchy_i in range(args.num_hierarchy-1):
            hierarchy_i_back = (args.num_hierarchy-1)-1-hierarchy_i
            macro_action_img = utils.actions_onehot_visualize(
                actions_onehot = np.expand_dims(
                    input_actions_onehot_global[hierarchy_i_back][0].cpu().numpy(),
                    axis = 0,
                ),
                figsize = (self.obs.shape[2:][1], int(self.obs.shape[2:][1]/args.num_subpolicy[hierarchy_i_back]*1))
            )
            try:
                img = np.concatenate((img, macro_action_img),0)
            except Exception as e:
                img = macro_action_img

        bottom_action_img = utils.actions_onehot_visualize(
            actions_onehot = np.expand_dims(
                utils.action_to_onehot(
                    action = self.action.squeeze().cpu().numpy()[0],
                    action_space = bottom_envs.action_space,
                ),
                axis = 0,
            ),
            figsize = (self.obs.shape[2:][1], int(self.obs.shape[2:][1]/bottom_envs.action_space.n*1))
        )
        img = np.concatenate((img, bottom_action_img),0)

        state_img = utils.gray_to_rgb(self.obs[0,0])
        state_img = cv2.putText(
            state_img,
            'Reward: {}'.format(
                self.reward_raw_OR_reward[0],
            ),
            (30,10),
            cv2.FONT_HERSHEY_COMPLEX,
            0.2,
            (0,0,255),
        )
        img = np.concatenate((img, state_img),0)

        try:
            self.episode_visilize_stack['observation'] += [img]
        except Exception as e:
            self.episode_visilize_stack['observation'] = [img]

        '''Summery state_prediction'''
        if self.predicted_next_observations_to_downer_layer is not None:
            img = self.rollouts.observations[self.step_i][0,-self.envs.observation_space.shape[0]:,:,:].permute(1,2,0)
            for action_i in range(self.envs.action_space.n):
                img = torch.cat([img,self.predicted_next_observations_to_downer_layer[action_i,0,:,:,:].permute(1,2,0)],1)
            img = img.cpu().numpy()
            try:
                self.episode_visilize_stack['state_prediction'] += [img]
            except Exception as e:
                self.episode_visilize_stack['state_prediction'] = [img]

    def summary_behavior_at_done(self):

        print('[H-{:1}] Log behavior done at {}.'.format(
            self.hierarchy_id,
            self.num_trained_frames,
        ))

        for episode_visilize_stack_name in self.episode_visilize_stack.keys():
            self.episode_visilize_stack[episode_visilize_stack_name] = np.stack(
                self.episode_visilize_stack[episode_visilize_stack_name]
            )
            image_summary_op = tf.summary.image(
                'H-{}_F-{}_{}'.format(
                    self.hierarchy_id,
                    self.num_trained_frames,
                    episode_visilize_stack_name,
                ),
                self.episode_visilize_stack[episode_visilize_stack_name],
                max_outputs = self.episode_visilize_stack[episode_visilize_stack_name].shape[0],
            )
            self.episode_visilize_stack[episode_visilize_stack_name] = None
            image_summary = sess.run(image_summary_op)
            summary_writer.add_summary(image_summary, self.num_trained_frames)
        summary_writer.flush()

    def get_sleeping(self, env_index):
        return self.envs.get_sleeping(env_index)

def main():

    hierarchy_layer = []
    hierarchy_layer += [HierarchyLayer(
        envs = bottom_envs,
        hierarchy_id = 0,
    )]
    for hierarchy_i in range(1, args.num_hierarchy):
        hierarchy_layer += [HierarchyLayer(
            envs = hierarchy_layer[hierarchy_i-1],
            hierarchy_id=hierarchy_i,
        )]

    for hierarchy_i in range(0,args.num_hierarchy-1):
        hierarchy_layer[hierarchy_i].set_upper_layer(hierarchy_layer[hierarchy_i+1])

    hierarchy_layer[-1].reset()

    while True:

        '''as long as the top hierarchy layer is stepping forward,
        the downer layers is controlled and kept running.
        Note that the top hierarchy does no have to call step,
        calling one_step is enough'''
        hierarchy_layer[-1].one_step(
            predicted_next_observations_by_upper_layer = None,
            is_final_step_by_upper_layer = False,
        )

if __name__ == "__main__":
    main()
