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
from scipy import ndimage

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
    os.makedirs(args.save_dir)
    print('Dir empty, making new log dir...')
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
    args.num_steps = [args.num_steps[0]]*(args.num_hierarchy)

if bottom_envs.action_space.__class__.__name__ == "Discrete":
    action_shape = 1
else:
    action_shape = bottom_envs.action_space.shape[0]

input_actions_onehot_global = []
for hierarchy_i in range(args.num_hierarchy):
    input_actions_onehot_global += [torch.zeros(args.num_processes, args.num_subpolicy[hierarchy_i]).cuda()]
'''init top layer input_actions'''
input_actions_onehot_global[-1][:,0]=1.0

sess = tf.Session()

if args.act_deterministically:
    print('==========================================================================')
    print("================ Note that I am acting deterministically =================")
    print('==========================================================================')

if args.distance in ['match']:
    sift = cv2.xfeatures2d.SIFT_create()
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

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
            num_subpolicy = args.num_subpolicy[self.hierarchy_id],
        ).cuda()

        if args.reward_bounty > 0.0 and self.hierarchy_id not in [0]:
            from model import TransitionModel
            self.transition_model = TransitionModel(
                input_observation_shape = obs_shape if not args.mutual_information else self.envs.observation_space.shape,
                input_action_space = self.envs.action_space,
                output_observation_shape = self.envs.observation_space.shape,
                num_subpolicy = args.num_subpolicy[self.hierarchy_id-1],
                mutual_information = args.mutual_information,
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
        self.episode_reward['bounty_clip'] = 0.0
        self.episode_reward['len'] = 0.0

        if self.hierarchy_id in [0]:
            '''for hierarchy_id=0, we need to summarize reward_raw'''
            self.episode_reward['raw'] = 0.0
            self.episode_reward_raw_all = 0.0
            self.episode_count = 0.0

        '''initialize final_reward, since it is possible that the episode length is longer than num_steps'''
        for episode_reward_type in self.episode_reward.keys():
            self.final_reward[episode_reward_type] = self.episode_reward[episode_reward_type]

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
            self.checkpoint_loaded = True
        except Exception as e:
            self.num_trained_frames = 0
            self.checkpoint_loaded = False

        print('[H-{:1}] Learner has been trained to step: {}'.format(self.hierarchy_id, self.num_trained_frames))
        self.num_trained_frames_at_start = self.num_trained_frames

        self.start = time.time()
        self.step_i = 0
        self.update_i = 0

        self.refresh_update_type()

        self.last_time_log_behavior = 0.0
        self.log_behavior = False
        self.episode_visilize_stack = {}

        self.predicted_next_observations_to_downer_layer = None

        self.agent.set_this_layer(self)

        if args.test_reward_bounty:
            if self.hierarchy_id == 1.0:
                self.macros = [0]*5+[1]*5+[2]*5+[3]*5+[4]*5
                self.macros_count = 0
                print(self.macros)

            if self.hierarchy_id == 0.0:
                self.actions = ([0,0,0,0]+[4,12,8,16]+[1,9,5,13]+[3,11,7,15]+[2,10,6,14])*5
                self.actions_count = 0
                print(self.actions)

                self.bounty_results = []

        if args.test_action_vis:
            if self.hierarchy_id == 1.0:
                self.macros = [0]*5+[1]*5+[2]*5+[3]*5+[4]*5
                self.macros_count = 0
                print(self.macros)

            if self.hierarchy_id == 0.0:
                self.action_sum = 5*5*4
                self.actions_count = 0
                self.action_dic = {}

        self.bounty_clip = torch.zeros(args.num_processes).cuda()
        self.reward_bounty_raw_to_return = torch.zeros(args.num_processes).cuda()
        self.reward_bounty = torch.zeros(args.num_processes).cuda()

    def set_upper_layer(self, upper_layer):
        self.upper_layer = upper_layer
        self.agent.set_upper_layer(self.upper_layer)

    def step(self, inputs):
        '''as a environment, it has step method'''
        if args.reward_bounty > 0.0 and (not args.mutual_information):
            input_cpu_actions = inputs[0]
            self.predicted_next_observations_by_upper_layer = inputs[1]
            self.predicted_reward_bounty_by_upper_layer = inputs[2]
        else:
            input_cpu_actions = inputs
            self.predicted_next_observations_by_upper_layer = None
            self.predicted_reward_bounty_by_upper_layer = None

        '''convert: input_cpu_actions >> input_actions_onehot_global[self.hierarchy_id]'''
        input_actions_onehot_global[self.hierarchy_id].fill_(0.0)
        input_actions_onehot_global[self.hierarchy_id].scatter_(1,torch.from_numpy(input_cpu_actions).long().unsqueeze(1).cuda(),1.0)

        '''macro step forward'''
        reward_macro = None
        for macro_step_i in range(self.hierarchy_interval):

            self.is_final_step_by_upper_layer = (macro_step_i in [self.hierarchy_interval-1])

            self.one_step()

            if reward_macro is None:
                reward_macro = self.reward
            else:
                reward_macro += self.reward

        return self.obs, reward_macro, self.reward_bounty_raw_to_return, self.done, self.info

    def one_step(self):
        '''as a environment, it has step method.
        But the step method step forward for args.hierarchy_interval times,
        as a macro action, this method is to step forward for a singel step'''

        '''for each one_step, interact with env for one step'''
        self.interact_one_step()

        self.step_i += 1
        if self.step_i==args.num_steps[self.hierarchy_id]:
            '''if reach args.num_steps[self.hierarchy_id], update agent for one step with the experiences stored in rollouts'''
            self.update_agent_one_step()
            self.step_i = 0

    def specify_action(self):
        '''this method is used to speicfy actions to the agent,
        so that we can get insight on with is happening'''

        if args.test_action:
            if self.hierarchy_id in [0]:
                self.action[0,0] = int(
                    input(
                        '[Macro Action {}, actual action {}], Act: '.format(
                            utils.onehot_to_index(input_actions_onehot_global[0][0].cpu().numpy()),
                            self.action[0,0].item(),
                        )
                    )
                )
            if self.hierarchy_id in [1]:
                self.action[0,0] = int(
                    input(
                        '[Macro Action {}], Act: '.format(
                            self.action[0,0].item(),
                        )
                    )
                )
            if self.hierarchy_id in [2]:
                self.action[0,0] = int(
                    input(
                        '[top Action {}], Act: '.format(
                            self.action[0,0].item(),
                        )
                    )
                )

        if args.test_reward_bounty:
            if self.hierarchy_id in [0]:
                if self.episode_reward['len'] < 4.0:
                    self.action[0,0] = self.actions[self.actions_count]
                    self.actions_count += 1
                    print('set action to {}'.format(self.action[0,0].item()))
            if self.hierarchy_id in [1]:
                if self.episode_reward['len']==0.0:
                    self.action[0,0] = self.macros[self.macros_count]
                    self.macros_count += 1
                    print('set macro action: {}'.format(self.action[0,0].item()))

        if args.test_action_vis:
            if self.hierarchy_id in [0]:
                if self.episode_reward['len'] < 16.0:
                    new_key = False
                    try:
                        self.action_dic[str(utils.onehot_to_index(input_actions_onehot_global[0][0].cpu().numpy()))].append(self.action[0,0].cpu().item())
                    except Exception as e:
                        new_key = True
                        self.action_dic[str(utils.onehot_to_index(input_actions_onehot_global[0][0].cpu().numpy()))] = [self.action[0,0].cpu().item()]
                    self.actions_count += 1
                    if self.actions_count%4 == 0 and not new_key:
                        self.action_dic[str(utils.onehot_to_index(input_actions_onehot_global[0][0].cpu().numpy()))] += [' ']

            if self.hierarchy_id in [1]:
                if self.episode_reward['len']==0.0:
                    self.action[0,0] = self.macros[self.macros_count]
                    self.macros_count += 1
                    print('set macro action: {}'.format(self.action[0,0].item()))

    def log_for_specify_action(self):

        if (args.test_reward_bounty or args.test_action or args.test_action_vis) and self.hierarchy_id in [0]:
            if args.test_action_vis:
                if self.episode_reward['len'] == 3.0:
                    if self.actions_count == self.action_sum:
                        for action_keys in self.action_dic.keys():
                            print('macro action: {}, action list: {}'.format(action_keys, self.action_dic[action_keys]))
                        print(s)

            print_str = ''
            print_str += '[reward {} ][done {}][masks {}]'.format(
                self.reward_raw_OR_reward[0],
                self.done[0],
                self.masks[0].item(),
            )
            if args.reward_bounty > 0.0:
                print_str += '[reward_bounty {}]'.format(
                    self.reward_bounty[0],
                )
                if args.test_reward_bounty:
                    if self.episode_reward['len'] == 3.0:
                        self.bounty_results += [self.reward_bounty[0]]
                        if self.actions_count == (len(self.actions)):
                            for x in range(5):
                                print_str = ''
                                max_value = 0.0
                                max_index = -1
                                for y in range(5):
                                    temp = self.bounty_results[x*5+y]
                                    print_str += '{}\t'.format(temp)
                                    if temp>max_value:
                                        max_value = temp
                                        max_index = y
                                print('{} max_index: {}'.format(print_str, max_index))
                            print(s)

            print(print_str)

    def generate_actions_to_step(self):
        '''this method generate actions_to_step controlled by many logic'''

        if args.use_fake_reward_bounty:

            if self.hierarchy_id in [0]:
                '''for fake reward bounty, actions_to_step in hierarchy 0 need to have macro-actions from all levels'''
                self.actions_to_step = [None]*args.num_processes
                for process_i in range(args.num_processes):
                    self.actions_to_step[process_i] = []
                    self.actions_to_step[process_i] += [self.action[process_i,0].item()]
                    for hierarchy_i in range(args.num_hierarchy-1):
                        self.actions_to_step[process_i] += [utils.onehot_to_index(
                            input_actions_onehot_global[hierarchy_i][process_i].cpu().numpy()
                        )]

            if self.hierarchy_id not in [0]:
                self.actions_to_step = self.action.squeeze(1).cpu().numpy()

        else:

            if (self.hierarchy_id not in [0]) and (args.reward_bounty > 0.0) and (not args.mutual_information):

                '''predict states'''
                self.transition_model.eval()
                with torch.no_grad():
                    self.predicted_next_observations_to_downer_layer, self.predicted_reward_bounty_to_downer_layer = self.transition_model(
                        inputs = self.rollouts.observations[self.step_i].repeat(self.envs.action_space.n,1,1,1),
                        input_action = self.action_onehot_batch,
                    )
                self.predicted_next_observations_to_downer_layer = self.predicted_next_observations_to_downer_layer.view(self.envs.action_space.n,args.num_processes,*self.predicted_next_observations_to_downer_layer.size()[1:])
                self.predicted_reward_bounty_to_downer_layer = self.predicted_reward_bounty_to_downer_layer.view(self.envs.action_space.n,args.num_processes,*self.predicted_reward_bounty_to_downer_layer.size()[1:]).squeeze()

                self.actions_to_step = [self.action.squeeze(1).cpu().numpy(), self.predicted_next_observations_to_downer_layer, self.predicted_reward_bounty_to_downer_layer]

            else:
                self.actions_to_step = self.action.squeeze(1).cpu().numpy()

    def generate_reward_bounty(self):
        '''this method generate reward bounty'''

        self.bounty_clip *= 0.0
        self.reward_bounty_raw_to_return *= 0.0
        self.reward_bounty *= 0.0

        if (args.reward_bounty>0) and (self.hierarchy_id not in [args.num_hierarchy-1]) and (self.is_final_step_by_upper_layer):

            action_rb = self.rollouts.input_actions[self.step_i].nonzero()[:,1]

            if not args.mutual_information:
                obs_rb = self.obs
                prediction_rb = self.predicted_next_observations_by_upper_layer.cpu().numpy()

            else:
                self.upper_layer.transition_model.eval()
                with torch.no_grad():
                    predicted_action_resulted_from, self.predicted_reward_bounty_by_upper_layer = self.upper_layer.transition_model(
                        inputs = torch.from_numpy(self.obs).float().cuda(),
                    )
                    predicted_action_resulted_from = predicted_action_resulted_from.exp()

            for process_i in range(args.num_processes):

                if not args.mutual_information:
                    difference_list = []
                    for action_i in range(prediction_rb.shape[0]):

                        if args.distance in ['l1','l1_mass_center']:
                            difference_l1 = np.mean(
                                np.abs(
                                    (obs_rb[process_i]-prediction_rb[action_i,process_i])
                                )
                            )/255.0

                        if args.distance in ['mass_center','l1_mass_center']:
                            mass_center_0 = np.asarray(
                                ndimage.measurements.center_of_mass(
                                    obs_rb[process_i][0].astype(np.uint8)
                                )
                            )
                            mass_center_1 = np.asarray(
                                ndimage.measurements.center_of_mass(
                                    prediction_rb[action_i,process_i][0].astype(np.uint8)
                                )
                            )
                            difference_mass_center = np.linalg.norm(mass_center_0-mass_center_1)

                        if args.distance in ['match']:
                            raise DeprecationWarning
                            # p0_kp, des0 = sift.detectAndCompute(obs_rb[process_i][0].astype(np.uint8),None)
                            # p1_kp, des1 = sift.detectAndCompute(prediction_rb[action_i,process_i][0].astype(np.uint8),None)
                            #
                            # difference = 0.0
                            # if des1 is not None:
                            #     '''it is possible that des1 is None, since the prediction is not ready yet'''
                            #     matches = flann.knnMatch(des0,des1,k=2)
                            #     for m,n in matches:
                            #         difference += m.distance

                        if args.distance in ['l1']:
                            difference = difference_l1
                        elif args.distance in ['mass_center']:
                            difference = difference_mass_center
                        elif args.distance in ['l1_mass_center']:
                            difference = difference_l1*5.0+difference_mass_center
                        elif args.distance in ['match']:
                            raise DeprecationWarning
                        else:
                            raise NotImplementedError

                        if action_i==action_rb[process_i]:
                            continue
                        difference_list += [difference]

                    self.reward_bounty_raw_to_return[process_i] = float(np.amin(difference_list))

                else:
                    self.reward_bounty_raw_to_return[process_i] = predicted_action_resulted_from[process_i, action_rb[process_i]].log()

            self.reward_bounty = self.reward_bounty_raw_to_return

            if args.clip_reward_bounty:

                if not args.mutual_information:
                    self.bounty_clip = self.predicted_reward_bounty_by_upper_layer[action_rb[process_i]]
                else:
                    self.bounty_clip = self.predicted_reward_bounty_by_upper_layer

                delta = (self.reward_bounty-self.bounty_clip)

                if args.clip_reward_bounty_active_function in ['linear']:
                    self.reward_bounty = delta
                elif args.clip_reward_bounty_active_function in ['u']:
                    self.reward_bounty = delta.sign().clamp(min=0.0,max=1.0)
                elif args.clip_reward_bounty_active_function in ['relu']:
                    self.reward_bounty = F.relu(delta)
                elif args.clip_reward_bounty_active_function in ['shrink_relu']:
                    positive_active = delta.sign().clamp(min=0.0,max=1.0)
                    self.reward_bounty = delta * positive_active + positive_active - 1
                else:
                    raise Exception('No Supported')

            self.reward_bounty = self.reward_bounty*args.reward_bounty

            '''mask reward bounty, since the final state is start state,
            and the estimation from transition model is not accurate'''
            self.reward_bounty *= self.masks.squeeze()

        if args.reward_bounty>0:
            if self.hierarchy_id in [args.num_hierarchy-1]:
                '''top level only receive reward from env'''
                self.reward_final = self.reward

            else:
                '''other levels only receives reward_bounty'''
                self.reward_final = self.reward_bounty

        else:
            self.reward_final = self.reward

        if args.reward_bounty>0:
            if self.is_final_step_by_upper_layer:
                '''mask it and stop reward function'''
                self.masks = self.masks * 0.0


    def interact_one_step(self):
        '''interact with self.envs for one step and store experience into self.rollouts'''

        self.rollouts.input_actions[self.step_i].copy_(input_actions_onehot_global[self.hierarchy_id])

        '''Sample actions'''
        with torch.no_grad():
            self.value, self.action, self.action_log_prob, self.states = self.actor_critic.act(
                inputs = self.rollouts.observations[self.step_i],
                states = self.rollouts.states[self.step_i],
                masks = self.rollouts.masks[self.step_i],
                deterministic = self.deterministic,
                input_action = self.rollouts.input_actions[self.step_i],
            )

        self.specify_action()

        self.generate_actions_to_step()

        env_0_sleeping = self.envs.get_sleeping(env_index=0)

        '''Obser reward and next obs'''
        fetched = self.envs.step(self.actions_to_step)
        if self.hierarchy_id in [0]:
            self.obs, self.reward_raw_OR_reward, self.done, self.info = fetched
        else:
            self.obs, self.reward_raw_OR_reward, self.reward_bounty_raw_returned, self.done, self.info = fetched

        self.masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in self.done]).cuda()

        if self.hierarchy_id in [(args.num_hierarchy-1)]:
            '''top hierarchy layer is responsible for reseting env if all env has done'''
            if self.masks.sum() == 0.0:
                self.obs = self.reset()

        if self.hierarchy_id in [0]:
            '''only when hierarchy_id is 0, the envs is returning reward_raw from the basic game emulator'''
            self.reward_raw = torch.from_numpy(self.reward_raw_OR_reward).float()
            self.reward = self.reward_raw.sign()
        else:
            '''otherwise, this is reward'''
            self.reward = self.reward_raw_OR_reward

        self.generate_reward_bounty()

        self.log_for_specify_action()

        if not env_0_sleeping:
            self.step_summary_from_env_0()

        '''If done then clean the history of observations'''
        if self.current_obs.dim() == 4:
            self.current_obs *= self.masks.unsqueeze(2).unsqueeze(2)
        else:
            self.current_obs *= self.masks

        self.update_current_obs(self.obs)

        if self.hierarchy_id not in [0]:
            self.rollouts.reward_bounty_raw[self.rollouts.step].copy_(self.reward_bounty_raw_returned.unsqueeze(1))

        self.rollouts.insert(
            self.current_obs,
            self.states,
            self.action,
            self.action_log_prob,
            self.value,
            self.reward_final.unsqueeze(1),
            self.masks,
        )

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

        if args.test_reward_bounty or args.test_action_vis or args.test_action:
            self.update_type = 'none'
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
        if (self.update_i % args.save_interval == 0 and args.save_dir != "") or (self.update_i in [1,2]):
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

            for episode_reward_type in self.final_reward.keys():
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
        if (self.update_i % args.vis_interval == 0) and (not (args.test_reward_bounty or args.test_action or args.test_action_vis)):
            '''we use tensorboard since its better when comparing plots'''
            self.summary = tf.Summary()
            if args.env_name in ['OverCooked']:
                action_count = np.zeros(4)
                for info_index in range(len(self.info)):
                    action_count += self.info[info_index]['action_count']
                if args.see_leg_fre:
                    leg_count = np.zeros(17)
                    for leg_index in range(len(self.info)):
                        leg_count += self.info[leg_index]['leg_count']

            if args.env_name in ['OverCooked']:
                if self.hierarchy_id in [0]:
                    for index_action in range(4):
                        self.summary.value.add(
                            tag = 'hierarchy_{}/action_{}'.format(
                                0,
                                index_action,
                            ),
                            simple_value = action_count[index_action],
                        )
                    if args.see_leg_fre:
                        for index_leg in range(17):
                            self.summary.value.add(
                                tag = 'hierarchy_{}/leg_{}_in_one_eposide'.format(
                                    0,
                                    index_leg,
                                ),
                                simple_value = leg_count[index_leg],
                            )

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

        if self.hierarchy_id > 0:
            '''for log behavior, hierarchy_id=0 is very long, ignore logging it'''
            if ((time.time()-self.last_time_log_behavior)/60.0 > args.log_behavior_interval) and (not (args.test_reward_bounty or args.test_action or args.test_action_vis)):
                '''log behavior every x minutes'''
                if self.episode_reward['len']==0:
                    self.last_time_log_behavior = time.time()
                    self.log_behavior = True

        if self.log_behavior:
            self.summary_behavior_at_step()

        '''summarize reward'''
        self.episode_reward['norm'] += self.reward[0].item()
        self.episode_reward['bounty'] += self.reward_bounty[0].item()
        self.episode_reward['bounty_clip'] += self.bounty_clip[0].item()
        if self.hierarchy_id in [0]:
            '''for hierarchy_id=0, summarize reward_raw'''
            self.episode_reward['raw'] += self.reward_raw[0].item()

        self.episode_reward['len'] += 1

        if self.done[0]:
            for episode_reward_type in self.episode_reward.keys():
                self.final_reward[episode_reward_type] = self.episode_reward[episode_reward_type]
                self.episode_reward[episode_reward_type] = 0.0

            if self.hierarchy_id in [0]:
                self.episode_reward_raw_all += self.final_reward['raw']
                self.episode_count += 1
                self.final_reward['raw_all'] = self.episode_reward_raw_all / self.episode_count

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
        hierarchy_layer[-1].predicted_next_observations_by_upper_layer = None
        hierarchy_layer[-1].predicted_reward_bounty_by_upper_layer = None
        hierarchy_layer[-1].is_final_step_by_upper_layer = False
        hierarchy_layer[-1].one_step()

if __name__ == "__main__":
    main()
