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
from visualize import visdom_plot
import tensorflow as tf

from policy_network import EHRL_Policy
from policy_storage import EHRL_RolloutStorage

import math
import gym.spaces as space

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

    if len(envs.observation_space.shape) == 1 and args.env_name not in ['OverCooked']:
        envs = VecNormalize(envs, gamma=args.gamma)

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

    def update_current_obs(obs):
        shape_dim0 = envs.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        if args.num_stack > 1:
            current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
        current_obs[:, -shape_dim0:] = obs

    def get_onehot(num_class, action):
        one_hot = np.zeros(num_class)
        one_hot[action] = 1
        one_hot = torch.from_numpy(one_hot).float()

        return one_hot

    if args.policy_type == 'shared_policy':

        actor_critic = Policy(obs_shape, envs.action_space, args.recurrent_policy)

        if envs.action_space.__class__.__name__ == "Discrete":
            action_shape = 1
        else:
            action_shape = envs.action_space.shape[0]

        if args.cuda:
            actor_critic.cuda()

        if args.algo == 'a2c':
            agent = algo.A2C_ACKTR(
                actor_critic, args.value_loss_coef, args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                alpha=args.alpha,
                max_grad_norm=args.max_grad_norm,
            )
        elif args.algo == 'ppo':
            agent = algo.PPO(
                actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch, args.value_loss_coef, args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                max_grad_norm=args.max_grad_norm,
            )
        elif args.algo == 'acktr':
            agent = algo.A2C_ACKTR(
                actor_critic, args.value_loss_coef, args.entropy_coef,
                acktr=True,
            )

        rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space, actor_critic.state_size)
        current_obs = torch.zeros(args.num_processes, *obs_shape)

        obs = envs.reset()
        update_current_obs(obs)

        rollouts.observations[0].copy_(current_obs)

        episode_reward_raw = 0.0
        final_reward_raw = 0.0

        if args.cuda:
            current_obs = current_obs.cuda()
            rollouts.cuda()

        # try to load checkpoint
        try:
            num_trained_frames = np.load(args.save_dir+'/num_trained_frames.npy')[0]
            try:
                actor_critic.load_state_dict(torch.load(args.save_dir+'/trained_learner.pth'))
                print('Load learner previous point: Successed')
            except Exception as e:
                print('Load learner previous point: Failed')
        except Exception as e:
            num_trained_frames = 0
        print('Learner has been trained to step: '+str(num_trained_frames))

        start = time.time()
        j = 0
        while True:
            if num_trained_frames > args.num_frames:
                break

            for step in range(args.num_steps):
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, states = actor_critic.act(
                        rollouts.observations[step],
                        rollouts.states[step],
                        rollouts.masks[step],
                    )
                cpu_actions = action.squeeze(1).cpu().numpy()

                # Obser reward and next obs
                obs, reward_raw, done, info = envs.step(cpu_actions)

                episode_reward_raw += reward_raw[0]
                if done[0]:
                    final_reward_raw = episode_reward_raw
                    episode_reward_raw = 0.0
                reward = np.sign(reward_raw)
                reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()

                # If done then clean the history of observations.
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

                if args.cuda:
                    masks = masks.cuda()

                if current_obs.dim() == 4:
                    current_obs *= masks.unsqueeze(2).unsqueeze(2)
                else:
                    current_obs *= masks

                update_current_obs(obs)
                rollouts.insert(current_obs, states, action, action_log_prob, value, reward, masks)

            with torch.no_grad():
                next_value = actor_critic.get_value(
                    rollouts.observations[-1],
                    rollouts.states[-1],
                    rollouts.masks[-1],
                ).detach()

            rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

            value_loss, action_loss, dist_entropy = agent.update(rollouts)

            rollouts.after_update()

            num_trained_frames += (args.num_steps*args.num_processes)
            j += 1

            # save checkpoint
            if j % args.save_interval == 0 and args.save_dir != "":
                try:
                    np.save(
                        args.save_dir+'/num_trained_frames.npy',
                        np.array([num_trained_frames]),
                    )
                    actor_critic.save_model(save_path=args.save_dir)
                except Exception as e:
                    print("Save checkpoint failed")

            # print info
            if j % args.log_interval == 0:
                end = time.time()
                total_num_steps = (j + 1) * args.num_processes * args.num_steps
                print("[{}/{}], FPS {}, final_reward_raw {:.2f}, remaining {} hours".
                    format(
                        num_trained_frames, args.num_frames,
                        int(num_trained_frames / (end - start)),
                        final_reward_raw,
                        (end - start)/num_trained_frames*(args.num_frames-num_trained_frames)/60.0/60.0
                    )
                )

            # visualize results
            if args.vis and j % args.vis_interval == 0:
                '''we use tensorboard since its better when comparing plots'''
                summary = tf.Summary()
                summary.value.add(
                    tag = 'final_reward_raw',
                    simple_value = final_reward_raw,
                )
                summary.value.add(
                    tag = 'value_loss',
                    simple_value = value_loss,
                )
                summary.value.add(
                    tag = 'action_loss',
                    simple_value = action_loss,
                )
                summary.value.add(
                    tag = 'dist_entropy',
                    simple_value = dist_entropy,
                )
                summary_writer.add_summary(summary, num_trained_frames)
                summary_writer.flush()

    elif args.policy_type == 'hierarchical_policy':
        num_subpolicy = args.num_subpolicy
        update_interval = args.hierarchy_interval

        while len(num_subpolicy)<args.num_hierarchy-1:
            num_subpolicy.append(num_subpolicy[-1])
        while len(update_interval)<args.num_hierarchy-1:
            update_interval.append(update_interval[-1])

        # print(update_interval)
        # print(num_subpolicy)

        actor_critic = {}
        rollouts = {}
        actor_critic['top'] = EHRL_Policy(obs_shape, space.Discrete(num_subpolicy[-1]), None, 128, args.recurrent_policy,'top')
        rollouts['top'] = EHRL_RolloutStorage(int(args.num_steps/update_interval[-1]), args.num_processes, obs_shape, space.Discrete(num_subpolicy[-1]), None, actor_critic['top'].state_size)
        for hie_id in range(args.num_hierarchy-1):
            if hie_id>0:
                actor_critic[str(hie_id)] = EHRL_Policy(obs_shape, space.Discrete(num_subpolicy[hie_id-1]), np.zeros(num_subpolicy[hie_id]), 128, args.recurrent_policy,str(hie_id))
                rollouts[str(hie_id)] = EHRL_RolloutStorage(int(args.num_steps/update_interval[hie_id-1]), args.num_processes, obs_shape, space.Discrete(num_subpolicy[hie_id-1]), np.zeros(num_subpolicy[hie_id]), actor_critic[str(hie_id)].state_size)
            else:
                actor_critic[str(hie_id)] = EHRL_Policy(obs_shape, envs.action_space, np.zeros(num_subpolicy[hie_id]), 128, args.recurrent_policy,str(hie_id))
                rollouts[str(hie_id)] = EHRL_RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space, np.zeros(num_subpolicy[hie_id]), actor_critic[str(hie_id)].state_size)

        if envs.action_space.__class__.__name__ == "Discrete":
            action_shape = 1
        else:
            action_shape = envs.action_space.shape[0]

        if args.cuda:
            for key in actor_critic:
                actor_critic[key].cuda()

        agent = {}
        for ac_key in actor_critic:
            if args.algo == 'a2c':
                agent[ac_key] = algo.A2C_ACKTR(
                    actor_critic[ac_key], args.value_loss_coef, args.entropy_coef,
                    lr=args.lr,
                    eps=args.eps,
                    alpha=args.alpha,
                    max_grad_norm=args.max_grad_norm,
                )
            elif args.algo == 'ppo':
                agent[ac_key] = algo.PPO(
                    actor_critic[ac_key], args.clip_param, args.ppo_epoch, args.num_mini_batch, args.value_loss_coef, args.entropy_coef,
                    lr=args.lr,
                    eps=args.eps,
                    max_grad_norm=args.max_grad_norm,
                )
            elif args.algo == 'acktr':
                agent[ac_key] = algo.A2C_ACKTR(
                    actor_critic[ac_key], args.value_loss_coef, args.entropy_coef,
                    acktr=True,
                )

        current_obs = torch.zeros(args.num_processes, *obs_shape)

        obs = envs.reset()
        update_current_obs(obs)

        for obs_key in rollouts:
            rollouts[obs_key].observations[0].copy_(current_obs)

        episode_reward_raw = 0.0
        final_reward_raw = 0.0

        if args.cuda:
            current_obs = current_obs.cuda()
            for rol_key in rollouts:
                rollouts[rol_key].cuda()

        # try to load checkpoint
        try:
            num_trained_frames = np.load(args.save_dir+'/num_trained_frames.npy')[0]
            try:
                for save_key in actor_critic:
                    actor_critic[save_key].load_state_dict(torch.load(args.save_dir+'/trained_learner_'+save_key+'.pth'))
                print('Load learner previous point: Successed')
            except Exception as e:
                print('Load learner previous point: Failed')
        except Exception as e:
            num_trained_frames = 0
        print('Learner has been trained to step: '+str(num_trained_frames))

        start = time.time()
        j = 0
        onehot_mem = {}
        reward_mem = {}
        update_flag = np.zeros(args.num_hierarchy-1, dtype=np.uint8)
        step_count = 0

        value = {}
        next_value = {}
        action = {}
        action_log_prob = {}
        states = {}
        while True:
            if num_trained_frames > args.num_frames:
                break
            step_count = 0

            for step in range(args.num_steps):
                if step_count % update_interval[-1] == 0:
                    # print
                    with torch.no_grad():
                        value['top'], action['top'], action_log_prob['top'], states['top'] = actor_critic['top'].act(
                            rollouts['top'].observations[update_flag[-1]],
                            None,
                            rollouts['top'].states[update_flag[-1]],
                            rollouts['top'].masks[update_flag[-1]],
                        )
                    update_flag[-1]+=1
                    onehot_mem[str(args.num_hierarchy-1)] = get_onehot(num_subpolicy[-1],action['top'])
                if len(update_interval)>1:
                    for interval_id in range(len(update_interval)-1):
                        if step_count % update_interval[interval_id] == 0:
                            with torch.no_grad():
                                value[str(interval_id+1)], action[str(interval_id+1)], action_log_prob[str(interval_id+1)], states[str(interval_id+1)] = \
                                actor_critic[str(interval_id+1)].act(
                                    rollouts[str(interval_id+1)].observations[update_flag[interval_id]],
                                    onehot_mem[str(interval_id+2)],
                                    rollouts[str(interval_id+1)].states[update_flag[interval_id]],
                                    rollouts[str(interval_id+1)].masks[update_flag[interval_id]],
                                )
                            update_flag[interval_id]+=1
                            onehot_mem[str(interval_id+1)] = get_onehot(num_subpolicy[interval_id],action[str(interval_id+1)])
                # Sample actions
                if step_count>=127:
                    print(stop)
                with torch.no_grad():
                    value['0'], action['0'], action_log_prob['0'], states['0'] = actor_critic['0'].act(
                        rollouts['0'].observations[step],
                        rollouts['0'].one_hot[step],
                        rollouts['0'].states[step],
                        rollouts['0'].masks[step],
                    )
                cpu_actions = action['0'].squeeze(1).cpu().numpy()

                # Obser reward and next obs
                obs, reward_raw, done, info = envs.step(cpu_actions)

                for reward_id in range(args.num_hierarchy-1):
                    try:
                        reward_mem[str(reward_id)] += reward_raw[0]
                    except Exception as e:
                        reward_mem[str(reward_id)] = reward_raw[0]

                episode_reward_raw += reward_raw[0]

                if done[0]:
                    final_reward_raw = episode_reward_raw
                    episode_reward_raw = 0.0

                reward = np.sign(reward_raw)
                reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()

                # If done then clean the history of observations.
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

                if args.cuda:
                    masks = masks.cuda()

                if current_obs.dim() == 4:
                    current_obs *= masks.unsqueeze(2).unsqueeze(2)
                else:
                    current_obs *= masks

                update_current_obs(obs)
                rollouts['0'].insert(current_obs, states['0'], action['0'], onehot_mem['1'], action_log_prob['0'], value['0'], reward, masks)
                if step_count % update_interval[-1] == 0 or done[0]:
                    reward_mean = np.mean(np.array(reward_mem[str(args.num_hierarchy-2)]))
                    reward_mean = torch.from_numpy(np.ones(1)*reward_mean).float()
                    rollouts['top'].insert(current_obs, states['top'], action['top'], None, action_log_prob['top'], value['top'], reward_mean , masks)
                    reward_mem[str(args.num_hierarchy-2)] = []
                if len(update_interval)>1:
                    for interval_id in range(len(update_interval)-1):
                        if step_count % update_interval[interval_id] == 0 or done[0]:
                            reward_mean = np.mean(np.array(reward_mem[str(interval_id)]))
                            reward_mean = torch.from_numpy(np.ones(1)*reward_mean).float()
                            rollouts[str(interval_id+1)].insert(current_obs, states[str(interval_id+1)], action[str(interval_id+1)], onehot_mem[str(interval_id+2)], action_log_prob[str(interval_id+1)], value[str(interval_id+1)], reward_mean, masks)
                            reward_mem[str(interval_id)] = []
                step_count+=1

            with torch.no_grad():
                next_value['0'] = actor_critic['0'].get_value(
                    rollouts['0'].observations[-1],
                    rollouts['0'].one_hot[-1],
                    rollouts['0'].states[-1],
                    rollouts['0'].masks[-1],
                ).detach()

            rollouts['0'].compute_returns(next_value['0'], args.use_gae, args.gamma, args.tau)

            value_loss, action_loss, dist_entropy = agent['0'].update(rollouts['0'],add_onehot = True)

            rollouts['0'].after_update()

            with torch.no_grad():
                next_value['top'] = actor_critic['top'].get_value(
                    rollouts['top'].observations[-1],
                    None,
                    rollouts['top'].states[-1],
                    rollouts['top'].masks[-1],
                ).detach()

            rollouts['top'].compute_returns(next_value['top'], args.use_gae, args.gamma, args.tau)
            _, _, _ = agent['top'].update(rollouts['top'], ismaster_policy = True)
            rollouts['top'].after_update()
            update_flag[-1] = 0

            if len(update_interval)>1:
                for interval_id in range(len(update_interval)-1):
                    with torch.no_grad():
                        next_value[str(interval_id+1)] = actor_critic[str(interval_id+1)].get_value(
                            rollouts[str(interval_id+1)].observations[-1],
                            rollouts[str(interval_id+1)].one_hot[-1],
                            rollouts[str(interval_id+1)].states[-1],
                            rollouts[str(interval_id+1)].masks[-1],
                        ).detach()

                    rollouts[str(interval_id+1)].compute_returns(next_value[str(interval_id+1)], args.use_gae, args.gamma, args.tau)
                    _, _, _ = agent[str(interval_id+1)].update(rollouts[str(interval_id+1)],add_onehot = True)
                    rollouts[str(interval_id+1)].after_update()
                    update_flag[interval_id] = 0


            num_trained_frames += (args.num_steps*args.num_processes)
            j += 1

            # save checkpoint
            if j % args.save_interval == 0 and args.save_dir != "":
                try:
                    np.save(
                        args.save_dir+'/num_trained_frames.npy',
                        np.array([num_trained_frames]),
                    )
                    actor_critic.save_model(save_path=args.save_dir)
                except Exception as e:
                    print("Save checkpoint failed")

            # print info
            if j % args.log_interval == 0:
                end = time.time()
                total_num_steps = (j + 1) * args.num_processes * args.num_steps
                print("[{}/{}], FPS {}, final_reward_raw {:.2f}, remaining {} hours".
                    format(
                        num_trained_frames, args.num_frames,
                        int(num_trained_frames / (end - start)),
                        final_reward_raw,
                        (end - start)/num_trained_frames*(args.num_frames-num_trained_frames)/60.0/60.0
                    )
                )

            # visualize results
            if args.vis and j % args.vis_interval == 0:
                '''we use tensorboard since its better when comparing plots'''
                summary = tf.Summary()
                summary.value.add(
                    tag = 'final_reward_raw',
                    simple_value = final_reward_raw,
                )
                summary.value.add(
                    tag = 'value_loss',
                    simple_value = value_loss,
                )
                summary.value.add(
                    tag = 'action_loss',
                    simple_value = action_loss,
                )
                summary.value.add(
                    tag = 'dist_entropy',
                    simple_value = dist_entropy,
                )
                summary_writer.add_summary(summary, num_trained_frames)
                summary_writer.flush()

if __name__ == "__main__":
    main()
