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

    if len(envs.observation_space.shape) == 1:
        envs = VecNormalize(envs, gamma=args.gamma)

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

    actor_critic = Policy(obs_shape, envs.action_space, args.recurrent_policy)

    if envs.action_space.__class__.__name__ == "Discrete":
        action_shape = 1
    else:
        action_shape = envs.action_space.shape[0]

    if args.cuda:
        actor_critic.cuda()

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, lr=args.lr,
                               eps=args.eps,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, acktr=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space, actor_critic.state_size)
    current_obs = torch.zeros(args.num_processes, *obs_shape)

    def update_current_obs(obs):
        shape_dim0 = envs.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        if args.num_stack > 1:
            current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
        current_obs[:, -shape_dim0:] = obs

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



if __name__ == "__main__":
    main()
