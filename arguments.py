import os
import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='RL')

    '''basic save and log'''
    parser.add_argument('--save-dir', default='../results/',
                        help='directory to save agent logs')
    parser.add_argument('--exp', type=str,
                        help='exp')

    '''following settings are seen as default in this project'''
    parser.add_argument('--algo', default='a2c',
                        help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=16,
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--num-steps', type=int, default=5,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--num-stack', type=int, default=4,
                        help='number of frames to stack (default: 4)')
    parser.add_argument('--log-interval', type=int, default=1,
                        help='log interval, one log per n updates')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='save interval, one save per n updates')
    parser.add_argument('--vis-interval', type=int, default=1,
                        help='vis interval, one log per n updates')
    parser.add_argument('--num-frames', type=int, default=10e7,
                        help='number of frames to train (default: 10e6)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--add-timestep', action='store_true', default=False,
                        help='add timestep to observations')
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='use a recurrent policy')
    parser.add_argument('--no-vis', action='store_true', default=False,
                        help='disables visdom visualization')

    '''environment details'''
    parser.add_argument('--obs-type', type=str, default='image',
                        help='observation type: image or ram' )
    parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                        help='environment to train on')
    parser.add_argument('--reward-level', type=int, default=2,
                        help='level of reward in games like OverCooked')

    '''policy details'''
    parser.add_argument('--num-hierarchy', type=int, default = 1,
                        help='num of the hierarchical_policy' )
    parser.add_argument('--num-subpolicy', type=int, nargs='*', default = 1,
                        help='num of the subpolicies per hierarchy' )
    parser.add_argument('--hierarchy-interval', type=int, nargs='*', default = 1,
                        help='the interval between the subpolicies')

    '''reward bounty details'''
    parser.add_argument('--reward-bounty', type=float,
                        help='the discount for the reward bounty, it would be different for shared_policy and hierarchical_policy' )

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.vis = not args.no_vis

    '''basic save path'''
    args.save_dir = os.path.join(args.save_dir, args.exp)

    '''environment details'''
    args.save_dir = os.path.join(args.save_dir, 'obs-type-{}'.format(args.obs_type))
    args.save_dir = os.path.join(args.save_dir, 'env_name-{}'.format(args.env_name))
    if args.env_name in ['OverCooked']:
        args.save_dir = os.path.join(args.save_dir, 'reward_level-{}'.format(args.reward_level))

    '''policy details'''
    args.save_dir = os.path.join(args.save_dir, 'num_hierarchy-{}'.format(args.num_hierarchy))
    args.save_dir = os.path.join(args.save_dir, 'num_subpolicy-{}'.format(args.num_subpolicy[0]))
    args.save_dir = os.path.join(args.save_dir, 'hierarchy_interval-{}'.format(args.hierarchy_interval[0]))

    '''reward bounty details'''
    args.save_dir = os.path.join(args.save_dir, 'reward_bounty-{}'.format(args.reward_bounty))

    return args
