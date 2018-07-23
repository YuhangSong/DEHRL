import os
import argparse
import torch
import utils

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
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--actor-critic-mini-batch-size', type=int, default=32,
                        help='mini batch size for ppo (default: 32)')
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
    parser.add_argument('--add-timestep', action='store_true', default=False,
                        help='add timestep to observations')
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='use a recurrent policy')
    parser.add_argument('--render', action='store_true',
                        help='render environment in a window')

    '''environment details'''
    parser.add_argument('--obs-type', type=str, default='image',
                        help='observation type: image or ram' )
    parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                        help='environment to train on')
    parser.add_argument('--reward-level', type=int, default=2,
                        help='level of reward in games like OverCooked')
    parser.add_argument('--use-fake-reward-bounty', action='store_true',
                        help='if use fake reward bounty')

    '''policy details'''
    parser.add_argument('--num-hierarchy',      type=int,
                        help='num of the hierarchical_policy' )
    parser.add_argument('--num-subpolicy',      type=int, nargs='*',
                        help='num of the subpolicies per hierarchy' )
    parser.add_argument('--hierarchy-interval', type=int, nargs='*',
                        help='the interval between the subpolicies')
    parser.add_argument('--num-steps',          type=int, nargs='*',
                        help='number of forward steps before update agent')

    '''reward bounty details'''
    parser.add_argument('--reward-bounty', type=float,
                        help='the discount for the reward bounty, it would be different for shared_policy and hierarchical_policy' )
    parser.add_argument('--encourage-ac-connection', type=str,
                        help='encourage connection to action conditional input on: transition_model, actor_critic, both, none' )
    parser.add_argument('--encourage-ac-connection-coefficient', type=float,
                        help='coefficient of encourage-ac-connection')

    '''for log behavior'''
    parser.add_argument('--log-behavior-interval', type=int, default=10,
                        help='log behavior every x minutes')

    parser.add_argument('--aux', type=str, default='',
                        help='some aux information you may want to record along with this run')

    args = parser.parse_args()
    args.transition_model_mini_batch_size = int(args.actor_critic_mini_batch_size/4)
    args.transition_model_epoch = int(args.ppo_epoch*4)

    '''basic save path'''
    args.save_dir = os.path.join(args.save_dir, args.exp)

    '''environment details'''
    args.save_dir = os.path.join(args.save_dir, 'o_t-{}'.format(args.obs_type))
    args.save_dir = os.path.join(args.save_dir, 'e_n-{}'.format(args.env_name))
    if args.env_name in ['OverCooked']:
        args.save_dir = os.path.join(args.save_dir, 'r_l-{}'.format(args.reward_level))
        args.save_dir = os.path.join(args.save_dir, 'u_f_r_b-{}'.format(args.use_fake_reward_bounty))

    '''policy details'''
    args.save_dir = os.path.join(args.save_dir, 'n_h-{}'.format(args.num_hierarchy))
    args.save_dir = os.path.join(args.save_dir, 'n_s-{}'.format(utils.list_to_str(args.num_subpolicy)))
    args.save_dir = os.path.join(args.save_dir, 'h_i-{}'.format(utils.list_to_str(args.hierarchy_interval)))
    args.save_dir = os.path.join(args.save_dir, 'n_s-{}'.format(utils.list_to_str(args.num_steps)))

    '''reward bounty details'''
    args.save_dir = os.path.join(args.save_dir, 'r_b-{}'.format(args.reward_bounty))

    if args.reward_bounty > 0.0:
        '''has a transition_model'''
        args.save_dir = os.path.join(args.save_dir, 't_m_m_b_s-{}'.format(args.transition_model_mini_batch_size))
        args.save_dir = os.path.join(args.save_dir, 't_m_e-{}'.format(args.transition_model_epoch))

    if (args.reward_bounty > 0.0) or args.use_fake_reward_bounty:
        '''for encourage_ac_connection'''

        if args.use_fake_reward_bounty:
            '''if use_fake_reward_bounty, encourage_ac_connection can only be applied on actor_critic, or not applied'''
            assert args.encourage_ac_connection in ['none','actor_critic']

        args.save_dir = os.path.join(args.save_dir, 'e_a_c-{}'.format(args.encourage_ac_connection))
        if args.encourage_ac_connection not in ['none']:
            args.save_dir = os.path.join(args.save_dir, 'e_a_c_c-{}'.format(args.encourage_ac_connection_coefficient))

    args.save_dir = os.path.join(args.save_dir, 'a-{}'.format(args.aux))

    return args
