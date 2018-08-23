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

    parser.add_argument('--log-interval', type=int, default=1,
                        help='log interval, one log per n updates')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='save interval, one save per n updates')
    parser.add_argument('--vis-interval', type=int, default=1,
                        help='vis interval, one log per n updates')

    '''environment details'''
    parser.add_argument('--obs-type', type=str, default='image',
                        help='observation type: image or ram' )
    parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                        help='environment to train on')
    parser.add_argument('--reward-level', type=int, default=2,
                        help='level of reward in games like OverCooked')
    parser.add_argument('--use-fake-reward-bounty', action='store_true',
                        help='if use fake reward bounty')
    parser.add_argument('--reset-leg', action='store_true',
                        help='if reset four legs after four steps')
    parser.add_argument('--add-goal-color', action='store_true',
                        help='if add area color when get the goal')
    parser.add_argument('--setup-goal', type=str, default='random',
                        help='The setup for goal: fix, random, any')

    '''for log behavior'''
    parser.add_argument('--log-behavior-interval', type=int, default=10,
                        help='log behavior every x minutes')
    parser.add_argument('--act-deterministically', action='store_true',
                        help='if act deterministically when interactiong')

    parser.add_argument('--aux', type=str, default='',
                        help='some aux information you may want to record along with this run')

    parser.add_argument('--test-reward-bounty', action='store_true',
                        help='to test what reward bounty will each macro-action produce')
    parser.add_argument('--test-action', action='store_true',
                        help='specify actions at every level')
    parser.add_argument('--test-action-vis', action='store_true',
                        help='see actions at every level')
    parser.add_argument('--run-overcooked', action='store_true',
                        help='run overcooked to debug the game')
    parser.add_argument('--see-leg-fre', action='store_true',
                        help='see the frequency of each leg through tensorboard')
    parser.add_argument('--render', action='store_true',
                        help='render environment in a window')

    args = parser.parse_args()

    '''basic save path'''
    args.save_dir = os.path.join(args.save_dir, args.exp)
    args.save_dir = os.path.join(args.save_dir, 'option_critic')

    '''environment details'''
    args.save_dir = os.path.join(args.save_dir, 'o_t-{}'.format(args.obs_type))
    args.save_dir = os.path.join(args.save_dir, 'e_n-{}'.format(args.env_name))
    if args.env_name in ['OverCooked']:
        args.save_dir = os.path.join(args.save_dir, 'r_l-{}'.format(args.reward_level))
        args.save_dir = os.path.join(args.save_dir, 'u_f_r_b-{}'.format(args.use_fake_reward_bounty))
        args.save_dir = os.path.join(args.save_dir, 'r_lg-{}'.format(args.reset_leg))
        args.save_dir = os.path.join(args.save_dir, 'a_g_c-{}'.format(args.add_goal_color))
        args.save_dir = os.path.join(args.save_dir, 's_g-{}'.format(args.setup_goal))

    args.save_dir = os.path.join(args.save_dir, 'a-{}'.format(args.aux))

    return args
