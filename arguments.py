import os
import argparse
import torch
import utils

def get_args():
    parser = argparse.ArgumentParser(description='RL')

    '''basic save and log'''
    parser.add_argument('--exp', type=str,
                        help='Give a top description of this experiment')

    '''following settings are seen as default in this project'''
    parser.add_argument('--algo', default='a2c',
                        help='Algorithm to use: a2c | ppo | acktr (Currently supported: PPO)')
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='Learning rate')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor for rewards')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='If use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='GAE parameter')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='Entropy term coefficient. Only for bottom_layer, other layers are default to be 0.01')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='Value loss coefficient')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='Max norm of gradients')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--num-processes', type=int, default=16,
                        help='How many training CPU processes to use')
    parser.add_argument('--actor-critic-epoch', type=int, default=4,
                        help='Number of ppo epochs')
    parser.add_argument('--actor-critic-mini-batch-size', type=int, default=32,
                        help='Mini batch size for PPO')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='PPO clip parameter')
    parser.add_argument('--num-stack', type=int, default=1,
                        help='Number of frames to stack')
    parser.add_argument('--log-interval', type=int, default=1,
                        help='In updates')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='In updates')
    parser.add_argument('--vis-curves-interval', type=int, default=1,
                        help='In updates')
    parser.add_argument('--num-frames', type=int, default=20e7,
                        help='number of frames to train')
    parser.add_argument('--add-timestep', action='store_true', default=False,
                        help='add timestep to observations (depreciated)')
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='use a recurrent policy (depreciated)')
    parser.add_argument('--render', action='store_true',
                        help='render environment in a window (depreciated)')

    '''environment details'''
    parser.add_argument('--obs-type', type=str, default='image',
                        help='Observation type: image or ram (depreciated)' )
    parser.add_argument('--env-name',
                        help='Name of the environment')
    parser.add_argument('--reward-level', type=int,
                        help='Level of reward in some games, including: OverCooked')
    parser.add_argument('--episode-length-limit', type=int,
                        help='Episode length limit in game, including: Explore2D')
    parser.add_argument('--use-fake-reward-bounty', action='store_true',
                        help='If use fake reward bounty (depreciated)')
    parser.add_argument('--reset-leg', action='store_true',
                        help='If reset four legs after four steps (for OverCooked, depreciated)')
    parser.add_argument('--add-goal-color', action='store_true',
                        help='If add area color when get the goal (for OverCooked, depreciated)')
    parser.add_argument('--setup-goal', type=str, default='random',
                        help='The setup for goal type: fix, random, any')
    parser.add_argument('--new-overcooked', action='store_true',
                        help='If use new overcooked or not')

    '''policy details'''
    parser.add_argument('--num-hierarchy',      type=int,
                        help='Num of the hierarchical_policy, i.e., L')
    parser.add_argument('--num-subpolicy',      type=int, nargs='*',
                        help='Num of the subpolicies per hierarchy, i.e., A^l')
    parser.add_argument('--hierarchy-interval', type=int, nargs='*',
                        help='Interval between the subpolicies, i.e., T^l')
    parser.add_argument('--num-steps',          type=int, nargs='*',
                        help='Number of forward steps before update agent')

    '''reward bounty details'''
    parser.add_argument('--reward-bounty', type=float,
                        help='Coefficient of the raw reward bounty, set to 0 to disable reward bounty' )
    parser.add_argument('--distance', type=str,
                        help='Distance to meansure the difference between states: l1, mass_center, l1_mass_center' )
    parser.add_argument('--train-mode', type=str,
                        help='Train mode for transition_model and actor_critic: switch, together' )
    parser.add_argument('--unmask-value-function', action='store_true',
                        help='Whether unmask value functions' )
    parser.add_argument('--clip-reward-bounty', action='store_true',
                        help='Whether clip the reward bounty' )
    parser.add_argument('--clip-reward-bounty-active-function', type=str,
                        help='Active function of clip reward bounty: linear, u, relu, shrink_relu' )
    parser.add_argument('--transition-model-mini-batch-size', type=int, nargs='*',
                        help='Num of the subpolicies per hierarchy' )

    parser.add_argument('--mutual-information', action='store_true',
                        help='Whether use mutual information as bounty reward' )

    '''inverse mask model'''
    parser.add_argument('--inverse-mask', action='store_true',
                        help='Whether use inverse mask to avoid the influence from uncontrollable parts of state' )
    parser.add_argument('--num-grid', type=int,
                        help='Num grid of inverse_mask_model' )

    parser.add_argument('--aux', type=str, default='',
                        help='Some aux information you may want to record along with this run')


    '''for summarize behavior'''
    parser.add_argument('--summarize-behavior-interval', type=int, default=10,
                        help='Interval for summarizing behavior (in minutes)')

    parser.add_argument('--summarize-observation', action='store_true',
                        help='Whether summarize observation as video' )
    parser.add_argument('--summarize-rendered-behavior', action='store_true',
                        help='Whether summarize rendered behavior as video (only for envs with render()')
    parser.add_argument('--summarize-state-prediction', action='store_true',
                        help='Whether summarize state and prediction as video' )

    parser.add_argument('--summarize-one-episode',  type=str, default='None',
                        help='Whether only summarize one episode, if not None, log with it as a log_header')
    parser.add_argument('--act-deterministically', action='store_true',
                        help='Whether act deterministically when interactiong')

    '''for debug'''
    parser.add_argument('--test-action', action='store_true',
                        help='Specify actions at every level')
    parser.add_argument('--see-leg-fre', action='store_true',
                        help='See the frequency of each leg through tensorboard')

    args = parser.parse_args()

    args.summarize_behavior = args.summarize_observation or args.summarize_rendered_behavior or args.summarize_state_prediction

    '''none = []'''
    if args.num_subpolicy is None:
        args.num_subpolicy = []
    if args.hierarchy_interval is None:
        args.hierarchy_interval = []
    if args.num_steps is None:
        args.num_steps = []

    args.transition_model_epoch = args.actor_critic_epoch

    args.save_dir = ''

    '''In the following, if the line is commended, it means the property is depreciated or treated as default'''

    '''environment details'''
    # args.save_dir = os.path.join(args.save_dir, 'o_t-{}'.format(args.obs_type))
    args.save_dir = os.path.join(args.save_dir, 'e_n-{}'.format(args.env_name))
    if args.env_name in ['OverCooked']:
        args.save_dir = os.path.join(args.save_dir, 'r_l-{}'.format(args.reward_level))
        # args.save_dir = os.path.join(args.save_dir, 'u_f_r_b-{}'.format(args.use_fake_reward_bounty))
        # args.save_dir = os.path.join(args.save_dir, 'r_lg-{}'.format(args.reset_leg))
        # args.save_dir = os.path.join(args.save_dir, 'a_g_c-{}'.format(args.add_goal_color))
        args.save_dir = os.path.join(args.save_dir, 's_g-{}'.format(args.setup_goal))
        # args.save_dir = os.path.join(args.save_dir, 'n_o-{}'.format(args.new_overcooked))

    if args.env_name in ['Explore2D']:
        args.save_dir = os.path.join(args.save_dir, 'e_l_l-{}'.format(args.episode_length_limit))

    '''policy details'''
    args.save_dir = os.path.join(args.save_dir, 'n_h-{}'.format(args.num_hierarchy))
    args.save_dir = os.path.join(args.save_dir, 'n_s-{}'.format(utils.list_to_str(args.num_subpolicy)))
    args.save_dir = os.path.join(args.save_dir, 'h_i-{}'.format(utils.list_to_str(args.hierarchy_interval)))
    # args.save_dir = os.path.join(args.save_dir, 'n_s-{}'.format(utils.list_to_str(args.num_steps)))

    '''reward bounty details'''
    args.save_dir = os.path.join(args.save_dir, 'r_b-{}'.format(args.reward_bounty))

    '''actor_critic training details'''
    # args.save_dir = os.path.join(args.save_dir, 'a_c_m_b_s-{}'.format(args.actor_critic_mini_batch_size))
    # args.save_dir = os.path.join(args.save_dir, 'a_c_e-{}'.format(args.actor_critic_epoch))
    if args.reward_bounty > 0.0:
        '''distance'''
        args.save_dir = os.path.join(args.save_dir, 'd-{}'.format(args.distance))
        '''transition_model training details'''
        # args.save_dir = os.path.join(args.save_dir, 't_m_e-{}'.format(args.transition_model_epoch))
        '''train mode'''
        # args.save_dir = os.path.join(args.save_dir, 't_m-{}'.format(args.train_mode))
        '''mask value function'''
        args.save_dir = os.path.join(args.save_dir, 'u_v_f-{}'.format(args.unmask_value_function))
        '''mutual information'''
        args.save_dir = os.path.join(args.save_dir, 'm_i-{}'.format(args.mutual_information))
        '''clip reward bounty'''
        args.save_dir = os.path.join(args.save_dir, 'c_r_b-{}'.format(args.clip_reward_bounty))
        if args.clip_reward_bounty:
            args.save_dir = os.path.join(args.save_dir, 'c_r_b_a_f-{}'.format(args.clip_reward_bounty_active_function))
        '''inverse mask'''
        args.save_dir = os.path.join(args.save_dir, 'i_m-{}'.format(args.inverse_mask))
        if args.inverse_mask:
            args.save_dir = os.path.join(args.save_dir, 'n_g-{}'.format(args.num_grid))

    args.save_dir = os.path.join(args.save_dir, 'a-{}'.format(args.aux))

    args.save_dir = args.save_dir.replace('/','--')
    args.save_dir = os.path.join(args.exp, args.save_dir)
    args.save_dir = os.path.join('../results', args.save_dir)

    return args
