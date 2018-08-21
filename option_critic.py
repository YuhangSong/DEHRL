#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from deep_rl import *
import tensorflow as tf

sess = tf.Session()

from arguments_for_option_critic import get_args
args = get_args()
try:
    os.makedirs(args.save_dir)
    print('Dir empty, making new log dir...')
except Exception as e:
    if e.__class__.__name__ in ['FileExistsError']:
        print('Dir exsit, checking checkpoint...')
    else:
        raise e

summary_writer = tf.summary.FileWriter(args.save_dir)

# Option-Critic
def option_ciritc_pixel_atari(name):
    config = Config()
    config.history_length = 4

    if name in ['OverCooked']:
        task_fn = lambda log_dir: PixelAtari(name, frame_skip=1, history_length=config.history_length, log_dir=log_dir, args=args)
    else:
        task_fn = lambda log_dir: PixelAtari(name, frame_skip=4, history_length=config.history_length, log_dir=log_dir)

    config.num_workers = 16
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers,
                                              log_dir=get_default_log_dir(option_ciritc_pixel_atari.__name__, args),
                                              single_process=True)
    config.eval_env = task_fn(None)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=1e-4, alpha=0.99, eps=1e-5)
    config.network_fn = lambda: OptionCriticNet(NatureConvBody(), config.action_dim, num_options=4)
    config.random_option_prob = LinearSchedule(0.1)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.rollout_length = 5
    config.gradient_clip = 5
    config.max_steps = int(100000000)
    config.entropy_weight = 0.01
    config.termination_regularizer = 0.01
    config.logger = get_logger(log_dir=args.save_dir)
    run_steps(OptionCriticAgent(config), summary_writer, args)

if __name__ == '__main__':

    set_one_thread()
    select_device(0)

    game = args.env_name
    option_ciritc_pixel_atari(game)
