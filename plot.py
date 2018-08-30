# # Python imports.
# import sys
#
# # Other imports.
# import os
# import sys
# parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
# sys.path.insert(0, parent_dir)
# from simple_rl.agents import QLearningAgent, RandomAgent
# from simple_rl.tasks import GridWorldMDP
# from simple_rl.run_experiments import run_agents_on_mdp
#
# def main(open_plot=True):
#
#     # Setup MDP.
#     mdp = GridWorldMDP(width=4, height=3, init_loc=(1, 1), goal_locs=[(4, 3)], lava_locs=[(4, 2)], gamma=0.95, walls=[(2, 2)], slip_prob=0.05)
#
#     # Make agents.
#     ql_agent = QLearningAgent(actions=mdp.get_actions())
#     rand_agent = RandomAgent(actions=mdp.get_actions())
#
#     # Run experiment and make plot.
#     run_agents_on_mdp([ql_agent, rand_agent], mdp, instances=5, episodes=50, steps=10, open_plot=open_plot)
#
# if __name__ == "__main__":
#     main(open_plot=not sys.argv[-1] == "no_plot")
import simple_rl
simple_rl.utils.chart_utils.make_plots(experiment_dir='/home/yuhangsong/git/EHRL/results/gridworld_h-3_w-4', experiment_agents=['dd'], plot_file_name="", cumulative=False, use_cost=False, episodic=True, open_plot=True, track_disc_reward=False)
