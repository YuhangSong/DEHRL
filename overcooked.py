"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c
New gym game OverCooked, support by Iceclear,
A game with three tasks.
"""

import logging
import math
import gym
from gym import error, spaces
from gym.utils import seeding
import numpy as np
import cv2
import random

logger = logging.getLogger(__name__)

class OverCooked(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, args=None):

        self.args = args

        self.action_space = spaces.Discrete(17)
        self.screen_width = 400
        self.screen_height = 400
        self.leg_num = 4
        self.goal_num = 4
        self.eposide_length = 0
        self.action_count = np.zeros(4)
        self.leg_count = np.zeros(self.leg_num*4+1)
        self.info = {}
        self.color_area = []

        if self.args.new_overcooked:
            self.img = np.ones((int(self.screen_width + self.screen_width / 8), int(self.screen_height), 3), np.uint8) * 255

        self.max_y = self.screen_height-self.screen_height/10
        self.min_y = self.screen_height/10
        self.max_x = self.screen_width-self.screen_width/10
        self.min_x = self.screen_width/10

        '''move steps: default:1---3 step, must be int'''
        self.body_steps = 3
        '''body thickness, default -- 2, -1 means solid'''
        self.body_thickness = -1
        '''leg size, default -- self.screen_width/40'''
        self.leg_size = self.screen_width/30
        '''body size, default -- self.screen_width/10'''
        self.body_size = self.screen_width/10
        '''leg position indent'''
        if self.args.new_overcooked:
            self.leg_indent = self.leg_size/2
        else:
            self.leg_indent = 0
        '''leg move distance'''
        self.leg_move_dis = self.screen_width/40
        self.body_move_dis = (int(self.screen_width/2)-int(self.min_x)-self.body_size/2-self.leg_size+self.leg_indent)/self.body_steps




        assert self.args.obs_type in ('ram', 'image')
        if self.args.obs_type == 'ram':
            self.observation_space = spaces.Box(low=0, high=1.0, dtype=np.float64, shape=(26,))
        elif self.args.obs_type == 'image':
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 1),dtype=np.uint8)
        else:
            raise error.Error('Unrecognized observation type: {}'.format(self.args.obs_type))

        if self.args.reward_level in [0]:
            self.episode_length_limit = 5
        elif self.args.reward_level in [1]:
            self.episode_length_limit = 4*6*2
        elif self.args.reward_level in [2]:
            if self.goal_num in [4]:
                # get 4 food in sequence
                self.episode_length_limit = 6+12+6+12
            elif self.goal_num in [3]:
                # get 3 food in sequence
                self.episode_length_limit = 6+12+6
            elif self.goal_num in [2]:
                # get 2 food in sequence
                self.episode_length_limit = 6+12
            elif self.goal_num in [1]:
                # get 1 food in sequence
                self.episode_length_limit = 6
            else:
                raise NotImplementedError
            self.episode_length_limit = self.episode_length_limit*4*2
        else:
            raise NotImplementedError

        if self.args.setup_goal in ['random','fix','any']:
            pass
        else:
            raise NotImplementedError

        self.realgoal = np.arange(1,self.goal_num+1)
        self.cur_goal = np.zeros(self.goal_num)
        self.viewer = None
        self.leg_id = 0
        self.goal_id = 0
        self.action_mem = np.zeros(self.leg_num)

        if self.args.new_overcooked:
            '''load pic'''
            self.background = self.adjust_color(cv2.imread('./game_pic/background.png'))
            self.background = cv2.resize(self.background,(self.screen_width,self.screen_height))

            self.goal_0 = self.adjust_color(cv2.imread('./game_pic/lemen.png',cv2.IMREAD_UNCHANGED))
            self.goal_0 = cv2.resize(self.goal_0,(int(self.screen_width/10),int(self.screen_height/10)))
            self.goal_1 = self.adjust_color(cv2.imread('./game_pic/orange_pepper.png',cv2.IMREAD_UNCHANGED))
            self.goal_1 = cv2.resize(self.goal_1,(int(self.screen_width/10),int(self.screen_height/10)))
            self.goal_2 = self.adjust_color(cv2.imread('./game_pic/padan.png',cv2.IMREAD_UNCHANGED))
            self.goal_2 = cv2.resize(self.goal_2,(int(self.screen_width/10),int(self.screen_height/10)))
            self.goal_3 = self.adjust_color(cv2.imread('./game_pic/cabbage.png',cv2.IMREAD_UNCHANGED))
            self.goal_3 = cv2.resize(self.goal_3,(int(self.screen_width/10),int(self.screen_height/10)))

            self.body = self.adjust_color(cv2.imread('./game_pic/body.png', cv2.IMREAD_UNCHANGED))
            self.body = cv2.resize(self.body,(int(self.body_size),int(self.body_size)))
            self.leg = self.adjust_color(cv2.imread('./game_pic/leg.png', cv2.IMREAD_UNCHANGED))
            self.leg = cv2.resize(self.leg,(int(self.leg_size),int(self.leg_size)))

            self.stove = self.adjust_color(cv2.imread('./game_pic/stove.png'))
            self.stove = cv2.resize(self.stove,(int(self.screen_height), int(self.screen_width + self.screen_width / 8)-int(self.screen_width)))

            self.img[int(self.screen_width):int(self.screen_width + self.screen_width / 8),0:int(self.screen_height),:] = self.stove

        # Just need to initialize the relevant attributes
        self.configure()

        self.goal_position = []
        self.goal_position.append(np.array([self.min_x, self.min_y]))
        self.goal_position.append(np.array([self.max_x, self.min_y]))
        self.goal_position.append(np.array([self.min_x, self.max_y]))
        self.goal_position.append(np.array([self.max_x, self.max_y]))

        self.goal_color = []
        self.goal_color.append(np.array([55, 255, 155]))
        self.goal_color.append(np.array([155, 0, 155]))
        self.goal_color.append(np.array([0, 255, 255]))
        self.goal_color.append(np.array([255, 0, 0]))

        self.triangle_line = []
        self.triangle_line.append(np.array([int(0), int(self.screen_height)]))
        self.triangle_line.append(
            np.array([int(self.screen_width / 20), int(self.screen_height - self.screen_height / 10)]))
        self.triangle_line.append(np.array([int(self.screen_width / 10), int(self.screen_height)]))
        self.triangle_line = np.array(self.triangle_line)
        self.triangle_line = self.triangle_line.reshape((-1, 1, 2))

        self.canvas_clear()

        self.goal_ram = np.zeros(self.goal_num)

    def canvas_clear(self):
        if self.args.new_overcooked:
            self.show_next_goal(self.goal_id)
            self.img[0:self.screen_width,0:self.screen_height,:] = self.background
        else:
            # canvas
            self.img = np.ones((int(self.screen_width + self.screen_width / 4.5), int(self.screen_height), 3), np.uint8) * 255

            # goals
            cv2.circle(self.img, (int(self.screen_width / 20), int(self.screen_height / 20)), int(self.screen_height / 20),
                       (int(self.goal_color[0][0]), int(self.goal_color[0][1]), int(self.goal_color[0][2])), -1)
            cv2.circle(self.img, (int(self.screen_width - self.screen_width / 20), int(self.screen_height / 20)),
                       int(self.screen_height / 20 - 1.5),
                       (int(self.goal_color[1][0]), int(self.goal_color[1][1]), int(self.goal_color[1][2])), 1)
            cv2.rectangle(self.img, (
            int(self.screen_width - self.screen_width / 10 + 2), int(self.screen_height - self.screen_height / 10 + 2)),
                          (int(self.screen_width - 2), int(self.screen_height - 2)),
                          (int(self.goal_color[2][0]), int(self.goal_color[2][1]), int(self.goal_color[2][2])), 1)
            cv2.polylines(self.img, [self.triangle_line], True,
                          (int(self.goal_color[3][0]), int(self.goal_color[3][1]), int(self.goal_color[3][2])), 1)
            # stoves
            cv2.rectangle(self.img, (0, int(self.screen_height / 10)),
                          (int(self.screen_width / 10), int(self.screen_height - self.screen_height / 10)), (255, 228, 225),
                          -1)
            cv2.rectangle(self.img, (int(self.screen_width - self.screen_width / 10), int(self.screen_height / 10)),
                          (self.screen_width, int(self.screen_height - self.screen_height / 10)), (255, 228, 225), -1)
            cv2.rectangle(self.img, (int(self.screen_width / 10), 0),
                          (int(self.screen_width - self.screen_width / 10), int(self.screen_height / 10)), (255, 228, 225),
                          -1)
            cv2.rectangle(self.img, (int(self.screen_width / 10), int(self.screen_height - self.screen_height / 10)),
                          (int(self.screen_width - self.screen_width / 10), self.screen_height), (255, 228, 225), -1)

    def adjust_color(self, input):
        '''change RGB to BGR'''
        pic = input.copy()
        pic[:,:,0] = input[:,:,2]
        pic[:,:,1] = input[:,:,1]
        pic[:,:,2] = input[:,:,0]
        return pic

    def setgoal(self):
        if self.args.reward_level == 1:
            if self.args.new_overcooked:
                position = np.array([self.screen_width/13,self.screen_height*1.01])
            else:
                position = np.array([0,self.screen_height])
            self.draw_goals(self.single_goal+1,position,self.img)
        elif self.args.reward_level == 2:
            for i in range(self.goal_num):
                if self.args.new_overcooked:
                    position = np.array([self.screen_width/13+i*self.screen_width/10,self.screen_height*1.01])
                else:
                    position = np.array([i*self.screen_width/10,self.screen_height])
                self.draw_goals(self.realgoal[i],position,self.img)

    def show_next_goal(self,goal_num):
        if self.args.reward_level == 2:
            if self.args.setup_goal in ['random', 'fix']:
                if self.args.new_overcooked:
                    show_position = np.array([int(self.screen_width*0.375),int(self.screen_height*0.885)])
                else:
                    show_position = np.array([int(self.screen_width*0.7),int(self.screen_height*1.05)])
                    cv2.rectangle(self.img, (show_position[0], show_position[1]),
                                  (int(show_position[0]+self.screen_width/9), int(show_position[1]+self.screen_height/9)),
                                  (255,255,255), -1)
                if goal_num<len(self.realgoal):
                    self.draw_goals(self.realgoal[goal_num],show_position,self.img)


    def draw_goals(self,goal_num,position,canvas):
        if goal_num == 1:
            if self.args.new_overcooked:
                self.overlay_image_alpha(canvas,self.goal_0,[int(position[0]),int(position[1])],self.goal_0[:,:,3]/255.0)
            else:
                cv2.circle(canvas, (int(position[0]+self.screen_height/20), int(position[1]+self.screen_height/20)),int(self.screen_height / 20),(int(self.goal_color[0][0]), int(self.goal_color[0][1]), int(self.goal_color[0][2])), -1)
        elif goal_num == 2:
            if self.args.new_overcooked:
                self.overlay_image_alpha(canvas,self.goal_1,[int(position[0]),int(position[1])],self.goal_1[:,:,3]/255.0)
            else:
                cv2.circle(canvas, (int(position[0]+self.screen_height/20), int(position[1]+self.screen_height/20)),int(self.screen_height / 20 - 1.5),(int(self.goal_color[1][0]), int(self.goal_color[1][1]), int(self.goal_color[1][2])), 1)
        elif goal_num == 3:
            if self.args.new_overcooked:
                self.overlay_image_alpha(canvas,self.goal_2,[int(position[0]),int(position[1])],self.goal_2[:,:,3]/255.0)
            else:
                triangle_line = []
                triangle_line.append(np.array([int(position[0]), int(position[1]+self.screen_height/10)]))
                triangle_line.append(np.array([int(position[0]+self.screen_width/20), int(position[1]+self.screen_height/10-self.screen_height/10)]))
                triangle_line.append(np.array([int(position[0]+self.screen_width/10), int(position[1]+self.screen_height/10)]))
                triangle_line = np.array(triangle_line)
                triangle_line = triangle_line.reshape((-1, 1, 2))
                cv2.polylines(canvas, [triangle_line], True,(int(self.goal_color[3][0]), int(self.goal_color[3][1]), int(self.goal_color[3][2])), 1)
        elif goal_num == 4:
            if self.args.new_overcooked:
                self.overlay_image_alpha(canvas,self.goal_3,[int(position[0]),int(position[1])],self.goal_3[:,:,3]/255.0)
            else:
                cv2.rectangle(canvas, (int(position[0] + 2), int(position[1] + 2)),
                              (int(position[0]+self.screen_width/10 - 2), int(position[1]+self.screen_height/10 - 2)),
                              (int(self.goal_color[2][0]), int(self.goal_color[2][1]), int(self.goal_color[2][2])), 1)

    def configure(self, display=None):
        self.display = display

    def seed(self, seed):
        np.random.seed(seed)

    def reset_leg_position(self):
        self.leg_position = []
        self.leg_position.append(
            np.array([self.position[0]-self.leg_size+self.leg_indent, self.position[1]-self.leg_size+self.leg_indent]))
        self.leg_position.append(
            np.array([self.position[0]-self.leg_size+self.leg_indent, self.position[1]+self.body_size-self.leg_indent]))
        self.leg_position.append(
            np.array([self.position[0]+self.body_size-self.leg_indent, self.position[1]-self.leg_size+self.leg_indent]))
        self.leg_position.append(
            np.array([self.position[0]+self.body_size-self.leg_indent, self.position[1]+self.body_size-self.leg_indent]))

        self.reset_legposi = []
        self.reset_legposi.append(
            np.array([self.position[0]-self.leg_size+self.leg_indent, self.position[1]-self.leg_size+self.leg_indent]))
        self.reset_legposi.append(
            np.array([self.position[0]-self.leg_size+self.leg_indent, self.position[1]+self.body_size-self.leg_indent]))
        self.reset_legposi.append(
            np.array([self.position[0]+self.body_size-self.leg_indent, self.position[1]-self.leg_size+self.leg_indent]))
        self.reset_legposi.append(
            np.array([self.position[0]+self.body_size-self.leg_indent, self.position[1]+self.body_size-self.leg_indent]))

    def step(self, action_list):
        reset_body = False

        if self.args.use_fake_reward_bounty:
            # for use_fake_reward_bounty
            # action_list is a list, the first element is the bottom action
            # the second is input_actions_onehot_global[0]
            # ...
            action_id = action_list[0]
            # raise Exception('Do not support this any more')
        else:
            action_id = action_list

        done = False
        self.eposide_length += 1
        reward = 0.0
        if action_id<17:

            self.leg_count[action_id] += 1
            self.leg_move_count += 1
            self.leg_id = int((action_id - 1) / 4)
            action = action_id-self.leg_id*4
            self.leg_position[self.leg_id][0] = self.reset_legposi[self.leg_id][0]
            self.leg_position[self.leg_id][1] = self.reset_legposi[self.leg_id][1]

            if action == 1:
                self.state[self.leg_id][0] = self.leg_move_dis
                self.state[self.leg_id][1] = 0

            elif action == 2:
                self.state[self.leg_id][0] = -self.leg_move_dis
                self.state[self.leg_id][1] = 0

            elif action == 3:
                self.state[self.leg_id][0] = 0
                self.state[self.leg_id][1] = self.leg_move_dis

            elif action == 4:
                self.state[self.leg_id][0] = 0
                self.state[self.leg_id][1] = -self.leg_move_dis

            else:
                self.state[self.leg_id][0] = 0
                self.state[self.leg_id][1] = 0

            self.action_mem[self.leg_id] = action
            self.leg_position[self.leg_id][0] = self.leg_position[self.leg_id][0]+self.state[self.leg_id][0]
            self.leg_position[self.leg_id][1] = self.leg_position[self.leg_id][1]+self.state[self.leg_id][1]

        if 0 not in self.action_mem:
            action_box = np.unique(self.action_mem)
            if action_box.shape[0]==1:
                body_action = action_box[0]

                if body_action == 1:
                    self.position[0] = self.position[0]+self.body_move_dis
                    self.action_count[0] += 1
                    if self.args.use_fake_reward_bounty:
                        if action_list[1] == 0:
                            reward = 1.0

                elif body_action == 2:
                    self.position[0] = self.position[0]-self.body_move_dis
                    self.action_count[1] += 1
                    if self.args.use_fake_reward_bounty:
                        if action_list[1] == 1:
                            reward = 1.0

                elif body_action == 3:
                    self.position[1] = self.position[1]+self.body_move_dis
                    self.action_count[2] += 1
                    if self.args.use_fake_reward_bounty:
                        if action_list[1] == 2:
                            reward = 1.0

                elif body_action == 4:
                    self.position[1] = self.position[1]-self.body_move_dis
                    self.action_count[3] += 1
                    if self.args.use_fake_reward_bounty:
                        if action_list[1] == 3:
                            reward = 1.0

                if self.args.reward_level == 0:

                    if body_action in [self.single_goal]:
                        reward = 1
                        done = True
                    else:
                        reward = 0
                        done = False


                self.position = self.position_constrain(self.position,[self.max_x,self.max_y],[self.min_x,self.min_y])
                self.action_mem = np.zeros(self.leg_num)

                self.reset_leg_position()

        if self.args.reset_leg:
            if self.leg_move_count%4 == 0:
                self.action_mem = np.zeros(self.leg_num)
                self.reset_leg_position()
        # if action_id==17:
        distance_1 = math.sqrt(abs(self.position[0] + self.body_size/2 - self.min_x) ** 2 + abs(self.position[1] + self.body_size/2 - self.min_y) ** 2)
        distance_2 = math.sqrt(abs(self.position[0] + self.body_size/2 - self.max_x) ** 2 + abs(self.position[1] + self.body_size/2 - self.min_y) ** 2)
        distance_3 = math.sqrt(abs(self.position[0] + self.body_size/2 - self.min_x) ** 2 + abs(self.position[1] + self.body_size/2 - self.max_y) ** 2)
        distance_4 = math.sqrt(abs(self.position[0] + self.body_size/2 - self.max_x) ** 2 + abs(self.position[1] + self.body_size/2 - self.max_y) ** 2)


        if distance_1 <= self.screen_width/20+self.leg_size+self.body_size/2:
            reset_body = True
            if 1 not in self.color_area:
                self.color_area += [1]
            if 1 not in self.cur_goal:
                self.cur_goal[self.goal_id] = 1
                self.goal_id += 1
                if self.args.use_fake_reward_bounty:
                    if len(action_list)>2:
                        if action_list[2] == 0:
                            reward = 1
                if self.args.reward_level == 1:
                    if self.single_goal == 0 or self.args.setup_goal in ['any']:
                        reward = 1
                    done = True
        elif distance_2 <= self.screen_width/20+self.leg_size+self.body_size/2:
            reset_body = True
            if 2 not in self.color_area:
                self.color_area += [2]
            if 2 not in self.cur_goal:
                self.cur_goal[self.goal_id] = 2
                self.goal_id += 1
                if self.args.use_fake_reward_bounty:
                    if len(action_list)>2:
                        if action_list[2] == 1:
                            reward = 1
                if self.args.reward_level == 1:
                    if self.single_goal == 1 or self.args.setup_goal in ['any']:
                        reward = 1
                    done = True
        elif distance_3 <= self.screen_width/20+self.leg_size+self.body_size/2:
            reset_body = True
            if 3 not in self.color_area:
                self.color_area += [3]
            if 3 not in self.cur_goal:
                self.cur_goal[self.goal_id] = 3
                self.goal_id += 1
                if self.args.use_fake_reward_bounty:
                    if len(action_list)>2:
                        if action_list[2] == 2:
                            reward = 1
                if self.args.reward_level == 1:
                    if self.single_goal == 2 or self.args.setup_goal in ['any']:
                        reward = 1
                    done = True
        elif distance_4 <= self.screen_width/20+self.leg_size+self.body_size/2:
            reset_body = True
            if 4 not in self.color_area:
                self.color_area += [4]
            if 4 not in self.cur_goal:
                self.cur_goal[self.goal_id] = 4
                self.goal_id += 1
                if self.args.use_fake_reward_bounty:
                    if len(action_list)>2:
                        if action_list[2] == 3:
                            reward = 1
                if self.args.reward_level == 1:
                    if self.single_goal == 3 or self.args.setup_goal in ['any']:
                        reward = 1
                    done = True

        if self.args.reward_level == 2:
            if (self.realgoal==self.cur_goal).all():
                reward = 1
                done = True
            elif self.cur_goal[self.goal_num-1]>0:
                if self.args.setup_goal in ['any']:
                    reward = 1
                else:
                    reward = 0
                done = True

            self.show_next_goal(self.goal_id)



        # if reset_body:
        #     self.reset_after_goal()
        obs = self.obs()

        if self.episode_length_limit > 0:
            if self.eposide_length >= self.episode_length_limit:
                # reward = 0.0
                done = True
        self.info['action_count'] = self.action_count
        self.info['leg_count'] = self.leg_count

        return obs, reward, done, self.info

    def obs(self):
        if self.args.obs_type == 'ram':
            return self.get_ram()
        elif self.args.obs_type == 'image':
            img = self.render()
            if not self.args.run_overcooked:
                img = self.processes_obs(img)
            return img

    def get_ram(self):
        if self.args.reward_level == 1:
            obs_position = np.concatenate([self.position,
                                      self.leg_position[0],
                                      self.leg_position[1],
                                      self.leg_position[2],
                                      self.leg_position[3],
                                      self.goal_position[0],
                                      self.goal_position[1],
                                      self.goal_position[2],
                                      self.goal_position[3],
                                      ])
            obs_position = (obs_position-self.min_x)/(self.max_x-self.min_x)
            obs_label = np.concatenate([self.goal_label/4,
                                        self.cur_goal/4
                                      ])
            obs_vec = np.concatenate([obs_position,obs_label])
            return obs_vec
        elif self.args.reward_level == 2:
            obs_position = np.concatenate([self.position,
                                      self.leg_position[0],
                                      self.leg_position[1],
                                      self.leg_position[2],
                                      self.leg_position[3],
                                      self.goal_position[0],
                                      self.goal_position[1],
                                      self.goal_position[2],
                                      self.goal_position[3],
                                      ])
            obs_position = (obs_position-self.min_x)/(self.max_x-self.min_x)
            obs_label = np.concatenate([self.realgoal/4,
                                        self.cur_goal/4
                                      ])
            obs_vec = np.concatenate([obs_position,obs_label])

            return obs_vec

        elif self.args.reward_level == 0:
            obs_position = np.concatenate([self.position,
                                      self.leg_position[0],
                                      self.leg_position[1],
                                      self.leg_position[2],
                                      self.leg_position[3],
                                      self.goal_position[0],
                                      self.goal_position[1],
                                      self.goal_position[2],
                                      self.goal_position[3],
                                      ])
            obs_position = (obs_position-self.min_x)/(self.max_x-self.min_x)
            obs_vec = np.concatenate([obs_position,np.zeros(8)])
            return obs_vec

    def reset(self):
        self.leg_id = 0
        self.goal_id = 0
        self.eposide_length = 0
        self.action_mem = np.zeros(self.leg_num)
        self.realgoal = np.arange(1,self.goal_num+1)
        self.cur_goal = np.zeros(self.goal_num)
        self.goal_ram = np.zeros(self.goal_num)
        self.leg_move_count = 0
        self.color_area = []
        # self.action_count = np.zeros(4)
        self.leg_count = np.zeros(self.leg_num*4+1)

        if self.args.new_overcooked:
             self.img[int(self.screen_width):int(self.screen_width + self.screen_width / 8),0:int(self.screen_height),:] = self.stove

        if self.args.reward_level == 1:
            if self.args.setup_goal in ['random']:
                self.single_goal = np.random.randint(0,self.goal_num)
            else:
                self.single_goal = 0
            self.goal_label = np.zeros(4)
            self.goal_label[0] = self.single_goal+1
        elif self.args.reward_level == 0:
            if self.args.setup_goal in ['random']:
                raise Exception('Not goal representation is presented in level 0')
            self.single_goal = 1

        self.position = [self.screen_width/2-self.body_size/2, self.screen_height/2-self.body_size/2]
        self.state = np.zeros((self.leg_num,2))
        self.reset_leg_position()

        self.canvas_clear()

        if self.args.setup_goal in ['random']:
            np.random.shuffle(self.realgoal)
            self.setgoal()
        elif self.args.setup_goal in ['fix']:
            self.setgoal()

        self.show_next_goal(self.goal_id)

        obs = self.obs()

        return obs

    def reset_after_goal(self):
        self.action_mem = np.zeros(self.leg_num)
        self.leg_move_count = 0

        self.position = [self.screen_width/2-self.body_size/2, self.screen_height/2-self.body_size/2]
        self.state = np.zeros((self.leg_num,2))
        self.reset_leg_position()

    def processes_obs(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        obs = np.expand_dims(cv2.resize(obs, (84, 84)), 2)
        return obs

    def get_keys_to_action(self):
        button = input()
        keys_to_action = 0
        '''
        leg control: up:w, down:s, left:a, right:d
        body control: up:v, down:c, left:x, right:z
        get:b
        Donot control leg and body at the same time
        '''
        key = {
            'd': 1,
            'a': 2,
            's': 3,
            'w': 4,
            'h': 5,
            'f': 6,
            'g': 7,
            't': 8,
            'l': 9,
            'j': 10,
            'k': 11,
            'i': 12,
            '6': 13,
            '4': 14,
            '2': 15,
            '8': 16,
            'b': 17,
            'z': 20,
            'x': 21,
            'c': 22,
            'v': 23,
        }

        keys_to_action = key[button[0]]

        return keys_to_action

    def position_constrain(self,cur_position,position_max,position_min):
        if cur_position[0]+self.body_size/2+self.leg_size>=position_max[0]:
            cur_position[0] = cur_position[0]-self.body_move_dis
        if cur_position[1]+self.body_size/2+self.leg_size>=position_max[1]:
            cur_position[1] = cur_position[1]-self.body_move_dis
        if cur_position[0]<=position_min[0]:
            cur_position[0] = cur_position[0]+self.body_move_dis
        if cur_position[1]<=position_min[1]:
            cur_position[1] = cur_position[1]+self.body_move_dis
        return cur_position

    def render(self):
        canvas = self.img.copy()
        if self.args.add_goal_color:
            if len(self.color_area) > 0:
                if 1 in self.color_area:
                    cv2.rectangle(canvas, (int(self.min_x), int(self.min_y)), (int((self.min_x+self.max_x)/2), int((self.min_y+self.max_y)/2)), (170,255,127), -1)
                if 2 in self.color_area:
                    cv2.rectangle(canvas, (int(self.max_x), int(self.min_y)), (int((self.min_x+self.max_x)/2), int((self.min_y+self.max_y)/2)), (170,255,127), -1)
                if 3 in self.color_area:
                    cv2.rectangle(canvas, (int(self.min_x), int(self.max_y)), (int((self.min_x+self.max_x)/2), int((self.min_y+self.max_y)/2)), (170,255,127), -1)
                if 4 in self.color_area:
                    cv2.rectangle(canvas, (int(self.max_x), int(self.max_y)), (int((self.min_x+self.max_x)/2), int((self.min_y+self.max_y)/2)), (170,255,127), -1)
        if self.args.new_overcooked:
            self.canvas_clear()
            self.overlay_image_alpha(canvas,self.body,[int(self.position[0]),int(self.position[1])],self.body[:,:,3]/255.0)
            # legs
            self.overlay_image_alpha(canvas,self.leg,[int(self.leg_position[0][0]),int(self.leg_position[0][1])],self.leg[:,:,3]/255.0)
            self.overlay_image_alpha(canvas,self.leg,[int(self.leg_position[1][0]),int(self.leg_position[1][1])],self.leg[:,:,3]/255.0)
            self.overlay_image_alpha(canvas,self.leg,[int(self.leg_position[2][0]),int(self.leg_position[2][1])],self.leg[:,:,3]/255.0)
            self.overlay_image_alpha(canvas,self.leg,[int(self.leg_position[3][0]),int(self.leg_position[3][1])],self.leg[:,:,3]/255.0)
        else:
            cv2.rectangle(canvas, (int(self.position[0]), int(self.position[1])), (int(self.position[0]+self.body_size), int(self.position[1]+self.body_size)), (92,92,205), self.body_thickness)
            # legs
            cv2.rectangle(canvas, (int(self.leg_position[0][0]), int(self.leg_position[0][1])),(int(self.leg_position[0][0] + self.leg_size), int(self.leg_position[0][1] + self.leg_size)),(0, 92, 205), -1)
            cv2.rectangle(canvas, (int(self.leg_position[1][0]), int(self.leg_position[1][1])), (int(self.leg_position[1][0] + self.leg_size), int(self.leg_position[1][1] + self.leg_size)),(0, 92, 205), -1)
            cv2.rectangle(canvas, (int(self.leg_position[2][0]), int(self.leg_position[2][1])), (int(self.leg_position[2][0] + self.leg_size), int(self.leg_position[2][1] + self.leg_size)),(0, 92, 205), -1)
            cv2.rectangle(canvas, (int(self.leg_position[3][0]), int(self.leg_position[3][1])), (int(self.leg_position[3][0] + self.leg_size), int(self.leg_position[3][1] + self.leg_size)),(0, 92, 205), -1)

        # self.color_area = 0
        if np.sum(self.cur_goal)>0:
            for i in range(self.goal_num):
                if self.cur_goal[i]>0:
                    if self.args.new_overcooked:
                        position = np.array([self.screen_width*0.55+self.screen_width/10*i, self.screen_height*1.01])
                    else:
                        position = np.array([self.screen_width/10*i, self.screen_height+self.screen_height/10])
                    self.draw_goals(self.cur_goal[i], position, canvas)
        # cv2.imwrite('C:\\Users\\IceClear\\Desktop' + '\\' + 'frame' + '.jpg', canvas)  # 存储为图像
        if self.args.render:
            cv2.imshow('overcooked',canvas)
            cv2.waitKey(2)

        return canvas

    def overlay_image_alpha(self,img, img_overlay, pos, alpha_mask):
        """Overlay img_overlay on top of img at the position specified by
        pos and blend using alpha_mask.

        Alpha mask must contain values within the range [0, 1] and be the
        same size as img_overlay.
        """

        x, y = pos

        # Image ranges
        y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
        x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

        # Overlay ranges
        y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
        x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

        # Exit if nothing to do
        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return

        channels = img.shape[2]

        alpha = alpha_mask[y1o:y2o, x1o:x2o]
        alpha_inv = 1.0 - alpha

        for c in range(channels):
            img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                    alpha_inv * img[y1:y2, x1:x2, c])

if __name__ == '__main__':
    from visdom import Visdom
    from arguments import get_args
    from scipy import ndimage
    viz = Visdom()
    win = None
    win_dic = {}
    win_dic['Obs'] = None
    args = get_args()

    env = OverCooked(args)
    difference_mass_center = 0
    for i_episode in range(20):
        observation = env.reset()
        last_mass_ceter = np.asarray(
            ndimage.measurements.center_of_mass(
                observation.astype(np.uint8)
            )
        )
        mass_center = last_mass_ceter
        win_dic['Obs'] = viz.images(
            observation.transpose(2,0,1),
            win=win_dic['Obs'],
            opts=dict(title=' ')
        )
        win_dic['Obs'] = viz.images(
            observation.transpose(2,0,1),
            win=win_dic['Obs'],
            opts=dict(title=' ')
        )
        for t in range(100):
            # env.render(True)
            key = env.get_keys_to_action()
            if key<20:
                action = key
                observation, reward, done, info = env.step(action)
            else:
                if key==20:
                    observation, reward, done, info = env.step(1)
                    observation, reward, done, info = env.step(5)
                    observation, reward, done, info = env.step(9)
                    observation, reward, done, info = env.step(13)
                elif key==21:
                    observation, reward, done, info = env.step(2)
                    observation, reward, done, info = env.step(6)
                    observation, reward, done, info = env.step(10)
                    observation, reward, done, info = env.step(14)
                elif key == 22:
                    observation, reward, done, info = env.step(3)
                    observation, reward, done, info = env.step(7)
                    observation, reward, done, info = env.step(11)
                    observation, reward, done, info = env.step(15)
                elif key == 23:
                    observation, reward, done, info = env.step(4)
                    observation, reward, done, info = env.step(8)
                    observation, reward, done, info = env.step(12)
                    observation, reward, done, info = env.step(16)

            gray_img = observation

            try:
                last_mass_ceter = mass_center
            except Exception as e:
                print('no last')

            mass_center = np.asarray(
                ndimage.measurements.center_of_mass(
                    gray_img.astype(np.uint8)
                )
            )

            try:
                difference_mass_center = np.linalg.norm(last_mass_ceter-mass_center)
            except Exception as e:
                last_mass_ceter =  mass_center

            win_dic['Obs'] = viz.images(
                gray_img.transpose(2,0,1),
                win=win_dic['Obs'],
                opts=dict(title=' ')
            )
            # cv2.imshow('gray_img_rezised', gray_img_rezised)
            cv2.waitKey(2)
            print(reward)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
