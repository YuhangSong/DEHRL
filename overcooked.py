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
        self.screen_width = 84
        self.screen_height = 84
        self.leg_num = 4
        self.goal_num = 4
        self.eposide_length = 0
        self.action_count = np.zeros(4)
        self.leg_count = np.zeros(self.leg_num*4+1)
        self.info = {}
        self.color_area = 0

        '''move distance: screen_width/move_discount, default:10---3 step'''
        self.move_discount = 10/3
        '''body thickness, default -- 2, -1 means solid'''
        self.body_thickness = -1
        '''leg size, default -- self.screen_width/20'''
        self.leg_size = self.screen_width/40


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
            self.episode_length_limit = 24*4

        self.realgoal = np.zeros(self.goal_num)
        self.cur_goal = np.zeros(self.goal_num)
        self.seed()
        self.viewer = None
        self.leg_id = 0
        self.goal_id = 0
        self.action_mem = np.zeros(self.leg_num)
        self.canvas_clear()
        # Just need to initialize the relevant attributes
        self.configure()

        self.max_y = self.screen_height-self.screen_height/10
        self.min_y = self.screen_height/10
        self.max_x = self.screen_width-self.screen_width/10
        self.min_x = self.screen_width/10

        self.goal_position = []
        self.goal_position.append(np.array([self.min_x, self.min_y]))
        self.goal_position.append(np.array([self.max_x, self.min_y]))
        self.goal_position.append(np.array([self.min_x, self.max_y]))
        self.goal_position.append(np.array([self.max_x, self.max_y]))

        self.goal_ram = np.zeros(self.goal_num)

    def canvas_clear(self):
        # canvas
        self.img = np.ones((int(self.screen_width + self.screen_width / 5), int(self.screen_height), 3), np.uint8) * 255
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
        # goals
        cv2.circle(self.img, (int(self.screen_width / 20), int(self.screen_height / 20)), int(self.screen_height / 20),
                   (int(self.goal_color[0][0]), int(self.goal_color[0][1]), int(self.goal_color[0][2])), -1)
        cv2.circle(self.img, (int(self.screen_width - self.screen_width / 20), int(self.screen_height / 20)),
                   int(self.screen_height / 20 - 1.5),
                   (int(self.goal_color[1][0]), int(self.goal_color[1][1]), int(self.goal_color[1][2])), 3)
        cv2.rectangle(self.img, (
        int(self.screen_width - self.screen_width / 10 + 2), int(self.screen_height - self.screen_height / 10 + 2)),
                      (int(self.screen_width - 2), int(self.screen_height - 2)),
                      (int(self.goal_color[2][0]), int(self.goal_color[2][1]), int(self.goal_color[2][2])), 4)
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

    def setgoal(self,goal_arr):
        self.realgoal = np.array(goal_arr)
        if self.args.reward_level == 1:
            position = np.array([0,self.screen_height])
            self.draw_goals(self.single_goal+1,position,self.img)
        elif self.args.reward_level == 2:
            for i in range(4):
                position = np.array([i*self.screen_width/10,self.screen_height])
                self.draw_goals(self.realgoal[i],position,self.img)

    def draw_goals(self,goal_num,position,canvas):
        if goal_num == 1:
            cv2.circle(canvas, (int(position[0]+self.screen_height/20), int(position[1]+self.screen_height/20)),int(self.screen_height / 20),(int(self.goal_color[0][0]), int(self.goal_color[0][1]), int(self.goal_color[0][2])), -1)
        elif goal_num == 2:
            cv2.circle(canvas, (int(position[0]+self.screen_height/20), int(position[1]+self.screen_height/20)),int(self.screen_height / 20 - 1.5),(int(self.goal_color[1][0]), int(self.goal_color[1][1]), int(self.goal_color[1][2])), 3)
        elif goal_num == 3:
            triangle_line = []
            triangle_line.append(np.array([int(position[0]), int(position[1]+self.screen_height/10)]))
            triangle_line.append(np.array([int(position[0]+self.screen_width/20), int(position[1]+self.screen_height/10-self.screen_height/10)]))
            triangle_line.append(np.array([int(position[0]+self.screen_width/10), int(position[1]+self.screen_height/10)]))
            triangle_line = np.array(triangle_line)
            triangle_line = triangle_line.reshape((-1, 1, 2))
            cv2.polylines(canvas, [triangle_line], True,(int(self.goal_color[3][0]), int(self.goal_color[3][1]), int(self.goal_color[3][2])), 1)
        elif goal_num == 4:
            cv2.rectangle(canvas, (int(position[0] + 2), int(position[1] + 2)),
                          (int(position[0]+self.screen_width/10 - 2), int(position[1]+self.screen_height/10 - 2)),
                          (int(self.goal_color[2][0]), int(self.goal_color[2][1]), int(self.goal_color[2][2])), 4)

    def configure(self, display=None):
        self.display = display

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action_list):

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
                self.state[self.leg_id][0] = self.screen_width/40
                self.state[self.leg_id][1] = 0

            elif action == 2:
                self.state[self.leg_id][0] = -self.screen_width/40
                self.state[self.leg_id][1] = 0

            elif action == 3:
                self.state[self.leg_id][0] = 0
                self.state[self.leg_id][1] = self.screen_height/40

            elif action == 4:
                self.state[self.leg_id][0] = 0
                self.state[self.leg_id][1] = -self.screen_height/40

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
                    self.position[0] = self.position[0]+self.screen_width/self.move_discount
                    self.action_count[0] += 1
                    if self.args.use_fake_reward_bounty:
                        if action_list[1] == 0:
                            reward = 1.0

                elif body_action == 2:
                    self.position[0] = self.position[0]-self.screen_width/self.move_discount
                    self.action_count[1] += 1
                    if self.args.use_fake_reward_bounty:
                        if action_list[1] == 1:
                            reward = 1.0

                elif body_action == 3:
                    self.position[1] = self.position[1]+self.screen_height/self.move_discount
                    self.action_count[2] += 1
                    if self.args.use_fake_reward_bounty:
                        if action_list[1] == 2:
                            reward = 1.0

                elif body_action == 4:
                    self.position[1] = self.position[1]-self.screen_height/self.move_discount
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

                self.leg_position = []
                self.leg_position.append(
                    np.array([self.position[0] - self.leg_size, self.position[1] - self.leg_size]))
                self.leg_position.append(
                    np.array([self.position[0] - self.leg_size, self.position[1] + self.screen_height / 10]))
                self.leg_position.append(
                    np.array([self.position[0] + self.screen_width / 10, self.position[1] - self.leg_size]))
                self.leg_position.append(
                    np.array([self.position[0] + self.screen_width / 10, self.position[1] + self.screen_height / 10]))

                self.reset_legposi = []
                self.reset_legposi.append(
                    np.array([self.position[0] - self.leg_size, self.position[1] - self.leg_size]))
                self.reset_legposi.append(
                    np.array([self.position[0] - self.leg_size, self.position[1] + self.screen_height / 10]))
                self.reset_legposi.append(
                    np.array([self.position[0] + self.screen_width / 10, self.position[1] - self.leg_size]))
                self.reset_legposi.append(
                    np.array([self.position[0] + self.screen_width / 10, self.position[1] + self.screen_height / 10]))

        if self.args.reset_leg:
            if self.leg_move_count%4 == 0:
                self.action_mem = np.zeros(self.leg_num)
                self.leg_position = []
                self.leg_position.append(
                    np.array([self.position[0] - self.leg_size, self.position[1] - self.leg_size]))
                self.leg_position.append(
                    np.array([self.position[0] - self.leg_size, self.position[1] + self.screen_height / 10]))
                self.leg_position.append(
                    np.array([self.position[0] + self.screen_width / 10, self.position[1] - self.leg_size]))
                self.leg_position.append(
                    np.array([self.position[0] + self.screen_width / 10, self.position[1] + self.screen_height / 10]))

                self.reset_legposi = []
                self.reset_legposi.append(
                    np.array([self.position[0] - self.leg_size, self.position[1] - self.leg_size]))
                self.reset_legposi.append(
                    np.array([self.position[0] - self.leg_size, self.position[1] + self.screen_height / 10]))
                self.reset_legposi.append(
                    np.array([self.position[0] + self.screen_width / 10, self.position[1] - self.leg_size]))
                self.reset_legposi.append(
                    np.array([self.position[0] + self.screen_width / 10, self.position[1] + self.screen_height / 10]))
        # if action_id==17:
        distance_1 = math.sqrt(abs(self.position[0] + self.screen_width/20 - self.min_x) ** 2 + abs(self.position[1] + self.screen_height/20 - self.min_y) ** 2)
        distance_2 = math.sqrt(abs(self.position[0] + self.screen_width/20 - self.max_x) ** 2 + abs(self.position[1] + self.screen_height/20 - self.min_y) ** 2)
        distance_3 = math.sqrt(abs(self.position[0] + self.screen_width/20 - self.min_x) ** 2 + abs(self.position[1] + self.screen_height/20 - self.max_y) ** 2)
        distance_4 = math.sqrt(abs(self.position[0] + self.screen_width / 20 - self.max_x) ** 2 + abs(self.position[1] + self.screen_height / 20 - self.max_y) ** 2)


        if distance_1 <= self.screen_width/20+self.screen_height/20+self.screen_height/20:
            self.color_area = 1
            if 1 not in self.cur_goal:
                self.cur_goal[self.goal_id] = 1
                self.goal_id += 1
                if self.args.use_fake_reward_bounty:
                    if len(action_list)>2:
                        if action_list[2] == 0:
                            reward = 10
                if self.args.reward_level == 1:
                    if self.single_goal == 0:
                        reward = 10
                    done = True
        elif distance_2 <= self.screen_width/20+self.screen_height/20+self.screen_height/20:
            self.color_area = 2
            if 2 not in self.cur_goal:
                self.cur_goal[self.goal_id] = 2
                self.goal_id += 1
                if self.args.use_fake_reward_bounty:
                    if len(action_list)>2:
                        if action_list[2] == 1:
                            reward = 10
                if self.args.reward_level == 1:
                    if self.single_goal == 1:
                        reward = 10
                    done = True
        elif distance_3 <= self.screen_width/20+self.screen_height/20+self.screen_height/20:
            self.color_area = 3
            if 3 not in self.cur_goal:
                self.cur_goal[self.goal_id] = 3
                self.goal_id += 1
                if self.args.use_fake_reward_bounty:
                    if len(action_list)>2:
                        if action_list[2] == 2:
                            reward = 10
                if self.args.reward_level == 1:
                    if self.single_goal == 2:
                        reward = 10
                    done = True
        elif distance_4 <= self.screen_width/20+self.screen_height/20+self.screen_height/20:
            self.color_area = 4
            if 4 not in self.cur_goal:
                self.cur_goal[self.goal_id] = 4
                self.goal_id += 1
                if self.args.use_fake_reward_bounty:
                    if len(action_list)>2:
                        if action_list[2] == 3:
                            reward = 10
                if self.args.reward_level == 1:
                    if self.single_goal == 3:
                        reward = 10
                    done = True

        if self.args.reward_level == 2:
            if (self.realgoal==self.cur_goal).all():
                reward = 100
                done = True
            elif self.cur_goal[self.goal_num-1]>0:
                reward = 0
                done = True

        obs = self.obs()

        if self.episode_length_limit > 0:
            if self.eposide_length >= self.episode_length_limit:
                reward = 0.0
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
        self.realgoal = np.zeros(self.goal_num)
        self.cur_goal = np.zeros(self.goal_num)
        self.goal_ram = np.zeros(self.goal_num)
        self.leg_move_count = 0
        self.color_area = 0
        # self.action_count = np.zeros(4)
        self.leg_count = np.zeros(self.leg_num*4+1)

        if self.args.reward_level == 1:
            self.single_goal = np.random.randint(0,self.goal_num)
            self.goal_label = np.zeros(4)
            self.goal_label[0] = self.single_goal+1
        elif self.args.reward_level == 0:
            self.single_goal = 1

        self.position = [self.screen_width/2-self.screen_width/20, self.screen_height/2-self.screen_height/20]
        self.state = np.zeros((self.leg_num,2))
        self.leg_position = []
        self.leg_position.append(np.array([self.position[0]-self.leg_size, self.position[1]-self.leg_size]))
        self.leg_position.append(np.array([self.position[0]-self.leg_size, self.position[1]+self.screen_height/10]))
        self.leg_position.append(np.array([self.position[0]+self.screen_width/10, self.position[1]-self.leg_size]))
        self.leg_position.append(np.array([self.position[0]+self.screen_width/10, self.position[1]+self.screen_height/10]))

        self.reset_legposi = []
        self.reset_legposi.append(
            np.array([self.position[0] - self.leg_size, self.position[1] - self.leg_size]))
        self.reset_legposi.append(
            np.array([self.position[0] - self.leg_size, self.position[1] + self.screen_height / 10]))
        self.reset_legposi.append(
            np.array([self.position[0] + self.screen_width / 10, self.position[1] - self.leg_size]))
        self.reset_legposi.append(
            np.array([self.position[0] + self.screen_width / 10, self.position[1] + self.screen_height / 10]))

        self.canvas_clear()

        goal_list = [1, 2, 4, 3]
        random.shuffle(goal_list)
        self.setgoal(goal_list)

        obs = self.obs()
        # obs = self.processes_obs(obs)

        return obs

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
        if cur_position[0]+self.screen_width/10>=position_max[0]:
            cur_position[0] = position_max[0]-self.screen_width/20-self.screen_width/10
        if cur_position[1]+self.screen_height/10>=position_max[1]:
            cur_position[1] = position_max[1]-self.screen_height/20-self.screen_height/10
        if cur_position[0]<=position_min[0]:
            cur_position[0] = position_min[0]+self.screen_width/20
        if cur_position[1]<=position_min[1]:
            cur_position[1] = position_min[1]+self.screen_height/20
        return cur_position

    def render(self):
        canvas = self.img.copy()
        if self.args.add_goal_color:
            if self.color_area > 0:
                if self.color_area == 1:
                    cv2.rectangle(canvas, (int(self.min_x), int(self.min_y)), (int((self.min_x+self.max_x)/2), int((self.min_y+self.max_y)/2)), (170,255,127), -1)
                elif self.color_area == 2:
                    cv2.rectangle(canvas, (int(self.max_x), int(self.min_y)), (int((self.min_x+self.max_x)/2), int((self.min_y+self.max_y)/2)), (170,255,127), -1)
                elif self.color_area == 3:
                    cv2.rectangle(canvas, (int(self.min_x), int(self.max_y)), (int((self.min_x+self.max_x)/2), int((self.min_y+self.max_y)/2)), (170,255,127), -1)
                elif self.color_area == 4:
                    cv2.rectangle(canvas, (int(self.max_x), int(self.max_y)), (int((self.min_x+self.max_x)/2), int((self.min_y+self.max_y)/2)), (170,255,127), -1)

        cv2.rectangle(canvas, (int(self.position[0]), int(self.position[1])), (int(self.position[0]+self.screen_width/10), int(self.position[1]+self.screen_height/10)), (92,92,205), self.body_thickness)
        # legs
        cv2.rectangle(canvas, (int(self.leg_position[0][0]), int(self.leg_position[0][1])),(int(self.leg_position[0][0] + self.leg_size), int(self.leg_position[0][1] + self.leg_size)),(0, 92, 205), -1)
        cv2.rectangle(canvas, (int(self.leg_position[1][0]), int(self.leg_position[1][1])), (int(self.leg_position[1][0] + self.leg_size), int(self.leg_position[1][1] + self.leg_size)),(0, 92, 205), -1)
        cv2.rectangle(canvas, (int(self.leg_position[2][0]), int(self.leg_position[2][1])), (int(self.leg_position[2][0] + self.leg_size), int(self.leg_position[2][1] + self.leg_size)),(0, 92, 205), -1)
        cv2.rectangle(canvas, (int(self.leg_position[3][0]), int(self.leg_position[3][1])), (int(self.leg_position[3][0] + self.leg_size), int(self.leg_position[3][1] + self.leg_size)),(0, 92, 205), -1)

        self.color_area = 0
        if np.sum(self.cur_goal)>0:
            if self.cur_goal[0]>0:
                position = np.array([0, self.screen_height+self.screen_height/10])
                self.draw_goals(self.cur_goal[0], position, canvas)
            if self.cur_goal[1]>0:
                position = np.array([self.screen_width/10, self.screen_height+self.screen_height/10])
                self.draw_goals(self.cur_goal[1], position, canvas)
            if self.cur_goal[2]>0:
                position = np.array([2*self.screen_width/10, self.screen_height+self.screen_height/10])
                self.draw_goals(self.cur_goal[2], position, canvas)
            if self.cur_goal[3]>0:
                position = np.array([3*self.screen_width/10, self.screen_height+self.screen_height/10])
                self.draw_goals(self.cur_goal[3], position, canvas)
        # cv2.imwrite('C:\\Users\\IceClear\\Desktop' + '\\' + 'frame' + '.jpg', canvas)  # 存储为图像
        if self.args.render:
            cv2.imshow('overcooked',canvas)
            cv2.waitKey(2)

        return canvas

if __name__ == '__main__':
    from visdom import Visdom
    from arguments import get_args
    viz = Visdom()
    win = None
    win_dic = {}
    win_dic['Obs'] = None
    args = get_args()

    env = OverCooked(args)
    for i_episode in range(20):
        observation = env.reset()
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
