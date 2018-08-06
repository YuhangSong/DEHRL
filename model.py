import torch
import torch.nn as nn
import torch.nn.functional as F
from distributions import Categorical, DiagGaussian
from utils import init, init_normc_
import numpy as np


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class DeFlatten(nn.Module):
    def __init__(self, shape):
        super(DeFlatten, self).__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(x.size(0), *self.shape)

class Policy(nn.Module):
    def __init__(self, obs_shape, input_action_space,output_action_space, num_subpolicy, recurrent_policy):
        super(Policy, self).__init__()
        self.num_subpolicy = num_subpolicy

        '''build base model'''
        if len(obs_shape) == 3:
            self.base = CNNBase(obs_shape[0], recurrent_policy)
        elif len(obs_shape) == 1:
            assert not recurrent_policy, \
                "Recurrent policy is not implemented for the MLP controller"
            self.base = MLPBase(obs_shape[0])
        else:
            raise NotImplementedError
        self.state_size = self.base.state_size

        '''build actor model'''
        self.output_action_space = output_action_space
        if self.output_action_space.__class__.__name__ == "Discrete":
            num_outputs = self.output_action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs,self.num_subpolicy)
        elif self.output_action_space.__class__.__name__ == "Box":
            num_outputs = self.output_action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

        '''build critic model'''
        if self.num_subpolicy > 1:
            self.critic_linear = []
            for linear_i in range(self.num_subpolicy):
                self.critic_linear += [self.base.linear_init_(nn.Linear(self.base.linear_size, 1))]
            self.critic_linear = nn.ModuleList(self.critic_linear)
        else:
            self.critic_linear = self.base.linear_init_(nn.Linear(self.base.linear_size, 1))

        self.input_action_space = input_action_space


    def forward(self, inputs, states, input_action, masks):
        raise NotImplementedError

    def get_final_features(self, inputs, states, masks, input_action=None):
        base_features, states = self.base(inputs, states, masks)
        return base_features, states

    def get_value_dist(self, base_features, input_action):

        if self.num_subpolicy > 1:
            index_dic = {}
            tensor_dic = {}
            y_dic = {}
            action_index = np.where(input_action==1)[1]
            for dic_i in range(self.num_subpolicy):
                index_dic[str(dic_i)] = torch.from_numpy(np.where(action_index==dic_i)[0]).long().cuda()
                if index_dic[str(dic_i)].size()[0] != 0:
                    tensor_dic[str(dic_i)] = torch.index_select(base_features,0,index_dic[str(dic_i)])
                    y_dic[str(dic_i)] = self.critic_linear[dic_i](tensor_dic[str(dic_i)])

            value = torch.zeros((input_action.size()[0],1)).cuda()
            for y_i in range(self.num_subpolicy):
                if str(y_i) in y_dic:
                    value.index_add_(0,index_dic[str(y_i)],y_dic[str(y_i)])

            dist, dist_features = self.dist(base_features,action_index)

        else:
            value = self.critic_linear(base_features)
            dist, dist_features = self.dist(base_features)

        return value, dist, dist_features

    def act(self, inputs, states, masks, deterministic=False, input_action=None):
        base_features, states = self.get_final_features(inputs, states, masks, input_action)

        value, dist, dist_features = self.get_value_dist(base_features, input_action)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, states

    def get_value(self, inputs, states, masks, input_action=None):
        base_features, states = self.get_final_features(inputs, states, masks, input_action)
        value, dist, dist_features = self.get_value_dist(base_features, input_action)
        return value

    def evaluate_actions(self, inputs, states, masks, action, input_action=None):
        base_features, states = self.get_final_features(inputs, states, masks, input_action)
        value, dist, dist_features = self.get_value_dist(base_features, input_action)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy()

        return value, action_log_probs, dist_entropy, states, dist_features

    def save_model(self, save_path):
        torch.save(self.state_dict(), save_path)

class CNNBase(nn.Module):
    def __init__(self, num_inputs, use_gru, linear_size=512):
        super(CNNBase, self).__init__()

        self.linear_size = linear_size

        self.relu_init_ = lambda m: init(m,
                      nn.init.orthogonal_,
                      lambda x: nn.init.constant_(x, 0),
                      nn.init.calculate_gain('relu'))

        self.leakrelu_init_ = lambda m: init(m,
                      nn.init.orthogonal_,
                      lambda x: nn.init.constant_(x, 0),
                      nn.init.calculate_gain('leaky_relu'))

        self.main = nn.Sequential(
            self.leakrelu_init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.LeakyReLU(),
            self.leakrelu_init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.LeakyReLU(),
            self.leakrelu_init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.LeakyReLU(),
            Flatten(),
            self.leakrelu_init_(nn.Linear(32 * 7 * 7, self.linear_size)),
            nn.LeakyReLU()
        )

        if use_gru:
            self.gru = nn.GRUCell(self.linear_size, self.linear_size)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

        self.linear_init_ = lambda m: init(m,
          nn.init.orthogonal_,
          lambda x: nn.init.constant_(x, 0))

        self.train()

    @property
    def state_size(self):
        if hasattr(self, 'gru'):
            return self.linear_size
        else:
            return 1

    @property
    def output_size(self):
        return self.linear_size

    def forward(self, inputs, states, masks):
        x = self.main(inputs / 255.0)

        if hasattr(self, 'gru'):
            if inputs.size(0) == states.size(0):
                x = states = self.gru(x, states * masks)
            else:
                x = x.view(-1, states.size(0), x.size(1))
                masks = masks.view(-1, states.size(0), 1)
                outputs = []
                for i in range(x.size(0)):
                    hx = states = self.gru(x[i], states * masks[i])
                    outputs.append(hx)
                x = torch.cat(outputs, 0)

        return x, states

class MLPBase(nn.Module):
    def __init__(self, num_inputs, linear_size=64):
        super(MLPBase, self).__init__()

        self.linear_size = linear_size

        self.linear_init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))

        self.main = nn.Sequential(
            self.linear_init_(nn.Linear(num_inputs, self.linear_size)),
            nn.Tanh(),
            self.linear_init_(nn.Linear(self.linear_size, self.linear_size)),
            nn.Tanh()
        )

        self.train()

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return 64

    def forward(self, inputs, states, masks):
        x = self.main(inputs)

        return x, states

class TransitionModel(nn.Module):
    def __init__(self, input_observation_shape, input_action_space, output_observation_shape, num_subpolicy, mutual_information, linear_size=512):
        super(TransitionModel, self).__init__()
        '''if mutual_information, transition_model is act as a regressor to fit p(Z|c)'''

        self.input_observation_shape = input_observation_shape
        self.output_observation_shape = output_observation_shape
        self.mutual_information = mutual_information

        self.linear_size = linear_size
        self.num_subpolicy = num_subpolicy

        self.linear_init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        self.relu_init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        self.leakrelu_init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('leaky_relu'))

        self.conv = nn.Sequential(
            self.leakrelu_init_(nn.Conv2d(self.input_observation_shape[0], 32, 8, stride=4)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            self.leakrelu_init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            self.leakrelu_init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            Flatten(),
            self.leakrelu_init_(nn.Linear(32 * 7 * 7, self.linear_size)),
            nn.BatchNorm1d(self.linear_size),
            nn.LeakyReLU()
        )

        self.reward_bounty_linear = nn.Sequential(
            self.linear_init_(nn.Linear(self.linear_size, 1)),
        )

        if not self.mutual_information:

            self.deconv = []
            for decon_i in range(self.num_subpolicy):
                self.deconv += [nn.Sequential(
                    self.leakrelu_init_(nn.Linear(self.linear_size, 32 * 7 * 7)),
                    # nn.BatchNorm1d(32 * 7 * 7),
                    nn.LeakyReLU(),
                    DeFlatten((32,7,7)),
                    self.leakrelu_init_(nn.ConvTranspose2d(32, 64, 3, stride=1)),
                    # nn.BatchNorm2d(64),
                    nn.LeakyReLU(),
                    self.leakrelu_init_(nn.ConvTranspose2d(64, 32, 4, stride=2)),
                    # nn.BatchNorm2d(32),
                    nn.LeakyReLU(),
                    self.leakrelu_init_(nn.ConvTranspose2d(32, self.output_observation_shape[0], 8, stride=4)),
                    # output do not normalize
                    nn.Sigmoid(),
                )]
            self.deconv = nn.ModuleList(self.deconv)

        else:

            self.label_linear = nn.Sequential(
                self.linear_init_(nn.Linear(self.linear_size, num_subpolicy)),
            )


    def forward(self, inputs, input_action=None):

        before_deconv = self.conv(inputs/255.0)

        if not self.mutual_information:

            index_dic = {}
            tensor_dic = {}
            y_dic = {}
            action_index = np.where(input_action==1)[1]
            for dic_i in range(self.num_subpolicy):
                index_dic[str(dic_i)] = torch.from_numpy(np.where(action_index==dic_i)[0]).long().cuda()
                if index_dic[str(dic_i)].size()[0] != 0:
                    tensor_dic[str(dic_i)] = torch.index_select(before_deconv,0,index_dic[str(dic_i)])
                    y_dic[str(dic_i)] = self.deconv[dic_i](tensor_dic[str(dic_i)])
            after_deconv = torch.zeros((inputs.size()[0],*self.output_observation_shape)).cuda()
            for y_i in range(self.num_subpolicy):
                if str(y_i) in y_dic:
                    after_deconv.index_add_(0,index_dic[str(y_i)],y_dic[str(y_i)])

            return after_deconv*255.0, before_deconv, self.reward_bounty_linear(before_deconv)

        else:

            return F.log_softmax(self.label_linear(before_deconv), dim=1), self.reward_bounty_linear(before_deconv)

    def save_model(self, save_path):
        torch.save(self.state_dict(), save_path)
