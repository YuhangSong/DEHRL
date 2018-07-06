import torch
import torch.nn as nn
import torch.nn.functional as F
from distributions import Categorical, DiagGaussian
from utils import init, init_normc_
import numpy as np


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class EHRL_Policy(nn.Module):
    def __init__(self, obs_shape, action_space, one_hot, hid_size, recurrent_policy, label):
        super(EHRL_Policy, self).__init__()

        self.hid_size = hid_size
        self.label = label
        # self.num_hid_layers = num_hid_layers
        # self.num_subpolicies = num_subpolicies
        # self.gaussian_fixed_var = gaussian_fixed_var

        if len(obs_shape) == 3:
            self.base = CNNBase(obs_shape[0], one_hot, self.hid_size, recurrent_policy)
        elif len(obs_shape) == 1:
            assert not recurrent_policy, \
                "Recurrent policy is not implemented for the MLP controller"
            self.base = MLPBase(obs_shape[0], one_hot, self.hid_size)
        else:
            raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

        self.state_size = self.base.state_size

    def forward(self, inputs, one_hot, states, masks):
        raise NotImplementedError

    def act(self, inputs, one_hot, states, masks, deterministic=False):
        value, actor_features, states = self.base(inputs, one_hot, states, masks)
        # print(actor_features)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, states

    def get_value(self, inputs, one_hot, states, masks):
        value, _, _ = self.base(inputs, one_hot, states, masks)
        return value

    def evaluate_actions(self, inputs, one_hot, states, masks, action):
        value, actor_features, states = self.base(inputs, one_hot, states, masks)

        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, states


    def save_model(self, save_path):
        torch.save(self.state_dict(), save_path+'/trained_learner_'+self.label+'.pth')

class CNNBase(nn.Module):
    def __init__(self, num_inputs, one_hot, hid_size, use_gru):
        super(CNNBase, self).__init__()

        init_ = lambda m: init(m,
                      nn.init.orthogonal_,
                      lambda x: nn.init.constant_(x, 0),
                      nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, 512)),
            nn.ReLU()
        )

        if use_gru:
            self.gru = nn.GRUCell(512, 512)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

        init_ = lambda m: init(m,
          nn.init.orthogonal_,
          lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(512, 1))
        '''
        action-conditional
        '''

        # self.feature_linear = init_(nn.Linear(512, hid_size))
        self.label_linear = init_(nn.Linear(one_hot.shape[0], 512))
        # self.combine_linear = init_(nn.Linear(hid_size, 512))

        self.train()

    @property
    def state_size(self):
        if hasattr(self, 'gru'):
            return 512
        else:
            return 1

    @property
    def output_size(self):
        return 512

    def forward(self, inputs, one_hot, states, masks):
        x = self.main(inputs / 255.0)
        # print(inputs)
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

        # x_feature = self.feature_linear(x)
        # x_feature = F.tanh(x_feature)
        label = self.label_linear(one_hot)
        label = F.tanh(label)
        x_multiply = x*label
        # x_ac = self.combine_linear(x_multiply)
        # x_ac = F.tanh(x_ac)
        return self.critic_linear(x_multiply), x_multiply, states
        # return self.critic_linear(x), x, states


class MLPBase(nn.Module):
    def __init__(self, num_inputs, one_hot, hid_size):
        super(MLPBase, self).__init__()

        init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, 64)),
            nn.Tanh(),
            init_(nn.Linear(64, 64)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, 64)),
            nn.Tanh(),
            init_(nn.Linear(64, 64)),
            nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(64, 1))

        '''
        action-conditional
        '''
        self.label_linear = init_(nn.Linear(one_hot.shape[0], 64))

        self.train()

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return 64

    def forward(self, inputs, one_hot, states, masks):

        hidden_critic = self.critic(inputs)
        hidden_actor = self.actor(inputs)

        label = self.label_linear(one_hot)
        label = F.tanh(label)

        hidden_critic = hidden_critic*label
        hidden_actor = hidden_actor*label

        return self.critic_linear(hidden_critic), hidden_actor, states
