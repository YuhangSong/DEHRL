import torch
import torch.nn as nn
import torch.nn.functional as F
from distributions import Categorical, DiagGaussian
from utils import init, init_normc_


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, input_action_space,output_action_space, recurrent_policy):
        super(Policy, self).__init__()

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
        if output_action_space.__class__.__name__ == "Discrete":
            num_outputs = output_action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif output_action_space.__class__.__name__ == "Box":
            num_outputs = output_action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

        '''build critic model'''
        self.critic_linear = self.base.linear_init_(nn.Linear(self.base.linear_size, 1))

        self.input_action_space = input_action_space

    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def act(self, inputs, states, masks, deterministic=False):
        actor_features, states = self.base(inputs, states, masks)
        value = self.critic_linear(actor_features)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, states

    def get_value(self, inputs, states, masks):
        actor_features, _ = self.base(inputs, states, masks)
        value = self.critic_linear(actor_features)
        return value

    def evaluate_actions(self, inputs, states, masks, action):
        actor_features, states = self.base(inputs, states, masks)
        value = self.critic_linear(actor_features)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, states


    def save_model(self, save_path):
        torch.save(self.state_dict(), save_path+'/trained_learner.pth')

class CNNBase(nn.Module):
    def __init__(self, num_inputs, use_gru, linear_size=512):
        super(CNNBase, self).__init__()

        self.linear_size = linear_size

        self.conv_init_ = lambda m: init(m,
                      nn.init.orthogonal_,
                      lambda x: nn.init.constant_(x, 0),
                      nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            self.conv_init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            self.conv_init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            self.conv_init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            self.conv_init_(nn.Linear(32 * 7 * 7, self.linear_size)),
            nn.ReLU()
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

        self.actor = nn.Sequential(
            self.linear_init_(nn.Linear(num_inputs, self.linear_size)),
            nn.Tanh(),
            self.linear_init_(nn.Linear(self.linear_size, self.linear_size)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
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
        hidden_critic = self.critic(inputs)
        hidden_actor = self.actor(inputs)

        return hidden_actor, states
