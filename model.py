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

        self.input_action_linear_critic = nn.Sequential(
            self.base.relu_init_(nn.Linear(self.input_action_space.n, self.base.linear_size)),
            nn.ReLU(),
        )
        self.input_action_linear_dist = nn.Sequential(
            self.base.relu_init_(nn.Linear(self.input_action_space.n, self.base.linear_size)),
            nn.ReLU(),
        )

        self.final_feature_linear_critic = nn.Sequential(
            self.base.relu_init_(nn.Linear(self.base.linear_size, self.base.linear_size)),
            nn.ReLU(),
        )
        self.final_feature_linear_dist = nn.Sequential(
            self.base.relu_init_(nn.Linear(self.base.linear_size, self.base.linear_size)),
            nn.ReLU(),
        )

        self.final_feature_linear_critic = nn.Sequential(
            self.base.relu_init_(nn.Linear(self.base.linear_size, self.base.linear_size)),
            nn.ReLU(),
        )
        self.final_feature_linear_dist = nn.Sequential(
            self.base.relu_init_(nn.Linear(self.base.linear_size, self.base.linear_size)),
            nn.ReLU(),
        )

    def forward(self, inputs, states, input_action, masks):
        raise NotImplementedError

    def get_final_features(self, inputs, states, masks, input_action=None):

        base_features, states = self.base(inputs, states, masks)

        input_action_features_critic = self.input_action_linear_critic(input_action)
        input_action_features_dist = self.input_action_linear_dist(input_action)

        final_features_critic = base_features * input_action_features_critic
        final_features_dist = base_features * input_action_features_dist

        return final_features_critic, final_features_dist, states

    def act(self, inputs, states, masks, deterministic=False, input_action=None):
        final_features_critic, final_features_dist, states = self.get_final_features(inputs, states, masks, input_action)
        value = self.critic_linear(final_features_critic)
        dist = self.dist(final_features_dist)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, states

    def get_value(self, inputs, states, masks, input_action=None):
        final_features_critic, final_features_dist, states = self.get_final_features(inputs, states, masks, input_action)
        value = self.critic_linear(final_features_critic)
        return value

    def evaluate_actions(self, inputs, states, masks, action, input_a