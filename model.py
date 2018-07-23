import torch
import torch.nn as nn
import torch.nn.functional as F
from distributions import Categorical, DiagGaussian
from utils import init, init_normc_


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
        self.output_action_space = output_action_space
        if self.output_action_space.__class__.__name__ == "Discrete":
            num_outputs = self.output_action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif self.output_action_space.__class__.__name__ == "Box":
            num_outputs = self.output_action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

        '''build critic model'''
        self.critic_linear = self.base.linear_init_(nn.Linear(self.base.linear_size, 1))

        self.input_action_space = input_action_space
        self.input_action_linear = nn.Sequential(
            self.base.leakrelu_init_(nn.Linear(self.input_action_space.n, self.base.linear_size)),
            nn.LayerNorm(self.base.linear_size),
            nn.LeakyReLU(),
        )

        self.final_feature_linear_critic = nn.Sequential(
            self.base.leakrelu_init_(nn.Linear(self.base.linear_size, self.base.linear_size)),
            nn.LayerNorm(self.base.linear_size),
            nn.LeakyReLU(),
        )
        self.final_feature_linear_dist = nn.Sequential(
            self.base.leakrelu_init_(nn.Linear(self.base.linear_size, self.base.linear_size)),
            nn.LayerNorm(self.base.linear_size),
            nn.LeakyReLU(),
        )

    def forward(self, inputs, states, input_action, masks):
        raise NotImplementedError

    def get_final_features(self, inputs, states, masks, input_action=None):
        # input_action = torch.zeros(inputs.size()[0],self.input_action_space.n).cuda()
        base_features, states = self.base(inputs, states, masks)
        input_action_features = self.input_action_linear(input_action)
        final_features = base_features*input_action_features
        final_features_critic = self.final_feature_linear_critic(final_features)
        final_features_dist = self.final_feature_linear_dist(final_features)
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

    def evaluate_actions(self, inputs, states, masks, action, input_action=None):
        final_features_critic, final_features_dist, states = self.get_final_features(inputs, states, masks, input_action)
        value = self.critic_linear(final_features_critic)
        dist = self.dist(final_features_dist)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, states

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
            nn.LayerNorm(self.linear_size),
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
    def __init__(self, input_observation_shape, input_action_space, output_observation_space, linear_size=512):
        super(TransitionModel, self).__init__()

        self.input_observation_shape = input_observation_shape
        self.output_observation_space = output_observation_space
        self.linear_size = linear_size

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

        self.input_action_space = input_action_space
        self.input_action_linear = nn.Sequential(
            self.leakrelu_init_(nn.Linear(self.input_action_space.n, self.linear_size)),
            nn.BatchNorm1d(self.linear_size),
            nn.LeakyReLU(),
        )

        self.deconv = nn.Sequential(
            self.leakrelu_init_(nn.Linear(self.linear_size, 32 * 7 * 7)),
            nn.BatchNorm1d(32 * 7 * 7),
            nn.LeakyReLU(),
            DeFlatten((32,7,7)),
            self.leakrelu_init_(nn.ConvTranspose2d(32, 64, 3, stride=1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            self.leakrelu_init_(nn.ConvTranspose2d(64, 32, 4, stride=2)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            self.leakrelu_init_(nn.ConvTranspose2d(32, self.output_observation_space.shape[0], 8, stride=4)),
            # output do not normalize
            nn.Sigmoid(),
        )

    def forward(self, inputs, input_action):
        predict_bef_deconv = self.conv(inputs/255.0)*self.input_action_linear(input_action)
        return self.deconv(predict_bef_deconv)*255.0, predict_bef_deconv
