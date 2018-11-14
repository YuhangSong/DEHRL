import torch
import torch.nn as nn
import torch.nn.functional as F
from distributions import Categorical, DiagGaussian
from utils import init, init_normc_
import numpy as np
from arguments import get_args
args = get_args()

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def flatten(x):
    return x.contiguous().view(x.size(0), -1)

class DeFlatten(nn.Module):
    def __init__(self, shape):
        super(DeFlatten, self).__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(x.size(0), *self.shape)

def get_obs_size_flatten(obs_shape):
    if len(obs_shape) in [3]:
        obs_size_flatten = obs_shape[0]*obs_shape[1]*obs_shape[2]
    elif len(obs_shape) in [1]:
        obs_size_flatten = obs_shape[0]
    else:
        raise NotImplemented
    return obs_size_flatten

class Policy(nn.Module):

    def __init__(self, obs_shape, state_type, input_action_space,output_action_space, num_subpolicy, recurrent_policy):
        super(Policy, self).__init__()

        if ('Bullet' in args.env_name) or (args.env_name in ['Explore2DContinuous']):
            self.linear_size = 64
        elif args.env_name in ['OverCooked','MineCraft','Explore2D','MontezumaRevengeNoFrameskip-v4','GridWorld']:
            self.linear_size = 256
        else:
            raise NotImplemented

        self.num_subpolicy = num_subpolicy
        self.base = StateEncoder(
            use_gru = recurrent_policy,
            obs_shape = obs_shape,
            state_type = state_type,
            linear_size = self.linear_size,
        )

        self.state_size = self.base.state_size

        '''build actor model'''
        self.output_action_space = output_action_space
        if self.output_action_space.__class__.__name__ == "Discrete":
            num_outputs = self.output_action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs, self.num_subpolicy)
        elif self.output_action_space.__class__.__name__ == "Box":
            num_outputs = self.output_action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs, self.num_subpolicy)
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

            '''value forward'''
            index_dic = {}
            tensor_dic = {}
            y_dic = {}
            action_index = np.where(input_action==1)[1]
            for dic_i in range(self.num_subpolicy):
                index_dic[str(dic_i)] = torch.from_numpy(np.where(action_index==dic_i)[0]).long().cuda()
                if index_dic[str(dic_i)].size()[0] != 0:
                    tensor_dic[str(dic_i)] = torch.index_select(base_features['critic'],0,index_dic[str(dic_i)])
                    y_dic[str(dic_i)] = self.critic_linear[dic_i](tensor_dic[str(dic_i)])

            value = torch.zeros((input_action.size()[0],1)).cuda()
            for y_i in range(self.num_subpolicy):
                if str(y_i) in y_dic:
                    value.index_add_(0,index_dic[str(y_i)],y_dic[str(y_i)])

            '''dist forward'''
            dist, dist_features = self.dist(base_features['actor'],action_index)

        else:
            '''value forward'''
            value = self.critic_linear(base_features['critic'])
            'dist forward'
            dist, dist_features = self.dist(base_features['actor'])

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

class StateEncoder(nn.Module):

    def __init__(self, use_gru, obs_shape, state_type, linear_size=256):
        super(StateEncoder, self).__init__()

        self.linear_size = linear_size
        self.state_type = state_type

        self.relu_init_ = lambda m: init(m,
                      nn.init.orthogonal_,
                      lambda x: nn.init.constant_(x, 0),
                      nn.init.calculate_gain('relu'))

        self.leakrelu_init_ = lambda m: init(m,
                      nn.init.orthogonal_,
                      lambda x: nn.init.constant_(x, 0),
                      nn.init.calculate_gain('leaky_relu'))

        self.linear_init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))

        if self.state_type in ['standard_image']:
            self.main = nn.Sequential(
                self.leakrelu_init_(nn.Conv2d(obs_shape[0], 16, 8, stride=4)),
                nn.LeakyReLU(),
                self.leakrelu_init_(nn.Conv2d(16, 32, 4, stride=2)),
                nn.LeakyReLU(),
                self.leakrelu_init_(nn.Conv2d(32, 16, 3, stride=1)),
                nn.LeakyReLU(),
                Flatten(),
                self.leakrelu_init_(nn.Linear(16 * 7 * 7, self.linear_size)),
                nn.LeakyReLU()
            )
        elif self.state_type in ['vector']:
            obs_size_flatten = get_obs_size_flatten(obs_shape)
            init_ = lambda m: init(m,
                init_normc_,
                lambda x: nn.init.constant_(x, 0))
            self.main_actor = nn.Sequential(
                init_(nn.Linear(obs_size_flatten, self.linear_size)),
                nn.Tanh(),
                init_(nn.Linear(self.linear_size, self.linear_size)),
                nn.Tanh()
            )
            self.main_critic = nn.Sequential(
                init_(nn.Linear(obs_size_flatten, self.linear_size)),
                nn.Tanh(),
                init_(nn.Linear(self.linear_size, self.linear_size)),
                nn.Tanh()
            )

        if use_gru:
            self.gru = nn.GRUCell(self.linear_size, self.linear_size)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

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
        if self.state_type in ['standard_image']:
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

            return {'actor': x, 'critic': x}, states

        elif self.state_type in ['vector']:
            x_actor = self.main_actor(
                inputs.view(inputs.size()[0],-1)
            )
            x_critic = self.main_critic(
                inputs.view(inputs.size()[0],-1)
            )
            return {'actor': x_actor, 'critic': x_critic}, states

        else:
            raise NotImplemented

class InverseMaskModel(nn.Module):
    def __init__(self, predicted_action_space, num_grid):
        super(InverseMaskModel, self).__init__()

        self.predicted_action_space = predicted_action_space
        self.num_grid = num_grid
        self.size_grid = int(84/self.num_grid)

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

        self.tanh_init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('tanh'))

        self.mlp_e = nn.Sequential(
            self.relu_init_(nn.Linear(int((self.size_grid**2)*2), 256)),
            nn.ReLU(inplace=True),
            self.relu_init_(nn.Linear(256, 128)),
            nn.ReLU(inplace=True),
            self.linear_init_(nn.Linear(128, self.predicted_action_space)),
        )

        self.mlp_alpha = nn.Sequential(
            self.relu_init_(nn.Linear(int(self.size_grid**2), 64)),
            nn.ReLU(inplace=True),
            self.relu_init_(nn.Linear(64, 64)),
            nn.ReLU(inplace=True),
            self.linear_init_(nn.Linear(64, 1)),
        )

    def get_alpha(self, states):
        alpha_bar = []
        for i in range(self.num_grid):
            for j in range(self.num_grid):
                alpha_bar += [self.mlp_alpha(
                    flatten(
                        self.slice_grid(
                            states=states,
                            i=i,
                            j=j,
                        )
                    )
                )]
        alpha = F.softmax(torch.cat(alpha_bar, 1), dim=1)
        return alpha

    def slice_grid(self, states, i, j):
        return states [:,:,i*self.size_grid:(i+1)*self.size_grid,j*self.size_grid:(j+1)*self.size_grid]

    def get_e(self, conved_last_states, conved_now_states):
        e = []
        for i in range(self.num_grid):
            for j in range(self.num_grid):
                e += [
                    torch.unsqueeze(
                        self.mlp_e(
                            torch.cat(
                                [
                                    flatten(
                                        self.slice_grid(
                                            states=conved_now_states,
                                            i=i,
                                            j=j,
                                        ) - self.slice_grid(
                                            states=conved_last_states,
                                            i=i,
                                            j=j,
                                        )
                                    ),
                                    flatten(
                                        self.slice_grid(
                                            states=conved_now_states,
                                            i=i,
                                            j=j,
                                        )
                                    )
                                ],
                                dim = 1,
                            )
                        ),
                        dim = 1,
                    )
                ]
        e = torch.cat(e, dim=1)
        return e

    def get_predicted_action_log_probs(self, e, alpha):
        return F.log_softmax(
            (e*alpha.unsqueeze(2).expand(-1,-1,self.predicted_action_space)).sum(
                dim = 1,
                keepdim = False,
            ),
            dim=1,
        )

    def forward(self, last_states, now_states):

        conved_last_states = (last_states/255.0)
        conved_now_states  = (now_states /255.0)

        alpha = self.get_alpha(
            states = conved_now_states,
        )

        e = self.get_e(
            conved_last_states = conved_last_states,
            conved_now_states  = conved_now_states ,
        )

        predicted_action_log_probs = self.get_predicted_action_log_probs(
            e = e,
            alpha = alpha,
        )

        predicted_action_log_probs_each = F.log_softmax(e,dim=2)
        # predicted_action_log_probs_each = None

        loss_ent = (alpha*alpha.log()).sum(dim=1,keepdim=False).mean(dim=0,keepdim=False)

        return predicted_action_log_probs, loss_ent, predicted_action_log_probs_each

    def alpha_to_mask(self, alpha):
        alpha = alpha.unsqueeze(2).expand(-1,-1,self.size_grid)
        alpha = alpha.contiguous().view(alpha.size()[0], self.num_grid, -1)
        alpha = torch.cat([alpha]*self.size_grid,dim=2).view(alpha.size()[0],self.size_grid*self.num_grid,self.size_grid*self.num_grid)
        '''alpha is kept to be softmax'''
        return alpha

    def get_mask(self, last_states):
        conved_last_states  = (last_states /255.0)

        alpha = self.get_alpha(
            states = conved_last_states,
        )
        mask = self.alpha_to_mask(
            alpha = alpha,
        ).unsqueeze(1)*255.0
        return mask

    def save_model(self, save_path):
        torch.save(self.state_dict(), save_path)

class TransitionModel(nn.Module):
    def __init__(self, input_observation_shape, state_type, input_action_space, output_observation_shape, num_subpolicy, mutual_information, linear_size=256):
        super(TransitionModel, self).__init__()
        '''if mutual_information, transition_model is act as a regressor to fit p(Z|c)'''

        self.input_observation_shape = input_observation_shape
        self.state_type = state_type
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

        self.tanh_init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('tanh'))

        if self.state_type in ['standard_image']:
            self.conv = nn.Sequential(
                self.leakrelu_init_(nn.Conv2d(self.input_observation_shape[0], 16, 8, stride=4)),
                # input do not normalize
                nn.LeakyReLU(inplace=True),

                self.leakrelu_init_(nn.Conv2d(16, 32, 4, stride=2)),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(inplace=True),

                self.leakrelu_init_(nn.Conv2d(32, 16, 3, stride=1)),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(inplace=True),

                Flatten(),

                self.linear_init_(nn.Linear(16 * 7 * 7, self.linear_size)),
                # fc donot normalize
                # fc linear
            )
        elif self.state_type in ['vector']:
            obs_size_flatten = get_obs_size_flatten(self.input_observation_shape)
            self.conv = nn.Sequential(
                self.linear_init_(nn.Linear(obs_size_flatten, self.linear_size)),
                # nn.BatchNorm1d(self.linear_size),
                nn.Tanh(),
                self.linear_init_(nn.Linear(self.linear_size, self.linear_size)),
                # nn.BatchNorm1d(self.linear_size),
                nn.Tanh()
            )

        self.reward_bounty_linear = nn.Sequential(
            self.linear_init_(nn.Linear(self.linear_size, 1)),
            # output do not normalize
            # linear output
        )

        if not self.mutual_information:

            self.input_action_space = input_action_space
            self.input_action_linear = nn.Sequential(
                self.linear_init_(nn.Linear(self.input_action_space.n, self.linear_size)),
                # fc donot normalize
                # fc linear
            )

            if self.state_type in ['standard_image']:
                self.deconv = nn.Sequential(
                    self.leakrelu_init_(nn.Linear(self.linear_size, 16 * 7 * 7)),
                    # project donot normalize
                    nn.LeakyReLU(),

                    DeFlatten((16,7,7)),

                    self.leakrelu_init_(nn.ConvTranspose2d(16, 32, 3, stride=1)),
                    nn.BatchNorm2d(32),
                    nn.LeakyReLU(),

                    self.leakrelu_init_(nn.ConvTranspose2d(32, 16, 4, stride=2)),
                    nn.BatchNorm2d(16),
                    nn.LeakyReLU(),

                    self.tanh_init_(nn.ConvTranspose2d(16, self.output_observation_shape[0], 8, stride=4)),
                    # output do not normalize
                    nn.Tanh(),
                )
            elif self.state_type in ['vector']:
                self.deconv = nn.Sequential(
                    self.linear_init_(nn.Linear(self.linear_size, self.linear_size)),
                    # nn.BatchNorm1d(self.linear_size),
                    nn.Tanh(),
                    self.linear_init_(nn.Linear(self.linear_size, obs_size_flatten)),
                    # output do not normalize
                )
                if args.env_name in ['Explore2D','MinitaurBulletEnv-v0','MinitaurBulletEnv-v1','MinitaurBulletEnv-v2','ReacherBulletEnv-v1','Explore2DContinuous']:
                    pass
                elif args.env_name in ['env with obs bound'] > -1:
                    self.deconv.add_module(
                        'output_active',
                        nn.Tanh(),
                    )
                else:
                    raise NotImplemented

        else:

            self.label_linear = nn.Sequential(
                self.linear_init_(nn.Linear(self.linear_size, num_subpolicy)),
            )

    def forward(self, inputs, input_action=None):
        if self.state_type in ['standard_image']:
            conved = self.conv(
                inputs/255.0,
            )
        elif self.state_type in ['vector']:
            conved = self.conv(
                inputs.view(inputs.size()[0],-1),
            )

        before_deconv = conved*self.input_action_linear(input_action)

        predicted_reward_bounty = self.reward_bounty_linear(before_deconv)

        if not self.mutual_information:

            if self.state_type in ['standard_image']:

                predicted_state = self.deconv(before_deconv)*255.0

            elif self.state_type in ['vector']:

                predicted_state = self.deconv(before_deconv)
                predicted_state = predicted_state.view(predicted_state.size()[0],*self.output_observation_shape)

            return predicted_state, predicted_reward_bounty

        else:

            predicted_action_resulted_from = F.log_softmax(self.label_linear(before_deconv), dim=1)

            return predicted_action_resulted_from, predicted_reward_bounty

    def save_model(self, save_path):
        torch.save(self.state_dict(), save_path)
