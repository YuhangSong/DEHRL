import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import init, init_normc_, AddBias

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=1, keepdim=True)

FixedNormal = torch.distributions.Normal
log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)

entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, interval):
        super(Categorical, self).__init__()
        self.interval = interval

        init_ = lambda m: init(m,
              nn.init.orthogonal_,
              lambda x: nn.init.constant_(x, 0),
              gain=0.01)

        if self.interval is None:
            self.linear = init_(nn.Linear(num_inputs, num_outputs))
        else:
            self.linear = []
            for linear_i in range(self.interval):
                self.linear += [init_(nn.Linear(num_inputs, num_outputs))]
            self.linear = nn.ModuleList(self.linear)


    def forward(self, x, index = None):
        if self.interval is None:
            y = self.linear(x)
        else:
            for x_i in range(x.size()[0]):
                try:
                    y = torch.cat((y,self.linear[index[x_i]](x[x_i]).unsqueeze(0)),0)
                except Exception as e:
                    y = self.linear[index[x_i]](x[x_i]).unsqueeze(0)
            
        return FixedCategorical(logits=y), y


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))

        if self.interval is None:
            self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        else:
            self.fc_mean = []
            for linear_i in range(self.interval):
                self.fc_mean += [init_(nn.Linear(num_inputs, num_outputs))]
            self.fc_mean = nn.ModuleList(self.fc_mean)

        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x, index = None):

        if self.interval is None:
            action_mean = self.fc_mean(x)
        else:
            action_mean = self.fc_mean[index](x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())
