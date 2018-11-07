import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import init, init_normc_, AddBias
import numpy as np

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
    def __init__(self, num_inputs, num_outputs, num_subpolicy):
        super(Categorical, self).__init__()
        self.num_subpolicy = num_subpolicy

        init_ = lambda m: init(m,
              nn.init.orthogonal_,
              lambda x: nn.init.constant_(x, 0),
              gain=0.01)

        if self.num_subpolicy <= 1:
            self.linear = init_(nn.Linear(num_inputs, num_outputs))
        else:
            self.linear = []
            for linear_i in range(self.num_subpolicy):
                self.linear += [init_(nn.Linear(num_inputs, num_outputs))]
            self.linear = nn.ModuleList(self.linear)
            self.index_dic = {}
            self.y_dic = {}
            self.num_outputs = num_outputs

    def forward(self, x, index = None):
        if self.num_subpolicy <= 1:
            y_ = self.linear(x)
        else:
            self.y_dic = {}
            for dic_i in range(self.num_subpolicy):
                self.index_dic[str(dic_i)] = torch.from_numpy(np.where(index==dic_i)[0]).long().cuda()
                if self.index_dic[str(dic_i)].size()[0] != 0:
                    self.y_dic[str(dic_i)] = self.linear[dic_i](
                        torch.index_select(x,0,self.index_dic[str(dic_i)])
                    )
            y_ = torch.zeros((x.size()[0],self.num_outputs)).cuda()
            for y_i in range(self.num_subpolicy):
                if str(y_i) in self.y_dic:
                    y_.index_add_(0,self.index_dic[str(y_i)],self.y_dic[str(y_i)])

        return FixedCategorical(logits=y_), y_

class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_subpolicy):
        super(DiagGaussian, self).__init__()
        self.num_subpolicy = num_subpolicy

        init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))

        if self.num_subpolicy <= 1:
            self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
            self.logstd = AddBias(torch.zeros(num_outputs))
        else:
            self.fc_mean = []
            self.logstd = []
            for linear_i in range(self.num_subpolicy):
                self.fc_mean += [init_(nn.Linear(num_inputs, num_outputs))]
                self.logstd += [AddBias(torch.zeros(num_outputs))]
            self.fc_mean = nn.ModuleList(self.fc_mean)
            self.logstd = nn.ModuleList(self.logstd)
            self.index_dic = {}
            self.y_dic = {}
            self.num_outputs = num_outputs

    def forward(self, x, index = None):

        if self.num_subpolicy <= 1:
            action_mean = self.fc_mean(x)
            action_logstd = self.logstd(torch.zeros(action_mean.size()).cuda())
        else:
            self.y_dic = {}
            self.std_dic = {}
            for dic_i in range(self.num_subpolicy):
                self.index_dic[str(dic_i)] = torch.from_numpy(np.where(index==dic_i)[0]).long().cuda()
                if self.index_dic[str(dic_i)].size()[0] != 0:
                    self.y_dic[str(dic_i)] = self.fc_mean[dic_i](
                        torch.index_select(x,0,self.index_dic[str(dic_i)])
                    )
                    self.std_dic[str(dic_i)] = self.logstd[dic_i](
                        torch.zeros(self.y_dic[str(dic_i)].size()).cuda()
                    )
            action_mean   = torch.zeros((x.size()[0],self.num_outputs)).cuda()
            action_logstd = torch.zeros((x.size()[0],self.num_outputs)).cuda()
            for y_i in range(self.num_subpolicy):
                if str(y_i) in self.y_dic:
                    action_mean  .index_add_(0,self.index_dic[str(y_i)],self.y_dic  [str(y_i)])
                    action_logstd.index_add_(0,self.index_dic[str(y_i)],self.std_dic[str(y_i)])

        return FixedNormal(action_mean, action_logstd.exp()), action_mean
