import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
import numpy as np


class PPO(object):

    def init_acforce_obj(self, onehot_length):
        self.onehot_length = onehot_length
        obj_collect = []
        full_block = np.arange(onehot_length)
        full_block = full_block[:,np.newaxis]
        for i in range(onehot_length):
            block_i = np.delete(full_block, i, axis = 0)
            try:
                obj_collect += [block_i]
            except Exception as e:
                obj_collect = block_i
        self.other_action_collect = obj_collect

    def set_this_layer(self, this_layer):
        self.this_layer = this_layer
        self.optimizer_actor_critic = optim.Adam(self.this_layer.actor_critic.parameters(), lr=self.this_layer.args.lr, eps=self.this_layer.args.eps)
        self.one = torch.FloatTensor([1]).cuda()
        self.mone = self.one * -1

    def set_upper_layer(self, upper_layer):
        '''this method will be called if we have a transition_model to generate reward bounty'''
        self.upper_layer = upper_layer

        '''build essential things for training transition_model'''
        self.mse_loss_model = torch.nn.MSELoss(size_average=True,reduce=True)
        self.optimizer_transition_model = optim.Adam(self.upper_layer.transition_model.parameters(), lr=1e-4, betas=(0.0, 0.9))

    def get_grad_norm(self, inputs, outputs):

        gradients = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=torch.ones(outputs.size()).cuda(),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.contiguous()
        gradients_fl = gradients.view(gradients.size()[0],-1)
        gradients_norm = gradients_fl.norm(2, dim=1) / ((gradients_fl.size()[1])**0.5)

        return gradients_norm

    def update(self, update_type):

        epoch_loss = {}

        if update_type in ['actor_critic']:
            advantages = self.this_layer.rollouts.returns[:-1] - self.this_layer.rollouts.value_preds[:-1]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
            epoch_loss['value'] = 0
            epoch_loss['action'] = 0
            epoch_loss['dist_entropy'] = 0
            epoch = self.this_layer.args.ppo_epoch

        elif update_type in ['transition_model']:
            epoch_loss['mse'] = 0
            epoch = self.this_layer.args.transition_model_epoch

        else:
            raise Exception('Not Supported')

        if self.this_layer.args.encourage_ac_connection in ['transition_model','actor_critic','both']:
            if update_type in [self.this_layer.args.encourage_ac_connection]:
                epoch_loss['gradients_reward'] = 0

        for e in range(epoch):

            if update_type in ['actor_critic']:
                data_generator = self.this_layer.rollouts.feed_forward_generator(
                    advantages = advantages,
                    mini_batch_size = self.this_layer.args.actor_critic_mini_batch_size,
                )

            elif update_type in ['transition_model']:
                data_generator = self.upper_layer.rollouts.transition_model_feed_forward_generator(
                    mini_batch_size = self.this_layer.args.transition_model_mini_batch_size,
                    recent_steps = int(self.this_layer.rollouts.num_steps/self.this_layer.hierarchy_interval)-1,
                    recent_at = self.upper_layer.step_i,
                )

            for sample in data_generator:

                if update_type in ['actor_critic']:

                    observations_batch, input_actions_batch, states_batch, actions_batch, \
                       return_batch, masks_batch, old_action_log_probs_batch, \
                            adv_targ = sample

                    if self.this_layer.args.encourage_ac_connection in ['actor_critic','both']:
                        input_actions_batch = torch.autograd.Variable(input_actions_batch, requires_grad=True)

                    # Reshape to do in a single forward pass for all steps
                    if self.this_layer.args.ac_force and self.this_layer.hierarchy_id < (self.this_layer.args.num_hierarchy-1):
                        values_all, action_log_probs_all, dist_entropy_all, states_all = self.this_layer.actor_critic.evaluate_actions(
                            inputs = observations_batch.repeat(self.onehot_length,1,1,1),
                            states = states_batch.repeat(self.onehot_length,1),
                            masks = masks_batch.repeat(self.onehot_length,1,1,1),
                            action = actions_batch.repeat(self.onehot_length,1),
                            input_action = input_actions_batch.repeat(self.onehot_length,1),
                        )

                        input_actions_index = np.where(input_actions_batch==1)[1]
                        other_action_index_for_batch = []
                        for batch_index in range(input_actions_batch.size()[0]):
                            try:
                                other_action_index_for_batch = np.vstack((other_action_index_for_batch, np.array(self.other_action_collect[input_actions_index[batch_index]])))
                            except Exception as e:
                                other_action_index_for_batch = np.array(self.other_action_collect[input_actions_index[batch_index]])

                        input_actions_index_for_batch = torch.from_numpy(input_actions_index)*input_actions_batch.size()[0]
                        input_actions_index_for_batch = input_actions_index_for_batch.unsqueeze(1)
                        other_action_index_for_batch = torch.from_numpy(other_action_index_for_batch)*input_actions_batch.size()[0]
                        input_actions_index_for_batch = input_actions_index_for_batch.cuda()
                        other_action_index_for_batch = other_action_index_for_batch.cuda()

                        one_hot_mask = torch.arange(0,input_actions_batch.size()[0])
                        one_hot_mask = one_hot_mask.unsqueeze(1)
                        one_hot_mask = one_hot_mask.cuda()
                        one_hot_mask = one_hot_mask.long()

                        input_actions_index_for_batch = input_actions_index_for_batch + one_hot_mask
                        one_hot_mask = one_hot_mask.repeat(self.onehot_length-1,1)
                        other_action_index_for_batch = other_action_index_for_batch + one_hot_mask
                        input_actions_index_for_batch = input_actions_index_for_batch.squeeze(1)
                        other_action_index_for_batch = other_action_index_for_batch.squeeze(1)

                        values = torch.index_select(values_all,0,input_actions_index_for_batch)
                        values_other = torch.index_select(values_all,0,other_action_index_for_batch)
                        action_log_probs = torch.index_select(action_log_probs_all,0,input_actions_index_for_batch)
                        # action_log_probs_other = torch.index_select(action_log_probs_all,0,other_action_index_for_batch)
                        dist_entropy = dist_entropy_all
                        # dist_entropy_other = torch.index_select(dist_entropy_all,0,other_action_index_for_batch)
                        states = torch.index_select(states_all,0,input_actions_index_for_batch)
                        # states_other = torch.index_select(states_all,0,other_action_index_for_batch)

                    else:
                        values, action_log_probs, dist_entropy, states = self.this_layer.actor_critic.evaluate_actions(
                            inputs = observations_batch,
                            states = states_batch,
                            masks = masks_batch,
                            action = actions_batch,
                            input_action = input_actions_batch,
                        )

                    ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(ratio, 1.0 - self.this_layer.args.clip_param,
                                               1.0 + self.this_layer.args.clip_param) * adv_targ
                    action_loss = -torch.min(surr1, surr2).mean()

                    value_loss = F.mse_loss(return_batch, values)
                    
                    if self.this_layer.args.ac_force and self.this_layer.hierarchy_id < (self.this_layer.args.num_hierarchy-1):
                        value_loss_other = F.mse_loss(values_other.detach(), values_other)
                        value_loss = value_loss + value_loss_other

                    self.optimizer_actor_critic.zero_grad()
                    (value_loss * self.this_layer.args.value_loss_coef + action_loss - dist_entropy * self.this_layer.args.entropy_coef).backward(
                        retain_graph = (self.this_layer.args.encourage_ac_connection in ['actor_critic','both']),
                    )
                    if self.this_layer.args.encourage_ac_connection in ['actor_critic','both']:
                        gradients_norm = self.get_grad_norm(
                            inputs = input_actions_batch,
                            outputs = values,
                        )
                        gradients_reward = (gradients_norm+1.0).log().mean()*self.this_layer.args.encourage_ac_connection_coefficient
                        epoch_loss['gradients_reward'] += gradients_reward.item()
                        gradients_reward.backward(self.mone)
                    nn.utils.clip_grad_norm_(self.this_layer.actor_critic.parameters(),
                                             self.this_layer.args.max_grad_norm)
                    self.optimizer_actor_critic.step()

                    epoch_loss['value'] += value_loss.item()
                    epoch_loss['action'] += action_loss.item()
                    epoch_loss['dist_entropy'] += dist_entropy.item()

                elif update_type in ['transition_model']:

                    observations_batch, next_observations_batch, actions_batch, next_masks_batch = sample

                    action_onehot_batch = torch.zeros(observations_batch.size()[0],self.upper_layer.actor_critic.output_action_space.n).cuda()

                    '''convert actions_batch to action_onehot_batch'''
                    action_onehot_batch.fill_(0.0)
                    action_onehot_batch.scatter_(1,actions_batch.long(),1.0)

                    '''generate indexs'''
                    next_masks_batch_index = next_masks_batch.squeeze().nonzero().squeeze()
                    next_masks_batch_index_observations_batch = next_masks_batch_index.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(next_masks_batch_index.size()[0],*observations_batch.size()[1:])
                    next_masks_batch_index_next_observations_batch = next_masks_batch_index.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(next_masks_batch_index.size()[0],*next_observations_batch.size()[1:])
                    next_masks_batch_index_action_onehot_batch = next_masks_batch_index.unsqueeze(1).expand(next_masks_batch_index.size()[0],*action_onehot_batch.size()[1:])

                    observations_batch = observations_batch.gather(0,next_masks_batch_index_observations_batch)
                    action_onehot_batch = action_onehot_batch.gather(0,next_masks_batch_index_action_onehot_batch)

                    if self.this_layer.args.encourage_ac_connection in ['transition_model','both']:
                        action_onehot_batch = torch.autograd.Variable(action_onehot_batch, requires_grad=True)

                    '''forward'''
                    self.upper_layer.transition_model.train()
                    predicted_next_observations_batch, before_deconv = self.upper_layer.transition_model(
                        inputs = observations_batch,
                        input_action = action_onehot_batch,
                    )

                    '''compute mse loss'''
                    mse_loss = self.mse_loss_model(
                        input = predicted_next_observations_batch,
                        target = next_observations_batch.gather(0,next_masks_batch_index_next_observations_batch),
                    )

                    '''backward'''
                    self.optimizer_transition_model.zero_grad()
                    mse_loss.backward(
                        retain_graph = (self.this_layer.args.encourage_ac_connection in ['transition_model','both']),
                    )
                    if self.this_layer.args.encourage_ac_connection in ['transition_model','both']:
                        gradients_norm = self.get_grad_norm(
                            inputs = action_onehot_batch,
                            outputs = predicted_next_observations_batch,
                        )
                        gradients_reward = (gradients_norm+1.0).log().mean()*self.this_layer.args.encourage_ac_connection_coefficient
                        epoch_loss['gradients_reward'] += gradients_reward.item()
                        gradients_reward.backward(self.mone)
                    self.optimizer_transition_model.step()

                    epoch_loss['mse'] += mse_loss.item()

        return epoch_loss
