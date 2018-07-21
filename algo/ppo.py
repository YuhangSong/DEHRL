import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils


class PPO(object):

    def set_this_layer(self, this_layer):
        self.this_layer = this_layer
        self.optimizer_actor_critic = optim.Adam(self.this_layer.actor_critic.parameters(), lr=self.this_layer.args.lr, eps=self.this_layer.args.eps)

    def set_upper_layer(self, upper_layer):
        '''this method will be called if we have a transition_model to generate reward bounty'''
        self.upper_layer = upper_layer

        '''build essential things for training transition_model'''
        self.mse_loss_model = torch.nn.MSELoss(size_average=True,reduce=True)
        self.optimizer_transition_model = optim.Adam(self.upper_layer.transition_model.parameters(), lr=1e-4, betas=(0.0, 0.9))

    def update(self, update_type):

        epoch_loss = {}

        if update_type in ['actor_critic']:
            advantages = self.this_layer.rollouts.returns[:-1] - self.this_layer.rollouts.value_preds[:-1]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
            epoch_loss['value'] = 0
            epoch_loss['action'] = 0
            epoch_loss['dist_entropy'] = 0

        elif update_type in ['transition_model']:
            epoch_loss['mse'] = 0

        else:
            raise Exception('Not Supported')

        for e in range(self.this_layer.args.ppo_epoch):

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

                    # Reshape to do in a single forward pass for all steps
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

                    self.optimizer_actor_critic.zero_grad()
                    (value_loss * self.this_layer.args.value_loss_coef + action_loss -
                     dist_entropy * self.this_layer.args.entropy_coef).backward()
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

                    '''forward'''
                    self.upper_layer.transition_model.train()
                    predicted_next_observations_batch = self.upper_layer.transition_model(
                        inputs = observations_batch.gather(0,next_masks_batch_index_observations_batch),
                        input_action = action_onehot_batch.gather(0,next_masks_batch_index_action_onehot_batch),
                    )

                    '''compute mse loss'''
                    mse_loss = self.mse_loss_model(
                        input = predicted_next_observations_batch,
                        target = next_observations_batch.gather(0,next_masks_batch_index_next_observations_batch),
                    )

                    '''backward'''
                    self.optimizer_transition_model.zero_grad()
                    mse_loss.backward()
                    self.optimizer_transition_model.step()

                    epoch_loss['mse'] += mse_loss.item()

        return epoch_loss
