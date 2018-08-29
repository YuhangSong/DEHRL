import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
import numpy as np

class PPO(object):

    def set_this_layer(self, this_layer):
        self.this_layer = this_layer
        self.init_actor_critic()

    def init_actor_critic(self):
        self.optimizer_actor_critic = optim.Adam(self.this_layer.actor_critic.parameters(), lr=self.this_layer.args.lr, eps=self.this_layer.args.eps)
        self.one = torch.FloatTensor([1]).cuda()
        self.mone = self.one * -1

        self.actor_critic_gradients_reward = False
        self.actor_critic_preserve_prediction = False
        self.transition_model_gradients_reward = False
        self.transition_model_preserve_prediction = False

        if self.this_layer.hierarchy_id != (self.this_layer.args.num_hierarchy-1):

            if self.this_layer.args.encourage_ac_connection in ['actor_critic','both']:

                if self.this_layer.args.encourage_ac_connection_type in ['gradients_reward']:
                    self.actor_critic_gradients_reward = True

                if self.this_layer.args.encourage_ac_connection_type in ['preserve_prediction']:
                    self.actor_critic_preserve_prediction = True

            if self.this_layer.args.encourage_ac_connection in ['transition_model','both']:

                if self.this_layer.args.encourage_ac_connection_type in ['gradients_reward']:
                    self.transition_model_gradients_reward = True

                if self.this_layer.args.encourage_ac_connection_type in ['preserve_prediction']:
                    self.transition_model_preserve_prediction = True

        if self.actor_critic_preserve_prediction:

            self.action_onehot_travel_batch = torch.zeros(self.this_layer.args.actor_critic_mini_batch_size*self.this_layer.action_space.n,self.this_layer.action_space.n).cuda()
            batch_i = 0
            for action_i in range(self.this_layer.action_space.n):
                for mini_batch_i in range(self.this_layer.args.actor_critic_mini_batch_size):
                    self.action_onehot_travel_batch[batch_i][action_i] = 1.0
                    batch_i += 1

            self.batch_index_travel = np.array(range(self.this_layer.args.actor_critic_mini_batch_size*self.this_layer.action_space.n))

    def set_upper_layer(self, upper_layer):
        '''this method will be called if we have a transition_model to generate reward bounty'''
        self.upper_layer = upper_layer
        if self.upper_layer.transition_model is not None:
            self.init_transition_model()

    def init_transition_model(self):
        '''build essential things for training transition_model'''
        self.optimizer_transition_model = optim.Adam(self.upper_layer.transition_model.parameters(), lr=1e-4, betas=(0.0, 0.9))
        self.NLLLoss = nn.NLLLoss()

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

        '''train actor_critic'''
        if update_type in ['actor_critic','both']:

            '''compute advantages'''
            advantages = self.this_layer.rollouts.returns[:-1] - self.this_layer.rollouts.value_preds[:-1]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

            '''prepare epoch'''
            epoch = self.this_layer.args.actor_critic_epoch
            if self.this_layer.update_i < 3:
                print('[H-{}] {}-th time train actor_critic, skip, since transition_model need to be trained first.'.format(
                    self.this_layer.hierarchy_id,
                    self.this_layer.update_i,
                ))
                epoch *= 0

            for e in range(epoch):

                data_generator = self.this_layer.rollouts.feed_forward_generator(
                    advantages = advantages,
                    mini_batch_size = self.this_layer.args.actor_critic_mini_batch_size,
                )

                for sample in data_generator:

                    self.optimizer_actor_critic.zero_grad()

                    observations_batch, input_actions_batch, states_batch, actions_batch, \
                       return_batch, masks_batch, old_action_log_probs_batch, \
                            adv_targ = sample

                    input_actions_index = input_actions_batch.nonzero()[:,1]

                    # Reshape to do in a single forward pass for all steps
                    values, action_log_probs, dist_entropy, _, dist_features = self.this_layer.actor_critic.evaluate_actions(
                        inputs       = observations_batch,
                        states       = states_batch,
                        masks        = masks_batch,
                        action       = actions_batch,
                        input_action = input_actions_batch,
                    )

                    ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(ratio, 1.0 - self.this_layer.args.clip_param,
                                               1.0 + self.this_layer.args.clip_param) * adv_targ
                    action_loss = -torch.min(surr1, surr2)
                    action_loss = action_loss.mean()

                    value_loss = (return_batch-values).pow(2) * self.this_layer.args.value_loss_coef
                    value_loss = value_loss.mean()

                    dist_entropy = dist_entropy * self.this_layer.args.entropy_coef
                    dist_entropy = dist_entropy.mean()

                    final_loss = value_loss + action_loss - dist_entropy

                    final_loss.backward(
                        retain_graph = (self.this_layer.args.encourage_ac_connection in ['actor_critic','both']),
                    )

                    nn.utils.clip_grad_norm_(self.this_layer.actor_critic.parameters(),
                                             self.this_layer.args.max_grad_norm)

                    self.optimizer_actor_critic.step()

            if epoch != 0:
                epoch_loss['action'] = action_loss.item()
                epoch_loss['value'] = value_loss.item()
                epoch_loss['dist_entropy'] = dist_entropy.item()

        '''train transition_model'''
        if update_type in ['transition_model','both']:
            '''prepare epoch'''
            epoch = self.this_layer.args.transition_model_epoch
            if self.this_layer.update_i < 3:
                print('[H-{}] {}-th time train transition_model, train more epoch'.format(
                    self.this_layer.hierarchy_id,
                    self.this_layer.update_i,
                ))
                if not self.this_layer.checkpoint_loaded:
                    epoch = 800

            for e in range(epoch):

                data_generator = self.upper_layer.rollouts.transition_model_feed_forward_generator(
                    mini_batch_size = int(self.this_layer.args.transition_model_mini_batch_size[self.this_layer.hierarchy_id]),
                    recent_steps = int(self.this_layer.rollouts.num_steps/self.this_layer.hierarchy_interval)-1,
                    recent_at = self.upper_layer.step_i,
                )

                for sample in data_generator:

                    observations_batch, next_observations_batch, action_onehot_batch, reward_bounty_raw_batch = sample

                    self.optimizer_transition_model.zero_grad()

                    if self.this_layer.args.encourage_ac_connection in ['transition_model','both']:
                        action_onehot_batch = torch.autograd.Variable(action_onehot_batch, requires_grad=True)

                    '''forward'''
                    self.upper_layer.transition_model.train()

                    predicted_next_observations_batch, next_state_feature_detached, reward_bounty, predicted_action = self.upper_layer.transition_model(
                        cur_state = observations_batch,
                        next_state = next_observations_batch,
                        input_action = action_onehot_batch,
                    )

                    '''inverse model loss'''
                    inverse_model_loss = self.NLLLoss(predicted_action, action_onehot_batch.nonzero()[:,1])

                    if self.this_layer.update_i > 0:
                        '''transition loss'''
                        transition_loss = F.mse_loss(
                            input = predicted_next_observations_batch,
                            target = next_state_feature_detached,
                            reduction='elementwise_mean',
                        )

                        if self.this_layer.update_i > 1:
                            '''reward bounty loss'''
                            reward_bounty_loss = F.mse_loss(
                                input = reward_bounty,
                                target = reward_bounty_raw_batch,
                                reduction='elementwise_mean',
                            )

                    if self.this_layer.update_i == 0:
                        final_loss = inverse_model_loss
                    elif self.this_layer.update_i == 1:
                        final_loss = inverse_model_loss + transition_loss
                    elif self.this_layer.update_i >= 2:
                        final_loss = inverse_model_loss + transition_loss + reward_bounty_loss

                    '''backward'''
                    final_loss.backward(
                        retain_graph = (self.this_layer.args.encourage_ac_connection in ['transition_model','both']),
                    )

                    self.optimizer_transition_model.step()

                if self.this_layer.update_i < 3:
                    print_str = ''
                    print_str += '[H-{}] {}-th time train transition_model, epoch {}'.format(
                        self.this_layer.hierarchy_id,
                        self.this_layer.update_i,
                        e,
                    )
                    print_str += ', iml {}'.format(
                        inverse_model_loss.item(),
                    )
                    if self.this_layer.update_i >= 1:
                        print_str += ', tl {}'.format(
                            transition_loss.item(),
                        )
                    if self.this_layer.update_i >= 2:
                        print_str += ', rbl {}'.format(
                            reward_bounty_loss.item(),
                        )
                    print(print_str)

            epoch_loss['inverse_model_loss'] = inverse_model_loss
            if self.this_layer.update_i >= 1:
                epoch_loss['transition_loss'] = transition_loss
            if self.this_layer.update_i >= 2:
                epoch_loss['reward_bounty_loss'] = reward_bounty_loss

        return epoch_loss
