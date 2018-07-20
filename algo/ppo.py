import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils


class PPO(object):
    def __init__(self,args,actor_critic,hierarchy_id,transition_model=None):

        self.actor_critic = actor_critic
        self.args = args
        self.hierarchy_id = hierarchy_id
        self.transition_model = transition_model

        self.optimizer_actor_critic = optim.Adam(actor_critic.parameters(), lr=self.args.lr, eps=self.args.eps)

        if self.transition_model is not None:
            self.action_onehot_batch = torch.zeros(self.args.mini_batch_size,self.actor_critic.output_action_space.n).cuda()
            self.mse_loss_model = torch.nn.MSELoss(size_average=True,reduce=True)
            self.optimizer_transition_model = optim.Adam(self.transition_model.parameters(), lr=1e-4, betas=(0.0, 0.9))
            self.batch_index = torch.LongTensor(range(self.args.mini_batch_size)).cuda()

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)


        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        mse_loss_epoch = 0

        for e in range(self.args.ppo_epoch):
            if hasattr(self.actor_critic.base, 'gru'):
                data_generator = rollouts.recurrent_generator(
                    advantages, self.args.mini_batch_size)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.args.mini_batch_size)

            for sample in data_generator:
                observations_batch, next_observations_batch, input_actions_batch, states_batch, actions_batch, \
                   return_batch, masks_batch, next_masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, states = self.actor_critic.evaluate_actions(
                    inputs = observations_batch,
                    states = states_batch,
                    masks = masks_batch,
                    action = actions_batch,
                    input_action = input_actions_batch,
                )

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.args.clip_param,
                                           1.0 + self.args.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(return_batch, values)

                self.optimizer_actor_critic.zero_grad()
                (value_loss * self.args.value_loss_coef + action_loss -
                 dist_entropy * self.args.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.args.max_grad_norm)
                self.optimizer_actor_critic.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

                if self.transition_model is not None:

                    '''convert actions_batch to action_onehot_batch'''
                    self.action_onehot_batch.fill_(0.0)
                    self.action_onehot_batch.scatter_(1,actions_batch.long(),1.0)

                    '''generate indexs'''
                    next_masks_batch_index = next_masks_batch.squeeze().nonzero().squeeze()
                    next_masks_batch_index_observations_batch = next_masks_batch_index.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(next_masks_batch_index.size()[0],*observations_batch.size()[1:])
                    next_masks_batch_index_next_observations_batch = next_masks_batch_index.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(next_masks_batch_index.size()[0],*next_observations_batch.size()[1:])
                    next_masks_batch_index_action_onehot_batch = next_masks_batch_index.unsqueeze(1).expand(next_masks_batch_index.size()[0],*self.action_onehot_batch.size()[1:])

                    '''forward'''
                    self.transition_model.train()
                    predicted_next_observations_batch = self.transition_model(
                        inputs = observations_batch.gather(0,next_masks_batch_index_observations_batch),
                        input_action = self.action_onehot_batch.gather(0,next_masks_batch_index_action_onehot_batch),
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

                    mse_loss_epoch += mse_loss.item()

                else:

                    mse_loss_epoch = None

        num_updates = self.args.ppo_epoch * (self.args.num_processes * self.args.num_steps[self.hierarchy_id]//self.args.mini_batch_size)

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        if mse_loss_epoch is not None:
            mse_loss_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, mse_loss_epoch
