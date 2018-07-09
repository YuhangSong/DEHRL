import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PPO(object):
    def __init__(self,args,actor_critic,hierarchy_id):

        self.actor_critic = actor_critic
        self.args = args
        self.hierarchy_id = hierarchy_id

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=self.args.lr, eps=self.args.eps)

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)


        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.args.ppo_epoch):
            if hasattr(self.actor_critic.base, 'gru'):
                data_generator = rollouts.recurrent_generator(
                    advantages, self.args.mini_batch_size)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.args.mini_batch_size)

            for sample in data_generator:
                observations_batch, input_actions_batch, states_batch, actions_batch, \
                   return_batch, masks_batch, old_action_log_probs_batch, \
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

                self.optimizer.zero_grad()
                (value_loss * self.args.value_loss_coef + action_loss -
                 dist_entropy * self.args.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.args.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.args.ppo_epoch * (self.args.num_processes * self.args.num_steps[self.hierarchy_id]//self.args.mini_batch_size)

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
