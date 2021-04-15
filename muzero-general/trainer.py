import numpy as np
import copy
import time

import numpy
import ray
import torch

import models
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence

@ray.remote
class Trainer:
    """
    Class which run in a dedicated thread to train a neural network and save it
    in the shared storage.
    """

    def __init__(self, initial_checkpoint, config):
        self.config = config
        # NOTE: 'mock_stochastic' uses stochastic dynamics, but is not a stochastic model
        self.is_stochastic_model = config.network in ['stochastic', 'stochastic_concat']

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Initialize the network
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(copy.deepcopy(initial_checkpoint["weights"]))
        self.model.to(torch.device("cuda" if self.config.train_on_gpu else "cpu"))
        self.model.train()

        if hasattr(self.config, "freeze_dynamics") and self.config.freeze_dynamics:
            self.model.freeze_dynamics()

        self.training_step = initial_checkpoint["training_step"]

        if "cuda" not in str(next(self.model.parameters()).device):
            print("You are not training on GPU.\n")

        # Initialize the optimizer
        if self.config.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.lr_init,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.lr_init,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise NotImplementedError(
                f"{self.config.optimizer} is not implemented. You can change the optimizer manually in trainer.py."
            )

        if initial_checkpoint["optimizer_state"] is not None:
            print("Loading optimizer...\n")
            self.optimizer.load_state_dict(
                copy.deepcopy(initial_checkpoint["optimizer_state"])
            )

    def save_batch_stats(self, batch, shared_storage):
        (
            observation_batch,
            action_batch,
            target_value,
            target_reward,
            target_policy, # B x N x  5
            weight_batch,
            target_terminal,
            target_reconstruction,
            gradient_scale_batch,
        ) = batch
         
        info = {}
        visit0 = np.array(target_policy)[:,0]
        info['avg_visit_dist'] = visit0.mean(0)
        info['visit_dist_entropy'] = - (visit0 * np.log(visit0 + 1e-3)).sum(1).mean(0)
        info['best_return'] = np.array(target_value).max()

        shared_storage.set_info.remote(info)       


    def continuous_update_weights(self, replay_buffer, shared_storage):
        # Wait for the replay buffer to be filled
        while ray.get(shared_storage.get_info.remote("num_played_games")) < 1:
            time.sleep(0.1)

        next_batch = replay_buffer.get_batch.remote()
        # Training loop
        while self.training_step < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            index_batch, batch = ray.get(next_batch)
            next_batch = replay_buffer.get_batch.remote()
            self.update_lr()
            
            (
                priorities,
                total_loss,
                value_loss,
                reward_loss,
                terminal_loss,
                policy_loss,
                reconstruction_loss,
                kl_loss,
            ) = self.update_weights(batch, shared_storage)

            if self.config.PER:
                # Save new priorities in the replay buffer (See https://arxiv.org/abs/1803.00933)
                replay_buffer.update_priorities.remote(priorities, index_batch)

            # Save to the shared storage
            if self.training_step % self.config.checkpoint_interval == 0:
                shared_storage.set_info.remote(
                    {
                        "weights": copy.deepcopy(self.model.get_weights()),
                        "optimizer_state": copy.deepcopy(
                            models.dict_to_cpu(self.optimizer.state_dict())
                        ),
                    }
                )
                if self.config.save_model:
                    shared_storage.save_checkpoint.remote()

            self.save_batch_stats(batch, shared_storage)

            shared_storage.set_info.remote(
                {
                    "training_step": self.training_step,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "total_loss": total_loss,
                    "value_loss": value_loss,
                    "reward_loss": reward_loss,
                    "policy_loss": policy_loss,
                    "terminal_loss": terminal_loss,
                    "reconstruction_loss": reconstruction_loss,
                    "kl_loss": kl_loss,
                }
            )

            # Managing the self-play / training ratio
            if self.config.training_delay:
                time.sleep(self.config.training_delay)
            if self.config.ratio:
                while (
                    self.training_step
                    / max(
                        1, ray.get(shared_storage.get_info.remote("num_played_steps"))
                    )
                    > self.config.ratio
                    and self.training_step < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    time.sleep(0.5)

    def update_weights(self, batch, shared_storage=None):
        """
        Perform one training step.
        """

        (
            observation_batch,
            action_batch,
            target_value,
            target_reward,
            target_policy,
            weight_batch,
            target_terminal,
            target_reconstruction,
            gradient_scale_batch,
        ) = batch

        # Keep values as scalars for calculating the priorities for the prioritized replay
        target_value_scalar = numpy.array(target_value, dtype="float32")
        priorities = numpy.zeros_like(target_value_scalar)

        device = next(self.model.parameters()).device
        if self.config.PER:
            weight_batch = torch.tensor(weight_batch.copy()).float().to(device)
        # TODO: does this include observations across all timesteps
        # NOTE: no, it doesn't. We need to modify the replay buffer sampling logic 
        #       to return with observations for all steps, instead of just
        #       the initial step with some past history
        # NOTE: this is fixed now, by changing upstream `replay_buffer.get_batch()` behavior
        #       if the model is stochastic
        # non-stochastic: [N, encoding_dims]
        # stochastic: [N, T, encoding_dims]
        observation_batch = torch.tensor(observation_batch).float().to(device)
        action_batch = torch.tensor(action_batch).long().to(device).unsqueeze(-1)
        target_value = torch.tensor(target_value).float().to(device)
        target_reward = torch.tensor(target_reward).float().to(device)
        target_policy = torch.tensor(target_policy).float().to(device)
        target_terminal = torch.tensor(target_terminal).float().to(device)
        target_reconstruction = torch.tensor(target_reconstruction).float().to(device)
        gradient_scale_batch = torch.tensor(gradient_scale_batch).float().to(device)
        # observation_batch: batch, channels, height, width
        # action_batch: batch, num_unroll_steps+1, 1 (unsqueeze)
        # target_value: batch, num_unroll_steps+1
        # target_reward: batch, num_unroll_steps+1
        # target_policy: batch, num_unroll_steps+1, len(action_space)
        # target_terminal: batch, num_unroll_steps+1
        # target_reconstruction: batch, num_unroll_steps+1, channels, width, height
        # gradient_scale_batch: batch, num_unroll_steps+1

        target_value = models.scalar_to_support(target_value, self.config.support_size)
        target_reward = models.scalar_to_support(
            target_reward, self.config.support_size
        )
        # target_value: batch, num_unroll_steps+1, 2*support_size+1
        # target_reward: batch, num_unroll_steps+1, 2*support_size+1

        if self.is_stochastic_model:
            ## Generate predictions
            value, reward, _, policy_logits, _, hidden_state = self.model.initial_inference(
                observation_batch[:, 0]
            )

            hidden_states_from_observation = [hidden_state]
            predictions = [(value, reward, None, policy_logits, None)]
            # NOTE: this is just to make the indexing work out
            transition_logits = [None]  

            # TODO: we need to iterate over all timesteps, and run model.representation
            #       to obtain all the hidden states across time
            # NOTE: done
            for i in range(1, action_batch.shape[1]):
                # NOTE: we will receive nans when the index is past an absorbing state
                #       in this case, we bypass the representation function, and directly set the hidden state to zeros
                if torch.isnan(observation_batch[:, i]).any():
                    hidden_state = torch.zeros_like(hidden_states_from_observation[0])
                else:
                    hidden_state = self.model.representation(observation_batch[:, i])
                # NOTE: we don't want the gradient flowing through representation function applied to future states
                hidden_state = hidden_state.detach()
                hidden_states_from_observation.append(hidden_state)

            # TODO: we need to pass in the next_hidden_state as well here (computed in a loop in the previous TODO)
            #       this is because we need it to do inference for the posterior transition logits
            # NOTE: done
            hidden_state = hidden_states_from_observation[0]
            for i in range(1, action_batch.shape[1]):
                # NOTE: Don't mind me, just debugging
                # print(f'Per timestep hidden state size is: {hidden_state.shape}')
                # print(f'Per timestep action size is: {action_batch[:, i].shape}')
                value, reward, policy_logits, hidden_state, transition_logits_post, transition_logits_prior = self.model.recurrent_inference(
                    hidden_state, action_batch[:, i], hidden_states_from_observation[i]
                )
                # Scale the gradient at the start of the dynamics function (See paper appendix Training)
                hidden_state.register_hook(lambda grad: grad * 0.5)
                predictions.append((value, reward, None, policy_logits, None))
                transition_logits.append((transition_logits_post, transition_logits_prior))
        else:
            ## Generate predictions
            value, reward, terminal_logits, policy_logits, reconstruction, hidden_state = self.model.initial_inference(
                observation_batch
            )
            predictions = [(value, reward, terminal_logits, policy_logits, reconstruction)]
            for i in range(1, action_batch.shape[1]):
                value, reward, terminal_logits, policy_logits, reconstruction, hidden_state = (
                    self.model.recurrent_inference(hidden_state, action_batch[:, i])
                )
                # Scale the gradient at the start of the dynamics function (See paper appendix Training)
                if not (hasattr(self.config, "freeze_dynamics") and self.config.freeze_dynamics):
                    hidden_state.register_hook(lambda grad: grad * 0.5)
                predictions.append((value, reward, terminal_logits, policy_logits, reconstruction))
            # predictions: num_unroll_steps+1, 3, batch, 2*support_size+1 | 2*support_size+1 | 9 (according to the 2nd dim)

        ## Compute losses
        kl_loss = 0.
        value_loss, reward_loss, terminal_loss, policy_loss, reconstruction_loss = (0, 0, 0, 0, 0)
        value, reward, terminal_logits, policy_logits, reconstruction = predictions[0]

        # Ignore reward loss for the first batch step
        current_value_loss, _, _, current_policy_loss, current_reconstruction_loss = self.loss_function(
            value.squeeze(-1),
            reward.squeeze(-1),
            terminal_logits,
            policy_logits,
            reconstruction,
            target_value[:, 0],
            target_reward[:, 0],
            target_terminal[:, 0],
            target_policy[:, 0],
            target_reconstruction[:, 0]
        )
        # NOTE: there's no KL loss for the first timestep
        value_loss += current_value_loss
        policy_loss += current_policy_loss
        reconstruction_loss += current_reconstruction_loss
        # Compute priorities for the prioritized replay (See paper appendix Training)
        pred_value_scalar = (
            models.support_to_scalar(value, self.config.support_size)
            .detach()
            .cpu()
            .numpy()
            .squeeze()
        )
        priorities[:, 0] = (
            numpy.abs(pred_value_scalar - target_value_scalar[:, 0])
            ** self.config.PER_alpha
        )

        for i in range(1, len(predictions)):
            value, reward, terminal_logits, policy_logits, reconstruction = predictions[i]
            (
                current_value_loss,
                current_reward_loss,
                current_terminal_loss,
                current_policy_loss,
                current_reconstruction_loss,
            ) = self.loss_function(
                value.squeeze(-1),
                reward.squeeze(-1),
                terminal_logits,
                policy_logits,
                reconstruction,
                target_value[:, i],
                target_reward[:, i],
                target_terminal[:, i],
                target_policy[:, i],
                target_reconstruction[:, i]
            )

            # Scale gradient by the number of unroll steps (See paper appendix Training)
            if not (hasattr(self.config, "freeze_dynamics") and self.config.freeze_dynamics):
                current_value_loss.register_hook(
                    lambda grad: grad / gradient_scale_batch[:, i]
                )
                current_reward_loss.register_hook(
                    lambda grad: grad / gradient_scale_batch[:, i]
                )
                current_policy_loss.register_hook(
                    lambda grad: grad / gradient_scale_batch[:, i]
                )
                if terminal_logits is not None:
                    current_terminal_loss.register_hook(
                        lambda grad: grad / gradient_scale_batch[:, i]
                    )
                if reconstruction is not None:
                    current_reconstruction_loss.register_hook(
                        lambda grad: grad / gradient_scale_batch[:, i]
                    )
            else:
                current_value_loss.register_hook(
                    lambda grad: grad / gradient_scale_batch[:, i]
                )
                current_policy_loss.register_hook(
                    lambda grad: grad / gradient_scale_batch[:, i]
                )

            # ignore losses for terminal states
            if hasattr(self.config, "mask_absorbing_states") and self.config.mask_absorbing_states:
                current_value_loss = current_value_loss * (1 - target_terminal[:, i])  # don't predict value of the first terminal state
                current_reward_loss = current_reward_loss * (1 - target_terminal[:, i - 1])  # predict reward at the first terminal state
                current_terminal_loss = current_terminal_loss * (1 - target_terminal[:, i - 1])  # predict terminal at the first terminal state
                current_policy_loss = current_policy_loss * (1 - target_terminal[:, i])  # don't predict policy at first terminal state
                current_reconstruction_loss = current_reconstruction_loss * (1 - target_terminal[:, i - 1])  # prediction reconstruction at first terminal state

            if self.is_stochastic_model:
                # TODO: compute KL divergence loss between the posterior transition logits and prior transition logits
                transition_logits_post, transition_logits_prior = transition_logits[i]
                current_kl_loss = kl_loss_criterion(transition_logits_post, transition_logits_prior)
                current_kl_loss.register_hook(lambda grad: grad / gradient_scale_batch[:, i])
                kl_loss += current_kl_loss

            value_loss += current_value_loss
            reward_loss += current_reward_loss
            terminal_loss += current_terminal_loss
            policy_loss += current_policy_loss
            reconstruction_loss += current_reconstruction_loss

            # Compute priorities for the prioritized replay (See paper appendix Training)
            pred_value_scalar = (
                models.support_to_scalar(value, self.config.support_size)
                .detach()
                .cpu()
                .numpy()
                .squeeze()
            )
            priorities[:, i] = (
                numpy.abs(pred_value_scalar - target_value_scalar[:, i])
                ** self.config.PER_alpha
            )

        # Scale the value loss, paper recommends by 0.25 (See paper appendix Reanalyze)
        loss = (
            value_loss * self.config.value_loss_weight
            + reward_loss * self.config.reward_loss_weight
            + terminal_loss * getattr(self.config, "terminal_loss_weight", 0.)
            + policy_loss * getattr(self.config, "policy_loss_weight", 1.)
            + reconstruction_loss * getattr(self.config, "reconstruction_loss_weight", 0.)
        )
        if self.config.PER:
            # Correct PER bias by using importance-sampling (IS) weights
            loss *= weight_batch
        if self.is_stochastic_model:
            loss += self.config.kl_loss_weight * kl_loss

        # Mean over batch dimension (pseudocode do a sum)
        loss = loss.mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_step += 1

        return (
            priorities,
            # For log purpose
            loss.item(),
            value_loss.mean().item() if torch.is_tensor(value_loss) else 0.,
            reward_loss.mean().item() if torch.is_tensor(reward_loss) else 0.,
            terminal_loss.mean().item() if torch.is_tensor(terminal_loss) else 0.,
            policy_loss.mean().item() if torch.is_tensor(policy_loss) else 0.,
            reconstruction_loss.mean().item() if torch.is_tensor(reconstruction_loss) else 0.,
            kl_loss.mean().item() if torch.is_tensor(kl_loss) else 0.
        )

    def update_lr(self):
        """
        Update learning rate
        """
        lr = self.config.lr_init * self.config.lr_decay_rate ** (
            self.training_step / self.config.lr_decay_steps
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    @staticmethod
    def loss_function(
        value,
        reward,
        terminal_logits,
        policy_logits,
        reconstruction,
        target_value,
        target_reward,
        target_terminal,
        target_policy,
        target_reconstruction,
    ):
        # Cross-entropy seems to have a better convergence than MSE
        value_loss = (-target_value * torch.nn.LogSoftmax(dim=1)(value)).sum(1)
        reward_loss = (-target_reward * torch.nn.LogSoftmax(dim=1)(reward)).sum(1)
        policy_loss = (-target_policy * torch.nn.LogSoftmax(dim=1)(policy_logits)).sum(
            1
        )

        if terminal_logits is not None:
            terminal_loss = torch.nn.functional.binary_cross_entropy_with_logits(terminal_logits, target_terminal, reduction="none")
        else:
            terminal_loss = torch.zeros_like(policy_loss)

        if reconstruction is not None:
            reconstruction_loss = torch.nn.functional.mse_loss(reconstruction, target_reconstruction, reduction="none")
            reconstruction_loss = torch.mean(reconstruction_loss.view(reconstruction_loss.size(0), -1), dim=-1)
        else:
            reconstruction_loss = torch.zeros_like(policy_loss)

        return value_loss, reward_loss, terminal_loss, policy_loss, reconstruction_loss


def kl_loss_criterion(transition_logits_post, transition_logits_prior):
    post_dist = Categorical(logits=transition_logits_post)
    prior_dist = Categorical(logits=transition_logits_prior)
    # TODO: make sure that we do want to have KL(q|p) in that order
    return kl_divergence(post_dist, prior_dist)
