import math
from abc import ABC, abstractmethod

import torch
from torch import nn

from torch.distributions import OneHotCategoricalStraightThrough


class MuZeroNetwork:
    def __new__(cls, config):
        stochastic_dynamics = hasattr(config, 'stochastic_dynamics') and config.stochastic_dynamics
        if config.network == "fullyconnected" and not stochastic_dynamics:
            assert not stochastic_dynamics, 'Cannot use stochastic MCTS if we use standard model'
            return MuZeroFullyConnectedNetwork(
                config.observation_shape,
                config.stacked_observations,
                len(config.action_space),
                config.encoding_size,
                config.fc_reward_layers,
                config.fc_value_layers,
                config.fc_policy_layers,
                config.fc_representation_layers,
                config.fc_dynamics_layers,
                config.fc_reconstruction_layers,
                config.support_size,
            )
        elif config.network == "mock_stochastic":
            assert stochastic_dynamics, 'Must use stochastic MCTS if we use mock stochastic model'
            return MockMuZeroStochastic(
                config.observation_shape,
                config.stacked_observations,
                len(config.action_space),
                config.encoding_size,
                config.fc_reward_layers,
                config.fc_value_layers,
                config.fc_policy_layers,
                config.fc_representation_layers,
                config.fc_dynamics_layers,
                config.support_size,
            )
        elif config.network == 'stochastic':
            assert stochastic_dynamics, 'Must use stochastic MCTS if we use stochastic model'
            return MuZeroStochastic(
                config.observation_shape,
                config.stacked_observations,
                len(config.action_space),
                config.encoding_size,
                config.fc_reward_layers,
                config.fc_value_layers,
                config.fc_policy_layers,
                config.fc_representation_layers,
                config.fc_dynamics_layers,
                config.support_size,
                config.n_futures,
                config.fc_prior_layers,
                config.fc_posterior_layers,
            )
        elif config.network == 'stochastic_concat':
            assert stochastic_dynamics, 'Must use stochastic MCTS if we use stochastic model'
            return MuZeroStochasticConcat(
                config.observation_shape,
                config.stacked_observations,
                len(config.action_space),
                config.encoding_size,
                config.fc_reward_layers,
                config.fc_value_layers,
                config.fc_policy_layers,
                config.fc_representation_layers,
                config.fc_dynamics_layers,
                config.support_size,
                config.n_futures,
                config.fc_prior_layers,
                config.fc_posterior_layers,
            )
        elif config.network == "resnet":
            return MuZeroResidualNetwork(
                config.observation_shape,
                config.stacked_observations,
                len(config.action_space),
                config.blocks,
                config.channels,
                config.reduced_channels_reward,
                config.reduced_channels_value,
                config.reduced_channels_policy,
                config.reduced_channels_reconstruction,
                config.resnet_fc_reward_layers,
                config.resnet_fc_value_layers,
                config.resnet_fc_policy_layers,
                config.resnet_fc_reconstruction_layers,
                config.support_size,
                config.downsample,
            )
        else:
            raise NotImplementedError(
                'The network parameter should be "fullyconnected" or "resnet".'
            )


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict


class AbstractNetwork(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def initial_inference(self, observation):
        pass

    @abstractmethod
    def recurrent_inference(self, encoded_state, action):
        pass

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights, strict=False)


##################################
######## Fully Connected #########


class MuZeroFullyConnectedNetwork(AbstractNetwork):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        action_space_size,
        encoding_size,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        fc_representation_layers,
        fc_dynamics_layers,
        fc_reconstruction_layers,
        support_size,
    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1
        self.observation_shape = observation_shape

        self.representation_network = torch.nn.DataParallel(
            mlp(
                observation_shape[0]
                * observation_shape[1]
                * observation_shape[2]
                * (stacked_observations + 1)
                + stacked_observations * observation_shape[1] * observation_shape[2],
                fc_representation_layers,
                encoding_size,
            )
        )

        self.dynamics_encoded_state_network = torch.nn.DataParallel(
            mlp(
                encoding_size + self.action_space_size,
                fc_dynamics_layers,
                encoding_size,
            )
        )
        self.dynamics_reward_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_reward_layers, self.full_support_size)
        )
        self.dynamics_terminal_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_reward_layers, 1)
        )

        self.prediction_policy_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_policy_layers, self.action_space_size)
        )
        self.prediction_value_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_value_layers, self.full_support_size)
        )

        self.reconstruction_network = torch.nn.DataParallel(
            mlp(
                encoding_size,
                fc_reconstruction_layers,
                observation_shape[0] * observation_shape[1] * observation_shape[2]
            )
        )

    def freeze_dynamics(self):
        for param in self.representation_network.parameters():
            param.requires_grad_(False)
        for param in self.dynamics_encoded_state_network.parameters():
            param.requires_grad_(False)
        for param in self.dynamics_reward_network.parameters():
            param.requires_grad_(False)
        for param in self.dynamics_terminal_network.parameters():
            param.requires_grad_(False)
        for param in self.reconstruction_network.parameters():
            param.requires_grad_(False)

    def prediction(self, encoded_state):
        policy_logits = self.prediction_policy_network(encoded_state)
        value = self.prediction_value_network(encoded_state)
        return policy_logits, value

    def reconstruction(self, encoded_state):
        reconstruction = self.reconstruction_network(encoded_state)
        reconstruction = reconstruction.view(reconstruction.size(0), *self.observation_shape)
        return reconstruction

    def representation(self, observation):
        encoded_state = self.representation_network(
            observation.view(observation.shape[0], -1)
        )
        # Scale encoded state between [0, 1] (See appendix paper Training)
        min_encoded_state = encoded_state.min(1, keepdim=True)[0]
        max_encoded_state = encoded_state.max(1, keepdim=True)[0]
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (
            encoded_state - min_encoded_state
        ) / scale_encoded_state
        return encoded_state_normalized

    def dynamics(self, encoded_state, action):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = (
            torch.zeros((action.shape[0], self.action_space_size))
            .to(action.device)
            .float()
        )
        action_one_hot.scatter_(1, action.long(), 1.0)
        x = torch.cat((encoded_state, action_one_hot), dim=1)

        next_encoded_state = self.dynamics_encoded_state_network(x)

        reward = self.dynamics_reward_network(next_encoded_state)
        terminal = self.dynamics_terminal_network(next_encoded_state)

        # Scale encoded state between [0, 1] (See paper appendix Training)
        min_next_encoded_state = next_encoded_state.min(1, keepdim=True)[0]
        max_next_encoded_state = next_encoded_state.max(1, keepdim=True)[0]
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized = (
            next_encoded_state - min_next_encoded_state
        ) / scale_next_encoded_state

        return next_encoded_state_normalized, reward, terminal[..., 0]

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)
        reconstruction = self.reconstruction(encoded_state)
        # reward equal to 0 for consistency
        reward = torch.log(
            (
                torch.zeros(1, self.full_support_size)
                .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
                .repeat(len(observation), 1)
                .to(observation.device)
            )
        )
        terminal = torch.zeros((len(observation),), device=observation.device)

        return (
            value,
            reward,
            terminal,
            policy_logits,
            reconstruction,
            encoded_state,
        )

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward, terminal = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        reconstruction = self.reconstruction(next_encoded_state)
        return value, reward, terminal, policy_logits, reconstruction, next_encoded_state


###### End Fully Connected #######
##################################


##################################
###### Start Stochastic #######


class MuZeroStochastic(MuZeroFullyConnectedNetwork):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        action_space_size,
        encoding_size,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        fc_representation_layers,
        fc_dynamics_layers,
        support_size,
        # stochastic
        n_futures,
        fc_prior_layers,
        fc_posterior_layers
    ):
        super().__init__(
            observation_shape,
            stacked_observations,
            action_space_size,
            encoding_size,
            fc_reward_layers,
            fc_value_layers,
            fc_policy_layers,
            fc_representation_layers,
            fc_dynamics_layers,
            fc_dynamics_layers,  # dummy reconstruction layers
            support_size,
        )
        # TODO: might be clearer to name these transitions instead of futures
        self.n_futures = n_futures

        # Override
        # we use separate headers to predict different futures
        # h'_1, h'_2, .... h'_k = f(h, a)
        self.dynamics_encoded_state_network = nn.ModuleList([
                torch.nn.DataParallel(
                    mlp(
                        encoding_size + self.action_space_size,
                        fc_dynamics_layers,
                        encoding_size,
                    )
                )
                for _ in range(n_futures)
            ])

        # Prior distribution over discrete future choices: P(z_{prior} | h)
        self.transition_prior_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_prior_layers, self.n_futures)
        )

        # Posterior distribution over discrete future choices: P(z_{post} | h, h')
        self.transition_posterior_network = torch.nn.DataParallel(
            mlp(encoding_size + encoding_size, fc_posterior_layers, self.n_futures)
        )

    def dynamics(self, encoded_state, action):
        """
        NOTE: this should only be called during inference, 
              it returns the prior transition distribution

        NOTE: We augment this function to return a categorical distribution over next states

        Returns:
            transition_logits (list): transition logits
            next_encoded_state  (list): next states
            reward (list): reward associated with each next state
        """
        # both [N, hidden_state_dim, n_futures]
        next_encoded_states, rewards = self._dynamics_all_futures(encoded_state, action)  

        transition_logits = self.transition_prior_network(encoded_state)  # [N, n_futures]

        # dynamics should only be called in inference, with batch size 1
        assert next_encoded_states.dim() == 3
        assert next_encoded_states.shape[0] == 1
        assert rewards.shape[0] == 1
        assert transition_logits.shape[0] == 1

        # TODO: This currently violates the interface, we have the future dim at -1\
        #       Need to transpose(-1, 0, 1, ...) or change the interface
        # NOTE: fixed
        next_encoded_states = next_encoded_states.permute(2, 0, 1)  # [n_futures, N, hidden_state_dim]
        rewards = rewards.permute(2, 0, 1)  # [n_futures, N, support_dim]
        transition_logits = transition_logits.permute(1, 0)  # [n_futures, N]

        return transition_logits, next_encoded_states, rewards

    def _dynamics_all_futures(self, encoded_state, action):
        # TODO: could probably be more efficient by parallelizing
        next_encoded_state_list, reward_list = [], []
        for future_ind in range(self.n_futures):
            next_encoded_state, reward = self._dynamics(encoded_state, action, future_ind)
            next_encoded_state_list.append(next_encoded_state)
            reward_list.append(reward)

        next_encoded_states = torch.stack(next_encoded_state_list, dim=-1)
        rewards = torch.stack(reward_list, dim=-1)

        return next_encoded_states, rewards

    def _dynamics(self, encoded_state, action, future_ind):
        """
        This is identical to original implementation. 
        Only change is indexing into module dict to select the specified dynamic network
        """
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = (
            torch.zeros((action.shape[0], self.action_space_size))
            .to(action.device)
            .float()
        )
        action_one_hot.scatter_(1, action.long(), 1.0)
        x = torch.cat((encoded_state, action_one_hot), dim=1)

        next_encoded_state = self.dynamics_encoded_state_network[future_ind](x)

        reward = self.dynamics_reward_network(next_encoded_state)

        # Scale encoded state between [0, 1] (See paper appendix Training)
        min_next_encoded_state = next_encoded_state.min(1, keepdim=True)[0]
        max_next_encoded_state = next_encoded_state.max(1, keepdim=True)[0]
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized = (
            next_encoded_state - min_next_encoded_state
        ) / scale_next_encoded_state

        return next_encoded_state_normalized, reward

    def recurrent_inference(self, encoded_state, action, future_encoded_state):
        """
        NOTE: This is only called during training, so we use the posterior transition distribution
        NOTE: In inference, we should call prediction and dynamics separately, as we have modified in stochastic MCTS
        NOTE: We also need to change the interface, to return the prior and posterior transition distribution
              such that we can take KL divergence loss on them
        NOTE: we need to future encoded state to get posterior distribution ove transitions

        NOTE: interface is changed to include two more in return:
            - transition_logits_post
            - transition_logits_prior

        This is for taking the KL divergence loss
        """
        N, encoding_size = encoded_state.shape
        next_encoded_states, rewards = self._dynamics_all_futures(encoded_state, action)  # assuming shapes are [N, encoding_size, 3], [N, 3]

        transition_logits_prior = self.transition_prior_network(encoded_state)  # [N, 3]
        transition_logits_post = self.transition_posterior_network(torch.cat((encoded_state, future_encoded_state), dim=-1))  # [N, 3]

        # TODO: implement straight through (or find an existing implementation)
        # transition_sample_post = sample_straight_through(logits=transition_logits_post)
        # NOTE: found native pytorch impl 
        dist = OneHotCategoricalStraightThrough(logits=transition_logits_post)
        transition_sample_post = dist.rsample()
        # NOTE: unsqueeze to match next_encoding_states dims
        transition_sample_post = transition_sample_post.unsqueeze(dim=1)  # [N, 1, n_futures]

        next_encoded_state = (next_encoded_states * transition_sample_post).sum(-1)  # [N, encoding_size]
        reward = (rewards * transition_sample_post).sum(-1)  # [N, support_dim]

        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state, transition_logits_post, transition_logits_prior


class MuZeroStochasticConcat(MuZeroFullyConnectedNetwork):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        action_space_size,
        encoding_size,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        fc_representation_layers,
        fc_dynamics_layers,
        support_size,
        # stochastic
        n_futures,
        fc_prior_layers,
        fc_posterior_layers
    ):
        super().__init__(
            observation_shape,
            stacked_observations,
            action_space_size,
            encoding_size,
            fc_reward_layers,
            fc_value_layers,
            fc_policy_layers,
            fc_representation_layers,
            fc_dynamics_layers,
            fc_dynamics_layers,  # dummy reconstruction layers
            support_size,
        )
        # TODO: might be clearer to name these transitions instead of futures
        self.n_futures = n_futures

        # Override
        # we use shared header to predict different futures
        # h'_z = f(h, a, z)
        self.dynamics_encoded_state_network = torch.nn.DataParallel(
            mlp(
                encoding_size + self.action_space_size + n_futures,
                fc_dynamics_layers,
                encoding_size,
            )
        )

        # Prior distribution over discrete future choices: P(z_{prior} | h)
        self.transition_prior_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_prior_layers, self.n_futures)
        )

        # Posterior distribution over discrete future choices: P(z_{post} | h, h')
        self.transition_posterior_network = torch.nn.DataParallel(
            mlp(encoding_size + encoding_size, fc_posterior_layers, self.n_futures)
        )

    def dynamics(self, encoded_state, action):
        """
        NOTE: this should only be called during inference, 
              it returns the prior transition distribution

        NOTE: We augment this function to return a categorical distribution over next states

        Returns:
            transition_logits (list): transition logits
            next_encoded_state  (list): next states
            reward (list): reward associated with each next state
        """
        # both [N, hidden_state_dim, n_futures]
        next_encoded_states, rewards = self._dynamics_all_futures(encoded_state, action)  

        transition_logits = self.transition_prior_network(encoded_state)  # [N, n_futures]

        # dynamics should only be called in inference, with batch size 1
        assert next_encoded_states.dim() == 3
        assert next_encoded_states.shape[0] == 1
        assert rewards.shape[0] == 1
        assert transition_logits.shape[0] == 1

        # TODO: This currently violates the interface, we have the future dim at -1\
        #       Need to transpose(-1, 0, 1, ...) or change the interface
        # NOTE: fixed
        next_encoded_states = next_encoded_states.permute(2, 0, 1)  # [n_futures, N, hidden_state_dim]
        rewards = rewards.permute(2, 0, 1)  # [n_futures, N, support_dim]
        transition_logits = transition_logits.permute(1, 0)  # [n_futures, N]

        return transition_logits, next_encoded_states, rewards

    def _dynamics_all_futures(self, encoded_state, action):
        N, _ = encoded_state.shape
        # TODO: could probably be more efficient by parallelizing
        next_encoded_state_list, reward_list = [], []
        for future_ind in range(self.n_futures):
            future_one_hot = integer_to_one_hot(future_ind, self.n_futures)  # [n_futures]
            future_one_hot = future_one_hot.to(encoded_state.device).unsqueeze(0).expand(N, -1)  # [N, n_futures]
            next_encoded_state, reward = self._dynamics(encoded_state, action, future_one_hot)
            next_encoded_state_list.append(next_encoded_state)
            reward_list.append(reward)

        next_encoded_states = torch.stack(next_encoded_state_list, dim=-1)
        rewards = torch.stack(reward_list, dim=-1)

        return next_encoded_states, rewards

    def _dynamics(self, encoded_state, action, future_one_hot):
        """
        This is identical to original implementation. 
        Only change is indexing into module dict to select the specified dynamic network
        """
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = (
            torch.zeros((action.shape[0], self.action_space_size))
            .to(action.device)
            .float()
        )
        action_one_hot.scatter_(1, action.long(), 1.0)
        x = torch.cat((encoded_state, action_one_hot, future_one_hot), dim=1)

        next_encoded_state = self.dynamics_encoded_state_network(x)

        reward = self.dynamics_reward_network(next_encoded_state)

        # Scale encoded state between [0, 1] (See paper appendix Training)
        min_next_encoded_state = next_encoded_state.min(1, keepdim=True)[0]
        max_next_encoded_state = next_encoded_state.max(1, keepdim=True)[0]
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized = (
            next_encoded_state - min_next_encoded_state
        ) / scale_next_encoded_state

        return next_encoded_state_normalized, reward

    def recurrent_inference(self, encoded_state, action, future_encoded_state):
        """
        NOTE: This is only called during training, so we use the posterior transition distribution
        NOTE: In inference, we should call prediction and dynamics separately, as we have modified in stochastic MCTS
        NOTE: We also need to change the interface, to return the prior and posterior transition distribution
              such that we can take KL divergence loss on them
        NOTE: we need to future encoded state to get posterior distribution ove transitions

        NOTE: interface is changed to include two more in return:
            - transition_logits_post
            - transition_logits_prior

        This is for taking the KL divergence loss
        """
        N, encoding_size = encoded_state.shape

        transition_logits_prior = self.transition_prior_network(encoded_state)  # [N, 3]
        transition_logits_post = self.transition_posterior_network(torch.cat((encoded_state, future_encoded_state), dim=-1))  # [N, 3]

        # TODO: implement straight through (or find an existing implementation)
        # transition_sample_post = sample_straight_through(logits=transition_logits_post)
        # NOTE: found native pytorch impl 
        dist = OneHotCategoricalStraightThrough(logits=transition_logits_post)
        transition_sample_post = dist.rsample()  # [N, n_futures]

        next_encoded_state, reward = self._dynamics(encoded_state, action, transition_sample_post)  # [N, encoding_size], [N, support_dim]

        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state, transition_logits_post, transition_logits_prior


class MockMuZeroStochastic(MuZeroFullyConnectedNetwork):
    def dynamics(self, encoded_state, action):
        """
        We augment this function to return a categorical distribution over next states

        Returns:
            probs (list): probability of next state
            next_encoded_state  (list): next states
            reward (list): reward associated with each next state
        """
        next_encoded_state, reward = super().dynamics(encoded_state, action)

        return [1.0], [next_encoded_state], [reward]

    def recurrent_inference(self, encoded_state, action):
        _, next_encoded_states, rewards = self.dynamics(encoded_state, action)
        next_encoded_state, reward = next_encoded_states[0], rewards[0]
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state

###### End Stochastic #######
##################################


##################################
############# ResNet #############


def conv3x3(in_channels, out_channels, stride=1):
    return torch.nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


# Residual block
class ResidualBlock(torch.nn.Module):
    def __init__(self, num_channels, stride=1):
        super().__init__()
        self.conv1 = conv3x3(num_channels, num_channels, stride)
        self.bn1 = torch.nn.BatchNorm2d(num_channels)
        self.conv2 = conv3x3(num_channels, num_channels)
        self.bn2 = torch.nn.BatchNorm2d(num_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.nn.functional.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = torch.nn.functional.relu(out)
        return out


# Downsample observations before representation network (See paper appendix Network Architecture)
class DownSample(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.resblocks1 = torch.nn.ModuleList(
            [ResidualBlock(out_channels // 2) for _ in range(2)]
        )
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.resblocks2 = torch.nn.ModuleList(
            [ResidualBlock(out_channels) for _ in range(3)]
        )
        self.pooling1 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks3 = torch.nn.ModuleList(
            [ResidualBlock(out_channels) for _ in range(3)]
        )
        self.pooling2 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        for block in self.resblocks1:
            x = block(x)
        x = self.conv2(x)
        for block in self.resblocks2:
            x = block(x)
        x = self.pooling1(x)
        for block in self.resblocks3:
            x = block(x)
        x = self.pooling2(x)
        return x


class DownsampleCNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, h_w):
        super().__init__()
        mid_channels = (in_channels + out_channels) // 2
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, mid_channels, kernel_size=h_w[0] * 2, stride=4, padding=2
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d(h_w)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x


class RepresentationNetwork(torch.nn.Module):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        num_blocks,
        num_channels,
        downsample,
    ):
        super().__init__()
        self.downsample = downsample
        if self.downsample:
            if self.downsample == "resnet":
                self.downsample_net = DownSample(
                    observation_shape[0] * (stacked_observations + 1)
                    + stacked_observations,
                    num_channels,
                )
            elif self.downsample == "CNN":
                self.downsample_net = DownsampleCNN(
                    observation_shape[0] * (stacked_observations + 1)
                    + stacked_observations,
                    num_channels,
                    (
                        math.ceil(observation_shape[1] / 16),
                        math.ceil(observation_shape[2] / 16),
                    ),
                )
            else:
                raise NotImplementedError('downsample should be "resnet" or "CNN".')
        self.conv = conv3x3(
            observation_shape[0] * (stacked_observations + 1) + stacked_observations,
            num_channels,
        )
        self.bn = torch.nn.BatchNorm2d(num_channels)
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

    def forward(self, x):
        if self.downsample:
            x = self.downsample_net(x)
        else:
            x = self.conv(x)
            x = self.bn(x)
            x = torch.nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)
        return x


class DynamicsNetwork(torch.nn.Module):
    def __init__(
        self,
        num_blocks,
        num_channels,
        reduced_channels_reward,
        fc_reward_layers,
        full_support_size,
        block_output_size_reward,
    ):
        super().__init__()
        self.conv = conv3x3(num_channels, num_channels - 1)
        self.bn = torch.nn.BatchNorm2d(num_channels - 1)
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels - 1) for _ in range(num_blocks)]
        )

        self.conv1x1_reward = torch.nn.Conv2d(
            num_channels - 1, reduced_channels_reward, 1
        )
        self.block_output_size_reward = block_output_size_reward
        self.fc = mlp(
            self.block_output_size_reward, fc_reward_layers, full_support_size,
        )

        # NOTE(kwong): reuse reward network hyperparameters, since we won't
        # be tuning terminal-specific hyperparameters anyways.
        self.fc_terminal = mlp(
            self.block_output_size_reward, fc_reward_layers, 1,
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.nn.functional.relu(x)
        for block in self.resblocks:
            x = block(x)
        state = x
        x = self.conv1x1_reward(x)
        x = x.view(-1, self.block_output_size_reward)
        reward = self.fc(x)
        terminal = self.fc_terminal(x)
        return state, reward, terminal


class PredictionNetwork(torch.nn.Module):
    def __init__(
        self,
        action_space_size,
        num_blocks,
        num_channels,
        reduced_channels_value,
        reduced_channels_policy,
        fc_value_layers,
        fc_policy_layers,
        full_support_size,
        block_output_size_value,
        block_output_size_policy,
    ):
        super().__init__()
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

        self.conv1x1_value = torch.nn.Conv2d(num_channels, reduced_channels_value, 1)
        self.conv1x1_policy = torch.nn.Conv2d(num_channels, reduced_channels_policy, 1)
        self.block_output_size_value = block_output_size_value
        self.block_output_size_policy = block_output_size_policy
        self.fc_value = mlp(
            self.block_output_size_value, fc_value_layers, full_support_size
        )
        self.fc_policy = mlp(
            self.block_output_size_policy, fc_policy_layers, action_space_size,
        )

    def forward(self, x):
        for block in self.resblocks:
            x = block(x)
        value = self.conv1x1_value(x)
        policy = self.conv1x1_policy(x)
        value = value.view(-1, self.block_output_size_value)
        policy = policy.view(-1, self.block_output_size_policy)
        value = self.fc_value(value)
        policy = self.fc_policy(policy)
        return policy, value


class ReconstructionNetwork(torch.nn.Module):
    def __init__(
        self,
        observation_shape,
        num_blocks,
        num_channels,
        reduced_channels,
        fc_reconstruction_layers,
        block_output_size,
    ):
        super().__init__()
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

        self.conv1x1 = torch.nn.Conv2d(num_channels, reduced_channels, 1)
        self.upsample = torch.nn.Upsample(block_output_size, mode="bilinear")
        self.conv_reconstruction = cnn3x3(
            reduced_channels, fc_reconstruction_layers, observation_shape[0]
        )

    def forward(self, x):
        for block in self.resblocks:
            x = block(x)
        reconstruction = self.conv1x1(x)
        reconstruction = self.upsample(reconstruction)
        reconstruction = self.conv_reconstruction(reconstruction)
        return reconstruction


class MuZeroResidualNetwork(AbstractNetwork):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        action_space_size,
        num_blocks,
        num_channels,
        reduced_channels_reward,
        reduced_channels_value,
        reduced_channels_policy,
        reduced_channels_reconstruction,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        fc_reconstruction_layers,
        support_size,
        downsample,
    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1
        block_output_size_reward = (
            (
                reduced_channels_reward
                * math.ceil(observation_shape[1] / 16)
                * math.ceil(observation_shape[2] / 16)
            )
            if downsample
            else (reduced_channels_reward * observation_shape[1] * observation_shape[2])
        )

        block_output_size_value = (
            (
                reduced_channels_value
                * math.ceil(observation_shape[1] / 16)
                * math.ceil(observation_shape[2] / 16)
            )
            if downsample
            else (reduced_channels_value * observation_shape[1] * observation_shape[2])
        )

        block_output_size_policy = (
            (
                reduced_channels_policy
                * math.ceil(observation_shape[1] / 16)
                * math.ceil(observation_shape[2] / 16)
            )
            if downsample
            else (reduced_channels_policy * observation_shape[1] * observation_shape[2])
        )

        block_output_size_reconstruction = (
            (math.ceil(observation_shape[1] / 16), math.ceil(observation_shape[2] / 16))
            if downsample
            else (observation_shape[1], observation_shape[2])
        )

        self.representation_network = torch.nn.DataParallel(
            RepresentationNetwork(
                observation_shape,
                stacked_observations,
                num_blocks,
                num_channels,
                downsample,
            )
        )

        self.dynamics_network = torch.nn.DataParallel(
            DynamicsNetwork(
                num_blocks,
                num_channels + 1,
                reduced_channels_reward,
                fc_reward_layers,
                self.full_support_size,
                block_output_size_reward,
            )
        )

        self.prediction_network = torch.nn.DataParallel(
            PredictionNetwork(
                action_space_size,
                num_blocks,
                num_channels,
                reduced_channels_value,
                reduced_channels_policy,
                fc_value_layers,
                fc_policy_layers,
                self.full_support_size,
                block_output_size_value,
                block_output_size_policy,
            )
        )

        self.reconstruction_network = torch.nn.DataParallel(
            ReconstructionNetwork(
                observation_shape,
                num_blocks,
                num_channels,
                reduced_channels_reconstruction,
                fc_reconstruction_layers,
                block_output_size_reconstruction,
            )
        )

    def freeze_dynamics(self):
        for param in self.representation_network.parameters():
            param.requires_grad_(False)
        for param in self.dynamics_network.parameters():
            param.requires_grad_(False)
        for param in self.reconstruction_network.parameters():
            param.requires_grad_(False)

    def prediction(self, encoded_state):
        policy, value = self.prediction_network(encoded_state)
        return policy, value

    def reconstruction(self, encoded_state):
        reconstruction = self.reconstruction_network(encoded_state)
        return reconstruction

    def representation(self, observation):
        encoded_state = self.representation_network(observation)

        # Scale encoded state between [0, 1] (See appendix paper Training)
        min_encoded_state = (
            encoded_state.view(
                -1,
                encoded_state.shape[1],
                encoded_state.shape[2] * encoded_state.shape[3],
            )
            .min(2, keepdim=True)[0]
            .unsqueeze(-1)
        )
        max_encoded_state = (
            encoded_state.view(
                -1,
                encoded_state.shape[1],
                encoded_state.shape[2] * encoded_state.shape[3],
            )
            .max(2, keepdim=True)[0]
            .unsqueeze(-1)
        )
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (
            encoded_state - min_encoded_state
        ) / scale_encoded_state
        return encoded_state_normalized

    def dynamics(self, encoded_state, action):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = (
            torch.ones(
                (
                    encoded_state.shape[0],
                    1,
                    encoded_state.shape[2],
                    encoded_state.shape[3],
                )
            )
            .to(action.device)
            .float()
        )
        action_one_hot = (
            action[:, :, None, None] * action_one_hot / self.action_space_size
        )
        x = torch.cat((encoded_state, action_one_hot), dim=1)
        next_encoded_state, reward, terminal = self.dynamics_network(x)

        # Scale encoded state between [0, 1] (See paper appendix Training)
        min_next_encoded_state = (
            next_encoded_state.view(
                -1,
                next_encoded_state.shape[1],
                next_encoded_state.shape[2] * next_encoded_state.shape[3],
            )
            .min(2, keepdim=True)[0]
            .unsqueeze(-1)
        )
        max_next_encoded_state = (
            next_encoded_state.view(
                -1,
                next_encoded_state.shape[1],
                next_encoded_state.shape[2] * next_encoded_state.shape[3],
            )
            .max(2, keepdim=True)[0]
            .unsqueeze(-1)
        )
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized = (
            next_encoded_state - min_next_encoded_state
        ) / scale_next_encoded_state
        return next_encoded_state_normalized, reward, terminal[..., 0]

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)
        reconstruction = self.reconstruction(encoded_state)
        # reward equal to 0 for consistency
        reward = torch.log(
            (
                torch.zeros(1, self.full_support_size)
                .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
                .repeat(len(observation), 1)
                .to(observation.device)
            )
        )
        terminal = torch.zeros((len(observation),), device=observation.device)

        return (
            value,
            reward,
            terminal,
            policy_logits,
            reconstruction,
            encoded_state,
        )

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward, terminal = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        reconstruction = self.reconstruction(next_encoded_state)
        return value, reward, terminal, policy_logits, reconstruction, next_encoded_state


########### End ResNet ###########
##################################


def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=torch.nn.Identity,
    activation=torch.nn.ELU,
):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
    return torch.nn.Sequential(*layers)


def cnn3x3(
    input_size,
    layer_sizes,
    output_size,
    output_activation=torch.nn.Identity,
    activation=torch.nn.ELU,
):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Conv2d(sizes[i], sizes[i + 1], kernel_size=3, padding=1, stride=1), act()]
    return torch.nn.Sequential(*layers)


def support_to_scalar(logits, support_size):
    """
    Transform a categorical representation to a scalar
    See paper appendix Network Architecture
    """
    # Decode to a scalar
    probabilities = torch.softmax(logits, dim=1)
    support = (
        torch.tensor([x for x in range(-support_size, support_size + 1)])
        .expand(probabilities.shape)
        .float()
        .to(device=probabilities.device)
    )
    x = torch.sum(support * probabilities, dim=1, keepdim=True)

    # Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (
        ((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
        ** 2
        - 1
    )
    return x


def scalar_to_support(x, support_size):
    """
    Transform a scalar to a categorical representation with (2 * support_size + 1) categories
    See paper appendix Network Architecture
    """
    # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x

    # Encode on a vector
    x = torch.clamp(x, -support_size, support_size)
    floor = x.floor()
    prob = x - floor
    logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).to(x.device)
    logits.scatter_(
        2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
    )
    indexes = floor + support_size + 1
    prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
    indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
    logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
    return logits


def integer_to_one_hot(value, n_classes):
    tensor = torch.zeros((n_classes,))
    tensor[value] = 1.
    return tensor