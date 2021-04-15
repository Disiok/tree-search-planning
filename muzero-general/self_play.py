import copy
import math
import time

import gym
import numpy
import ray
import torch

import models


@ray.remote
class SelfPlay:
    """
    Class which run in a dedicated thread to play games and save them to the replay-buffer.
    """

    def __init__(self, initial_checkpoint, Game, config, seed):
        self.config = config

        if hasattr(config, "cfg_file"):
            self.game = Game(seed=seed, cfg_file=config.cfg_file)
        else:
            self.game = Game(seed=seed)

        # Fix random generator seed
        numpy.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize the network
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.eval()

    def continuous_self_play(self, shared_storage, replay_buffer, test_mode=False):
        while ray.get(
            shared_storage.get_info.remote("training_step")
        ) < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            self.model.set_weights(ray.get(shared_storage.get_info.remote("weights")))

            if not test_mode:
                game_history = self.play_game(
                    self.config.visit_softmax_temperature_fn(
                        trained_steps=ray.get(
                            shared_storage.get_info.remote("training_step")
                        )
                    ),
                    self.config.temperature_threshold,
                    False,
                    "self",
                    0,
                )
                replay_buffer.save_game.remote(game_history, shared_storage)

            else:
                # Take the best action (no exploration) in test mode
                game_history = self.play_game(
                    0,
                    self.config.temperature_threshold,
                    False,
                    "self" if len(self.config.players) == 1 else self.config.opponent,
                    self.config.muzero_player,
                )

                # Save to the shared storage
                shared_storage.set_info.remote(
                    {
                        "episode_length": len(game_history.action_history) - 1,
                        "total_reward": sum(game_history.reward_history),
                        "mean_value": numpy.mean(
                            [value for value in game_history.root_values if value]
                        ),
                    }
                )
                if 1 < len(self.config.players):
                    shared_storage.set_info.remote(
                        {
                            "muzero_reward": sum(
                                reward
                                for i, reward in enumerate(game_history.reward_history)
                                if game_history.to_play_history[i - 1]
                                == self.config.muzero_player
                            ),
                            "opponent_reward": sum(
                                reward
                                for i, reward in enumerate(game_history.reward_history)
                                if game_history.to_play_history[i - 1]
                                != self.config.muzero_player
                            ),
                        }
                    )

            # Managing the self-play / training ratio
            if not test_mode and self.config.self_play_delay:
                time.sleep(self.config.self_play_delay)
            if not test_mode and self.config.ratio:
                while (
                    ray.get(shared_storage.get_info.remote("training_step"))
                    / max(
                        1, ray.get(shared_storage.get_info.remote("num_played_steps"))
                    )
                    < self.config.ratio
                    and ray.get(shared_storage.get_info.remote("training_step"))
                    < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    time.sleep(0.5)

        self.close_game()

    def play_game(
        self, temperature, temperature_threshold, render, opponent, muzero_player, save_gif=False
    ):
        """
        Play one game with actions based on the Monte Carlo tree search at each moves.
        """
        game_history = GameHistory()
        observation = self.game.reset()
        game_history.action_history.append(0)
        game_history.observation_history.append(observation)
        game_history.reward_history.append(0)
        game_history.to_play_history.append(self.game.to_play())

        done = False

        if render:
            self.game.render()

        if save_gif:
            self.game.render_rgb()

        with torch.no_grad():
            while (
                not done and len(game_history.action_history) <= self.config.max_moves
            ):
                assert (
                    len(numpy.array(observation).shape) == 3
                ), f"Observation should be 3 dimensionnal instead of {len(numpy.array(observation).shape)} dimensionnal. Got observation of shape: {numpy.array(observation).shape}"
                assert (
                    numpy.array(observation).shape == self.config.observation_shape
                ), f"Observation should match the observation_shape defined in MuZeroConfig. Expected {self.config.observation_shape} but got {numpy.array(observation).shape}."
                stacked_observations = game_history.get_stacked_observations(
                    -1,
                    self.config.stacked_observations,
                )

                # Choose the action
                if opponent == "self" or muzero_player == self.game.to_play():
                    if hasattr(self.config, "mcts_pretrain") and self.config.mcts_pretrain:
                        root, mcts_info = VMCTS(self.config).run(
                            self.game,
                            self.game.legal_actions(),
                            self.game.to_play(),
                            True,
                        )
                    elif hasattr(self.config, "dynamics_model") and self.config.dynamics_model == "perfect":
                        root, mcts_info = AZMCTS(self.config).run(
                            self.model,
                            self.game,
                            game_history,
                            self.game.legal_actions(),
                            self.game.to_play(),
                            True,
                        )
                    else:
                        root, mcts_info = MCTS(self.config).run(
                            self.model,
                            stacked_observations,
                            self.game.legal_actions(),
                            self.game.to_play(),
                            True,
                        )

                    action = self.select_action(
                        root,
                        temperature
                        if not temperature_threshold
                        or len(game_history.action_history) < temperature_threshold
                        else 0,
                    )

                    if render:
                        print(f'Tree depth: {mcts_info["max_tree_depth"]}')
                        print(
                            f"Root value for player {self.game.to_play()}: {root.value():.2f}"
                        )
                else:
                    action, root = self.select_opponent_action(
                        opponent, stacked_observations, game_history
                    )

                observation, reward, done = self.game.step(action)
        
                if render:
                    print(f"Played action: {self.game.action_to_string(action)}")
                    self.game.render()
                
                if save_gif:
                    self.game.render_rgb()

                game_history.store_search_statistics(root, self.config.action_space)

                # Next batch
                game_history.action_history.append(action)
                game_history.observation_history.append(observation)
                game_history.reward_history.append(reward)
                game_history.to_play_history.append(self.game.to_play())
                # NOTE(suo): keep track of how many times we interacted with the real environment
                #            mcts_info['n_env_interactions'] is for environment interactions during MCTS for AlphaZero
                #            extra + 1 for stepping with the final selected action
                game_history.n_env_interactions_history.append(mcts_info['n_env_interactions'] + 1)

        if save_gif:
            self.game.save_gif()

        return game_history

    def close_game(self):
        self.game.close()

    def select_opponent_action(self, opponent, stacked_observations, game_history):
        """
        Select opponent action for evaluating MuZero level.
        """
        if opponent == "human":
            if hasattr(self.config, "mcts_pretrain") and self.config.mcts_pretrain:
                root, mcts_info = VMCTS(self.config).run(
                    self.game,
                    self.game.legal_actions(),
                    self.game.to_play(),
                    True,
                )
            elif hasattr(self.config, "dynamics_model") and self.config.dynamics_model == "perfect":
                root, mcts_info = AZMCTS(self.config).run(
                    self.model,
                    self.game,
                    game_history,
                    self.game.legal_actions(),
                    self.game.to_play(),
                    True,
                )
            else:
                root, mcts_info = MCTS(self.config).run(
                    self.model,
                    stacked_observations,
                    self.game.legal_actions(),
                    self.game.to_play(),
                    True,
                )
            print(f'Tree depth: {mcts_info["max_tree_depth"]}')
            print(f"Root value for player {self.game.to_play()}: {root.value():.2f}")
            print(
                f"Player {self.game.to_play()} turn. MuZero suggests {self.game.action_to_string(self.select_action(root, 0))}"
            )
            return self.game.human_to_action(), root
        elif opponent == "expert":
            return self.game.expert_agent(), None
        elif opponent == "random":
            assert (
                self.game.legal_actions()
            ), f"Legal actions should not be an empty array. Got {self.game.legal_actions()}."
            assert set(self.game.legal_actions()).issubset(
                set(self.config.action_space)
            ), "Legal actions should be a subset of the action space."

            return numpy.random.choice(self.game.legal_actions()), None
        else:
            raise NotImplementedError(
                'Wrong argument: "opponent" argument should be "self", "human", "expert" or "random"'
            )

    @staticmethod
    def select_action(node, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function
        in the config.
        """
        visit_counts = numpy.array(
            [child.visit_count for child in node.children.values()], dtype="int32"
        )
        actions = [action for action in node.children.keys()]
        if temperature == 0:
            action = actions[numpy.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = numpy.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = numpy.random.choice(actions, p=visit_count_distribution)

        return action


# Game independent
class MCTS:
    """
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """

    def __init__(self, config):
        self.config = config

    def run(
        self,
        model,
        observation,
        legal_actions,
        to_play,
        add_exploration_noise,
        override_root_with=None,
        policy_only=False,
        uniform_policy=False,
    ):
        """
        At the root of the search tree we use the representation function to obtain a
        hidden state given the current observation.
        We then run a Monte Carlo Tree Search using only action sequences and the model
        learned by the network.
        """
        if override_root_with:
            root = override_root_with
            root_predicted_value = None
        else:
            root = Node(0)
            observation = (
                torch.tensor(observation)
                .float()
                .unsqueeze(0)
                .to(next(model.parameters()).device)
            )
            (
                root_predicted_value,
                reward,
                _,
                policy_logits,
                _,
                hidden_state,
            ) = model.initial_inference(observation)
            root_predicted_value = models.support_to_scalar(
                root_predicted_value, self.config.support_size
            ).item()
            reward = models.support_to_scalar(reward, self.config.support_size).item()
            assert (
                legal_actions
            ), f"Legal actions should not be an empty array. Got {legal_actions}."
            assert set(legal_actions).issubset(
                set(self.config.action_space)
            ), "Legal actions should be a subset of the action space."

            if uniform_policy:
                policy_logits.zero_()

            root.expand(
                legal_actions,
                to_play,
                reward,
                policy_logits,
                hidden_state,
            )

        if add_exploration_noise:
            root.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction,
            )

        min_max_stats = MinMaxStats()

        max_tree_depth = 0
        num_simulations = 0 if policy_only else self.config.num_simulations 
        for _ in range(num_simulations):
            virtual_to_play = to_play
            node = root
            search_path = [node]
            current_tree_depth = 0

            while node.expanded():
                current_tree_depth += 1
                action, node = self.select_child(node, min_max_stats)
                search_path.append(node)

                # Players play turn by turn
                if virtual_to_play + 1 < len(self.config.players):
                    virtual_to_play = self.config.players[virtual_to_play + 1]
                else:
                    virtual_to_play = self.config.players[0]

            # Inside the search tree we use the dynamics function to obtain the next hidden
            # state given an action and the previous hidden state
            parent = search_path[-2]
            value, reward, terminal, policy_logits, reconstruction, hidden_state = model.recurrent_inference(
                parent.hidden_state,
                torch.tensor([[action]]).to(parent.hidden_state.device),
            )
            value = models.support_to_scalar(value, self.config.support_size).item()
            reward = models.support_to_scalar(reward, self.config.support_size).item()
            is_terminal = terminal.item() >= 0.  # hard threshold to determine terminal state

            if uniform_policy:
                policy_logits.zero_()

            # only expand node if we're not at a terminal state
            # or if we don't actually use the is_terminal prediction
            if is_terminal and hasattr(self.config, "mask_absorbing_states") and self.config.mask_absorbing_states:
                self.backpropagate(search_path, 0., virtual_to_play, min_max_stats)
                max_tree_depth = max(max_tree_depth, current_tree_depth)
            else:
                node.expand(
                    self.config.action_space,
                    virtual_to_play,
                    reward,
                    policy_logits,
                    hidden_state,
                )
                self.backpropagate(search_path, value, virtual_to_play, min_max_stats)
                max_tree_depth = max(max_tree_depth, current_tree_depth)

        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": root_predicted_value,
            "n_env_interactions": 0  # MuZero does not interact with the environment during MCTS
        }
        return root, extra_info

    def select_child(self, node, min_max_stats):
        """
        Select the child with the highest UCB score.
        """
        max_ucb = max(
            self.ucb_score(node, child, min_max_stats)
            for action, child in node.children.items()
        )
        action = numpy.random.choice(
            [
                action
                for action, child in node.children.items()
                if self.ucb_score(node, child, min_max_stats) == max_ucb
            ]
        )
        return action, node.children[action]

    def ucb_score(self, parent, child, min_max_stats):
        """
        The score for a node is based on its value, plus an exploration bonus based on the prior.
        """
        pb_c = (
            math.log(
                (parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base
            )
            + self.config.pb_c_init
        )
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior

        if child.visit_count > 0:
            # Mean value Q
            value_score = min_max_stats.normalize(
                child.reward
                + self.config.discount
                * (child.value() if len(self.config.players) == 1 else -child.value())
            )
        else:
            value_score = 0

        return prior_score + value_score

    def backpropagate(self, search_path, value, to_play, min_max_stats):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        if len(self.config.players) == 1:
            for node in reversed(search_path):
                node.value_sum += value
                node.visit_count += 1
                min_max_stats.update(node.reward + self.config.discount * node.value())

                value = node.reward + self.config.discount * value

        elif len(self.config.players) == 2:
            for node in reversed(search_path):
                node.value_sum += value if node.to_play == to_play else -value
                node.visit_count += 1
                min_max_stats.update(node.reward + self.config.discount * -node.value())

                value = (
                    -node.reward if node.to_play == to_play else node.reward
                ) + self.config.discount * value

        else:
            raise NotImplementedError("More than two player mode not implemented.")


class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, actions, to_play, reward, policy_logits, hidden_state):
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network.
        """
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state

        policy_values = torch.softmax(
            torch.tensor([policy_logits[0][a] for a in actions]), dim=0
        ).tolist()
        policy = {a: policy_values[i] for i, a in enumerate(actions)}
        for action, p in policy.items():
            self.children[action] = Node(p)

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        actions = list(self.children.keys())
        noise = numpy.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac


class GameHistory:
    """
    Store only usefull information of a self-play game.
    """

    def __init__(self):
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.to_play_history = []
        self.child_visits = []
        self.root_values = []
        self.reanalysed_predicted_root_values = None
        # For PER
        self.priorities = None
        self.game_priority = None
        self.n_env_interactions_history = []

    def store_search_statistics(self, root, action_space):
        # Turn visit count from root into a policy
        if root is not None:
            sum_visits = sum(child.visit_count for child in root.children.values())
            self.child_visits.append(
                [
                    root.children[a].visit_count / max(sum_visits, 1)
                    if a in root.children
                    else 0
                    for a in action_space
                ]
            )

            self.root_values.append(root.value())
        else:
            self.root_values.append(None)

    def get_stacked_observations(self, index, num_stacked_observations):
        """
        Generate a new observation with the observation at the index position
        and num_stacked_observations past observations and actions stacked.

        NOTE(kwong): This code is duplicated below. But we'll leave it here so
        that we don't introduce new bugs to the existing MuZero implementation.
        """
        # Convert to positive index
        index = index % len(self.observation_history)

        stacked_observations = self.observation_history[index].copy()
        for past_observation_index in reversed(
            range(index - num_stacked_observations, index)
        ):
            if 0 <= past_observation_index:
                previous_observation = numpy.concatenate(
                    (
                        self.observation_history[past_observation_index],
                        [
                            numpy.ones_like(stacked_observations[0])
                            * self.action_history[past_observation_index + 1]
                        ],
                    )
                )
            else:
                previous_observation = numpy.concatenate(
                    (
                        numpy.zeros_like(self.observation_history[index]),
                        [numpy.zeros_like(stacked_observations[0])],
                    )
                )

            stacked_observations = numpy.concatenate(
                (stacked_observations, previous_observation)
            )

        return stacked_observations


def _get_stacked_observations(observations, actions, index, num_stacked_observations):
    # Convert to positive index
    index = index % len(observations)

    stacked_observations = observations[index].copy()
    for past_observation_index in reversed(
        range(index - num_stacked_observations, index)
    ):
        if 0 <= past_observation_index:
            previous_observation = numpy.concatenate(
                (
                    observations[past_observation_index],
                    [
                        numpy.ones_like(stacked_observations[0])
                        * actions[past_observation_index + 1]
                    ],
                )
            )
        else:
            previous_observation = numpy.concatenate(
                (
                    numpy.zeros_like(observations[index]),
                    [numpy.zeros_like(stacked_observations[0])],
                )
            )

        stacked_observations = numpy.concatenate(
            (stacked_observations, previous_observation)
        )

    return stacked_observations



class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


def _safe_deepcopy_env(obj):
    """Return a deepcopy of the environmnet, but without copying its viewer."""
    cls = obj.__class__
    result = cls.__new__(cls)
    memo = {id(obj): result}
    for k, v in obj.__dict__.items():
        if k not in ['viewer', 'automatic_rendering_callback', 'automatic_record_callback', 'grid_render']:
            if isinstance(v, gym.Env):
                setattr(result, k, _safe_deepcopy_env(v))
            else:
                setattr(result, k, copy.deepcopy(v, memo=memo))
        else:
            setattr(result, k, None)
    return result


class AZMCTS(MCTS):
    r"""MCTS algorithm adapted to AlphaZero.
    
    NOTE(kwong): This class works for HighwayEnv only.
    """

    def run(
        self,
        model,
        game,
        game_history,
        legal_actions,
        to_play,
        add_exploration_noise,
        override_root_with=None,
        policy_only=False,
        uniform_policy=False,
    ):
        r"""Run MCTS for a number of simulations."""
        n_env_interactions = 0
        if override_root_with:
            root = override_root_with
            root_predicted_value = None
        else:
            root = Node(0)
            observation = _get_stacked_observations(
                game_history.observation_history,
                game_history.action_history,
                -1,
                self.config.stacked_observations
            )
            observation = (
                torch.tensor(observation)
                .float()
                .unsqueeze(0)
                .to(next(model.parameters()).device)
            )
            (
                _,
                _,
                _,
                policy_logits,
                _,
                _
            ) = model.initial_inference(observation)
            assert (
                legal_actions
            ), f"Legal actions should not be an empty array. Got {legal_actions}."
            assert set(legal_actions).issubset(
                set(self.config.action_space)
            ), "Legal actions should be a subset of the action space."

            if uniform_policy:
                policy_logits.zero_()

            hidden_state = (
                game.env.simplify(),  # NOTE(kwong): This works for highway-env only.
                game_history.observation_history[-1],
                game_history.action_history[-1],
                False
            )
            root.expand(
                legal_actions,
                to_play,
                0.,  # no rewards at the root node
                policy_logits,
                hidden_state,  # hidden state of the game
            )

        if add_exploration_noise:
            root.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction,
            )

        min_max_stats = MinMaxStats()

        max_tree_depth = 0
        num_simulations = 0 if policy_only else self.config.num_simulations
        for _ in range(num_simulations):
            node = root

            terminal = False
            search_path = [node]
            current_tree_depth = 0
            virtual_to_play = to_play

            while node.expanded() and not terminal:
                current_tree_depth += 1
                action, node = self.select_child(node, min_max_stats)
                if node.expanded():
                    _, _, _, terminal = node.hidden_state
                search_path.append(node)

                # Players play turn by turn
                if virtual_to_play + 1 < len(self.config.players):
                    virtual_to_play = self.config.players[virtual_to_play + 1]
                else:
                    virtual_to_play = self.config.players[0]

            if not terminal:
                parent_state, _, _, _ = search_path[-2].hidden_state
                parent_state_copy = _safe_deepcopy_env(parent_state)
                observation, reward, terminal, _ = parent_state_copy.step(action)
                next_state = parent_state_copy  # changed due to step above
                n_env_interactions += 1
                next_hidden_state = (next_state, observation, action, terminal)

                # first node in search_path (root) is already in observation_history
                # last node in search_path (current node) is not expanded yet
                observation = _get_stacked_observations(
                    game_history.observation_history + [n.hidden_state[1] for n in search_path[1:-1]] + [observation],
                    game_history.action_history + [n.hidden_state[2] for n in search_path[1:-1]] + [action],
                    -1,
                    self.config.stacked_observations
                )
                observation = (
                    torch.tensor(observation)
                    .float()
                    .unsqueeze(0)
                    .to(next(model.parameters()).device)
                )
                value, _, _, policy_logits, _, _ = model.initial_inference(observation)
                value = models.support_to_scalar(value, self.config.support_size).item()

                if uniform_policy:
                    policy_logits.zero_()

                node.expand(
                    self.config.action_space,
                    virtual_to_play,
                    reward,  # reward of action leading to this node
                    policy_logits,
                    next_hidden_state,  # new hidden state
                )
            else:
                value = 0.  # terminal states have no value

            self.backpropagate(search_path, value, virtual_to_play, min_max_stats)
            max_tree_depth = max(max_tree_depth, current_tree_depth)

        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": None,
            "n_env_interactions": n_env_interactions,
        }
        return root, extra_info


class VMCTS(MCTS):
    r"""Vanilla MCTS.

    NOTE(kwong): This class works for HighwayEnv only.
    """

    def run(
        self,
        game,
        legal_actions,
        to_play,
        add_exploration_noise,
        override_root_with=None,
    ):
        r"""Run MCTS for a number of simulations."""
        n_env_interactions = 0
        if override_root_with:
            root = override_root_with
            root_predicted_value = None
        else:
            root = Node(0)
            hidden_state = (game.env.simplify(), False)
            policy_logits = torch.zeros((1, len(self.config.action_space)))
            root.expand(
                legal_actions,
                to_play,
                0.,  # no rewards at the root
                policy_logits,
                hidden_state,
            )

        if add_exploration_noise:
            root.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction,
            )

        min_max_stats = MinMaxStats()

        max_tree_depth = 0
        for _ in range(self.config.num_simulations):
            node = root

            terminal = False
            search_path = [node]
            current_tree_depth = 0
            virtual_to_play = to_play

            while node.expanded() and not terminal:
                current_tree_depth += 1
                action, node = self.select_child(node, min_max_stats)
                if node.expanded():
                    _, terminal = node.hidden_state
                search_path.append(node)

                # Players play turn by turn
                if virtual_to_play + 1 < len(self.config.players):
                    virtual_to_play = self.config.players[virtual_to_play + 1]
                else:
                    virtual_to_play = self.config.players[0]

            if not terminal:
                parent_state, _ = search_path[-2].hidden_state
                parent_state_copy = _safe_deepcopy_env(parent_state)
                observation, reward, terminal, _ = parent_state_copy.step(action)
                next_state = parent_state_copy  # changed due to step above
                n_env_interactions += 1
                next_hidden_state = (next_state, terminal)

                policy_logits = torch.zeros((1, len(self.config.action_space)))
                value, more_env_interactions = estimate_value_via_rollout(next_state, self.config)
                n_env_interactions += more_env_interactions

                node.expand(
                    self.config.action_space,
                    virtual_to_play,
                    reward,  # reward of action leading to this node
                    policy_logits,
                    next_hidden_state,  # new hidden state
                )
            else:
                value = 0.  # terminal states have no value

            self.backpropagate(search_path, value, virtual_to_play, min_max_stats)
            max_tree_depth = max(max_tree_depth, current_tree_depth)

        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": None,
            "n_env_interactions": n_env_interactions,
        }
        return root, extra_info


def estimate_value_via_rollout(state, config):
    if config.num_rollout_steps is not None and config.num_rollout_steps == 0:
        return 0.0, 0  # don't copy environment if not necessary

    total_reward = 0.0
    num_env_interactions = 0
    state = _safe_deepcopy_env(state)

    step = 0
    while config.num_rollout_steps is None or step < config.num_rollout_steps:
        action = numpy.random.choice(config.action_space)
        _, reward, terminal, _ = state.step(action)
        num_env_interactions += 1
        total_reward += config.discount ** step * reward
        step += 1
        if terminal:
            break

    return total_reward, num_env_interactions
