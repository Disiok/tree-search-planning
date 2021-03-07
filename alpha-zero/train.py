r"""Toy AlphaZero implementation for HighwayEnv.

In this implementation, we spawn several actors whose job is to asynchronously
collect trajectories based on the latest policy/value network checkpoint.
These trajectories are returned to the trainer who updates the network and
then broadcasts the latest checkpoint to the actors.

Note that we do not implement asynchronous MCTS or batched inference.

Some todos:
    - Save checkpoints.
    - Add visualizations.
    - Log more information.
    - Tune hyperparameters + verify that it works.

The architecture of this code is based on:
    https://github.com/deepmind/open_spiel/
"""

import argparse
import copy
import itertools
import random
import traceback
from typing import Dict, List, NamedTuple, Optional, Tuple

import highway_env
import gym
import multiprocessing as mp
import numpy as np
import torch


Connection = mp.connection.Connection
PolicyValue = Tuple[torch.FloatTensor, torch.FloatTensor]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def deepcopy_env(env: gym.Env) -> gym.Env:
    r"""Deep copy the environment. Copied from rl_agents.

    Args:
        env (gym.Env): Gym environment to copy.

    Returns:
        (gym.Env): Copied Gym environment.
    """
    cls = env.__class__
    result = cls.__new__(cls)
    memo = {id(env): result}
    for k, v in env.__dict__.items():
        if k not in ['viewer', 'automatic_rendering_callback', 'automatic_record_callback', 'grid_render']:
            if isinstance(v, gym.Env):
                setattr(result, k, deepcopy_env(v))
            else:
                setattr(result, k, copy.deepcopy(v, memo=memo))
        else:
            setattr(result, k, None)
    return result


class Config(NamedTuple):
    # environment
    env_name: str = "highway-v0"
    discount: float = 1.
    l2_weight: float = 1.
    temperature: float = 1.

    # network
    hidden_sizes: List[int] = (32, 32)

    # mcts
    num_rollouts: int = 25
    max_steps_per_episode: Optional[int] = None

    c_uct: float = 1.
    d_eps: float = 0.25
    d_alpha: float = 0.25

    # training
    lr: float = 1e-2
    batch_size: int = 32

    buffer_size: int = 128
    buffer_turnover: float = 1.

    # logging
    log_frequency: int = 1


class MLP(torch.nn.Module):
    r"""MLP-based policy/value net.

    Args:
        env    (gym.Env): Gym environment.
        config (Config):  Model configuration.
    """
    def __init__(self, env: gym.Env, config: Config) -> None:
        super(MLP, self).__init__()
        self.num_actions = env.action_space.n

        net = []
        num_inputs = np.product(env.observation_space.shape)
        for hidden_size in config.hidden_sizes:
            net.extend([
                torch.nn.Linear(num_inputs, hidden_size),
                torch.nn.ReLU(inplace=True),
            ])
            num_inputs = hidden_size

        self.net = torch.nn.Sequential(*net)
        self.value_net = torch.nn.Linear(num_inputs, 1)
        self.prior_net = torch.nn.Linear(num_inputs, self.num_actions)

    def random(self) -> PolicyValue:
        value = torch.zeros((1,), device=self.device)
        prior = torch.ones((1, self.num_actions), device=self.device)
        prior = prior / torch.sum(prior)
        return (prior, value)

    def forward(self, xs: torch.FloatTensor, return_prob: bool = False) -> PolicyValue:
        xs = self.net(torch.flatten(xs, start_dim=1))
        value = self.value_net(xs)[..., 0]
        prior = torch.nn.functional.log_softmax(self.prior_net(xs), dim=-1)
        if return_prob:
            prior = torch.exp(prior)
        return (prior, value)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


def loss_fn(
    net: torch.nn.Module,
    observations: torch.FloatTensor,
    target_priors: torch.FloatTensor,
    target_values: torch.FloatTensor,
    l2_weight: float,
) -> torch.FloatTensor:
    r"""Compute AlphaZero policy/value matching loss.

    Args:
        net           (torch.nn.Module):   Policy/value net.
        observations  (torch.FloatTensor): [B x N x F] kinematic observations.
        target_priors (torch.FloatTensor): [B x N x num_actions] MCTS policies.
        target_values (torch.FloatTensor): [B x N x 1] values.
        lambda_       (float):             L2 regularization weight.

    Returns:
        (torch.FloatTensor): Total loss.
    """
    priors, values = net(observations)
    value_loss = torch.nn.functional.mse_loss(values, target_values)
    prior_loss = torch.mean(torch.sum(target_priors * priors, dim=-1))
    regularization = sum(torch.norm(x) for x in net.parameters())
    return value_loss - prior_loss + l2_weight * regularization


class MCTSNode(object):
    r"""MCTS node.

    Args:
        parent (Optional[MCTSNode]): Parent node of the current node.
        prior  (float):              Prior probability of this node's (state, action) pair.
    """
    def __init__(self, parent: Optional["MCTSNode"] = None, prior: float = 0.) -> None:
        super(MCTSNode, self).__init__()
        self.prior = prior
        self.value = 0.
        self.count = 0

        self.parent = parent
        self.children = None

    @property
    def q(self) -> float:
        return self.value / float(self.count) if self.count > 0 else 0.

    def add_child(self, child: "MCTSNode") -> None:
        if self.children is None:
            self.children = []
        self.children.append(child)

    def add_visit(self, value: float) -> None:
        self.value += value
        self.count += 1

    def __repr__(self) -> str:
        description = [
            f"q: {self.q:.2f}",
            f"prior: {self.prior:.2f}",
            f"count: {self.count:d}",
            f"value: {self.value:.2f}",
        ]

        if self.parent is not None:
            description.append(f"puct: {puct(self, 1.):.2f}")
            description.append(f"parent count: {self.parent.count:d}")

        return f"<MCTSNode {' '.join(description)}>"


def puct(node: MCTSNode, c: float) -> float:
    return node.q + c * np.sqrt(float(node.parent.count)) / float(1 + node.count)


class MCTSTree(object):
    r"""MCTS tree.

    Args:
        env     (gym.Env): Gym environment.
        net     (MLP):     Policy/value net.
        c_uct   (float):   PUCT exploration constant.
        d_eps   (float):   Dirichlet mixture weight.
        d_alpha (flato):   Dirichlet concentration.
    """
    def __init__(
        self,
        env: gym.Env,
        net: MLP,
        c_uct: float,
        d_eps: float,
        d_alpha: float,
        discount: float,
    ) -> None:
        super(MCTSTree, self).__init__()
        self.env = env
        self.net = net
        self.root = None

        self.c_uct = c_uct
        self.d_eps = d_eps
        self.d_alpha = d_alpha
        self.discount = discount

    def reset(self, observation: np.ndarray) -> None:
        observation = torch.from_numpy(observation).to(self.net.device, torch.float)
        with torch.no_grad():
            prior, value = self.net(observation[None], return_prob=True)
        noise = np.random.dirichlet([self.d_alpha] * self.env.action_space.n)

        self.root = MCTSNode()
        for action in range(self.env.action_space.n):
            psa = (1 - self.d_eps) * prior[0, action].item() + self.d_eps * noise[action].item()
            self.root.add_child(MCTSNode(self.root, psa))

    def simulate(self) -> None:
        node = self.root
        state = deepcopy_env(self.env)

        # selection
        rewards = []
        terminal = False
        while node.children is not None and not terminal:
            action = np.argmax([puct(x, self.c_uct) for x in node.children])
            observation, reward, terminal, _ = state.step(action)
            node = node.children[action]
            rewards.append(reward.item())

        # expand and evaluate
        if not terminal:
            observation = torch.from_numpy(observation).to(self.net.device, torch.float)
            with torch.no_grad():
                prior, value = self.net(observation[None], return_prob=True)
            for action_index in range(state.action_space.n):
                node.add_child(MCTSNode(node, prior[0, action_index].item()))
            value = value[0].item()
        else:
            value = 0.

        # backup
        for t in range(len(rewards) - 1, -1, -1):
            value = rewards[t] + self.discount * value
            node.add_visit(value)
            node = node.parent

        # visit root
        self.root.add_visit(0.)

    def step(self, action_index: int) -> None:
        self.root = self.root.children[action_index]
        self.root.parent = None

    def get_policy(self, temperature: float = 1.) -> np.ndarray:
        counts = np.array([x.count for x in self.root.children], dtype=np.float32)
        if temperature != 1.:
            counts = np.power(counts, 1. / temperature)
        return counts / max(np.sum(counts), 1.)

    def __repr__(self) -> str:
        if self.root is None:
            return f"<MCTSTree root: None>"

        description = [f"policy: {self.get_policy()}"]
        for child in self.root.children:
            description.append(f"  {repr(child)}")
        description = "\n".join(description)
        return f"<MCTSTree {description}>"


class Transition(NamedTuple):
    observation: np.ndarray
    action: int
    reward: float
    prior: np.ndarray
    value: Optional[float] = None


def run_episode_with_mcts(env: gym.Env, net: MLP, config: Config) -> Transition:
    r"""Collect trajectory from one episode using MCTS.

    Args:
        env    (gym.Env): Gym environment.
        net    (MLP):     Policy/value net.
        config (Config):  Experiment configuration.

    Returns:
        (List[Transition]): A trajectory from one episode.
    """
    tree = MCTSTree(env, net, config.c_uct, config.d_eps, config.d_alpha, config.discount)
    observation = env.reset()
    tree.reset(observation)

    trajectory = []
    for step in itertools.count(1):
        # run MCTS simulation
        num_rollouts = config.num_rollouts - tree.root.count
        for _ in range(max(num_rollouts, 0)):
            tree.simulate()

        # sample from tree policy
        action = np.argmax(tree.get_policy(config.temperature))
        next_observation, reward, terminal, _ = env.step(action)
        tree.step(action)

        # update trajectory with new transition
        transition = Transition(observation, action, reward.item(), tree.get_policy())
        trajectory.append(transition)
        observation = next_observation

        if terminal:
            break

        if (
            config.max_steps_per_episode is not None
            and step >= config.max_steps_per_episode
        ):
            break

    value = 0.
    for t in range(len(trajectory) - 1, -1, -1):
        observation, action, reward, prior, _ = trajectory[t]
        value = reward + config.discount * value
        trajectory[t] = Transition(observation, action, reward, prior, value)

    return trajectory


def Actor(
    conn: Connection,
    config: Config,
    *,
    seed: int = 42,
    device: Optional[torch.device] = None,
    state_dict: Optional[Dict] = None,
) -> None:
    r"""Asynchronously collect trajectories using MCTS.

    Args:
        conn       (Connection):             Connection to trainer.
        config     (Config):                 Experiment configuration.
        seed       (int):                    Random seed.
        device     (Optional[torch.device]): Policy/value net device.
        state_dict (Optional[Dict]):         Policy/value net initialization.
    """
    set_seed(seed)
    env = gym.make(config.env_name)
    net = MLP(env, config).to(device)
    if state_dict is not None:
        net.load_state_dict(state_dict)

    while True:
        if conn.poll():
            state_dict = conn.recv()
            if state_dict is None:
                break
            net.load_state_dict(state_dict)
        else:
            trajectory = run_episode_with_mcts(env, net, config)
            conn.send(trajectory)


class BatchTransition(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    priors: np.ndarray
    values: np.ndarray


class ReplayBuffer(object):
    r"""FIFO replay buffer.

    Args:
        maxlen (int): Maximum size of the buffer.
    """
    def __init__(self, maxlen: int) -> None:
        super(ReplayBuffer, self).__init__()
        self.index = -1
        self.buffer = []
        self.maxlen = maxlen

    def sample(self, batch_size: int) -> BatchTransition:
        batch_index = np.random.choice(len(self.buffer), batch_size)
        transitions = [self.buffer[i] for i in batch_index]

        observations = np.stack([x.observation for x in transitions], axis=0)
        actions = np.array([x.action for x in transitions])
        rewards = np.array([x.reward for x in transitions])
        priors = np.stack([x.prior for x in transitions], axis=0)
        values = np.array([x.value for x in transitions])

        batch = BatchTransition(observations, actions, rewards, priors, values)
        return batch

    def append(self, transition: Transition) -> None:
        self.index = (self.index + 1) % self.maxlen
        if len(self.buffer) <= self.index:
            self.buffer.append(None)
        self.buffer[self.index] = transition

    def extend(self, transitions: List[Transition]) -> None:
        for transition in transitions:
            self.append(transition)

    def __len__(self) -> int:
        return max(self.buffer, self.index + 1)


def Trainer(
    actors: List[Tuple[mp.Process, Connection]],
    config: Config,
    *,
    seed: int = 42,
    device: Optional[torch.device] = None,
    state_dict: Optional[Dict] = None,
) -> None:
    r"""Train model for an indefinite number of episodes.

    Args:
        actors     (List[Tuple[mp.Process, Connection]]): Connections to each actor.
        config     (Config):                              Experiment configuration.
        seed       (int):                                 Random seed.
        device     (Optional[torch.device]):              Policy/value net device.
        state_dict (Optional[Dict]):                      Policy/value net initialization.
    """
    set_seed(seed)
    env = gym.make(config.env_name)
    net = MLP(env, config).to(device)
    if state_dict is not None:
        net.load_state_dict(state_dict)

    optimizer = torch.optim.SGD(net.parameters(), lr=config.lr)
    replay_buffer = ReplayBuffer(config.buffer_size)

    def get_transitions(num_transitions: int) -> List[Transition]:
        transitions = []
        while len(transitions) < num_transitions:
            for (_, conn) in actors:
                if conn.poll():
                    trajectory = conn.recv()
                    transitions.extend(trajectory)
                if len(transitions) >= num_transitions:
                    break
        return transitions[:num_transitions]

    for epoch in itertools.count(1):
        # collect transitions into replay buffer
        num_transitions = int(config.buffer_size * config.buffer_turnover)
        replay_buffer.extend(get_transitions(num_transitions))

        # train policy/value net
        total_loss = 0.
        num_training_steps = config.buffer_size // config.batch_size
        for step in range(num_training_steps):
            batch = replay_buffer.sample(config.batch_size)
            loss = loss_fn(
                net,
                torch.from_numpy(batch.observations).to(net.device, torch.float),
                torch.from_numpy(batch.priors).to(net.device, torch.float),
                torch.from_numpy(batch.values).to(net.device, torch.float),
                config.l2_weight,
            )
            loss.backward()
            optimizer.step()

            if step % config.log_frequency == 0:
                print(
                    f"Train Epoch {epoch} "
                    f"[{step}/{num_training_steps}]: "
                    f"Loss - {loss.item():.2f} "
                )

        # evaluate new agent
        trajectory = run_episode_with_mcts(env, net, config)
        print(
            f"Evaluation Epoch {epoch} "
            f"Value - {trajectory[0].value:.2f} "
            f"Duration - {len(trajectory):d} "
        )

        # broadcast new parameters
        state_dict = net.state_dict()
        for (_, conn) in actors:
            conn.send(state_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-name", type=str, default="highway-v0")
    parser.add_argument("--num-actors", type=int, default=4)
    parser.add_argument("--use-cuda", action="store_true", default=False)
    parser.add_argument("--output-path", type=str, default=None)

    args = parser.parse_args()
    print(args)

    set_seed(args.seed)
    config = Config(args.env_name)
    device = torch.device("cuda") if args.use_cuda else None

    # set up actors
    actors = []
    for i in range(args.num_actors):
        head, tail = mp.Pipe(True)
        process = mp.Process(
            target=Actor,
            args=(tail, config),
            kwargs=dict(device=device, seed=args.seed + i)
        )
        actors.append((process, head))

    for (process, _) in actors:
        process.start()

    # start training
    try:
        Trainer(actors, config, device=device, seed=args.seed)
    except (KeyboardInterrupt, EOFError):
        print("Caught interrupt; cleaning up.")
    except:
        traceback.print_exc()
    finally:
        for (process, conn) in actors:
            conn.send(None)
            while conn.poll():
                _ = conn.recv()
            process.join(0.01)  # 10ms

        print("Done.")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()

