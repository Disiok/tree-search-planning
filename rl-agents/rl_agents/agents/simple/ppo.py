import logging
import torch

from rl_agents.agents.common.memory import ReplayMemory, Transition
from rl_agents.agents.common.abstract import AbstractAgent
from rl_agents.agents.common.models import model_factory, size_model_config, trainable_parameters
from rl_agents.agents.common.optimizers import loss_function_factory, optimizer_factory
from rl_agents.agents.common.utils import choose_device


logger = logging.getLogger(__name__)


class PPOAgent(AbstractAgent):

    batched = True  # run batch episodes

    def __init__(self, env, config=None) -> None:
        super(PPOAgent, self).__init__(config)
        self.env = env
        self.config = config

        size_model_config(self.env, self.config["model"])
        self.device = choose_device(self.config["device"])
        self.policy_net = model_factory(self.config["model"])
        self.policy_net.to(self.device)

        self.rollout_buffer = ReplayMemory(self.config)

    @classmethod
    def default_config(cls):
        return dict(model=dict(type="MultiLayerPerceptron"),
                    device="cuda:best")
                    # optimizer=dict(type="ADAM",
                    #                lr=5e-4,
                    #                weight_decay=0,
                    #                k=5),
                    # loss_function="l2",
                    # memory_capacity=50000,
                    # batch_size=100,
                    # gamma=0.99,
                    # device="cuda:best",
                    # exploration=dict(method="EpsilonGreedy"),
                    # target_update=1,
                    # double=True)

    def record(self, state, action, reward, next_state, done, info):
        """
            Record a transition of the environment to update the agent
        :param state: s, the current state of the agent
        :param action: a, the action performed
        :param reward: r(s, a), the reward collected
        :param next_state: s', the new state of the agent after the action was performed
        :param done: whether the next state is terminal
        :return:
        """
        logging.info("Debug!")
        pass

    def act(self, state):
        """
            Pick an action

        :param state: s, the current state of the agent
        :return: a, the action to perform
        """
        return 0

    def reset(self):
        """
            Reset the agent to its initial internal state
        """
        pass

    def seed(self, seed=None):
        """
            Seed the agent's random number generator
        :param seed: the seed to be used to generate random numbers
        :return: the used seed
        """
        pass

    def save(self, filename):
        """
            Save the model parameters to a file
        :param str filename: the path of the file to save the model parameters in
        """
        return filename

    def load(self, filename):
        """
            Load the model parameters from a file
        :param str filename: the path of the file to load the model parameters from
        """
        # checkpoint = torch.load(filename, map_location=self.device)
        # self.policy_net.load_state_dict
        return filename

    def eval(self):
        """
            Set to testing mode. Disable any unnecessary exploration.
        """
        pass
