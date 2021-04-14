import datetime
import os

from .cross_merge_env_1lane import Game
from .cross_merge_env_1lane import MuZeroConfig as _MuZeroConfig


class MuZeroConfig(_MuZeroConfig):
    def __init__(self):
        super().__init__()

        ### Training
        self.results_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../results", os.path.basename(__file__)[:-3],
            datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        )  # Path to store the model weights and TensorBoard logs

        self.reward_loss_weight = 5.0
        self.terminal_loss_weight = 1.0
        self.reconstruction_loss_weight = 1.0
        self.mask_absorbing_states = True # whether to mask absorbing states' losses

        self.value_loss_weight = 0.0  # do not train the value function!
        self.policy_loss_weight = 0.0  # do not train the policy!
        self.num_rollout_steps = 0  # this means we don't use value estimates to guide MCTS
        self.mcts_pretrain = True