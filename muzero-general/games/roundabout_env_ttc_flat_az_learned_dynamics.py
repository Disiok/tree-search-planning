import datetime
import os

from .roundabout_env_ttc_flat import Game
from .roundabout_env_ttc_flat import MuZeroConfig as _MuZeroConfig


class MuZeroConfig(_MuZeroConfig):
    def __init__(self):
        super().__init__()

        ### Training
        self.results_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../results", os.path.basename(__file__)[:-3],
            datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        )  # Path to store the model weights and TensorBoard logs

        # don't train the dynamics model!
        self.reward_loss_weight = 0.0
        self.terminal_loss_weight = 0.0
        self.reconstruction_loss_weight = 0.0
        self.mask_absorbing_states = True # whether to mask absorbing states' losses

        self.value_loss_weight = 0.75  # do not train the value function!
        self.policy_loss_weight = 1.0  # do not train the policy!
        self.freeze_dynamics = True