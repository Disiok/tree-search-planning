import datetime
import os

from .highway_env_ttc_flat import Game
from .highway_env_ttc_flat import MuZeroConfig as _MuZeroConfig


class MuZeroConfig(_MuZeroConfig):
    def __init__(self):
        super().__init__()

        ### Training
        self.results_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../results", os.path.basename(__file__)[:-3],
            datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        )  # Path to store the model weights and TensorBoard logs

        self.terminal_loss_weight = 0.0  # Scale the terminal loss 
        self.reconstruction_loss_weight = 1.0  # Scale the reconstruction loss
        self.mask_absorbing_states = False  # whether to mask absorbing states' losses
