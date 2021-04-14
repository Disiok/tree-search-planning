import datetime
import os

from .cross_merge_env import Game
from .cross_merge_env import MuZeroConfig as _MuZeroConfig


class MuZeroConfig(_MuZeroConfig):
    def __init__(self):
        super().__init__()

        ### Training
        self.results_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../results", os.path.basename(__file__)[:-3],
            datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        )  # Path to store the model weights and TensorBoard logs

        self.terminal_loss_weight = 0.0
        self.reconstruction_loss_weight = 1.0
        self.mask_absorbing_states = False