import datetime
import os

from .highway_env_ttc import Game
from .highway_env_ttc import MuZeroConfig as _MuZeroConfig


class MuZeroConfig(_MuZeroConfig):
    def __init__(self):
        super().__init__()

        ### Self-Play
        self.dynamics_model = "perfect"
        self.stacked_observations = 1  # Number of previous observations and previous actions to add to the current observation

        ### Replay Buffer
        self.num_unroll_steps = 0  # Number of game moves to keep for every batch element

        ### Training
        self.results_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../results", os.path.basename(__file__)[:-3],
            datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        )  # Path to store the model weights and TensorBoard logs
