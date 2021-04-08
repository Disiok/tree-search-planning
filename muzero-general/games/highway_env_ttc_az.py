

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
