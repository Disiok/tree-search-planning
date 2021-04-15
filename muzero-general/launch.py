import copy
import importlib
import math
import os
import pickle
import sys
import time
from glob import glob

import nevergrad
import numpy
import ray
import torch
from torch.utils.tensorboard import SummaryWriter

import diagnose_model
import models
import replay_buffer
import self_play
import shared_storage
import trainer

from muzero import MuZero


def load_model_menu(muzero, game_name):
    # Configure running options
    options = ["Specify paths manually"] + sorted(glob(f"results/{game_name}/*/"))
    options.reverse()
    print()
    for i in range(len(options)):
        print(f"{i}. {options[i]}")

    choice = input("Enter a number to choose a model to load: ")
    valid_inputs = [str(i) for i in range(len(options))]
    while choice not in valid_inputs:
        choice = input("Invalid input, enter a number listed above: ")
    choice = int(choice)

    if choice == (len(options) - 1):
        # manual path option
        checkpoint_path = input(
            "Enter a path to the model.checkpoint, or ENTER if none: "
        )
        while checkpoint_path and not os.path.isfile(checkpoint_path):
            checkpoint_path = input("Invalid checkpoint path. Try again: ")
        replay_buffer_path = input(
            "Enter a path to the replay_buffer.pkl, or ENTER if none: "
        )
        while replay_buffer_path and not os.path.isfile(replay_buffer_path):
            replay_buffer_path = input("Invalid replay buffer path. Try again: ")
    else:
        checkpoint_path = f"{options[choice]}model.checkpoint"
        replay_buffer_path = f"{options[choice]}replay_buffer.pkl"

    muzero.load_model(
        checkpoint_path=checkpoint_path, replay_buffer_path=replay_buffer_path,
    )


if __name__ == "__main__":
    # Train directly with "python muzero.py cartpole"
    env_name, env_cfg, exp_name = sys.argv[1:4]

    print(f'Training kicked off for environment {env_name}')
    muzero = MuZero(env_name, env_cfg_key=env_cfg, exp_name=exp_name)

    print(f'Saving results to {muzero.config.results_path}')
    muzero.train()
    
    
    ray.shutdown()
