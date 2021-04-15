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
    if len(sys.argv) == 2:
        # Train directly with "python muzero.py cartpole"
        print(f'Training kicked off for environment {sys.argv[1]}')
        muzero = MuZero(sys.argv[1])
        print(f'Saving results to {muzero.config.results_path}')
        muzero.train()
    elif len(sys.argv) == 3:
        # Train directly with "python muzero.py cartpole path_to_checkpoint"
        print(f'Training kicked off for environment {sys.argv[1]}')
        muzero = MuZero(sys.argv[1])

        print(f"Loading checkpoint from {sys.argv[2]}")
        muzero.load_model(checkpoint_path=sys.argv[2])

        print(f'Saving results to {muzero.config.results_path}')
        muzero.train()
    elif len(sys.argv) == 4:
        # Train directly with "python muzero.py cartpole path_to_checkpoint path_to_buffer"
        print(f'Training kicked off for environment {sys.argv[1]}')
        muzero = MuZero(sys.argv[1])

        print(f"Loading checkpoint from {sys.argv[2]} and {sys.argv[3]}")
        muzero.load_model(checkpoint_path=sys.argv[2], replay_buffer_path=sys.argv[3])

        print(f'Saving results to {muzero.config.results_path}')
        muzero.train()
    else:
        print("\nWelcome to MuZero! Here's a list of games:")
        # Let user pick a game
        games = [
            filename[:-3]
            for filename in sorted(
                os.listdir(os.path.dirname(os.path.realpath(__file__)) + "/games")
            )
            if filename.endswith(".py") and filename != "abstract_game.py"
        ]
        for i in range(len(games)):
            print(f"{i}. {games[i]}")
        choice = input("Enter a number to choose the game: ")
        valid_inputs = [str(i) for i in range(len(games))]
        while choice not in valid_inputs:
            choice = input("Invalid input, enter a number listed above: ")

        # Initialize MuZero
        choice = int(choice)
        game_name = games[choice]
        muzero = MuZero(game_name)

        while True:
            # Configure running options
            options = [
                "Train",
                "Load pretrained model",
                "Diagnose model",
                "Render some self play games",
                "Play against MuZero",
                "Test the game manually",
                "Save GIFs for self play games",
                "Exit",
            ]
            print()
            for i in range(len(options)):
                print(f"{i}. {options[i]}")

            choice = input("Enter a number to choose an action: ")
            valid_inputs = [str(i) for i in range(len(options))]
            while choice not in valid_inputs:
                choice = input("Invalid input, enter a number listed above: ")
            choice = int(choice)
            if choice == 0:
                muzero.train()
            elif choice == 1:
                load_model_menu(muzero, game_name)
            elif choice == 2:
                muzero.diagnose_model(30)
            elif choice == 3:
                muzero.test(render=True, opponent="self", muzero_player=None)
            elif choice == 4:
                muzero.test(render=True, opponent="human", muzero_player=0)
            elif choice == 5:
                env = muzero.Game()
                env.reset()
                env.render()

                done = False
                while not done:
                    action = env.human_to_action()
                    observation, reward, done = env.step(action)
                    print(f"\nAction: {env.action_to_string(action)}\nReward: {reward}")
                    env.render()
            elif choice == 6:
                muzero.test(render=False, opponent="self", muzero_player=None, save_gif=True)
            else:
                break
            print("\nDone")

    ray.shutdown()
