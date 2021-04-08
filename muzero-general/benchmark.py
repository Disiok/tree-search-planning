import copy
import argparse
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


def evaluate_muzero(env, checkpoint_path, n_episodes):
    # Initialize MuZero
    muzero = MuZero(env)
    muzero.load_model(checkpoint_path=checkpoint_path)

    result = muzero.test(render=False, opponent='self', muzero_player=None, num_tests=n_episodes, save_gif=False)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='highway_env_ttc')
    parser.add_argument('--checkpoint_path', type=str, default='/h/suo/dev/tree-search-planning/muzero-general/results/highway_env/2021-03-14--22-39-32/model.checkpoint')
    parser.add_argument('--n_episodes', type=int, default=50)
    args = parser.parse_args()

    evaluate_muzero(args.env, args.checkpoint_path, args.n_episodes)
    ray.shutdown()