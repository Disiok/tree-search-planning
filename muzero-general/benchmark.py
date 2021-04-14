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


def evaluate_muzero(env, checkpoint_path, n_episodes, num_gpus):
    # Initialize MuZero
    muzero = MuZero(env)
    if checkpoint_path is not None:
        muzero.load_model(checkpoint_path=checkpoint_path)

    result = muzero.test(render=False, opponent='self', muzero_player=None, num_tests=n_episodes, save_gif=False, num_gpus=num_gpus)
    ray.shutdown()

    return result


def evaluate(model, env, checkpoint_path, n_episodes, num_gpus):
    if model == 'muzero':
        result = evaluate_muzero(env, checkpoint_path, n_episodes, num_gpus)
    elif model == 'alphazero':
        raise NotImplementedError('TODO: Hook up alphazero from Kelvin')
    else: 
        raise NotImplementedError('TODO: Merge in rl-agent experiments.py')

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='muzero')
    parser.add_argument('--env', type=str, default='highway_env')
    parser.add_argument('--checkpoint_path', type=str, default='/h/suo/dev/tree-search-planning/muzero-general/results/highway_env/2021-03-14--22-39-32/model.checkpoint')
    parser.add_argument('--n_episodes', type=int, default=50)
    parser.add_argument('--num_gpus', type=int, default=1)
    args = parser.parse_args()

    result = evaluate(args.model, args.env, args.checkpoint_path, args.n_episodes, args.num_gpus)
    print(result)