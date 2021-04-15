import copy
import argparse
import importlib
import math
import os
import pathlib
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


def evaluate_muzero(env, checkpoint_path, n_episodes, num_gpus, output_path, policy_only, uniform_policy, num_simulations):
    # Initialize MuZero
    muzero = MuZero(env)
    if checkpoint_path is not None:
        muzero.load_model(checkpoint_path=checkpoint_path)

    result = muzero.test(
        render=False,
        opponent='self',
        muzero_player=None,
        num_tests=n_episodes,
        save_gif=False,
        num_gpus=num_gpus,
        output_path=output_path,
        policy_only=policy_only,
        uniform_policy=uniform_policy,
        num_simulations=num_simulations,
    )
    ray.shutdown()

    return result


def evaluate(model, env, checkpoint_path, n_episodes, num_gpus, output_path, policy_only, uniform_policy, num_simulations):
    if model == 'muzero':
        result = evaluate_muzero(env, checkpoint_path, n_episodes, num_gpus, output_path, policy_only, uniform_policy, num_simulations)
    elif model == 'alphazero':
        # NOTE(kwong): no need to do so; works with muzero.
        raise NotImplementedError('TODO: Hook up alphazero from Kelvin')
    else: 
        raise NotImplementedError('TODO: Merge in rl-agent `experiments.py`')

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='muzero')
    parser.add_argument('--env', type=str, default='highway_env')
    parser.add_argument('--checkpoint_path', type=str, default='/h/suo/dev/tree-search-planning/muzero-general/results/highway_env/2021-03-14--22-39-32/model.checkpoint')
    parser.add_argument('--n_episodes', type=int, default=5)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--output_path', type=str, default='/scratch/ssd002/home/kelvin/projects/mcts_planner/tsp/evaluation/')
    parser.add_argument('--policy-only', action='store_true', default=False)
    parser.add_argument('--uniform-policy', action='store_true', default=False)
    parser.add_argument('--num-simulations', type=int, default=None)
    args = parser.parse_args()

    output_path = pathlib.Path(args.output_path)
    if args.policy_only:
        output_path = output_path / f"{args.env}_policy_only"
    elif args.uniform_policy:
        output_path = output_path / f"{args.env}_no_policy"
    elif args.num_simulations is not None:
        output_path = output_path / f"{args.env}_{args.num_simulations}"
    else:
        output_path = output_path / args.env

    result = evaluate(
        args.model, args.env, args.checkpoint_path, args.n_episodes, args.num_gpus, output_path, args.policy_only, args.uniform_policy, args.num_simulations
    )
    print(result)