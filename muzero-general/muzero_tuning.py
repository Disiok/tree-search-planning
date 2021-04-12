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



def hyperparameter_search(
    game_name, parametrization, budget, parallel_experiments, num_tests
):
    """
    Search for hyperparameters by launching parallel experiments.

    Args:
        game_name (str): Name of the game module, it should match the name of a .py file
        in the "./games" directory.

        parametrization : Nevergrad parametrization, please refer to nevergrad documentation.

        budget (int): Number of experiments to launch in total.

        parallel_experiments (int): Number of experiments to launch in parallel.

        num_tests (int): Number of games to average for evaluating an experiment.
    """
    # optimizer = nevergrad.optimizers.OnePlusOne(
    #     parametrization=parametrization, budget=budget
    # )
    optimizer = nevergrad.optimizers.RandomSearch(
        parametrization=parametrization, budget=budget
    )

    running_experiments = []
    best_training = None
    try:
        # Launch initial experiments
        for i in range(parallel_experiments):
            if 0 < budget:
                param = optimizer.ask()
                print(f"Launching new experiment: {param.value}")
                muzero = MuZero(game_name, param.value, parallel_experiments, remote_logging=True)
                muzero.param = param
                muzero.train(True)
                running_experiments.append(muzero)
                budget -= 1

        while 0 < budget or any(running_experiments):
            for i, experiment in enumerate(running_experiments):
                if experiment and experiment.config.training_steps <= ray.get(
                    experiment.shared_storage_worker.get_info.remote("training_step")
                ):
                    experiment.terminate_workers()
                    result = experiment.test(False, num_tests=num_tests)
                    if not best_training or best_training["result"] < result:
                        best_training = {
                            "result": result,
                            "config": experiment.config,
                            "checkpoint": experiment.checkpoint,
                        }
                    print(f"Parameters: {experiment.param.value}")
                    print(f"Result: {result}")
                    optimizer.tell(experiment.param, -result)

                    if 0 < budget:
                        param = optimizer.ask()
                        print(f"Launching new experiment: {param.value}")
                        muzero = MuZero(game_name, param.value, parallel_experiments)
                        muzero.param = param
                        muzero.train(True)
                        running_experiments[i] = muzero
                        budget -= 1
                    else:
                        running_experiments[i] = None

    except KeyboardInterrupt:
        for experiment in running_experiments:
            if isinstance(experiment, MuZero):
                experiment.terminate_workers()

    recommendation = optimizer.provide_recommendation()
    print("Best hyperparameters:")
    print(recommendation.value)
    if best_training:
        # Save best training weights (but it's not the recommended weights)
        os.makedirs(best_training["config"].results_path, exist_ok=True)
        torch.save(
            best_training["checkpoint"],
            os.path.join(best_training["config"].results_path, "model.checkpoint"),
        )
        # Save the recommended hyperparameters
        text_file = open(
            os.path.join(best_training["config"].results_path, "best_parameters.txt"),
            "w",
        )
        text_file.write(str(recommendation.value))
        text_file.close()
    return recommendation.value


if __name__ == "__main__":

    # Initialize MuZero
    game_name = sys.argv[1]
    muzero = MuZero(game_name)

    # Define here the parameters to tune
    # Parametrization documentation: https://facebookresearch.github.io/nevergrad/parametrization.html
    muzero.terminate_workers()
    del muzero
    budget = 30
    parallel_experiments = 10
    num_tests = 40
    parametrization = nevergrad.p.Dict(
        lr_init=nevergrad.p.Log(lower=0.005, upper=0.01),
        # lr_decay_rate=nevergrad.p.Scalar(lower=0.1, upper=0.9),
        # lr_decay_steps = nevergrad.p.Log(lower=5e3, upper=3e4).set_integer_casting(),
        discount=nevergrad.p.Scalar(lower=0.95, upper=0.9999),
        td_steps=nevergrad.p.Scalar(lower=1, upper=10).set_integer_casting(),
        num_simulations=nevergrad.p.Log(lower=1, upper=50).set_integer_casting(),
        # checkpoint_interval=nevergrad.p.Log(lower=10, upper=1000).set_integer_casting(),
        # batch_size=nevergrad.p.Choice([256, 512]),
        # PER=nevergrad.p.Choice([True, False]),
        # PER_alpha=nevergrad.p.Scalar(lower=0.0, upper=1.0),
        value_loss_weight=nevergrad.p.Scalar(lower=0.25, upper=1.0),
        # optimizer=nevergrad.p.Choice(['Adam', 'SGD']),
        # use_last_model_value=nevergrad.p.Choice([True, False]),
        # stacked_observations=nevergrad.p.Choice([1, 5]),
        # ratio=nevergrad.p.Choice([None, 1.5]),
        # support_size=nevergrad.p.Choice([10, 300]),
        num_unroll_steps=nevergrad.p.Scalar(lower=1, upper=15).set_integer_casting(),
        # weight_decay=nevergrad.p.Choice([0.0, 1e-4]),
        root_dirichlet_alpha = nevergrad.p.Scalar(lower=0.0, upper=1.0),
        root_exploration_fraction = nevergrad.p.Scalar(lower=0.0, upper=1.0),
        # support_size = nevergrad.p.Choice([2, 5, 10, 50, 100]),
    )

    print("Searching hyperparameters")
    best_hyperparameters = hyperparameter_search(
        game_name, parametrization, budget, parallel_experiments, num_tests
    )
    muzero = MuZero(game_name, best_hyperparameters)
    print("\nDone")

    ray.shutdown()
