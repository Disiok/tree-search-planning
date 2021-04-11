import argparse
import datetime
import pathlib

import gym
import highway_env
import numpy as np
from rl_agents.trainer.monitor import MonitorV2
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# ==================================
#     Environment configuration
# ==================================


def get_env_kwargs(env_name):
    if env_name == "highway":
        env_kwargs = {
            'id': 'highway-v0',
            'config': {
                'observation': {
                    'type': 'FlatTimeToCollisionWithEgoVelocity',
                    'horizon': 10,
                    'num_lanes': 3,
                    'num_speeds': 5,
                    'fixed_velocity_grid': True,
                },
                'action': {'type': 'DiscreteMetaAction'},
                'simulation_frequency': 15,
                'policy_frequency': 1,
                'other_vehicles_type': 'highway_env.vehicle.behavior.IDMVehicle',
                'screen_width': 600,
                'screen_height': 150,
                'centering_position': [0.3, 0.5],
                'scaling': 5.5,
                'show_trajectories': False,
                'render_agent': True,
                'offscreen_rendering': False,
                'manual_control': False,
                'real_time_rendering': False,
                'lanes_count': 4,
                'controlled_vehicles': 1,
                'initial_lane_id': None,
                'duration': 40,
                'ego_spacing': 2,
                'vehicles_density': 1,
                'collision_reward': -1,
                'reward_speed_range': [20, 30],
                'offroad_terminal': False
            }
        }
    elif env_name == "roundabout":
        env_kwargs = {
            "id": "roundabout-v0",
            "config": {
                'observation': {
                    'type': 'FlatTimeToCollisionWithEgoVelocity',
                    'horizon': 10,
                    'num_lanes': 3,
                    'num_speeds': 5,
                    'project_speed': False,
                    'fixed_velocity_grid': True,
                },
                "action": {
                    "type": "DiscreteMetaAction"
                },
                "incoming_vehicle_destination": None,
                "duration": 11,  # [s]
                "simulation_frequency": 15,  # [Hz]
                "policy_frequency": 1,  # [Hz]
                "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "screen_width": 600,  # [px]
                "screen_height": 600,  # [px]
                "centering_position": [0.5, 0.6],
                "scaling": 5.5,
                "show_trajectories": False,
                "render_agent": True,
                "offscreen_rendering": False
            }
        }
    else:
        raise ValueError(f"{env_name} is not supported.")
    return env_kwargs


def make_configure_env(**kwargs):
    env = gym.make(kwargs["id"])
    env.configure(kwargs["config"])
    env.reset()
    return env


# ==================================
#        Main script
# ==================================

def _ppo(args):
    env_kwargs = get_env_kwargs(args.env_name)
    output_path = pathlib.Path(args.output_path) / args.env_name

    if args.train:
        env = make_vec_env(
            make_configure_env,
            n_envs=args.num_cpus,
            seed=args.seed,
            vec_env_cls=SubprocVecEnv,
            env_kwargs=env_kwargs
        )

        model = PPO(
            "MlpPolicy",
            env,
            seed=args.seed,
            n_steps=args.buffer_size // args.num_cpus,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            verbose=2,
            tensorboard_log=output_path / "tensorboard",
        )
        model.learn(total_timesteps=args.num_transitions, log_interval=args.log_interval)
        model.save(output_path / "model_checkpoint")

    evaluation_path = output_path / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

    model = PPO.load(output_path / "model_checkpoint", seed=args.seed)
    env = make_configure_env(**env_kwargs)
    env = MonitorV2(env, evaluation_path, video_callable=False)

    for _ in range(args.num_evaluation_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)

    print("done!")


def _a2c(args):
    env_kwargs = get_env_kwargs(args.env_name)
    output_path = pathlib.Path(args.output_path) / args.env_name

    if args.train:
        env = make_vec_env(
            make_configure_env,
            n_envs=args.num_cpus,
            seed=args.seed,
            vec_env_cls=SubprocVecEnv,
            env_kwargs=env_kwargs
        )

        model = A2C(
            "MlpPolicy",
            env,
            seed=args.seed,
            n_steps=args.buffer_size // args.num_cpus,
            learning_rate=args.lr,
            verbose=2,
            tensorboard_log=output_path / "tensorboard",
        )
        model.learn(total_timesteps=args.num_transitions, log_interval=args.log_interval)
        model.save(output_path / "model_checkpoint")

    evaluation_path = output_path / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

    model = A2C.load(output_path / "model_checkpoint", seed=args.seed)
    env = make_configure_env(**env_kwargs)
    env = MonitorV2(env, evaluation_path, video_callable=False)

    for _ in range(args.num_evaluation_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)

    print("done!")


def _main():
    parser = argparse.ArgumentParser("Train A2C/PPO on highway-env.")
    subparser = parser.add_subparsers(dest="algo")

    # PPO
    ppo_parser = subparser.add_parser("ppo")
    ppo_parser.add_argument("--seed", type=int, default=42)
    ppo_parser.add_argument("--env-name", choices=["highway", "roundabout"], default="highway")
    ppo_parser.add_argument("--num-cpus", type=int, default=16)

    ppo_parser.add_argument("--output-path", type=str, default="./results/ppo2")
    ppo_parser.add_argument("--num-evaluation-episodes", type=int, default=50)

    ppo_parser.add_argument("--train", action="store_true", default=False)
    ppo_parser.add_argument("--batch-size", type=int, default=64)
    ppo_parser.add_argument("--buffer-size", type=int, default=512)
    ppo_parser.add_argument("--num-transitions", type=int, default=100000)
    ppo_parser.add_argument("--lr", type=float, default=1e-3)
    ppo_parser.add_argument("--log-interval", type=int, default=1)

    # A2C
    a2c_parser = subparser.add_parser("a2c")
    a2c_parser.add_argument("--seed", type=int, default=42)
    a2c_parser.add_argument("--env-name", choices=["highway", "roundabout"], default="highway")
    a2c_parser.add_argument("--num-cpus", type=int, default=16)

    a2c_parser.add_argument("--output-path", type=str, default="./results/a2c2")
    a2c_parser.add_argument("--num-evaluation-episodes", type=int, default=50)

    a2c_parser.add_argument("--train", action="store_true", default=False)
    a2c_parser.add_argument("--buffer-size", type=int, default=512)
    a2c_parser.add_argument("--num-transitions", type=int, default=100000)
    a2c_parser.add_argument("--lr", type=float, default=1e-3)
    a2c_parser.add_argument("--log-interval", type=int, default=1)

    args = parser.parse_args()
    print(args)

    if args.algo == "ppo":
        _ppo(args)
    elif args.algo == "a2c":
        _a2c(args)
    else:
        raise ValueError(f"{args.algo} is not supported.")


if __name__ == "__main__":
    _main()
