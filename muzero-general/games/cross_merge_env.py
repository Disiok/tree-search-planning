import datetime
import os

import gym
import numpy
import torch
import imageio

import highway_env

from .abstract_game import AbstractGame
from rl_agents.trainer.monitor import MonitorV2
from rl_agents.agents.common.factory import load_environment

NUM_LANES = 3
NUM_SPEEDS = 5
HORIZON = 10
FIXED_VELOCITY_GRID = True


two = {'dim': 423,
       'name': '2lane',
       'cfg': '2lane.json',
       'batch_size': 1024,
       'encoding_size': 512,
       'fc_representation_layers': [512, 256, 256],
       'fc_dynamics_layers': [256],  # Define the hidden layers in the dynamics network
       'training_steps': 30000,
       }

one = {'dim': 183,
       'name': '1lane',
       'cfg': '1lane.json'
       }

one_dense = {'dim': 183,
       'name': '1lane_dense',
       'cfg': '1lane_dense.json'
       }

one_sparse = {'dim': 183,
       'name': '1lane_sparse',
       'cfg': '1lane_sparse.json'
       }

one_aggro = {'dim': 183,
       'name': '1lane_aggro',
       'cfg': '1lane_aggro.json'
       }

cfg = {}
cfgs = {'one': one, 'two': two, 'one_dense': one_dense, 'one_sparse': one_sparse, 'one_aggro': one_aggro}


class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available



        ### Game
        #self.observation_shape = (1, 1, NUM_SPEEDS * NUM_LANES * HORIZON + NUM_SPEEDS)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.observation_shape = (1, 1, cfg['dim'])
        self.action_space = list(range(5))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(1))  # List of players. You should only edit the length
        self.stacked_observations = 1  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 12 # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 500  # Maximum number of moves if game is not finished before
        self.num_simulations = 50  # Number of future moves self-simulated
        self.discount = 0.995  # 0.975  # 0.997  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.5


        ### Network
        self.network = "fullyconnected"  # "resnet" / "fullyconnected"
        self.support_size = 3  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 3  # Number of blocks in the ResNet
        self.channels = 64  # Number of channels in the ResNet
        self.reduced_channels_reward = 32  # Number of channels in reward head
        self.reduced_channels_value = 32  # Number of channels in value head
        self.reduced_channels_policy = 32  # Number of channels in policy head
        self.reduced_channels_reconstruction = 32  # Number of channels in reconstruction head
        self.resnet_fc_reward_layers = [32]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [32]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [32]  # Define the hidden layers in the policy head of the prediction network
        self.resnet_fc_reconstruction_layers = [32]  # Define the hidden layers in the reconstruction head of the reconstruction network

        # Fully Connected Network
        self.encoding_size = 256
        self.fc_representation_layers = [256, 128, 128]
        self.fc_dynamics_layers = [128]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [32]  # Define the hidden layers in the reward network
        self.fc_value_layers = [32]  # Define the hidden layers in the value network
        self.fc_policy_layers = [32]  # Define the hidden layers in the policy network
        self.fc_reconstruction_layers = [32]  # Define the hidden layers in the reconstruction network



        ### Training
        # self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results", os.path.basename(__file__)[:-3], 'hyperparameter_search_normalized_random_dirichlet', datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        
        if 'exp_name' in cfg:
            self.results_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "../results", os.path.basename(__file__)[:-3],
                f'{cfg["exp_name"]}'
            )  # Path to store the model weights and TensorBoard logs
        else:
            self.results_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "../results", os.path.basename(__file__)[:-3],
                f'{cfg["name"]}_day3' + datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
            )  # Path to store the model weights and TensorBoard logs

        self.cfg_file = cfg['cfg']
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 10000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 512 # Number of parts of games to train on at each training step
        self.checkpoint_interval = 300  # 10  # Number of training steps before using the model for self-playing
        self.policy_loss_weight = 1.0  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.value_loss_weight = 1.0  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.reward_loss_weight = 1.0  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.terminal_loss_weight = 0.0  # Scale the terminal loss 
        self.reconstruction_loss_weight = 0.0  # Scale the reconstruction loss
        self.mask_absorbing_states = False # whether to mask absorbing states' losses
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 0. #1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 1e-3 # 0.0075  # Initial learning rate
        self.lr_decay_rate = 1 # 0.5  # 0.9  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 15000


        ### Replay Buffer
        self.replay_buffer_size = int(1e3)  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 5  # Number of game moves to keep for every batch element
        self.td_steps = 17  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 1.0  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False


        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it

        for attr in dir(self):
            if attr in cfg:
                print("Overwriting config." + attr)
                setattr(self, attr, cfg[attr])
        

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        end = 1.0
        beg = 5
        return end + (1 - trained_steps / self.training_steps) * beg

        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25


class Game(AbstractGame):
    """
    Game wrapper.
    """

    ENV_NAME = 'cross-merge-v0'

    def __init__(self, cfg_file='', seed=None, monitor_path=None):
        # self.env = gym.make(self.ENV_NAME)
        this_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Using config {cfg_file}")
        self.env = load_environment(os.path.join(this_dir, 'cross_merge_configs', cfg_file))
        
        if monitor_path is not None:
            self.env = MonitorV2(self.env, monitor_path, video_callable=False)
        
        self.env.reset()
        if seed is not None:
            self.env.seed(seed)

        self.gif_imgs = []

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done, _ = self.env.step(action)
        return observation, reward, done

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return list(range(5))

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def close(self):
        """
        Properly close the game.
        """
        self.env.close()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def render_rgb(self):
        rgb_img = self.env.render(mode='rgb_array')
        self.gif_imgs.append(rgb_img)
    
    def save_gif(self):
        imageio.mimsave(
            f'/Users/sergio/tree-search-planning/gifs/{datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")}.gif',
            self.gif_imgs,
            fps=5,
        )
        self.gif_imgs = []

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        actions = {
            0: 'LANE_LEFT',
            1: 'IDLE',
            2: 'LANE_RIGHT',
            3: 'FASTER',
            4: 'SLOWER'
        }
        return f"{action_number}. {actions[action_number]}"