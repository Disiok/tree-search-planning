# Installation for Development
`conda create -n tsmp python=3.7`  
`conda activate tsmp`  
`conda install -c conda-forge git-lfs`

In `highway-env`: 
`python3 setup.py develop`  
In `rl-agents`: 
`python3 setup.py develop`  
In `muzero-general`: 
`pip install -r requirements.txt`

For logging to Weights and Biases
`wandb login`

# Experiments

## Training the models in the writeup

### MuZero
- All MuZero experiments are run from within `muzero-general`, our fork of https://github.com/werner-duvaud/muzero-general with all our modifications.
- Set the environment variables `RAY_TEMP_DIR` and `WANDB_USERNAME` for Ray and Weights and Biases `init`.

Each setting defines its own game file under `muzero-general/games`. All training are run by `python muzero_interactive GAME_NAME`.
Relevant game files are the following (by environment):

#### HighwayEnv
- MuZero: `highway_env_ttc_flat`

#### RoundaboutEnv
- MuZero: `roundabout_env_ttc_flat`
- Stochastic MuZero: `roundabout_env_ttc_flat_stochastic_concat`
- Risk-Sensitive MuZero: `roundabout_env_ttc_flat_risk_sensitive`

#### CrossMergeEnv


## Benchmark