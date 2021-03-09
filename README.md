# Installation for Development

## Creating a new environment
`conda create -n tsmp python=3.7`  
`conda activate tsmp`  
In `highway-env`: 
`python3 setup.py develop`  
In `rl-agents`: 
`python3 setup.py develop`  
In `muzero-general`: 
`pip install -r requirements.txt`

## Using Sergio's environment at Vector's q server
Run `. project.env`, which will set your `PATH` and `PYTHONPATH` (Let me know if this doesn't work due to permissions or other issues)

## Running muzero-general's training on highway-env
In `muzero-general`: `python muzero.py highway_env` (Kinematics obs)

In `muzero-general`: `python muzero.py highway_env` (Occupancy obs)