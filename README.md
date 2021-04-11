# Installation for Development

## Creating a new environment
`conda create -n tsmp python=3.7`  
`conda activate tsmp`  
`conda install -c conda-forge git-lfs`

In `highway-env`: 
`python3 setup.py develop`  
In `rl-agents`: 
`python3 setup.py develop`  
In `muzero-general`: 
`pip install -r requirements.txt`
