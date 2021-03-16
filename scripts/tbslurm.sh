#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -J  tensorboard_server    # name
#SBATCH -t 180:00:00               # max runtime is 180 hours (9 days)
#SBATCH --mem=6GB
#SBATCH -p gpu 
#SBATCH -o /h/suo/tbslurm/tb-%J.out # TODO: Where to save your output

# INSTRUCTIONS

# 0. Fill in all the "TODO" in this file (1 above, 4 below)
# 1. To run, use the following command:
# sbatch --array=0-0 tbslurm.sh

# 2. Then get the output, this will tell you your ip address and port number:
# cat ~/tb-%J.out  

# 3. Then ssh into q, forwarding that port:
# ssh -L PORT:IPADDR:PORT spitis@q.vectorinstitute.ai

source /h/suo/.bashrc #TODO: Your profile
conda activate treesearch

# source activate tfcpu #TODO: Your local cpu environment (with cpu tensorboard, until ops installs cuda on CPU only machines)
# MODEL_DIR=/scratch/gobi2/pitchan/her_results/protoge/slide #TODO: Your TF model directory
# MODEL_DIR=/h/suo/dev/tree-search-planning/muzero-gneral/results #TODO: Your TF model directory
MODEL_DIR=/scratch/gobi1/suo/experiments/tree-search-planning

ipnport=60315 # TODO set your port here
echo ipnport=$ipnport

ipnip=$(hostname -i)
echo ipnip=$ipnip

tensorboard --logdir="${MODEL_DIR}" --port=$ipnport
