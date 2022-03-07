#!/bin/bash

#SBATCH --job-name=test_diffusion
#SBATCH --gres=gpu:1
#SBATCH -o slurm.out
#SBATCH --time=14-0  # 10 hours

. /data/shinahyung/anaconda3/etc/profile.d/conda.sh
conda activate torch38gpu


python exe_physio.py
