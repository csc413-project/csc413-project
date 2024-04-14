#!/bin/bash

source /h/u7/c1/00/cuifuyan/opt/miniconda3/etc/profile.d/conda.sh

# Activate Conda environment
conda activate csc413proj

export MUJOCO_GL="osmesa"
# Execute Python script
python dreamer/main.py