#!/bin/bash
# 
# Installer for Stryde environment
# 
# Run: ./install_env.sh
# 
# M. Ravasi, 09/10/2023

echo 'Creating strydeenv environment'

# create conda env
conda env create -f environment.yml
source ~/miniconda3/etc/profile.d/conda.sh
conda activate strydeenv
conda env list
echo 'Created and activated environment:' $(which python)

echo 'Done!'

