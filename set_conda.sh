#!/bin/bash
# This script creates a Conda environment named FLStore and installs the required packages using pip.

# Ensure conda is available
if [ -z "$CONDA_EXE" ]; then
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        echo "Conda not found! Please install Anaconda or Miniconda and try again."
        exit 1
    fi
fi

# Create the conda environment (you can change the python version if needed)
conda create --name FLStore python=3.8 -y

# Activate the environment
conda activate FLStore

# Install packages using pip
pip install minio numpy torch torchvision scikit-learn timm

echo "Conda environment 'FLStore' created and packages installed successfully."
