#!/bin/bash
#SBATCH --job-name=fdk_reconstruction
#SBATCH --partition=gpu-a100-small
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=research-AS-RST
#SBATCH --output=/scratch/matildegoncalv/fdk_%j.out
#SBATCH --error=/scratch/matildegoncalv/fdk_%j.err

# Load necessary modules
module load 2025
module load cuda/11.6
module load miniconda3

# Change to work directory
cd /scratch/matildegoncalv/

# Run reconstruction using full path to Python
#/home/matildegoncalv/miniconda3/envs/cbct/bin/python fdk_astra.py

/scratch/matildegoncalv/miniconda3/envs/cbct/bin/python fdk_reconstruction.py
