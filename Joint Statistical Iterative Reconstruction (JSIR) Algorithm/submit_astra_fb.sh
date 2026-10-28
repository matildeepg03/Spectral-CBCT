#!/bin/bash
#SBATCH --job-name=astra_cone_cylinder
#SBATCH --partition=gpu-a100-small
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G
#SBATCH --account=research-AS-RST
#SBATCH --output=/home/matildegoncalv/astra_%j.out
#SBATCH --error=/home/matildegoncalv/astra_%j.err

# -----------------------
# Load modules
# -----------------------
module load 2025
module load cuda/11.6
module load miniconda3

# -----------------------
# Go to working directory
# -----------------------
cd /home/matildegoncalv/

# -----------------------
# Run ASTRA script
# -----------------------
/home/matildegoncalv/miniconda3/envs/cbct/bin/python astra_fp.py