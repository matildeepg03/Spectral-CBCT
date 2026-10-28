#!/bin/bash
#SBATCH --job-name=astra_cone_cylinder
#SBATCH --partition=gpu-a100
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=3G
#SBATCH --account=research-AS-RST
#SBATCH --output=/scratch/matildegoncalv/astra_%j.out
#SBATCH --error=/scratch/matildegoncalv/astra_%j.err

# -----------------------
# Load modules
# -----------------------
module load 2025
module load cuda/11.6
module load miniconda3

# -----------------------
# Go to working directory
# -----------------------
cd /scratch/matildegoncalv/

# -----------------------
# Run ASTRA script
# -----------------------
/scratch/matildegoncalv/miniconda3/envs/cbct/bin/python start.py