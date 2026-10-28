#!/bin/bash
#SBATCH --job-name=astra_cone_cylinder
#SBATCH --partition=gpu-a100-small
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G
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
/scratch/matildegoncalv/miniconda3/envs/cbct/bin/python astra_sinogram.py