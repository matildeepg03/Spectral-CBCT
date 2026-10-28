#!/bin/bash
#SBATCH --job-name=pixel_detector
#SBATCH --partition=gpu-a100-small
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=research-AS-RST
#SBATCH --output=/home/matildegoncalv/pixel_detector_%j.out
#SBATCH --error=/home/matildegoncalv/pixel_detector_%j.err

# -----------------------
# Load modules
# -----------------------
module load 2025
module load cuda/11.6
module load miniconda3

# -----------------------
# Working directory
# -----------------------
cd /home/matildegoncalv/

# -----------------------
# Run pixel detector script
# -----------------------
for i in {173..183}; do
    /home/matildegoncalv/miniconda3/envs/cbct/bin/python pixel_detector.py $i
done
