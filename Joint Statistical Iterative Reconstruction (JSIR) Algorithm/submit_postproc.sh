#!/bin/bash
#SBATCH --job-name=root_pixelv2
#SBATCH --partition=gpu-a100-small
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G
#SBATCH --account=research-AS-RST
#SBATCH --output=/home/matildegoncalv/astra_%j.out
#SBATCH --error=/home/matildegoncalv/astra_%j.err

module load 2025
module load cuda/11.6
module load miniconda3

cd /home/matildegoncalv/

/home/matildegoncalv/miniconda3/envs/cbct/bin/python root_pixelv2.py
