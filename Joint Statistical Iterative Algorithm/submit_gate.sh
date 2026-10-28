#!/bin/bash
#SBATCH --job-name=gate_simulation
#SBATCH --partition=gpu-a100-small
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=research-AS-RST
#SBATCH --output=/scratch/matildegoncalv/gate_simulation_%j.out
#SBATCH --error=/scratch/matildegoncalv/gate_simulation_%j.err

# Load modules
module load 2025
module load cuda/11.6
module load miniconda3

# Working directory

cd /scratch/matildegoncalv/

# Run pixel detector script
for i in {0..99}; do
    /scratch/matildegoncalv/miniconda3/envs/cbct/bin/python gate_iterative.py $i
done
