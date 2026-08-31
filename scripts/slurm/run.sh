#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodelist=node1
#SBATCH --gres=shard:20
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00

module load anaconda3-2024.2
module load cuda-12.8
source /apps/compilers/anaconda3-24.2/etc/profile.d/conda.sh
conda activate vlm

echo "PYTHON: $(which python)"

PORT=$(shuf -i 8900-9900 -n 1)

echo "NODE: $(hostname)"
echo "PORT: $PORT"

# Print IP addresses
echo "HOSTNAME: $(hostname)"
echo "IP(s): $(hostname -I)"
echo "Primary IP: $(hostname -I | awk '{print $1}')"

# Alternatively, print all IPv4 addresses with interfaces
ip -4 addr show

jupyter notebook --no-browser --ip=0.0.0.0 --port=$PORT