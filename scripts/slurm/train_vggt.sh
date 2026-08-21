#!/bin/bash
#SBATCH --job-name=vggt_train
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodelist=node2
#SBATCH --gres=shard:20
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

module load cuda-12.8

PYTHON_BIN=/home/mazaveri/.conda/envs/vlm/bin/python

echo "PYTHON: $PYTHON_BIN"
$PYTHON_BIN -c "import torch; print(torch.__version__, torch.cuda.is_available())"

cd ~/hpc-prog/sankarshan/vggt/training
$PYTHON_BIN -m torch.distributed.run --nproc_per_node=1 launch.py