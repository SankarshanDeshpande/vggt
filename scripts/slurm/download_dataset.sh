#!/bin/bash
#SBATCH --job-name=replica_download
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodelist=node1
#SBATCH --gres=shard:10
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00

module load anaconda3-2024.2
module load cuda-12.8
source /apps/compilers/anaconda3-24.2/etc/profile.d/conda.sh
conda activate vlm

echo "=========================================="
echo "ReplicaOcc Dataset Download"
echo "=========================================="

echo "PYTHON: $(which python)"
echo "HF: $(which hf)"

DATA_DIR="/home/mazaveri/hpc-prog/sankarshan/ReplicaOcc"

mkdir -p "$DATA_DIR"

echo ""
echo "NODE: $(hostname)"
echo "DATA DIR: $DATA_DIR"
echo "START TIME: $(date)"
echo ""

echo "Disk space before download:"
df -h "$DATA_DIR"

echo ""
echo "Downloading ReplicaOcc scenes:"
echo "  room0"
echo "  room1"
echo "  room2"
echo "  office0"
echo "  office1"
echo ""

hf download the-masses/ReplicaOcc \
    --repo-type dataset \
    --include "Replica_OCC/sequences/cam_params.json" \
    --include "Replica_OCC/sequences/room0/color/**" \
    --include "Replica_OCC/sequences/room0/pose/**" \
    --include "Replica_OCC/sequences/room0/intrinsic/**" \
    --include "Replica_OCC/sequences/room1/color/**" \
    --include "Replica_OCC/sequences/room1/pose/**" \
    --include "Replica_OCC/sequences/room1/intrinsic/**" \
    --include "Replica_OCC/sequences/room2/color/**" \
    --include "Replica_OCC/sequences/room2/pose/**" \
    --include "Replica_OCC/sequences/room2/intrinsic/**" \
    --include "Replica_OCC/sequences/office0/color/**" \
    --include "Replica_OCC/sequences/office0/pose/**" \
    --include "Replica_OCC/sequences/office0/intrinsic/**" \
    --include "Replica_OCC/sequences/office1/color/**" \
    --include "Replica_OCC/sequences/office1/pose/**" \
    --include "Replica_OCC/sequences/office1/intrinsic/**" \
    --local-dir "$DATA_DIR"

STATUS=$?

echo ""
echo "=========================================="
echo "Download finished"
echo "=========================================="

echo "END TIME: $(date)"
echo "Exit status: $STATUS"

echo ""
echo "Disk space after download:"
df -h "$DATA_DIR"

echo ""
echo "Dataset size:"
du -sh "$DATA_DIR"

echo ""
echo "Downloaded directories:"
find "$DATA_DIR/Replica_OCC/sequences" -maxdepth 2 -type d | sort

exit $STATUS