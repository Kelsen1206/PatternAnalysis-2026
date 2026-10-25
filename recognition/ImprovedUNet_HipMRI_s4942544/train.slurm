#!/bin/bash
# Full training run for one model and one random seed.
#
# Usage (from this folder):
#   sbatch --export=ALL,MODEL=baseline,SEED=0 train.slurm
#   sbatch --export=ALL,MODEL=improved,SEED=1 train.slurm
# Optional: EPOCHS=80 (default 50)
#
# Note: --cpus-per-task is deliberately absent -- the comp3710 partition left jobs
# pending (PartitionConfig) when it was set. Two loader workers suit the single
# CPU core the job then receives.
#SBATCH --job-name=hipmri-train
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

MODEL=${MODEL:-improved}
SEED=${SEED:-0}
EPOCHS=${EPOCHS:-50}

source $HOME/miniconda3/bin/activate
conda activate torch

echo "Job $SLURM_JOB_ID on $(hostname), started $(date) | model=$MODEL seed=$SEED epochs=$EPOCHS"
nvidia-smi --query-gpu=name,memory.total --format=csv

python train.py --model "$MODEL" --seed "$SEED" --epochs "$EPOCHS" --workers 2 --out "runs/${MODEL}_s${SEED}"

echo "Finished $(date)"
