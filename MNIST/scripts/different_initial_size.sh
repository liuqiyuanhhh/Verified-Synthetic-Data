#!/bin/bash
#SBATCH --job-name=init_sweep
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qiyuanliu@fe01.ds.uchicago.edu
#SBATCH --array=0
#SBATCH --output=/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST/logs/%x_%A_%a.out
#SBATCH --error=/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST/logs/%x_%A_%a.err

PROJECT_DIR="/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST/conv_cvae"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "$LOG_DIR"

# sweep these init sizes
sizes=(60000)
INIT_SIZE=${sizes[$SLURM_ARRAY_TASK_ID]}

conda activate vae_env

cd "$PROJECT_DIR"
python -u different_initial_sample_size.py --init-size "${INIT_SIZE}" --rounds 40 --synthetic-per-round 20000 --threshold 0.1 --seed 0
