#!/bin/bash
#SBATCH --job-name=fixed_size
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

# --- Paths ---
PROJECT_DIR="/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST/conv_cvae"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "$LOG_DIR"

# --- Sizes to sweep ---
#sizes=(2000 5000 10000 20000)
#FIXED_SIZE=${sizes[$SLURM_ARRAY_TASK_ID]}

#echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
#echo "Running fixed size = ${FIXED_SIZE}"

# --- Conda env (robust activation) ---
# If conda is not in non-interactive shells, source your profile first
conda activate vae_env

# --- Run ---
cd "$PROJECT_DIR"
python -u full_pipeline_no_filter.py #--fixed-size "${FIXED_SIZE}"
