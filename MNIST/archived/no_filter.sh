#!/bin/bash
#SBATCH --job-name=cvae_nofilter
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=2
#SBATCH --time=07:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qiyuanliu@fe01.ds.uchicago.edu
#SBATCH --output=/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST/logs/%x_%A.out
#SBATCH --error=/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST/logs/%x_%A.err
# --- Paths ---
PROJECT_DIR="/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST"
LOG_DIR="${PROJECT_DIR}/logs"
# --- Conda env (robust activation) ---
conda activate vae_env

python -u multiple_run_no_filter.py 