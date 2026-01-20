#!/bin/bash
#SBATCH --job-name=iterative_cvae_pipeline
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=2
#SBATCH --time=07:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qiyuanliu@fe01.ds.uchicago.edu
#SBATCH --output=/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST/logs/%x_%A.out
#SBATCH --error=/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST/logs/%x_%A.err

conda activate vae_env

# -------- Fixed parameters --------
SAMPLE_SIZE=5000
K_ROUNDS=10
LATENT_DIM=20
NUM_CLASSES=10
BATCH_SIZE=128
EPOCHS=150
LR=0.001
PATIENCE=100
SEED=0
DISC_PATH="model_saved/discriminator_mnist_cvae_2.pth"

TOTAL_ATTEMPTS=200000
UPPER_PERCENT=0.1
LOWER_PERCENT=0.15
LOWER_KEEP=0

WORK_DIR="conv_top${UPPER_PERCENT}_tot${TOTAL_ATTEMPTS}_2"

python -u multiple_run_filter_diverse_sample_conv.py \
  --sample_size "${SAMPLE_SIZE}" \
  --discriminator_path "${DISC_PATH}" \
  --k "${K_ROUNDS}" \
  --work_dir "${WORK_DIR}" \
  --total_sample_size "${TOTAL_ATTEMPTS}" \
  --upper_percent "${UPPER_PERCENT}" \
  --lower_percent "${LOWER_PERCENT}" \
  --lower_keep "${LOWER_KEEP}" \
  --latent_dim "${LATENT_DIM}" \
  --num_classes "${NUM_CLASSES}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --patience "${PATIENCE}" \
  --seed "${SEED}"
