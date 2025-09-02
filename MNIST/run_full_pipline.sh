#!/bin/bash
#SBATCH --job-name=iterative_cvae_pipeline
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=2
#SBATCH --time=07:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qiyuanliu@fe01.ds.uchicago.edu
#SBATCH --output=/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST/logs/%x_%A_%a.out
#SBATCH --error=/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST/logs/%x_%A_%a.err
#SBATCH --array=1-10

conda activate vae_env

SAMPLE_SIZE=5000
K_ROUNDS=5
LATENT_DIM=20
NUM_CLASSES=10
BATCH_SIZE=128
EPOCHS=150
LR=0.001
PATIENCE=100
SEED=0
DISC_PATH="model_saved/discriminator_mnist_cvae_2.pth"

# 固定 total attempts
TOTAL_ATTEMPTS=700000

P_LIST=(0.001 0.002 0.005 0.007 0.009 0.01 0.015 0.02 0.03 0.04)
IDX=$((SLURM_ARRAY_TASK_ID - 1))
LOWER_PERCENT=${P_LIST[$IDX]}
UPPER_PERCENT=$(awk -v p="$LOWER_PERCENT" 'BEGIN{printf "%.6g", p*1.5}')
LOWER_KEEP=0

WORK_DIR="diverse_up${UPPER_PERCENT}_low${LOWER_PERCENT}_up1p5x_low_keep${LOWER_KEEP}_tot${TOTAL_ATTEMPTS}"

echo "[Task ${SLURM_ARRAY_TASK_ID}] tot=${TOTAL_ATTEMPTS} lower=${LOWER_PERCENT} upper=${UPPER_PERCENT} (1.5x) keep=${LOWER_KEEP}"
echo "[Task ${SLURM_ARRAY_TASK_ID}] workdir=${WORK_DIR}"

python multiple_run_filter_diverse_sample.py \
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
