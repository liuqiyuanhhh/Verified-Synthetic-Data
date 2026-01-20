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
#SBATCH --array=1-36

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

TOTAL_ATTEMPTS=100000

UPPER_LIST=(0.1 0.2 0.3 0.4)
LOWER_LIST=(0.1 0.2 0.3)
KEEP_LIST=(0.1 0.3 0.5)

COMB_UP=()
COMB_LOW=()
COMB_KEEP=()
for up in "${UPPER_LIST[@]}"; do
  for low in "${LOWER_LIST[@]}"; do
    for keep in "${KEEP_LIST[@]}"; do
      COMB_UP+=("$up")
      COMB_LOW+=("$low")
      COMB_KEEP+=("$keep")
    done
  done
done

N_COMBOS=${#COMB_UP[@]}
if [[ $SLURM_ARRAY_TASK_ID -lt 1 || $SLURM_ARRAY_TASK_ID -gt $N_COMBOS ]]; then
  echo "Array index ${SLURM_ARRAY_TASK_ID} out of range (1..${N_COMBOS})."
  exit 2
fi

IDX=$((SLURM_ARRAY_TASK_ID - 1))
UPPER_PERCENT=${COMB_UP[$IDX]}
LOWER_PERCENT=${COMB_LOW[$IDX]}
LOWER_KEEP=${COMB_KEEP[$IDX]}

WORK_DIR="diverse_up${UPPER_PERCENT}_low${LOWER_PERCENT}_low${LOWER_KEEP}_tot${TOTAL_ATTEMPTS}_keep${LOWER_KEEP}"

echo "[Task ${SLURM_ARRAY_TASK_ID}/${N_COMBOS}] tot=${TOTAL_ATTEMPTS} upper=${UPPER_PERCENT} lower=${LOWER_PERCENT} keep=${LOWER_KEEP}"
echo "[Task ${SLURM_ARRAY_TASK_ID}] workdir=${WORK_DIR}"
which python; python -V || true

python -u multiple_run_filter_diverse_sample.py \
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
