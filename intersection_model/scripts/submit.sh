#!/bin/bash
#
# ==== SBATCH OPTIONS ==========================================================
#SBATCH --job-name=main_intersection_train
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:8
#SBATCH --gres=gpumem:23G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
# ==============================================================================

source config.env


#use for debuggin
#export NCCL_DEBUG=INFO
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=12345

# Dynamic log directory
DATE=$(date '+%Y-%m-%d_%H-%M-%S')
LOG_DIR="logs/${DATE}"
mkdir -p "${LOG_DIR}"
OUT_FILE="${LOG_DIR}/%j_${DATE}.out"
ERR_FILE="${LOG_DIR}/%j_${DATE}.err"

OUT_FILE_SCRIPT="${LOG_DIR}/script_%j_${DATE}.out"
ERR_FILE_SCRIPT="${LOG_DIR}/script%j_${DATE}.err"

echo "Requested GPUs: $SLURM_JOB_GPUS" > $OUT_FILE

echo "Running preprocessing step on a single node..."
srun --nodes=1 --ntasks=1 -A ${SLURM_ACCOUNT} --output=${OUT_FILE_SCRIPT} --error="${ERR_FILE_SCRIPT}" python -u scripts/prepare_data_parallel.py $@

if [ $? -ne 0 ]; then
    exit 1
fi
#ESCAPED_ARGS=$(printf " %q" "$@")
# Run Python
srun --kill-on-bad-exit=1 -A ${SLURM_ACCOUNT} --output=${OUT_FILE} --error="${ERR_FILE}" python -u main.py --run_id "${DATE}" $@