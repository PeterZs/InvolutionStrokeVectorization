#!/bin/bash

source config.env
#SBATCH --job-name=main_centerline_test
#SBATCH --time=40:00:00
PYTHON_FILE="$1"
shift  # Remaining args go to python



# Dynamic log directory
DATE=$(date '+%Y-%m-%d_%H-%M-%S')
LOG_DIR="logs/${DATE}"
mkdir -p "${LOG_DIR}"
OUT_FILE="${LOG_DIR}/%j_${DATE}.out"
ERR_FILE="${LOG_DIR}/%j_${DATE}.err"
OUT_FILE_SCRIPT="${LOG_DIR}/script_%j_${DATE}.out"
ERR_FILE_SCRIPT="${LOG_DIR}/script%j_${DATE}.err"

echo "Running preprocessing step on a single node..."
srun --nodes=1 --ntasks=1 --output=${OUT_FILE_SCRIPT} --error="${ERR_FILE_SCRIPT}" python -u scripts/prepare_data.py $@

#ESCAPED_ARGS=$(printf " %q" "$@")
# Run Python
srun --kill-on-bad-exit=1 -A ${SLURM_ACCOUNT} --output=${OUT_FILE} --error="${ERR_FILE}" python -u main.py --run_id "${DATE}" $@