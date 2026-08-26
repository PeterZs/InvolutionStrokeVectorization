#!/bin/bash
source config.env

PYTHON_FILE="scripts/make_tars.py" 

BASE_DIR="${OUT_DIR}/$1" 
TAR_DIR="${OUT_DIR}/$2" 

shift 2                     # Shift to capture remaining optional args (like --ext)
TIME="07:00:00"
CPUS=4
MEM_PER_CPU="16G"
DATE=$(date '+%Y-%m-%d_%H-%M-%S')
SCRIPT_NAME=$(basename "${PYTHON_FILE}")
JOB_BASE_NAME="${SCRIPT_NAME%%.*}"



# 2. Get list of folder names inside the first shard (e.g. a, b, c)
NAMES=$(ls "$BASE_DIR/shard_0") 

echo "Detected shard structure in: $FIRST_SHARD"
echo "Log directory: logs/${JOB_BASE_NAME}/${DATE}"

for NAME in $NAMES; do

    # Create specific log dir per run or keep them grouped
    LOG_DIR="logs/${JOB_BASE_NAME}/${DATE}"
    mkdir -p "${LOG_DIR}"
    
    # Include the subfolder name in the log filename for easy debugging
    OUT_FILE="${LOG_DIR}/${NAME}_%j.out"
    ERR_FILE="${LOG_DIR}/${NAME}_%j.err"

    SBATCH_CMD="sbatch \
      --job-name=${JOB_BASE_NAME}_${NAME} \
      --account=${SLURM_ACCOUNT} \
      --time=${TIME} \
      --ntasks=1 \
      --cpus-per-task=${CPUS} \
      --mem-per-cpu=${MEM_PER_CPU} \
      --output=${OUT_FILE} \
      --error=${ERR_FILE}"

    # Construct the python command for this specific folder name
    CMD="uv run python -u ${PYTHON_FILE} --base_dir ${BASE_DIR} --tar_dir ${TAR_DIR} --name ${NAME} $@"

    echo "Submitting job for target: ${NAME}"
    $SBATCH_CMD --wrap="${CMD}"

done