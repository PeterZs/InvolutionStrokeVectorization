#!/bin/bash
source config.env

PYTHON_FILE="$1"
shift  # Shift so all remaining arguments go to python
TIME="07:00:00"
CPUS=4
MEM_PER_CPU="16G"
DATE=$(date '+%Y-%m-%d_%H-%M-%S')
SCRIPT_NAME=$(basename "${PYTHON_FILE}")
NAME="${SCRIPT_NAME%%.*}"
LOG_DIR="logs/${NAME}/${DATE}"
mkdir -p "${LOG_DIR}"
OUT_FILE="${LOG_DIR}/%j_${DATE}.out"
ERR_FILE="${LOG_DIR}/%j_${DATE}.err"


SBATCH_CMD="sbatch \
  --job-name=${NAME} \
  --account=${SLURM_ACCOUNT} \
  --time=${TIME} \
  --ntasks=1 \
  --cpus-per-task=${CPUS} \
  --mem-per-cpu=${MEM_PER_CPU} \
  --output=${OUT_FILE} \
  --error=${ERR_FILE}"

ESCAPED_ARGS=$(printf " %q" "$@")
$SBATCH_CMD --wrap="uv run python -u ${PYTHON_FILE} ${ESCAPED_ARGS}"