#!/bin/bash

# to watch for errors:
# find logs -type f -name "*.err" -print -exec cat {} \;

source config.env

TIME="20:00:00"

#GeCreativeDataset
#TUBerlinDataset
#InstanceSegmentationDataset
#QuickDrawDataset
#SyntheticDataset
CLASS="GeCreativeDataset"

CPUS=1
MEM_PER_CPU="8G"


DATE=$(date '+%Y-%m-%d_%H-%M-%S')
LOG_DIR="logs/${DATE}"
mkdir -p "${LOG_DIR}"

START=0
STEP=100
MAX=20000

for ((offset=START; offset<MAX; offset+=STEP)); do
  to=$((offset + STEP -1))
  DATE=$(date '+%Y-%m-%d_%H-%M-%S')
  
  JOB_NAME_OFFSET="${CLASS}_${offset}"
  OUT_FILE="${LOG_DIR}/${JOB_NAME_OFFSET}_${DATE}.out"
  ERR_FILE="${LOG_DIR}/${JOB_NAME_OFFSET}_${DATE}.err"

  OUT_DIR_CURR="${OUT_DIR}/${CLASS}/shard_${offset}"
  mkdir -p "${OUT_DIR_CURR}"

  SBATCH_CMD="sbatch \
    --job-name=${JOB_NAME_OFFSET} \
    --account=${SLURM_ACCOUNT} \
    --time=${TIME} \
    --ntasks=1 \
    --cpus-per-task=${CPUS} \
    --mem-per-cpu=${MEM_PER_CPU} \
    --output=${OUT_FILE} \
    --error=${ERR_FILE}" \

  echo "Submitting job: ${JOB_NAME_OFFSET} (offset=${offset}, to=${to})"
  $SBATCH_CMD --wrap="uv run python -u process_everything.py --out_dir ${OUT_DIR_CURR} --offset ${offset} --to ${to} --class ${CLASS}"
done