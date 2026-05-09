#!/bin/sh

# FILE: /projects/zhanglab/users/ana/cts_peak_gene/experiments/job.pearsoncorr/pearsonr_batch.sh

MULTIOME_FILE=/projects/zhanglab/users/ana/multiome/processed/neurips_bmmc/bmmc.h5mu
LINKS_FILE=/projects/zhanglab/users/ana/multiome/simulations/pearsonr/full_abc/eval_df.csv
TARGET_PATH=/projects/zhanglab/users/ana/multiome/simulations/pearsonr/full_abc
GENOME_FILE=/projects/zhanglab/users/ana/bedtools2/ana_bedfiles/ref/GRCh38.p14.genome.fa.bgz
BIN_CONFIG='1.1.inf.inf.10000'
BIN_TYPE='mean_var'
PYBEDTOOLS_PATH=/projects/zhanglab/users/ana/bedtools2/bin
BATCH_SIZE=100
MAX_CONCURRENT_TASKS=20
EMAIL_USER="asprieto@andrew.cmu.edu"

# Get MaxArraySize 
HPC_MAX_ARRAY_SIZE=$(echo "$(scontrol show config | grep MaxArraySize)" | awk -F'=' '{print $2}' | tr -d '[:space:]')
if ! [[ "$HPC_MAX_ARRAY_SIZE" =~ ^[0-9]+$ ]] || [ "$HPC_MAX_ARRAY_SIZE" -le 0 ]; then
  echo "Error: Could not determine MaxArraySize from scontrol. Defaulting to 1001."
  HPC_MAX_ARRAY_SIZE=1001
fi

echo "Beginning $BIN_CONFIG create_ctrl job..."
CTRL_JOB_ID=$(sbatch --parsable -p mzhang -t 1-00:00:00 --mem=80Gb \
  -o "/home/asprieto/logs/create_ctrl_${BIN_CONFIG}_%J.err" \
  -J "ctrl_$BIN_CONFIG" \
  --wait \
  --wrap " \
source ~/.bashrc && \
conda activate ctar && \
python /projects/zhanglab/users/ana/cts_peak_gene/CLI_ctar.py \
    --job create_ctrl \
    --multiome_file $MULTIOME_FILE \
    --links_file $LINKS_FILE \
    --target_path $TARGET_PATH \
    --genome_file $GENOME_FILE \
    --binning_config $BIN_CONFIG \
    --binning_type $BIN_TYPE \
    --pybedtools_path $PYBEDTOOLS_PATH")

if [ -z "$CTRL_JOB_ID" ] || ! [[ "$CTRL_JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "Error: create_ctrl job submission failed or did not return a valid job ID ($CTRL_JOB_ID). Exiting."
    exit 1
fi
echo "create_ctrl job $CTRL_JOB_ID completed."

# Calculate TOTAL_ITEMS_TO_BATCH
if [[ $BIN_CONFIG =~ "inf" ]]; then
  NUM_LINKS=$(wc -l < "$LINKS_FILE")
  TOTAL_ITEMS_TO_BATCH=$((NUM_LINKS > 0 ? NUM_LINKS - 1 : 0)) # Assumes header
else
  CTRL_LINKS_DIR="$TARGET_PATH/ctrl_peaks/ctrl_links_$BIN_CONFIG"
  if [ -d "$CTRL_LINKS_DIR" ]; then
    TOTAL_ITEMS_TO_BATCH=$(find "$CTRL_LINKS_DIR" -type f -printf '.' | wc -c)
  else
    echo "Warning: Control links directory $CTRL_LINKS_DIR not found. Assuming 0 items to batch."
    TOTAL_ITEMS_TO_BATCH=0
  fi
fi

TOTAL_NUM_BATCHES=$(( (TOTAL_ITEMS_TO_BATCH + BATCH_SIZE - 1) / BATCH_SIZE ))
echo "Found $TOTAL_ITEMS_TO_BATCH items to process. This requires $TOTAL_NUM_BATCHES total batch(es) of size $BATCH_SIZE."

# Submit compute_corr job
if [ "$TOTAL_NUM_BATCHES" -gt 0 ]; then

  echo "Submitting $BIN_CONFIG compute_corr job(s) sequentially..."
  sbatch_submission_index=0
  PREVIOUS_SBATCH_CHUNK_JOB_ID=""

  # Calculate the number of sbatch chunks that will be submitted
  TOTAL_SBATCH_CHUNKS=$(( (TOTAL_NUM_BATCHES + HPC_MAX_ARRAY_SIZE - 1) / HPC_MAX_ARRAY_SIZE ))

  for (( batch_offset=0; batch_offset<TOTAL_NUM_BATCHES; batch_offset+=HPC_MAX_ARRAY_SIZE )); do
    sbatch_submission_index=$((sbatch_submission_index + 1)) # Increment chunk index (1-based for easier check)

    num_tasks_in_this_chunk=$((TOTAL_NUM_BATCHES - batch_offset))
    if [ "$num_tasks_in_this_chunk" -gt "$HPC_MAX_ARRAY_SIZE" ]; then
      num_tasks_in_this_chunk=$HPC_MAX_ARRAY_SIZE
    fi

    if [ "$num_tasks_in_this_chunk" -le 0 ]; then
        echo "Warning: num_tasks_in_this_chunk is zero or negative. Skipping sbatch submission."
        continue
    fi

    array_end_index=$((num_tasks_in_this_chunk - 1))
    ARRAY_OPT="--array=0-${array_end_index}%${MAX_CONCURRENT_TASKS}"
    
    JOB_NAME_SUFFIX=""
    # Use sbatch_submission_index-1 for 0-based chunk suffix if preferred
    if [ "$TOTAL_NUM_BATCHES" -gt "$HPC_MAX_ARRAY_SIZE" ]; then
      JOB_NAME_SUFFIX="_chunk$((sbatch_submission_index-1))"
    fi
    CURRENT_JOB_NAME="corr_${BIN_CONFIG}${JOB_NAME_SUFFIX}"
    LOG_FILE_PREFIX="/home/asprieto/logs/compute_corr_${BIN_CONFIG}${JOB_NAME_SUFFIX}"

    DEPENDENCY_OPT=""
    if [ "$sbatch_submission_index" -eq 1 ]; then # First chunk
        DEPENDENCY_OPT="--dependency=afterok:${CTRL_JOB_ID}"
    elif [ -n "$PREVIOUS_SBATCH_CHUNK_JOB_ID" ]; then
        DEPENDENCY_OPT="--dependency=afterok:${PREVIOUS_SBATCH_CHUNK_JOB_ID}"
    fi

    # Conditional Email
    EMAIL_OPTS=""
    if [ "$sbatch_submission_index" -eq "$TOTAL_SBATCH_CHUNKS" ]; then
        # This is the last sbatch chunk
        EMAIL_OPTS="--mail-type=FAIL,END --mail-user=$EMAIL_USER"
        echo "This is the final chunk, enabling email notification."
    fi

    echo "Submitting sbatch chunk $((sbatch_submission_index-1)): Effective batches ${batch_offset} to $((batch_offset + num_tasks_in_this_chunk - 1)). SLURM array tasks 0-${array_end_index}. Dependency: $DEPENDENCY_OPT"

    sbatch_command_core=" \
      source ~/.bashrc && \
      conda activate ctar && \
      python /projects/zhanglab/users/ana/cts_peak_gene/CLI_ctar.py \
          --job compute_corr \
          --multiome_file $MULTIOME_FILE \
          --batch_size $BATCH_SIZE \
          --batch_id "\$SLURM_ARRAY_TASK_ID" \
          --batch_offset ${batch_offset} \
          --links_file $LINKS_FILE \
          --target_path $TARGET_PATH \
          --binning_config $BIN_CONFIG \
          --binning_type $BIN_TYPE \
          --pybedtools_path $PYBEDTOOLS_PATH"
    
    CURRENT_SBATCH_CHUNK_JOB_ID=$(sbatch --parsable -p mzhang -t 1-00:00:00 --mem=80Gb \
      ${ARRAY_OPT} \
      ${EMAIL_OPTS} \
      -o "${LOG_FILE_PREFIX}_%A_%a.err" \
      -J "${CURRENT_JOB_NAME}" \
      ${DEPENDENCY_OPT} \
      --wrap "$sbatch_command_core")
    
    echo "Executing sbatch for chunk $((sbatch_submission_index-1)). Job ID: ${CURRENT_SBATCH_CHUNK_JOB_ID}"
    
    if [ -z "$CURRENT_SBATCH_CHUNK_JOB_ID" ] || ! [[ "$CURRENT_SBATCH_CHUNK_JOB_ID" =~ ^[0-9]+$ ]]; then
        echo "Error: Failed to submit sbatch chunk $((sbatch_submission_index-1)) or get a valid job ID. Exiting."
        if [ -n "$PREVIOUS_SBATCH_CHUNK_JOB_ID" ]; then scancel "$PREVIOUS_SBATCH_CHUNK_JOB_ID"; fi
        if [ -n "$CTRL_JOB_ID" ]; then scancel "$CTRL_JOB_ID"; fi
        exit 1
    fi

    PREVIOUS_SBATCH_CHUNK_JOB_ID=$CURRENT_SBATCH_CHUNK_JOB_ID
    # Note: sbatch_submission_index incremented at the start of the loop
  done
  echo "All compute_corr job chunks submitted."
fi
