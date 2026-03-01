#!/usr/bin/env bash
set -euo pipefail

# bash /projects/zhanglab/users/ana/cts_peak_gene/experiments/job.evaluation/mini_evaluate.sh

# Slurm settings
PARTITIONS="${PARTITIONS:-mzhang,pool1}"
LOG_DIR="${LOG_DIR:-/projects/zhanglab/users/ana/logs}"
CONDA_ENV="${CONDA_ENV:-ctar}"

# Dataset name used consistently across steps
DATASET_NAME="${DATASET_NAME:-pbmc}"

# Paths (match repository defaults in scripts)
REPO_ROOT="${REPO_ROOT:-/projects/zhanglab/users/ana/cts_peak_gene}"
SCRIPTS_DIR="${SCRIPTS_DIR:-${REPO_ROOT}/experiments/job.evaluation}"

# New file to add column of
NEW_FILE="${NEW_FILE:-/projects/zhanglab/users/ana/multiome/results/scent/myscent_pbmc_cauchy.csv}"
ORIGINAL_COL="${ORIGINAL_COL:-cauchy_p}"
NEW_COL="${NEW_COL:-scent_cauchy}"

# Be sure to include new col!
DEDUP="${DEDUP:-min}"
METHOD_COLS="${METHOD_COLS:-scent,scmm,signac,ctar_filt_z,ctar_filt,scmm_cauchy,scent_cauchy}"

# Merge outputs (using add_new_overlap.py)
MERGE_DIR="${MERGE_DIR:-/projects/zhanglab/users/ana/multiome/validation/evaluation/${DATASET_NAME}}"

# AUERC outputs (calculate_auerc.py default)
AURC_DIR="${AURC_DIR:-/projects/zhanglab/users/ana/multiome/validation/evaluation/tables/metrics_jitter/${DATASET_NAME}}"
REFERENCE_METHOD="${REFERENCE_METHOD:-ctar_filt}"
N_BOOTSTRAP="${N_BOOTSTRAP:-1000}"

echo "Evaluating ${DATASET_NAME} with ${NEW_COL}..."

mkdir -p "${LOG_DIR}"

submit() {
  sbatch --parsable -p "${PARTITIONS}" "$@"
}


echo "Submitting add_new_overlap.py ..."
MERGE_JOB_ID=$(submit \
  -t 0-06:00:00 --mem=32G -J "add_overlap_${DATASET_NAME}" \
  -o "${LOG_DIR}/add_overlap_${DATASET_NAME}-%j.out" \
  -e "${LOG_DIR}/add_overlap_${DATASET_NAME}-%j.err" \
  --wrap "source ~/.bashrc && conda activate ${CONDA_ENV} && \
          mkdir -p '${MERGE_DIR}' && \
          python ${SCRIPTS_DIR}/add_new_overlap.py \
            --merge_path '${MERGE_DIR}' \
            --new_file '${NEW_FILE}' \
            --dataset_name '${DATASET_NAME}' \
            --original_col '${ORIGINAL_COL}' \
            --new_col '${NEW_COL}' \
            --dedup '${DEDUP}'")
echo "MERGE_JOB_ID=${MERGE_JOB_ID}"

echo "Submitting calculate_auerc_seeds.py (after merge) ..."
AURC_JOB_ID=$(submit \
  --dependency=afterok:${MERGE_JOB_ID} \
  -t 3-00:00:00 --mem=128G -J "auerc_${DATASET_NAME}" \
  -o "${LOG_DIR}/auerc_${DATASET_NAME}-%j.out" \
  -e "${LOG_DIR}/auerc_${DATASET_NAME}-%j.err" \
  --wrap "source ~/.bashrc && conda activate ${CONDA_ENV} && \
          mkdir -p '${AURC_DIR}' && \
          python ${SCRIPTS_DIR}/calculate_auerc_seeds.py \
            --merge_path '${MERGE_DIR}' \
            --res_path '${AURC_DIR}' \
            --dataset_name '${DATASET_NAME}' \
            --n_bootstrap '${N_BOOTSTRAP}' \
            --reference_method '${REFERENCE_METHOD}' \
            --method_cols '${METHOD_COLS}'")
echo "AURC_JOB_ID=${AURC_JOB_ID}"

echo "Submitted jobs:"
echo "  merge:          ${MERGE_JOB_ID}"
echo "  auerc:          ${AURC_JOB_ID}"