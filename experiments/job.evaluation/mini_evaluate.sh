#!/usr/bin/env bash
set -euo pipefail

# bash /ocean/projects/bio240051p/aprieto/cts_peak_gene/experiments/job.evaluation/mini_evaluate.sh

# Slurm settings
PARTITIONS="${PARTITIONS:-RM}"
LOG_DIR="${LOG_DIR:-/ocean/projects/bio240051p/aprieto/logs}"
CONDA_ENV="${CONDA_ENV:-ctar}"

# Dataset name used consistently across steps
DATASET_NAME="${DATASET_NAME:-bmmc}"

# Paths (match repository defaults in scripts)
REPO_ROOT="${REPO_ROOT:-/ocean/projects/bio240051p/aprieto/cts_peak_gene}"
SCRIPTS_DIR="${SCRIPTS_DIR:-${REPO_ROOT}/experiments/job.evaluation}"

# New file to add column of
NEW_FILE="${NEW_FILE:-/ocean/projects/bio240051p/aprieto/multiome/results/scmultimap/scmultimap_bmmc_with_donor_cis.csv}"
ORIGINAL_COL="${ORIGINAL_COL:-pval}"
NEW_COL="${NEW_COL:-scmm_with_donor}"

# Be sure to include new col!
DEDUP="${DEDUP:-min}"
METHOD_COLS="${METHOD_COLS:-scent,scmm,signac,ctar_filt_z,ctar_filt,scmm_with_donor}"

# Merge outputs (using add_new_overlap.py)
MERGE_DIR="${MERGE_DIR:-/ocean/projects/bio240051p/aprieto/multiome/validation/evaluation/${DATASET_NAME}}"

# AUERC outputs (calculate_auerc.py default)
AURC_DIR="${AURC_DIR:-/ocean/projects/bio240051p/aprieto/multiome/validation/evaluation/tables/metrics_jitter/${DATASET_NAME}}"
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