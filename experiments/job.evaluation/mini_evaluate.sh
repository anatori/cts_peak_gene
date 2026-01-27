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
NEW_FILE="${NEW_FILE:-/projects/zhanglab/users/ana/multiome/results/ctar/final_eval/pbmc/pbmc_filtered_5.5.5.5.10000/cis_links_df.csv}"
ORIGINAL_COL="${ORIGINAL_COL:-5.5.5.5.10000_mcpval}"
NEW_COL="${NEW_COL:-ctar_filt_10k}"

# Be sure to include new col!
METHOD_COLS="${METHOD_COLS:-scmm,signac,ctar_filt_z,ctar_filt,ctar_filt_10k}"

# Intersect inputs/outputs
OVERLAP_DIR="${OVERLAP_DIR:-/projects/zhanglab/users/ana/bedtools2/ana_bedfiles/validation/overlap/${DATASET_NAME}}"
BEDTOOLS_BIN="${BEDTOOLS_BIN:-/projects/zhanglab/users/ana/bedtools2/bin/intersectBed}"

# Aggregate outputs (aggregate_overlaps.py default)
AGG_DIR="${AGG_DIR:-/projects/zhanglab/users/ana/multiome/validation/${DATASET_NAME}}"

# AUERC outputs (calculate_auerc.py default)
AURC_DIR="${AURC_DIR:-/projects/zhanglab/users/ana/multiome/validation/tables/${DATASET_NAME}}"
REFERENCE_METHOD="${REFERENCE_METHOD:-ctar_filt}"
N_BOOTSTRAP="${N_BOOTSTRAP:-1000}"

echo "Evaluating ${DATASET_NAME}..."

mkdir -p "${LOG_DIR}"

submit() {
  sbatch --parsable -p "${PARTITIONS}" "$@"
}


echo "Submitting add_new_overlap.py ..."
AGG_JOB_ID=$(submit \
  -t 0-06:00:00 --mem=32G -J "add_overlap_${DATASET_NAME}" \
  -o "${LOG_DIR}/add_overlap_${DATASET_NAME}-%j.out" \
  -e "${LOG_DIR}/add_overlap_${DATASET_NAME}-%j.err" \
  --wrap "source ~/.bashrc && conda activate ${CONDA_ENV} && \
          mkdir -p '${AGG_DIR}' && \
          python ${SCRIPTS_DIR}/add_new_overlap.py \
            --agg_path '${AGG_DIR}' \
            --new_file '${NEW_FILE}' \
            --dataset_name '${DATASET_NAME}' \
            --original_col '${ORIGINAL_COL}' \
            --new_col '${NEW_COL}'")
echo "AGG_JOB_ID=${AGG_JOB_ID}"

echo "Submitting calculate_auerc.py (after aggregate) ..."
AURC_JOB_ID=$(submit \
  --dependency=afterok:${AGG_JOB_ID} \
  -t 0-06:00:00 --mem=32G -J "auerc_${DATASET_NAME}" \
  -o "${LOG_DIR}/auerc_${DATASET_NAME}-%j.out" \
  -e "${LOG_DIR}/auerc_${DATASET_NAME}-%j.err" \
  --wrap "source ~/.bashrc && conda activate ${CONDA_ENV} && \
          mkdir -p '${AURC_DIR}' && \
          python ${SCRIPTS_DIR}/calculate_auerc.py \
            --agg_path '${AGG_DIR}' \
            --res_path '${AURC_DIR}' \
            --dataset_name '${DATASET_NAME}' \
            --n_bootstrap '${N_BOOTSTRAP}' \
            --reference_method '${REFERENCE_METHOD}' \
            --method_cols '${METHOD_COLS}'")
echo "AURC_JOB_ID=${AURC_JOB_ID}"

echo "Submitted jobs:"
echo "  aggregate:      ${AGG_JOB_ID}"
echo "  auerc:          ${AURC_JOB_ID}"