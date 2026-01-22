#!/usr/bin/env bash
set -euo pipefail

# bash /projects/zhanglab/users/ana/cts_peak_gene/experiments/job.evaluation/evaluate.sh

# Slurm settings
PARTITIONS="${PARTITIONS:-mzhang,pool1}"
LOG_DIR="${LOG_DIR:-/projects/zhanglab/users/ana/logs}"
CONDA_ENV="${CONDA_ENV:-ctar}"

# Dataset name used consistently across steps
DATASET_NAME="${DATASET_NAME:-pbmc}"

# Paths (match repository defaults in scripts)
REPO_ROOT="${REPO_ROOT:-/projects/zhanglab/users/ana/cts_peak_gene}"
SCRIPTS_DIR="${SCRIPTS_DIR:-${REPO_ROOT}/experiments/job.evaluation}"

# Make union output directory (used by make_union_bed.py)
UNION_DIR="${UNION_DIR:-/projects/zhanglab/users/ana/bedtools2/ana_bedfiles/validation/union_links}"
SCENT_FILE="${SCENT_FILE:-}"
SCMM_FILE="${SCMM_FILE:-/projects/zhanglab/users/ana/multiome/results/scmultimap/scmultimap_pbmc_cis.csv}"
SIGNAC_FILE="${SIGNAC_FILE:-/projects/zhanglab/users/ana/multiome/results/signac/signac_pbmc_links.csv}"
CTAR_FILE="${CTAR_FILE:-}"
CTAR_FILT_FILE="${CTAR_FILT_FILE:-/projects/zhanglab/users/ana/multiome/results/ctar/final_eval/pbmc/pbmc_filtered_5.5.5.5.1000/cis_links_df.csv}"

SCENT_COL="${SCENT_COL:-}"
SCMM_COL="${SCMM_COL:-pval}"
SIGNAC_COL="${SIGNAC_COL:-pvalue}"
CTAR_COL_Z="${CTAR_COL_Z:-}"
CTAR_COL="${CTAR_COL:-}"
CTAR_FILT_COL_Z="${CTAR_FILT_COL_Z:-5.5.5.5.1000_mcpval_z}"
CTAR_FILT_COL="${CTAR_FILT_COL:-5.5.5.5.1000_mcpval}"

METHOD_COLS="${METHOD_COLS:-scmm,signac,ctar_filt_z,ctar_filt}"

# Intersect inputs/outputs
EVAL_DIR="${EVAL_DIR:-/projects/zhanglab/users/ana/bedtools2/ana_bedfiles/validation/tissue_max}"
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

echo "Submitting make_union_bed.py ..."
UNION_JOB_ID=$(submit \
  -t 1-00:00:00 --mem=128G -J "union_${DATASET_NAME}" \
  -o "${LOG_DIR}/union_${DATASET_NAME}-%j.out" \
  -e "${LOG_DIR}/union_${DATASET_NAME}-%j.err" \
  --wrap "source ~/.bashrc && conda activate ${CONDA_ENV} && \
          mkdir -p '${UNION_DIR}' && \
          python ${SCRIPTS_DIR}/make_union_bed.py \
            --dataset_name '${DATASET_NAME}' \
            --bed_path '${UNION_DIR}' \
            --scent_file '${SCENT_FILE}' --scent_col '${SCENT_COL}' \
            --scmm_file '${SCMM_FILE}' --scmm_col '${SCMM_COL}' \
            --signac_file '${SIGNAC_FILE}' --signac_col '${SIGNAC_COL}' \
            --ctar_file '${CTAR_FILE}' --ctar_col '${CTAR_COL}' --ctar_col_z '${CTAR_COL_Z}' \
            --ctar_filt_file '${CTAR_FILT_FILE}' --ctar_filt_col '${CTAR_FILT_COL}' --ctar_filt_col_z '${CTAR_FILT_COL_Z}' \
            --method_cols '${METHOD_COLS}'")
echo "UNION_JOB_ID=${UNION_JOB_ID}"

echo "Submitting bedtools intersect (after union) ..."
INTERSECT_JOB_ID=$(submit \
  --dependency=afterok:${UNION_JOB_ID} \
  -t 0-06:00:00 --mem=16G -J "intersect_${DATASET_NAME}" \
  -o "${LOG_DIR}/intersect_${DATASET_NAME}-%j.out" \
  -e "${LOG_DIR}/intersect_${DATASET_NAME}-%j.err" \
  --wrap "bash ${SCRIPTS_DIR}/run_bedtools_intersect.sh \
           --dataset-name '${DATASET_NAME}' \
           --eval-dir '${EVAL_DIR}' \
           --union-dir '${UNION_DIR}' \
           --overlap-dir '${OVERLAP_DIR}' \
           --bedtools-bin '${BEDTOOLS_BIN}'")
echo "INTERSECT_JOB_ID=${INTERSECT_JOB_ID}"

echo "Submitting aggregate_overlaps.py (after intersect) ..."
AGG_JOB_ID=$(submit \
  --dependency=afterok:${INTERSECT_JOB_ID} \
  -t 0-06:00:00 --mem=32G -J "aggregate_${DATASET_NAME}" \
  -o "${LOG_DIR}/aggregate_${DATASET_NAME}-%j.out" \
  -e "${LOG_DIR}/aggregate_${DATASET_NAME}-%j.err" \
  --wrap "source ~/.bashrc && conda activate ${CONDA_ENV} && \
          mkdir -p '${AGG_DIR}' && \
          python ${SCRIPTS_DIR}/aggregate_overlaps.py \
            --bed_path '${OVERLAP_DIR}' \
            --res_path '${AGG_DIR}' \
            --dataset_name '${DATASET_NAME}' \
            --method_cols '${METHOD_COLS}'")
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
echo "  make_union_bed: ${UNION_JOB_ID}"
echo "  intersect:      ${INTERSECT_JOB_ID}"
echo "  aggregate:      ${AGG_JOB_ID}"
echo "  auerc:          ${AURC_JOB_ID}"