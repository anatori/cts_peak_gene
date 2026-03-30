#!/usr/bin/env bash
set -euo pipefail

# bash /projects/zhanglab/users/ana/cts_peak_gene/experiments/job.evaluation/evaluate_distances.sh

# Slurm settings
PARTITIONS="${PARTITIONS:-statgen}"
LOG_DIR="${LOG_DIR:-/projects/zhanglab/users/ana/logs}"
CONDA_ENV="${CONDA_ENV:-ctar}"

# Dataset / repo paths
DATASET_NAME="${DATASET_NAME:-bmmc_donor1}"
REPO_ROOT="${REPO_ROOT:-/projects/zhanglab/users/ana/cts_peak_gene}"
SCRIPTS_DIR="${SCRIPTS_DIR:-${REPO_ROOT}/experiments/job.evaluation}"

# Inputs / outputs
MERGE_DIR="${MERGE_DIR:-/projects/zhanglab/users/ana/multiome/validation/evaluation/${DATASET_NAME}}"
AURC_DIR="${AURC_DIR:-/projects/zhanglab/users/ana/multiome/validation/evaluation/tables/metrics_jitter_distance/${DATASET_NAME}}"
GENCODE_GTF="${GENCODE_GTF:-/projects/zhanglab/users/ana/reference/gencode.gtf.gz}"

# Evaluation settings
METHOD_COLS="${METHOD_COLS:-scmm,signac,ctar_filt_z,ctar_filt}"
REFERENCE_METHOD="${REFERENCE_METHOD:-ctar_filt}"
N_BOOTSTRAP="${N_BOOTSTRAP:-1000}"

# Optional distance-script settings
GTEX_SCORE_THRES="${GTEX_SCORE_THRES:-0.5}"
ABC_SCORE_THRES="${ABC_SCORE_THRES:-0.2}"
MIN_BIN_N="${MIN_BIN_N:-100}"
MIN_BIN_POS="${MIN_BIN_POS:-10}"
REQUIRE_SAME_CHR="${REQUIRE_SAME_CHR:-0}"
PRINT_BIN_SUMMARY="${PRINT_BIN_SUMMARY:-0}"
COMPUTE_OVERALL="${COMPUTE_OVERALL:-0}"
DISTANCE_BINS_BP="${DISTANCE_BINS_BP:-}"
DISTANCE_BIN_LABELS="${DISTANCE_BIN_LABELS:-}"

echo "Evaluating distance-stratified AUERC for ${DATASET_NAME}..."

mkdir -p "${LOG_DIR}"

submit() {
  sbatch --parsable -p "${PARTITIONS}" "$@"
}

EXTRA_ARGS=()
if [[ "${REQUIRE_SAME_CHR}" == "1" ]]; then
  EXTRA_ARGS+=(--require_same_chr)
fi
if [[ "${PRINT_BIN_SUMMARY}" == "1" ]]; then
  EXTRA_ARGS+=(--print_bin_summary)
fi
if [[ "${COMPUTE_OVERALL}" == "1" ]]; then
  EXTRA_ARGS+=(--compute_overall)
fi
if [[ -n "${DISTANCE_BINS_BP}" ]]; then
  EXTRA_ARGS+=(--distance_bins_bp "${DISTANCE_BINS_BP}")
fi
if [[ -n "${DISTANCE_BIN_LABELS}" ]]; then
  EXTRA_ARGS+=(--distance_bin_labels "${DISTANCE_BIN_LABELS}")
fi

echo "Submitting calculate_auerc_distances.py ..."
AURC_JOB_ID=$(submit \
  -t 3-00:00:00 --mem=128G -J "auerc_dist_${DATASET_NAME}" \
  -o "${LOG_DIR}/auerc_dist_${DATASET_NAME}-%j.out" \
  -e "${LOG_DIR}/auerc_dist_${DATASET_NAME}-%j.err" \
  --wrap "source ~/.bashrc && conda activate ${CONDA_ENV} && \
          mkdir -p '${AURC_DIR}' && \
          python ${SCRIPTS_DIR}/calculate_auerc_distances.py \
            --merge_path '${MERGE_DIR}' \
            --res_path '${AURC_DIR}' \
            --dataset_name '${DATASET_NAME}' \
            --gencode_gtf '${GENCODE_GTF}' \
            --n_bootstrap '${N_BOOTSTRAP}' \
            --reference_method '${REFERENCE_METHOD}' \
            --method_cols '${METHOD_COLS}' \
            --gtex_score_thres '${GTEX_SCORE_THRES}' \
            --abc_score_thres '${ABC_SCORE_THRES}' \
            --min_bin_n '${MIN_BIN_N}' \
            --min_bin_pos '${MIN_BIN_POS}' \
            ${EXTRA_ARGS[*]}")
echo "AURC_JOB_ID=${AURC_JOB_ID}"

echo "Submitted jobs:"
echo "  auerc_distance: ${AURC_JOB_ID}"
