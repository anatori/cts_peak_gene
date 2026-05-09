#!/usr/bin/env bash
set -euo pipefail

# bash /ocean/projects/bio240051p/aprieto/cts_peak_gene/experiments/job.evaluation/evaluate_distances.sh

# Slurm settings
PARTITIONS="${PARTITIONS:-RM-shared}"
LOG_DIR="${LOG_DIR:-/ocean/projects/bio240051p/aprieto/logs}"
CONDA_ENV="${CONDA_ENV:-ctar}"

# Dataset / repo paths
DATASET_NAME="${DATASET_NAME:-brain}"
REPO_ROOT="${REPO_ROOT:-/ocean/projects/bio240051p/aprieto/cts_peak_gene}"
SCRIPTS_DIR="${SCRIPTS_DIR:-${REPO_ROOT}/experiments/job.evaluation}"

# Inputs / outputs
MERGE_DIR="${MERGE_DIR:-/ocean/projects/bio240051p/aprieto/multiome/validation/evaluation_unfiltered/${DATASET_NAME}/gtex_tissues/gtex_best}"
AURC_DIR="${AURC_DIR:-/ocean/projects/bio240051p/aprieto/multiome/validation/tables/metrics_jitter_no_dedup_unfiltered_distances/${DATASET_NAME}/gtex_tissues}"
GENCODE_GTF="${GENCODE_GTF:-/ocean/projects/bio240051p/aprieto/bedfiles/ref/gencode.v49.primary_assembly.annotation.gtf.gz}"

# Evaluation settings
# METHOD_COLS can be set explicitly or derived from a methods manifest CSV
METHODS_CSV="${METHODS_CSV:-${SCRIPTS_DIR}/methods_manifests/${DATASET_NAME}_methods.csv}"
if [[ -z "${METHOD_COLS:-}" && -n "${METHODS_CSV}" ]]; then
  METHOD_COLS=$(python3 -c "
import csv, sys
with open('${METHODS_CSV}') as f:
    rows = list(csv.DictReader(f))
names = [r['method_name'].strip() for r in rows if r.get('method_name','').strip()]
if not names:
    sys.exit('ERROR: no method_name values found in ${METHODS_CSV}')
print(','.join(names))
")
fi
REFERENCE_METHOD="${REFERENCE_METHOD:-ctar_5.5.5.5}"
N_BOOTSTRAP="${N_BOOTSTRAP:-1000}"

# Optional distance-script settings
GTEX_SCORE_THRES="${GTEX_SCORE_THRES:-0.5}"
ABC_SCORE_THRES="${ABC_SCORE_THRES:-0.2}"
MIN_BIN_N="${MIN_BIN_N:-10}"
MIN_BIN_POS="${MIN_BIN_POS:-1}"
REQUIRE_SAME_CHR="${REQUIRE_SAME_CHR:-0}"
PRINT_BIN_SUMMARY="${PRINT_BIN_SUMMARY:-0}"
COMPUTE_OVERALL="${COMPUTE_OVERALL:-0}"
DISTANCE_BINS_BP="${DISTANCE_BINS_BP:-0,5000,10000,20000,50000,inf}"
DISTANCE_BIN_LABELS="${DISTANCE_BIN_LABELS:-0-5kb,5-10kb,10-20kb,20-50kb,50kb+}"

echo "Evaluating distance-stratified AUERC for ${DATASET_NAME}..."
echo "METHOD_COLS: ${METHOD_COLS}"

mkdir -p "${LOG_DIR}"

submit() {
  sbatch --parsable -p "${PARTITIONS}" -A bio240051p "$@"
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
  -t 3-00:00:00 --ntasks-per-node=64 -J "auerc_dist_${DATASET_NAME}" \
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
