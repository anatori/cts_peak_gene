#!/usr/bin/env bash
set -euo pipefail

# bash /ocean/projects/bio240051p/aprieto/cts_peak_gene/experiments/job.evaluation/evaluate_specificity_quantiles.sh

# Slurm settings
PARTITIONS="${PARTITIONS:-RM-shared}"
LOG_DIR="${LOG_DIR:-/ocean/projects/bio240051p/aprieto/logs}"
CONDA_ENV="${CONDA_ENV:-ctar}"

# Dataset / repo paths
DATASET_NAME="${DATASET_NAME:-pbmc}"
REPO_ROOT="${REPO_ROOT:-/ocean/projects/bio240051p/aprieto/cts_peak_gene}"
SCRIPTS_DIR="${SCRIPTS_DIR:-${REPO_ROOT}/experiments/job.evaluation}"

# Inputs / outputs
MERGE_DIR="${MERGE_DIR:-/ocean/projects/bio240051p/aprieto/multiome/validation/evaluation_unfiltered/${DATASET_NAME}/gtex_tissues/gtex_best}"
AURC_DIR="${AURC_DIR:-/ocean/projects/bio240051p/aprieto/multiome/validation/tables/metrics_jitter_no_dedup_unfiltered_specificity/${DATASET_NAME}/broad_no_weight}"
TRUTH_SPEC_PATH="${TRUTH_SPEC_PATH:-}"
GENE_SPEC_PATH="${GENE_SPEC_PATH:-/ocean/projects/bio240051p/aprieto/multiome/analysis/broad_ct_specificity_with_vectors/rna/${DATASET_NAME}_specificity.csv}"
PEAK_SPEC_PATH="${PEAK_SPEC_PATH:-/ocean/projects/bio240051p/aprieto/multiome/analysis/broad_ct_specificity_with_vectors/atac/${DATASET_NAME}_specificity.csv}"
SUBTYPES_MAP_PATH="${SUBTYPES_MAP_PATH:-}"

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
USE_QVAL="${USE_QVAL:-false}"
if [[ "${USE_QVAL}" == "true" ]]; then
  METHOD_COLS=$(python3 -c "print(','.join(c.strip()+'_qval' for c in '${METHOD_COLS}'.split(',')))")
  REFERENCE_METHOD="${REFERENCE_METHOD}_qval"
fi

# Optional specificity-script settings
GTEX_SCORE_THRES="${GTEX_SCORE_THRES:-0.5}"
ABC_SCORE_THRES="${ABC_SCORE_THRES:-0.2}"
SPECIFICITY_COL="${SPECIFICITY_COL:-link_specificity_joint}"
GENE_SPEC_COL="${GENE_SPEC_COL:-specificity}"
PEAK_SPEC_COL="${PEAK_SPEC_COL:-specificity}"
SPECIFICITY_QUANTILES="${SPECIFICITY_QUANTILES:-0,0.25,0.5,0.75,1}"
SPECIFICITY_QUANTILE_LABELS="${SPECIFICITY_QUANTILE_LABELS:-}"
QUANTILE_RANK_METHOD="${QUANTILE_RANK_METHOD:-average}"
MIN_BIN_N="${MIN_BIN_N:-100}"
MIN_BIN_POS="${MIN_BIN_POS:-5}"
PRINT_BIN_SUMMARY="${PRINT_BIN_SUMMARY:-0}"
COMPUTE_OVERALL="${COMPUTE_OVERALL:-0}"
DROP_MISSING_METHOD_SCORES="${DROP_MISSING_METHOD_SCORES:-0}"
FORCE_RECOMPUTE_SPECIFICITY="${FORCE_RECOMPUTE_SPECIFICITY:-0}"
PEAK_COL="${PEAK_COL:-peak}"
GENE_COL="${GENE_COL:-gene}"
MERGE_TRUTH_ID_COL="${MERGE_TRUTH_ID_COL:-gtex_id}"
MERGE_TRUTH_GENE_COL="${MERGE_TRUTH_GENE_COL:-gt_gene}"
TRUTH_ID_COL="${TRUTH_ID_COL:-gtex_id}"
TRUTH_GENE_COL="${TRUTH_GENE_COL:-gt_gene}"
TRUTH_SPEC_COL="${TRUTH_SPEC_COL:-specificity}"
TRUTH_DUPLICATE_STRATEGY="${TRUTH_DUPLICATE_STRATEGY:-error}"
WEIGHT_CONCORDANCE="${WEIGHT_CONCORDANCE:-false}"

echo "Evaluating specificity-quantile-stratified AUERC for ${DATASET_NAME}..."
echo "METHOD_COLS: ${METHOD_COLS}"

mkdir -p "${LOG_DIR}"

submit() {
  sbatch --parsable -p "${PARTITIONS}" -A bio240051p "$@"
}

EXTRA_ARGS=()
if [[ -n "${TRUTH_SPEC_PATH}" ]]; then
  EXTRA_ARGS+=(--truth_specificity_path "${TRUTH_SPEC_PATH}")
fi
if [[ -n "${GENE_SPEC_PATH}" ]]; then
  EXTRA_ARGS+=(--gene_specificity_path "${GENE_SPEC_PATH}")
fi
if [[ -n "${PEAK_SPEC_PATH}" ]]; then
  EXTRA_ARGS+=(--peak_specificity_path "${PEAK_SPEC_PATH}")
fi
if [[ -n "${SUBTYPES_MAP_PATH}" ]]; then
  EXTRA_ARGS+=(--subtypes_map_path "${SUBTYPES_MAP_PATH}")
fi
if [[ -n "${WEIGHT_CONCORDANCE}" ]]; then
  EXTRA_ARGS+=(--weight_concordance "${WEIGHT_CONCORDANCE}")
fi
if [[ "${PRINT_BIN_SUMMARY}" == "1" ]]; then
  EXTRA_ARGS+=(--print_bin_summary)
fi
if [[ "${COMPUTE_OVERALL}" == "1" ]]; then
  EXTRA_ARGS+=(--compute_overall)
fi
if [[ "${DROP_MISSING_METHOD_SCORES}" == "1" ]]; then
  EXTRA_ARGS+=(--drop_missing_method_scores)
fi
if [[ "${FORCE_RECOMPUTE_SPECIFICITY}" == "1" ]]; then
  EXTRA_ARGS+=(--force_recompute_specificity)
fi
if [[ -n "${SPECIFICITY_QUANTILE_LABELS}" ]]; then
  EXTRA_ARGS+=(--specificity_quantile_labels "${SPECIFICITY_QUANTILE_LABELS}")
fi

echo "Submitting calculate_auerc_specificity_quantiles.py ..."
AURC_JOB_ID=$(submit \
  -A bio240051p \
  -t 3-00:00:00 --ntasks-per-node=64 -J "auerc_spec_${DATASET_NAME}" \
  -o "${LOG_DIR}/auerc_spec_${DATASET_NAME}-%j.out" \
  -e "${LOG_DIR}/auerc_spec_${DATASET_NAME}-%j.err" \
  --wrap "source ~/.bashrc && conda activate ${CONDA_ENV} && \
          mkdir -p '${AURC_DIR}' && \
          python ${SCRIPTS_DIR}/calculate_auerc_specificity_quantiles.py \
            --merge_path '${MERGE_DIR}' \
            --res_path '${AURC_DIR}' \
            --dataset_name '${DATASET_NAME}' \
            --n_bootstrap '${N_BOOTSTRAP}' \
            --reference_method '${REFERENCE_METHOD}' \
            --method_cols '${METHOD_COLS}' \
            --gtex_score_thres '${GTEX_SCORE_THRES}' \
            --abc_score_thres '${ABC_SCORE_THRES}' \
            --specificity_col '${SPECIFICITY_COL}' \
            --gene_spec_col '${GENE_SPEC_COL}' \
            --peak_spec_col '${PEAK_SPEC_COL}' \
            --specificity_quantiles '${SPECIFICITY_QUANTILES}' \
            --quantile_rank_method '${QUANTILE_RANK_METHOD}' \
            --peak_col '${PEAK_COL}' \
            --gene_col '${GENE_COL}' \
            --merge_truth_id_col '${MERGE_TRUTH_ID_COL}' \
            --merge_truth_gene_col '${MERGE_TRUTH_GENE_COL}' \
            --truth_id_col '${TRUTH_ID_COL}' \
            --truth_gene_col '${TRUTH_GENE_COL}' \
            --truth_specificity_col '${TRUTH_SPEC_COL}' \
            --truth_duplicate_strategy '${TRUTH_DUPLICATE_STRATEGY}' \
            --min_bin_n '${MIN_BIN_N}' \
            --min_bin_pos '${MIN_BIN_POS}' \
            ${EXTRA_ARGS[*]}")
echo "AURC_JOB_ID=${AURC_JOB_ID}"

echo "Submitted jobs:"
echo "  auerc_specificity: ${AURC_JOB_ID}"