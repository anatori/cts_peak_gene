#!/usr/bin/env bash
set -euo pipefail

# bash /ocean/projects/bio240051p/aprieto/cts_peak_gene/experiments/job.evaluation/cts_evaluate.sh

# Slurm settings
PARTITIONS="${PARTITIONS:-RM-shared}"
LOG_DIR="${LOG_DIR:-/ocean/projects/bio240051p/aprieto/logs}"
CONDA_ENV="${CONDA_ENV:-ctar}"

# Paths
REPO_ROOT="${REPO_ROOT:-/ocean/projects/bio240051p/aprieto/cts_peak_gene}"
SCRIPTS_DIR="${SCRIPTS_DIR:-${REPO_ROOT}/experiments/job.evaluation}"

# Dataset name
DATASET_NAME="${DATASET_NAME:-bmmc}"
FILTER_TAG="${FILTER_TAG:-unfiltered}"

# Celltype config CSV: columns celltype, file_path, score_col, method_name, eval_dir (optional)
# Multiple rows per celltype (one per method)
CONFIG_FILE="${CONFIG_FILE:-${SCRIPTS_DIR}/methods_manifests/celltype_${DATASET_NAME}_methods.csv}"

# Global eval dir override — overrides any eval_dir value in the config CSV.
# If unset and a celltype has no eval_dir, evaluation steps (2-5) are skipped for that celltype.
EVAL_DIR="${EVAL_DIR:-/ocean/projects/bio240051p/aprieto/bedfiles/validation/tissue_max/crispr_k562}"

# Shared parameters
DEDUP="${DEDUP:-min}"
REFERENCE_METHOD="${REFERENCE_METHOD:-ctar_5.5.5.5.50k}"
N_BOOTSTRAP="${N_BOOTSTRAP:-1000}"
FDR_CORRECTION="${FDR_CORRECTION:-false}"
USE_QVAL="${USE_QVAL:-false}"

FDR_FLAG=""
[[ "${FDR_CORRECTION}" == "true" ]] && FDR_FLAG="--fdr_correction"
USE_QVAL_FLAG=""
[[ "${USE_QVAL}" == "true" ]] && USE_QVAL_FLAG="--use_qval"

mkdir -p "${LOG_DIR}"

submit() {
  sbatch --parsable -p "${PARTITIONS}" -A bio240051p "$@"
}

# METHOD_COLS can be set explicitly to override method columns for all celltypes.
# If unset, method columns are derived per celltype from the config CSV (handles missing methods).
# METHOD_COLS="${METHOD_COLS:-}"

echo "Reading cell type configuration from ${CONFIG_FILE}..."

# Store AURC job IDs for the consolidation dependency
TEMP_JOB_FILE=$(mktemp)

# Iterate over unique celltypes in config order
while IFS= read -r CELLTYPE; do

  echo ""
  echo "=========================================="
  echo "Processing cell type: ${CELLTYPE}"
  echo "=========================================="

  # Per-celltype directories
  UNION_DIR="/ocean/projects/bio240051p/aprieto/bedfiles/validation/union_links/${FILTER_TAG}/${CELLTYPE}"
  OVERLAP_DIR="/ocean/projects/bio240051p/aprieto/bedfiles/validation/overlap/${DATASET_NAME}/${FILTER_TAG}/${CELLTYPE}"
  AGG_DIR="/ocean/projects/bio240051p/aprieto/multiome/validation/overlap_${FILTER_TAG}/${DATASET_NAME}/${CELLTYPE}"
  MERGE_DIR="/ocean/projects/bio240051p/aprieto/multiome/validation/evaluation_${FILTER_TAG}/${DATASET_NAME}/${CELLTYPE}"
  AURC_DIR="/ocean/projects/bio240051p/aprieto/multiome/validation/tables/metrics_jitter_no_dedup_${FILTER_TAG}/${DATASET_NAME}/${CELLTYPE}"

  # Resolve eval_dir: global EVAL_DIR override > config CSV value (column may be absent)
  CSV_EVAL_DIR=$(python3 -c "
import csv
with open('${CONFIG_FILE}') as f:
    for row in csv.DictReader(f):
        if row.get('celltype','').strip() == '${CELLTYPE}':
            print(row.get('eval_dir','').strip())
            break
")
  EFFECTIVE_EVAL_DIR="${EVAL_DIR:-${CSV_EVAL_DIR}}"

  # Derive method columns for this celltype (global METHOD_COLS override takes precedence)
  if [[ -n "${METHOD_COLS:-}" ]]; then
    CELLTYPE_METHOD_COLS="${METHOD_COLS}"
  else
    CELLTYPE_METHOD_COLS=$(python3 -c "
import csv, sys
with open('${CONFIG_FILE}') as f:
    rows = list(csv.DictReader(f))
names = [r['method_name'].strip() for r in rows
         if r.get('celltype','').strip() == '${CELLTYPE}' and r.get('method_name','').strip()]
if not names:
    sys.exit('ERROR: no method_name values found for ${CELLTYPE} in ${CONFIG_FILE}')
print(','.join(names))
")
  fi
  echo "  METHOD_COLS: ${CELLTYPE_METHOD_COLS}"

  # Step 1: make_union_bed
  echo "Submitting make_union_bed for ${CELLTYPE}..."
  UNION_JOB_ID=$(submit \
    -t 1-00:00:00 --ntasks-per-node=32 -J "union_${DATASET_NAME}_${CELLTYPE}" \
    -o "${LOG_DIR}/union_${DATASET_NAME}_${CELLTYPE}-%j.out" \
    -e "${LOG_DIR}/union_${DATASET_NAME}_${CELLTYPE}-%j.err" \
    --wrap "source ~/.bashrc && conda activate ${CONDA_ENV} && \
            mkdir -p '${UNION_DIR}' && \
            python ${SCRIPTS_DIR}/make_union_bed.py \
              --dataset_name '${DATASET_NAME}_${CELLTYPE}' \
              --bed_path '${UNION_DIR}' \
              --config_file '${CONFIG_FILE}' \
              --celltype '${CELLTYPE}'")
  echo "  UNION_JOB_ID=${UNION_JOB_ID}"

  if [[ -z "${EFFECTIVE_EVAL_DIR}" ]]; then
    echo "  No eval_dir for ${CELLTYPE} and EVAL_DIR not set — skipping evaluation steps (2-5)."
    continue
  fi

  echo "  EVAL_DIR: ${EFFECTIVE_EVAL_DIR}"

  # Step 2: bedtools intersect via run_bedtools_intersect.sh
  echo "Submitting bedtools intersect for ${CELLTYPE}..."
  INTERSECT_JOB_ID=$(submit \
    --dependency=afterok:${UNION_JOB_ID} \
    -t 0-06:00:00 --ntasks-per-node=32 -J "intersect_${DATASET_NAME}_${CELLTYPE}" \
    -o "${LOG_DIR}/intersect_${DATASET_NAME}_${CELLTYPE}-%j.out" \
    -e "${LOG_DIR}/intersect_${DATASET_NAME}_${CELLTYPE}-%j.err" \
    --wrap "bash '${SCRIPTS_DIR}/run_bedtools_intersect.sh' \
              --dataset-name '${DATASET_NAME}_${CELLTYPE}' \
              --eval-dir '${EFFECTIVE_EVAL_DIR}' \
              --union-dir '${UNION_DIR}' \
              --overlap-dir '${OVERLAP_DIR}'")
  echo "  INTERSECT_JOB_ID=${INTERSECT_JOB_ID}"

  # Step 3: aggregate_overlaps
  echo "Submitting aggregate_overlaps for ${CELLTYPE}..."
  AGG_JOB_ID=$(submit \
    --dependency=afterok:${INTERSECT_JOB_ID} \
    -t 0-06:00:00 --ntasks-per-node=32 -J "aggregate_${DATASET_NAME}_${CELLTYPE}" \
    -o "${LOG_DIR}/aggregate_${DATASET_NAME}_${CELLTYPE}-%j.out" \
    -e "${LOG_DIR}/aggregate_${DATASET_NAME}_${CELLTYPE}-%j.err" \
    --wrap "source ~/.bashrc && conda activate ${CONDA_ENV} && \
            mkdir -p '${AGG_DIR}' && \
            python ${SCRIPTS_DIR}/aggregate_overlaps.py \
              --bed_path '${OVERLAP_DIR}' \
              --res_path '${AGG_DIR}' \
              --dataset_name '${DATASET_NAME}_${CELLTYPE}'")
  echo "  AGG_JOB_ID=${AGG_JOB_ID}"

  # Step 4: merge_methods
  echo "Submitting merge_methods for ${CELLTYPE}..."
  MERGE_JOB_ID=$(submit \
    --dependency=afterok:${AGG_JOB_ID} \
    -t 1-00:00:00 --ntasks-per-node=32 -J "merge_${DATASET_NAME}_${CELLTYPE}" \
    -o "${LOG_DIR}/merge_${DATASET_NAME}_${CELLTYPE}-%j.out" \
    -e "${LOG_DIR}/merge_${DATASET_NAME}_${CELLTYPE}-%j.err" \
    --wrap "source ~/.bashrc && conda activate ${CONDA_ENV} && \
            mkdir -p '${MERGE_DIR}' && \
            python ${SCRIPTS_DIR}/merge_methods.py \
              --dataset_name '${DATASET_NAME}_${CELLTYPE}' \
              --config_file '${CONFIG_FILE}' \
              --celltype '${CELLTYPE}' \
              --dedup '${DEDUP}' \
              --agg_path '${AGG_DIR}' \
              --res_path '${MERGE_DIR}' \
              ${FDR_FLAG}")
  echo "  MERGE_JOB_ID=${MERGE_JOB_ID}"

  # Step 5: calculate_auerc
  echo "Submitting calculate_auerc_seeds for ${CELLTYPE}..."
  AURC_JOB_ID=$(submit \
    --dependency=afterok:${MERGE_JOB_ID} \
    -t 3-00:00:00 --ntasks-per-node=32 -J "auerc_${DATASET_NAME}_${CELLTYPE}" \
    -o "${LOG_DIR}/auerc_${DATASET_NAME}_${CELLTYPE}-%j.out" \
    -e "${LOG_DIR}/auerc_${DATASET_NAME}_${CELLTYPE}-%j.err" \
    --wrap "source ~/.bashrc && conda activate ${CONDA_ENV} && \
            mkdir -p '${AURC_DIR}' && \
            python ${SCRIPTS_DIR}/calculate_auerc_seeds.py \
              --merge_path '${MERGE_DIR}' \
              --res_path '${AURC_DIR}' \
              --dataset_name '${DATASET_NAME}_${CELLTYPE}' \
              --n_bootstrap '${N_BOOTSTRAP}' \
              --reference_method '${REFERENCE_METHOD}' \
              --method_cols '${CELLTYPE_METHOD_COLS}' \
              ${USE_QVAL_FLAG}")
  echo "  AURC_JOB_ID=${AURC_JOB_ID}"

  echo "${AURC_JOB_ID}" >> "${TEMP_JOB_FILE}"

done < <(python3 -c "
import csv
seen = []
with open('${CONFIG_FILE}') as f:
    for row in csv.DictReader(f):
        ct = row.get('celltype','').strip()
        if ct and ct not in seen:
            seen.append(ct)
print('\n'.join(seen))
")

# Consolidate after all cell-type AURC jobs finish
AURC_DEPS=$(paste -sd: "${TEMP_JOB_FILE}")
rm -f "${TEMP_JOB_FILE}"

if [[ -n "${AURC_DEPS}" ]]; then
  BASE_AURC_DIR="/ocean/projects/bio240051p/aprieto/multiome/validation/tables/metrics_jitter_no_dedup_${FILTER_TAG}/${DATASET_NAME}"
  echo ""
  echo "Submitting consolidation job..."
  CONSOLIDATE_JOB_ID=$(submit \
    --dependency=afterok:${AURC_DEPS} \
    -t 0-02:00:00 --ntasks-per-node=8 -J "consolidate_${DATASET_NAME}" \
    -o "${LOG_DIR}/consolidate_${DATASET_NAME}-%j.out" \
    -e "${LOG_DIR}/consolidate_${DATASET_NAME}-%j.err" \
    --wrap "source ~/.bashrc && conda activate ${CONDA_ENV} && \
            python ${SCRIPTS_DIR}/consolidate_celltype_results.py \
              --config_file '${CONFIG_FILE}' \
              --dataset_name '${DATASET_NAME}' \
              --base_aurc_dir '${BASE_AURC_DIR}' \
              --output_dir '${BASE_AURC_DIR}/consolidated'")
  echo "  CONSOLIDATE_JOB_ID=${CONSOLIDATE_JOB_ID}"
fi

echo ""
echo "All jobs submitted!"