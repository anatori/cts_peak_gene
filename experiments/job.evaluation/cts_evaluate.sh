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

# Celltype config CSV: columns celltype, methods_csv, eval_bed
# Each row is one cell type; methods_csv is a per-celltype methods manifest
# (file_path, score_col, method_name) that drives make_union_bed and merge_methods.
CONFIG_FILE="${CONFIG_FILE:-${SCRIPTS_DIR}/celltype_config.csv}"

# Dataset name
DATASET_NAME="${DATASET_NAME:-pbmc}"
EVAL_NAME="${EVAL_NAME:-onek1k}"

# Shared parameters
DEDUP="${DEDUP:-min}"
REFERENCE_METHOD="${REFERENCE_METHOD:-ctar_filt}"
N_BOOTSTRAP="${N_BOOTSTRAP:-1000}"

mkdir -p "${LOG_DIR}"

submit() {
  sbatch --parsable -p "${PARTITIONS}" -A bio240051p "$@"
}

# Derive METHOD_COLS from the first cell type's methods CSV (all cell types must share the
# same method names and order; column names live in each per-celltype methods CSV).
# Can be overridden by setting METHOD_COLS explicitly before calling this script.
if [[ -z "${METHOD_COLS:-}" ]]; then
  FIRST_METHODS_CSV=$(tail -n +2 "${CONFIG_FILE}" | head -n 1 | cut -d',' -f2)
  METHOD_COLS=$(python3 -c "
import csv, sys
with open('${FIRST_METHODS_CSV}') as f:
    rows = list(csv.DictReader(f))
names = [r['method_name'].strip() for r in rows if r.get('method_name','').strip()]
if not names:
    sys.exit('ERROR: no method_name values found in ${FIRST_METHODS_CSV}')
print(','.join(names))
")
fi

echo "Reading cell type configuration from ${CONFIG_FILE}..."
echo "METHOD_COLS: ${METHOD_COLS}"

# Store AURC job IDs for the consolidation dependency
TEMP_JOB_FILE=$(mktemp)

# Skip header line and process each row
tail -n +2 "${CONFIG_FILE}" | while IFS=, read -r CELLTYPE METHODS_CSV EVAL_BED; do

  echo ""
  echo "=========================================="
  echo "Processing cell type: ${CELLTYPE}"
  echo "=========================================="
  echo "METHODS_CSV: ${METHODS_CSV}"
  echo "EVAL_BED:    ${EVAL_BED}"

  # Per-celltype directories (PSC)
  UNION_DIR="/ocean/projects/bio240051p/aprieto/bedfiles/validation/union_links/no_score/${CELLTYPE}"
  OVERLAP_DIR="/ocean/projects/bio240051p/aprieto/bedfiles/validation/overlap/${DATASET_NAME}/no_score/${CELLTYPE}"
  AGG_DIR="/ocean/projects/bio240051p/aprieto/multiome/validation/overlap/${DATASET_NAME}/${CELLTYPE}"
  MERGE_DIR="/ocean/projects/bio240051p/aprieto/multiome/validation/evaluation/${DATASET_NAME}/${CELLTYPE}"
  AURC_DIR="/ocean/projects/bio240051p/aprieto/multiome/validation/evaluation/tables/metrics/${DATASET_NAME}/${CELLTYPE}"

  # Step 1: make_union_bed
  echo "Submitting make_union_bed for ${CELLTYPE}..."
  UNION_JOB_ID=$(submit \
    -t 1-00:00:00 --mem=128G -J "union_${DATASET_NAME}_${CELLTYPE}" \
    -o "${LOG_DIR}/union_${DATASET_NAME}_${CELLTYPE}-%j.out" \
    -e "${LOG_DIR}/union_${DATASET_NAME}_${CELLTYPE}-%j.err" \
    --wrap "source ~/.bashrc && conda activate ${CONDA_ENV} && \
            mkdir -p '${UNION_DIR}' && \
            python ${SCRIPTS_DIR}/make_union_bed.py \
              --dataset_name '${DATASET_NAME}_${CELLTYPE}' \
              --bed_path '${UNION_DIR}' \
              --methods_csv '${METHODS_CSV}'")
  echo "  UNION_JOB_ID=${UNION_JOB_ID}"

  # Step 2: bedtools intersect
  echo "Submitting bedtools intersect for ${CELLTYPE}..."
  INTERSECT_JOB_ID=$(submit \
    --dependency=afterok:${UNION_JOB_ID} \
    -t 0-06:00:00 --mem=16G -J "intersect_${DATASET_NAME}_${CELLTYPE}" \
    -o "${LOG_DIR}/intersect_${DATASET_NAME}_${CELLTYPE}-%j.out" \
    -e "${LOG_DIR}/intersect_${DATASET_NAME}_${CELLTYPE}-%j.err" \
    --wrap "mkdir -p '${OVERLAP_DIR}' && \
            module load bedtools && \
            bedtools intersect -wo \
              -a '${EVAL_BED}' \
              -b '${UNION_DIR}/${DATASET_NAME}_${CELLTYPE}.bed' \
              > '${OVERLAP_DIR}/${EVAL_NAME}_${DATASET_NAME}_${CELLTYPE}.bed'")
  echo "  INTERSECT_JOB_ID=${INTERSECT_JOB_ID}"

  # Step 3: aggregate_overlaps
  echo "Submitting aggregate_overlaps for ${CELLTYPE}..."
  AGG_JOB_ID=$(submit \
    --dependency=afterok:${INTERSECT_JOB_ID} \
    -t 0-06:00:00 --mem=32G -J "aggregate_${DATASET_NAME}_${CELLTYPE}" \
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
    -t 1-00:00:00 --mem=128G -J "merge_${DATASET_NAME}_${CELLTYPE}" \
    -o "${LOG_DIR}/merge_${DATASET_NAME}_${CELLTYPE}-%j.out" \
    -e "${LOG_DIR}/merge_${DATASET_NAME}_${CELLTYPE}-%j.err" \
    --wrap "source ~/.bashrc && conda activate ${CONDA_ENV} && \
            mkdir -p '${MERGE_DIR}' && \
            python ${SCRIPTS_DIR}/merge_methods.py \
              --dataset_name '${DATASET_NAME}_${CELLTYPE}' \
              --methods_csv '${METHODS_CSV}' \
              --dedup '${DEDUP}' \
              --agg_path '${AGG_DIR}' \
              --res_path '${MERGE_DIR}'")
  echo "  MERGE_JOB_ID=${MERGE_JOB_ID}"

  # Step 5: calculate_auerc
  echo "Submitting calculate_auerc_seeds for ${CELLTYPE}..."
  AURC_JOB_ID=$(submit \
    --dependency=afterok:${MERGE_JOB_ID} \
    -t 3-00:00:00 --mem=32G -J "auerc_${DATASET_NAME}_${CELLTYPE}" \
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
              --method_cols '${METHOD_COLS}'")
  echo "  AURC_JOB_ID=${AURC_JOB_ID}"

  echo "${AURC_JOB_ID}" >> "${TEMP_JOB_FILE}"

done

# Consolidate after all cell-type AURC jobs finish
AURC_DEPS=$(paste -sd: "${TEMP_JOB_FILE}")
rm -f "${TEMP_JOB_FILE}"

if [[ -n "${AURC_DEPS}" ]]; then
  BASE_AURC_DIR="/ocean/projects/bio240051p/aprieto/multiome/validation/evaluation/tables/metrics/${DATASET_NAME}"
  echo ""
  echo "Submitting consolidation job..."
  CONSOLIDATE_JOB_ID=$(submit \
    --dependency=afterok:${AURC_DEPS} \
    -t 0-02:00:00 --mem=16G -J "consolidate_${DATASET_NAME}" \
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