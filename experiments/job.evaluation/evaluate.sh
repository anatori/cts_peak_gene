#!/usr/bin/env bash
set -euo pipefail

# bash /ocean/projects/bio240051p/aprieto/cts_peak_gene/experiments/job.evaluation/evaluate.sh

# Slurm settings
PARTITIONS="${PARTITIONS:-RM-shared}"
LOG_DIR="${LOG_DIR:-/ocean/projects/bio240051p/aprieto/logs}"
CONDA_ENV="${CONDA_ENV:-ctar}"

# Dataset name used consistently across steps
DATASET_NAME="${DATASET_NAME:-brain}"
FILTER_TAG="${FILTER_TAG:-unfiltered}"

# Paths
REPO_ROOT="${REPO_ROOT:-/ocean/projects/bio240051p/aprieto/cts_peak_gene}"
SCRIPTS_DIR="${SCRIPTS_DIR:-${REPO_ROOT}/experiments/job.evaluation}"

# Methods manifest CSV: columns file_path, score_col, method_name (row order = METHOD_COLS order)
METHODS_CSV="${METHODS_CSV:-${SCRIPTS_DIR}/methods_manifests/${DATASET_NAME}_methods.csv}"

# Derive METHOD_COLS from the manifest (override by setting METHOD_COLS explicitly)
if [[ -z "${METHOD_COLS:-}" ]]; then
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

# Directory paths
UNION_DIR="${UNION_DIR:-/ocean/projects/bio240051p/aprieto/bedfiles/validation/union_links/${FILTER_TAG}}"
EVAL_DIR="${EVAL_DIR:-/ocean/projects/bio240051p/aprieto/bedfiles/validation/tissue_max/gwas_pops}"
OVERLAP_DIR="${OVERLAP_DIR:-/ocean/projects/bio240051p/aprieto/bedfiles/validation/overlap/${DATASET_NAME}/${FILTER_TAG}/gwas_pops}"
AGG_DIR="${AGG_DIR:-/ocean/projects/bio240051p/aprieto/multiome/validation/overlap_${FILTER_TAG}/${DATASET_NAME}/gwas_pops}"
MERGE_DIR="${MERGE_DIR:-/ocean/projects/bio240051p/aprieto/multiome/validation/evaluation_${FILTER_TAG}/${DATASET_NAME}/gwas_pops}"
AURC_DIR="${AURC_DIR:-/ocean/projects/bio240051p/aprieto/multiome/validation/tables/metrics_jitter_no_dedup_${FILTER_TAG}/${DATASET_NAME}}"

DEDUP="${DEDUP:-min}"
REFERENCE_METHOD="${REFERENCE_METHOD:-ctar_5.5.5.5}"
N_BOOTSTRAP="${N_BOOTSTRAP:-1000}"

echo "Evaluating ${DATASET_NAME}, ${FILTER_TAG}..."
echo "METHOD_COLS: ${METHOD_COLS}"

mkdir -p "${LOG_DIR}"

submit() {
  sbatch --parsable -p "${PARTITIONS}" -A bio240051p "$@"
}

# echo "Submitting make_union_bed.py ..."
# UNION_JOB_ID=$(submit \
#   -A bio250028p \
#   -t 1-00:00:00 --ntasks-per-node=6 -J "union_${DATASET_NAME}" \
#   -o "${LOG_DIR}/union_${DATASET_NAME}-%j.out" \
#   -e "${LOG_DIR}/union_${DATASET_NAME}-%j.err" \
#   --wrap "source ~/.bashrc && conda activate ${CONDA_ENV} && \
#           mkdir -p '${UNION_DIR}' && \
#           python ${SCRIPTS_DIR}/make_union_bed.py \
#             --dataset_name '${DATASET_NAME}' \
#             --bed_path '${UNION_DIR}' \
#             --methods_csv '${METHODS_CSV}'")
# echo "UNION_JOB_ID=${UNION_JOB_ID}"

echo "Submitting bedtools intersect (after union) ..."
INTERSECT_JOB_ID=$(submit \
  -A bio250028p \
  -t 0-02:00:00 --ntasks-per-node=4 -J "intersect_${DATASET_NAME}" \
  -o "${LOG_DIR}/intersect_${DATASET_NAME}-%j.out" \
  -e "${LOG_DIR}/intersect_${DATASET_NAME}-%j.err" \
  --wrap "mkdir -p '${OVERLAP_DIR}' && \
          bash ${SCRIPTS_DIR}/run_bedtools_intersect.sh \
           --dataset-name '${DATASET_NAME}' \
           --eval-dir '${EVAL_DIR}' \
           --union-dir '${UNION_DIR}' \
           --overlap-dir '${OVERLAP_DIR}'")
echo "INTERSECT_JOB_ID=${INTERSECT_JOB_ID}"

echo "Submitting aggregate_overlaps.py (after intersect) ..."
AGG_JOB_ID=$(submit \
  -A bio250028p \
  --dependency=afterok:${INTERSECT_JOB_ID} \
  -t 0-06:00:00 --ntasks-per-node=32 -J "aggregate_${DATASET_NAME}" \
  -o "${LOG_DIR}/aggregate_${DATASET_NAME}-%j.out" \
  -e "${LOG_DIR}/aggregate_${DATASET_NAME}-%j.err" \
  --wrap "source ~/.bashrc && conda activate ${CONDA_ENV} && \
          mkdir -p '${AGG_DIR}' && \
          python ${SCRIPTS_DIR}/aggregate_overlaps.py \
            --bed_path '${OVERLAP_DIR}' \
            --res_path '${AGG_DIR}' \
            --dataset_name '${DATASET_NAME}'")
echo "AGG_JOB_ID=${AGG_JOB_ID}"

echo "Submitting merge_methods.py ..."
MERGE_JOB_ID=$(submit \
  -A bio250028p \
  --dependency=afterok:${AGG_JOB_ID} \
  -t 1-00:00:00 --ntasks-per-node=64 -J "merge_${DATASET_NAME}" \
  -o "${LOG_DIR}/merge_${DATASET_NAME}-%j.out" \
  -e "${LOG_DIR}/merge_${DATASET_NAME}-%j.err" \
  --wrap "source ~/.bashrc && conda activate ${CONDA_ENV} && \
          mkdir -p '${MERGE_DIR}' && \
          python ${SCRIPTS_DIR}/merge_methods.py \
            --dataset_name '${DATASET_NAME}' \
            --methods_csv '${METHODS_CSV}' \
            --dedup '${DEDUP}' \
            --agg_path '${AGG_DIR}' \
            --res_path '${MERGE_DIR}'")
echo "MERGE_JOB_ID=${MERGE_JOB_ID}"

echo "Submitting calculate_auerc_seeds.py (after merge) ..."
AURC_JOB_ID=$(submit \
  -A bio250028p \
  --dependency=afterok:${MERGE_JOB_ID} \
  -t 3-00:00:00 --ntasks-per-node=64 -J "auerc_${DATASET_NAME}" \
  -o "${LOG_DIR}/auerc_${DATASET_NAME}-%j.out" \
  -e "${LOG_DIR}/auerc_${DATASET_NAME}-%j.err" \
  --wrap "source ~/.bashrc && conda activate ${CONDA_ENV} && \
          mkdir -p '${AURC_DIR}' && \
          python ${SCRIPTS_DIR}/calculate_auerc_seeds.py \
            --merge_path '${MERGE_DIR}' \
            --res_path '${AURC_DIR}' \
            --dataset_name '${DATASET_NAME}' \
            --n_bootstrap '${N_BOOTSTRAP}' \
            --eqtl_pip_thres_list '0.5' \
            --eqtl_filter_mode 'all_only' \
            --reference_method '${REFERENCE_METHOD}' \
            --method_cols '${METHOD_COLS}'")
echo "AURC_JOB_ID=${AURC_JOB_ID}"

echo "Submitted jobs:"
# echo "  make_union_bed: ${UNION_JOB_ID}"
echo "  intersect:      ${INTERSECT_JOB_ID}"
echo "  aggregate:      ${AGG_JOB_ID}"
echo "  merge:          ${MERGE_JOB_ID}"
echo "  auerc:          ${AURC_JOB_ID}"