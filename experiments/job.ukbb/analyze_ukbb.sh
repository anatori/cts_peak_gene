#!/usr/bin/env bash
set -euo pipefail

# bash /ocean/projects/bio240051p/aprieto/cts_peak_gene/experiments/job.ukbb/analyze_ukbb.sh

# Slurm settings
PARTITIONS="${PARTITIONS:-RM-shared}"
LOG_DIR="${LOG_DIR:-/ocean/projects/bio240051p/aprieto/logs}"
CONDA_ENV="${CONDA_ENV:-ctar}"

# Datasets to include in the union BED
DATASETS="${DATASETS:-bmmc,pbmc,brain,neat,shareseq}"
FILTER_TAG="${FILTER_TAG:-unfiltered}"

# Paths
REPO_ROOT="${REPO_ROOT:-/ocean/projects/bio240051p/aprieto/cts_peak_gene}"
SCRIPTS_DIR="${SCRIPTS_DIR:-${REPO_ROOT}/experiments/job.ukbb}"

# Methods manifest CSV: columns file_path, dataset_name, method_name, pval_col, fdr_col
METHODS_CSV="${METHODS_CSV:-${SCRIPTS_DIR}/methods_manifests/ukbb_methods.csv}"

# Directory/file paths
DATASET_DIR="${DATASET_DIR:-/ocean/projects/bio240051p/aprieto/bedfiles/validation/union_links/${FILTER_TAG}}"
UNION_DIR="${UNION_DIR:-/ocean/projects/bio240051p/aprieto/bedfiles/analysis/${FILTER_TAG}}"
ANALYSIS_BED="${ANALYSIS_BED:-/ocean/projects/bio240051p/aprieto/bedfiles/analysis/ukbb_hg38.bed}"
UNION_BED="${UNION_BED:-${UNION_DIR}/multiome_union.bed}"
OVERLAP_BED="${OVERLAP_BED:-/ocean/projects/bio240051p/aprieto/bedfiles/analysis/overlap/${FILTER_TAG}/multiome_ukbb.bed}"
OUT_PATH="${OUT_PATH:-/ocean/projects/bio240051p/aprieto/multiome/results/ukbb/${FILTER_TAG}/ukbb_merged.csv}"

# Optional SNP annotation join
SNP_ANNOT_FILE="${SNP_ANNOT_FILE:-}"
SNP_ANNOT_ID_COL="${SNP_ANNOT_ID_COL:-snp_id}"
SNP_ANNOT_COLS="${SNP_ANNOT_COLS:-}"

DEDUP="${DEDUP:-min}"

echo "Datasets: ${DATASETS}, filter: ${FILTER_TAG}"

mkdir -p "${LOG_DIR}"

submit() {
  sbatch --parsable -p "${PARTITIONS}" -A bio250028p "$@"
}

echo "Submitting make_multiome_union_bed.py ..."
UNION_JOB_ID=$(submit \
  -t 1-00:00:00 --ntasks-per-node=64 -J "union_ukbb" \
  -o "${LOG_DIR}/union_ukbb-%j.out" \
  -e "${LOG_DIR}/union_ukbb-%j.err" \
  --wrap "source ~/.bashrc && conda activate ${CONDA_ENV} && \
          mkdir -p '${UNION_DIR}' && \
          python ${SCRIPTS_DIR}/make_multiome_union_bed.py \
            --datasets '${DATASETS}' \
            --dataset_path '${DATASET_DIR}' \
            --union_path '${UNION_DIR}'")
echo "UNION_JOB_ID=${UNION_JOB_ID}"

echo "Submitting bedtools intersect ..."
INTERSECT_JOB_ID=$(submit \
  --dependency=afterok:${UNION_JOB_ID} \
  -t 0-02:00:00 --ntasks-per-node=64 -J "intersect_ukbb" \
  -o "${LOG_DIR}/intersect_ukbb-%j.out" \
  -e "${LOG_DIR}/intersect_ukbb-%j.err" \
  --wrap "mkdir -p '$(dirname ${OVERLAP_BED})' && \
          bash ${SCRIPTS_DIR}/run_bedtools_intersect.sh \
            --analysis-bed '${ANALYSIS_BED}' \
            --union-bed '${UNION_BED}' \
            --overlap-bed '${OVERLAP_BED}'")
echo "INTERSECT_JOB_ID=${INTERSECT_JOB_ID}"

# Build optional SNP annotation args
ANNOT_ARGS=""
if [[ -n "${SNP_ANNOT_FILE}" ]]; then
  ANNOT_ARGS="--snp_annot_file '${SNP_ANNOT_FILE}' --snp_annot_id_col '${SNP_ANNOT_ID_COL}'"
  if [[ -n "${SNP_ANNOT_COLS}" ]]; then
    ANNOT_ARGS="${ANNOT_ARGS} --snp_annot_cols '${SNP_ANNOT_COLS}'"
  fi
fi

echo "Submitting merge_methods.py ..."
MERGE_JOB_ID=$(submit \
  --dependency=afterok:${INTERSECT_JOB_ID} \
  -t 1-00:00:00 --ntasks-per-node=64 -J "merge_ukbb" \
  -o "${LOG_DIR}/merge_ukbb-%j.out" \
  -e "${LOG_DIR}/merge_ukbb-%j.err" \
  --wrap "source ~/.bashrc && conda activate ${CONDA_ENV} && \
          mkdir -p '$(dirname ${OUT_PATH})' && \
          python ${SCRIPTS_DIR}/merge_methods.py \
            --overlap_bed '${OVERLAP_BED}' \
            --methods_csv '${METHODS_CSV}' \
            --out_path '${OUT_PATH}' \
            --dedup '${DEDUP}' \
            ${ANNOT_ARGS}")
echo "MERGE_JOB_ID=${MERGE_JOB_ID}"

echo "Submitted jobs:"
echo "  make_union_bed: ${UNION_JOB_ID}"
echo "  intersect:      ${INTERSECT_JOB_ID}"
echo "  merge_methods:  ${MERGE_JOB_ID}"
