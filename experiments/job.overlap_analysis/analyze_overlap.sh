#!/usr/bin/env bash
set -euo pipefail

# bash /projects/zhanglab/users/ana/cts_peak_gene/experiments/job.overlap_analysis/analyze_overlap.sh

# Slurm settings
PARTITIONS="${PARTITIONS:-mzhang,statgen,pool1}"
LOG_DIR="${LOG_DIR:-/projects/zhanglab/users/ana/logs}"
CONDA_ENV="${CONDA_ENV:-ctar}"

# Datasets to include in the union BED
DATASETS="${DATASETS:-brain,brain_10k,hdma_brain}"
FILTER_TAG="${FILTER_TAG:-unfiltered}"

# Paths
REPO_ROOT="${REPO_ROOT:-/projects/zhanglab/users/ana/cts_peak_gene}"
SCRIPTS_DIR="${SCRIPTS_DIR:-${REPO_ROOT}/experiments/job.overlap_analysis}"
BEDTOOLS_BIN="${BEDTOOLS_BIN:-/projects/zhanglab/users/ana/bedtools2/bin/intersectBed}"

# Methods manifest CSV: columns file_path, dataset_name, method_name, pval_col, fdr_col, stat_col
METHODS_CSV="${METHODS_CSV:-${SCRIPTS_DIR}/methods_manifests/brain_methods.csv}"

# Directory/file paths
DATASET_DIR="${DATASET_DIR:-/projects/zhanglab/users/ana/bedtools2/ana_bedfiles/validation/union_links/${FILTER_TAG}}"
UNION_DIR="${UNION_DIR:-/projects/zhanglab/users/ana/bedtools2/ana_bedfiles/analysis/${FILTER_TAG}}"
UNION_BED="${UNION_BED:-${UNION_DIR}/multiome_brain.bed}"
ANALYSIS_BED="${ANALYSIS_BED:-/projects/zhanglab/users/ana/bedtools2/ana_bedfiles/analysis/funqtl_eqtl_broad_cts/funqtl_eqtl_broad_cts.bed}"
OVERLAP_BED="${OVERLAP_BED:-/projects/zhanglab/users/ana/bedtools2/ana_bedfiles/analysis/overlap/${FILTER_TAG}/multiome_funqtl_eqtl_broad_cts.bed}"
OUT_PATH="${OUT_PATH:-/projects/zhanglab/users/ana/multiome/analysis/funqtl_eqtl_broad_cts/${FILTER_TAG}/funqtl_eqtl_broad_cts_merged.csv}"

# Optional SNP annotation join
SNP_ANNOT_FILE="${SNP_ANNOT_FILE:-/projects/zhanglab/users/ana/multiome/analysis/funqtl_brain/funqtl_brain_info.tsv}"
SNP_ANNOT_ID_COL="${SNP_ANNOT_ID_COL:-final_col}"
SNP_ANNOT_COLS="${SNP_ANNOT_COLS:-astrocytes_celltype,astrocytes_pip,neurons_celltype,neurons_pip,oligodendrocytes_celltype,oligodendrocytes_pip,microglia_celltype,microglia_pip}"

DEDUP="${DEDUP:-min}"

# Optional gene-match filter: keep only overlaps where the gene in the SNP ID matches the
# link gene. Requires the analysis BED col-4 to be in `variant;gene` compound format.
FILTER_GENE_MATCH="${FILTER_GENE_MATCH:-True}"

echo "Datasets: ${DATASETS}, filter: ${FILTER_TAG}"

mkdir -p "${LOG_DIR}"

submit() {
  sbatch --parsable -p "${PARTITIONS}" "$@"
}

echo "Submitting make_multiome_union_bed.py ..."
UNION_JOB_ID=$(submit \
  -t 1-00:00:00 --mem 128Gb -J "union_multiome" \
  -o "${LOG_DIR}/union_multiome-%j.out" \
  -e "${LOG_DIR}/union_multiome-%j.err" \
  --wrap "source ~/.bashrc && conda activate ${CONDA_ENV} && \
          mkdir -p '${UNION_DIR}' && \
          python ${SCRIPTS_DIR}/make_multiome_union_bed.py \
            --datasets '${DATASETS}' \
            --dataset_path '${DATASET_DIR}' \
            --union_bed '${UNION_BED}'")
echo "UNION_JOB_ID=${UNION_JOB_ID}"

echo "Submitting bedtools intersect ..."
INTERSECT_JOB_ID=$(submit \
  --dependency=afterok:${UNION_JOB_ID} \
  -t 0-02:00:00 --mem 32Gb -J "intersect_analysis" \
  -o "${LOG_DIR}/intersect_analysis-%j.out" \
  -e "${LOG_DIR}/intersect_analysis-%j.err" \
  --wrap "mkdir -p '$(dirname ${OVERLAP_BED})' && \
          bash ${SCRIPTS_DIR}/run_bedtools_intersect.sh \
            --analysis-bed '${ANALYSIS_BED}' \
            --union-bed '${UNION_BED}' \
            --overlap-bed '${OVERLAP_BED}' \
            --bedtools-bin '${BEDTOOLS_BIN}'")
echo "INTERSECT_JOB_ID=${INTERSECT_JOB_ID}"

# Build optional SNP annotation args
ANNOT_ARGS=""
if [[ -n "${SNP_ANNOT_FILE}" ]]; then
  ANNOT_ARGS="--snp_annot_file '${SNP_ANNOT_FILE}' --snp_annot_id_col '${SNP_ANNOT_ID_COL}'"
  if [[ -n "${SNP_ANNOT_COLS}" ]]; then
    ANNOT_ARGS="${ANNOT_ARGS} --snp_annot_cols '${SNP_ANNOT_COLS}'"
  fi
fi

# Build optional gene-match filter arg
GENE_MATCH_ARGS=""
if [[ "${FILTER_GENE_MATCH,,}" == "true" ]]; then
  GENE_MATCH_ARGS="--filter_gene_match"
fi

echo "Submitting merge_methods.py ..."
MERGE_JOB_ID=$(submit \
  -t 1-00:00:00 --mem 128Gb -J "merge_analysis" \
  --dependency=afterok:${INTERSECT_JOB_ID} \
  -o "${LOG_DIR}/merge_analysis-%j.out" \
  -e "${LOG_DIR}/merge_analysis-%j.err" \
  --wrap "source ~/.bashrc && conda activate ${CONDA_ENV} && \
          mkdir -p '$(dirname ${OUT_PATH})' && \
          python ${SCRIPTS_DIR}/merge_methods.py \
            --overlap_bed '${OVERLAP_BED}' \
            --methods_csv '${METHODS_CSV}' \
            --out_path '${OUT_PATH}' \
            --dedup '${DEDUP}' \
            ${ANNOT_ARGS} \
            ${GENE_MATCH_ARGS}")
echo "MERGE_JOB_ID=${MERGE_JOB_ID}"

echo "Submitted jobs:"
echo "  make_union_bed: ${UNION_JOB_ID}"
echo "  intersect:      ${INTERSECT_JOB_ID}"
echo "  merge_methods:  ${MERGE_JOB_ID}"