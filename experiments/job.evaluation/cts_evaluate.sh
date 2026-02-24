#!/usr/bin/env bash
set -euo pipefail

# bash /projects/zhanglab/users/ana/cts_peak_gene/experiments/job.evaluation/cts_evaluate.sh

# Slurm settings
PARTITIONS="${PARTITIONS:-mzhang,pool1}"
LOG_DIR="${LOG_DIR:-/projects/zhanglab/users/ana/logs}"
CONDA_ENV="${CONDA_ENV:-ctar}"

# Paths
REPO_ROOT="${REPO_ROOT:-/projects/zhanglab/users/ana/cts_peak_gene}"
SCRIPTS_DIR="${SCRIPTS_DIR:-${REPO_ROOT}/experiments/job.evaluation}"
CONFIG_FILE="${CONFIG_FILE:-${SCRIPTS_DIR}/celltype_config.csv}"

# Dataset name
DATASET_NAME="${DATASET_NAME:-pbmc}"

# Shared parameters
DEDUP="${DEDUP:-min}"
METHOD_COLS="${METHOD_COLS:-scent,scmm,signac,ctar_filt_z,ctar_filt}"
REFERENCE_METHOD="${REFERENCE_METHOD:-ctar_filt}"
N_BOOTSTRAP="${N_BOOTSTRAP:-1000}"

# Column name specs (these should be the same across cell types)
SCENT_COL="${SCENT_COL:-boot_basic_p}"
SCMM_COL="${SCMM_COL:-pval}"
SIGNAC_COL="${SIGNAC_COL:-pvalue}"
CTAR_FILT_COL_Z="${CTAR_FILT_COL_Z:-5.5.5.5.1000_mcpval_z}"
CTAR_FILT_COL="${CTAR_FILT_COL:-5.5.5.5.1000_mcpval}"

BEDTOOLS_BIN="${BEDTOOLS_BIN:-/projects/zhanglab/users/ana/bedtools2/bin/intersectBed}"

mkdir -p "${LOG_DIR}"

submit() {
  sbatch --parsable -p "${PARTITIONS}" "$@"
}

# Read config file and submit jobs for each cell type
echo "Reading cell type configuration from ${CONFIG_FILE}..."

# Store job IDs in a file
TEMP_JOB_FILE=$(mktemp)

# Skip header line and process each row
tail -n +2 "${CONFIG_FILE}" | while IFS=, read -r CELLTYPE SCENT_FILE SIGNAC_FILE CTAR_FILT_FILE SCMM_FILE ONEK1K_BED; do
  
  echo ""
  echo "=========================================="
  echo "Processing cell type: ${CELLTYPE}"
  echo "=========================================="

  echo "SCENT_FILE: ${SCENT_FILE}"
  echo "SCMM_FILE: ${SCMM_FILE}"
  echo "SIGNAC_FILE: ${SIGNAC_FILE}"
  echo "CTAR_FILT_FILE: ${CTAR_FILT_FILE}"
  echo "ONEK1K_BED: ${ONEK1K_BED}"
  
  # Set up cell-type-specific directories
  UNION_DIR="/projects/zhanglab/users/ana/bedtools2/ana_bedfiles/validation/union_links/no_score/${CELLTYPE}"
  OVERLAP_DIR="/projects/zhanglab/users/ana/bedtools2/ana_bedfiles/validation/overlap/${DATASET_NAME}/no_score/${CELLTYPE}"
  AGG_DIR="/projects/zhanglab/users/ana/multiome/validation/overlap/${DATASET_NAME}/${CELLTYPE}"
  MERGE_DIR="/projects/zhanglab/users/ana/multiome/validation/evaluation/${DATASET_NAME}/${CELLTYPE}"
  AURC_DIR="/projects/zhanglab/users/ana/multiome/validation/evaluation/tables/metrics/${DATASET_NAME}/${CELLTYPE}"
  
  # # Step 1: make_union_bed
  # echo "Submitting make_union_bed for ${CELLTYPE}..."
  # UNION_JOB_ID=$(submit \
  #   -t 1-00:00:00 --mem=128G -J "union_${DATASET_NAME}_${CELLTYPE}" \
  #   -o "${LOG_DIR}/union_${DATASET_NAME}_${CELLTYPE}-%j.out" \
  #   -e "${LOG_DIR}/union_${DATASET_NAME}_${CELLTYPE}-%j.err" \
  #   --wrap "source ~/.bashrc && conda activate ${CONDA_ENV} && \
  #           mkdir -p '${UNION_DIR}' && \
  #           python ${SCRIPTS_DIR}/make_union_bed.py \
  #             --dataset_name '${DATASET_NAME}_${CELLTYPE}' \
  #             --bed_path '${UNION_DIR}' \
  #             --scent_file '${SCENT_FILE}' \
  #             --scmm_file '${SCMM_FILE}' \
  #             --signac_file '${SIGNAC_FILE}' \
  #             --ctar_file '' \
  #             --ctar_filt_file '${CTAR_FILT_FILE}'")
  # echo "  UNION_JOB_ID=${UNION_JOB_ID}"
  
  # # Step 2: bedtools intersect
  # echo "Submitting bedtools intersect for ${CELLTYPE}..."
  # INTERSECT_JOB_ID=$(submit \
  #   --dependency=afterok:${UNION_JOB_ID} \
  #   -t 0-06:00:00 --mem=16G -J "intersect_${DATASET_NAME}_${CELLTYPE}" \
  #   -o "${LOG_DIR}/intersect_${DATASET_NAME}_${CELLTYPE}-%j.out" \
  #   -e "${LOG_DIR}/intersect_${DATASET_NAME}_${CELLTYPE}-%j.err" \
  #   --wrap "mkdir -p '${OVERLAP_DIR}' && \
  #           '${BEDTOOLS_BIN}' -wo \
  #             -a '${ONEK1K_BED}' \
  #             -b '${UNION_DIR}/${DATASET_NAME}_${CELLTYPE}.bed' \
  #             > '${OVERLAP_DIR}/onek1k_${DATASET_NAME}_${CELLTYPE}.bed'")
  # echo "  INTERSECT_JOB_ID=${INTERSECT_JOB_ID}"
  
  # # Step 3: aggregate_overlaps
  # echo "Submitting aggregate_overlaps for ${CELLTYPE}..."
  # AGG_JOB_ID=$(submit \
  #   --dependency=afterok:${INTERSECT_JOB_ID} \
  #   -t 0-06:00:00 --mem=32G -J "aggregate_${DATASET_NAME}_${CELLTYPE}" \
  #   -o "${LOG_DIR}/aggregate_${DATASET_NAME}_${CELLTYPE}-%j.out" \
  #   -e "${LOG_DIR}/aggregate_${DATASET_NAME}_${CELLTYPE}-%j.err" \
  #   --wrap "source ~/.bashrc && conda activate ${CONDA_ENV} && \
  #           mkdir -p '${AGG_DIR}' && \
  #           python ${SCRIPTS_DIR}/aggregate_overlaps.py \
  #             --bed_path '${OVERLAP_DIR}' \
  #             --res_path '${AGG_DIR}' \
  #             --dataset_name '${DATASET_NAME}_${CELLTYPE}'")
  # echo "  AGG_JOB_ID=${AGG_JOB_ID}"
  
  # # Step 4: merge_methods
  # echo "Submitting merge_methods for ${CELLTYPE}..."
  # MERGE_JOB_ID=$(submit \
  #   --dependency=afterok:${AGG_JOB_ID} \
  #   -t 1-00:00:00 --mem=128G -J "merge_${DATASET_NAME}_${CELLTYPE}" \
  #   -o "${LOG_DIR}/merge_${DATASET_NAME}_${CELLTYPE}-%j.out" \
  #   -e "${LOG_DIR}/merge_${DATASET_NAME}_${CELLTYPE}-%j.err" \
  #   --wrap "source ~/.bashrc && conda activate ${CONDA_ENV} && \
  #           mkdir -p '${MERGE_DIR}' && \
  #           python ${SCRIPTS_DIR}/merge_methods.py \
  #             --dataset_name '${DATASET_NAME}_${CELLTYPE}' \
  #             --scent_file '${SCENT_FILE}' --scent_col '${SCENT_COL}' \
  #             --scmm_file '${SCMM_FILE}' --scmm_col '${SCMM_COL}' \
  #             --signac_file '${SIGNAC_FILE}' --signac_col '${SIGNAC_COL}' \
  #             --ctar_file '' --ctar_col '' --ctar_col_z '' \
  #             --ctar_filt_file '${CTAR_FILT_FILE}' --ctar_filt_col '${CTAR_FILT_COL}' --ctar_filt_col_z '${CTAR_FILT_COL_Z}' \
  #             --dedup '${DEDUP}' \
  #             --agg_path '${AGG_DIR}' \
  #             --res_path '${MERGE_DIR}'")
  # echo "  MERGE_JOB_ID=${MERGE_JOB_ID}"

  #     --dependency=afterok:${MERGE_JOB_ID} \
  # Step 5: calculate_auerc
  echo "Submitting calculate_auerc_seeds for ${CELLTYPE}..."
  AURC_JOB_ID=$(submit \
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
  
  # Save job ID to temp file
  echo "${AURC_JOB_ID}" >> "${TEMP_JOB_FILE}"
  
done

# Read job IDs from temp file
AURC_DEPS=$(paste -sd: "${TEMP_JOB_FILE}")
rm -f "${TEMP_JOB_FILE}"

# Only submit consolidation if we have jobs
if [[ -n "${AURC_DEPS}" ]]; then
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
              --base_aurc_dir '/projects/zhanglab/users/ana/multiome/validation/evaluation/tables/metrics/${DATASET_NAME}' \
              --output_dir '/projects/zhanglab/users/ana/multiome/validation/evaluation/tables/metrics/${DATASET_NAME}/consolidated'")
  
  echo "  CONSOLIDATE_JOB_ID=${CONSOLIDATE_JOB_ID}"
fi

echo ""
echo "All jobs submitted!"