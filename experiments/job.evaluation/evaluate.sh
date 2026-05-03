#!/usr/bin/env bash
set -euo pipefail

# bash /ocean/projects/bio240051p/aprieto/job.gtex_evaluation/evaluate.sh

# Slurm settings
PARTITIONS="${PARTITIONS:-RM-shared}"
LOG_DIR="${LOG_DIR:-/ocean/projects/bio240051p/aprieto/logs}"
CONDA_ENV="${CONDA_ENV:-ctar}"

# Dataset name used consistently across steps
DATASET_NAME="${DATASET_NAME:-pbmc}"
FILTER_TAG="${FILTER_TAG:-unfiltered}"

# Paths (match repository defaults in scripts)
REPO_ROOT="${REPO_ROOT:-/ocean/projects/bio240051p/aprieto}"
SCRIPTS_DIR="${SCRIPTS_DIR:-${REPO_ROOT}/job.gtex_evaluation}"

# Make union output directory (used by make_union_bed.py)
UNION_DIR="${UNION_DIR:-/ocean/projects/bio240051p/aprieto/bedfiles/validation/union_links/${FILTER_TAG}}"
SCENT_FILE="${SCENT_FILE:-/ocean/projects/bio240051p/aprieto/multiome/results/scent/myscent_pbmc_cts.csv}"
SCMM_FILE="${SCMM_FILE:-/ocean/projects/bio240051p/aprieto/multiome/results/scmultimap/${FILTER_TAG}/scmultimap_${DATASET_NAME}_cis.csv}"
SIGNAC_FILE="${SIGNAC_FILE:-/ocean/projects/bio240051p/aprieto/multiome/results/signac/${FILTER_TAG}/signac_${DATASET_NAME}_links.csv}"
CTAR_FILE="${CTAR_FILE:-/ocean/projects/bio240051p/aprieto/multiome/results/ctar/final_eval/${DATASET_NAME}/${DATASET_NAME}_${FILTER_TAG}_5.5.5.5.1000/cis_links_df.csv}"
CTAR_FILT_FILE="${CTAR_FILT_FILE:-/ocean/projects/bio240051p/aprieto/multiome/results/ctar/final_eval/${DATASET_NAME}/${DATASET_NAME}_${FILTER_TAG}_corr_5.5.5.5.1000/cis_links_df.csv}"

SCENT_COL="${SCENT_COL:-boot_basic_p}"
SCMM_COL="${SCMM_COL:-pval}"
SIGNAC_COL="${SIGNAC_COL:-pvalue}"
CTAR_COL_Z="${CTAR_COL_Z:-5.5.5.5.1000_mcpval_z}"
CTAR_COL="${CTAR_COL:-5.5.5.5.1000_mcpval}"
CTAR_FILT_COL_Z="${CTAR_FILT_COL_Z:-5.5.5.5.1000_mcpval_z}"
CTAR_FILT_COL="${CTAR_FILT_COL:-5.5.5.5.1000_mcpval}"

DEDUP="${DEDUP:-min}"
METHOD_COLS="${METHOD_COLS:-scent,scmm,signac,ctar_z,ctar,ctar_filt_z,ctar_filt}"

# crispr_k562 or gtex_tissues
# Intersect inputs/outputs
EVAL_DIR="${EVAL_DIR:-/ocean/projects/bio240051p/aprieto/bedfiles/validation/tissue_max/gtex_tissues}"
OVERLAP_DIR="${OVERLAP_DIR:-/ocean/projects/bio240051p/aprieto/bedfiles/validation/overlap/${DATASET_NAME}/${FILTER_TAG}/gtex_tissues}"

# Aggregate outputs (aggregate_overlaps.py default)
AGG_DIR="${AGG_DIR:-/ocean/projects/bio240051p/aprieto/multiome/validation/overlap_${FILTER_TAG}/${DATASET_NAME}/gtex_tissues}"
MERGE_DIR="${MERGE_DIR:-/ocean/projects/bio240051p/aprieto/multiome/validation/evaluation_${FILTER_TAG}/${DATASET_NAME}/gtex_tissues}"

# AUERC outputs (calculate_auerc.py default)
AURC_DIR="${AURC_DIR:-/ocean/projects/bio240051p/aprieto/multiome/validation/evaluation/tables/metrics_jitter_no_dedup_${FILTER_TAG}/${DATASET_NAME}}"
REFERENCE_METHOD="${REFERENCE_METHOD:-ctar}"
N_BOOTSTRAP="${N_BOOTSTRAP:-1000}"

echo "Evaluating ${DATASET_NAME}, ${FILTER_TAG}..."

mkdir -p "${LOG_DIR}"

submit() {
  sbatch --parsable -p "${PARTITIONS}" "$@"
}

echo "Submitting make_union_bed.py ..."
UNION_JOB_ID=$(submit \
  --dependency=afterok:38796939 \
  -t 1-00:00:00 --ntasks-per-node=6 -J "union_${DATASET_NAME}" \
  -o "${LOG_DIR}/union_${DATASET_NAME}-%j.out" \
  -e "${LOG_DIR}/union_${DATASET_NAME}-%j.err" \
  --wrap "source ~/.bashrc && conda activate ${CONDA_ENV} && \
          mkdir -p '${UNION_DIR}' && \
          python ${SCRIPTS_DIR}/make_union_bed.py \
            --dataset_name '${DATASET_NAME}' \
            --bed_path '${UNION_DIR}' \
            --scent_file '${SCENT_FILE}' \
            --scmm_file '${SCMM_FILE}' \
            --signac_file '${SIGNAC_FILE}' \
            --ctar_file '${CTAR_FILE}' \
            --ctar_filt_file '${CTAR_FILT_FILE}'")
echo "UNION_JOB_ID=${UNION_JOB_ID}"

echo "Submitting bedtools intersect (after union) ..."
INTERSECT_JOB_ID=$(submit \
  --dependency=afterok:${UNION_JOB_ID} \
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
  --dependency=afterok:${AGG_JOB_ID} \
  -t 1-00:00:00 --ntasks-per-node=64 -J "merge_${DATASET_NAME}" \
  -o "${LOG_DIR}/merge_${DATASET_NAME}-%j.out" \
  -e "${LOG_DIR}/merge_${DATASET_NAME}-%j.err" \
  --wrap "source ~/.bashrc && conda activate ${CONDA_ENV} && \
          mkdir -p '${MERGE_DIR}' && \
          python ${SCRIPTS_DIR}/merge_methods.py \
            --dataset_name '${DATASET_NAME}' \
            --scent_file '${SCENT_FILE}' --scent_col '${SCENT_COL}' \
            --scmm_file '${SCMM_FILE}' --scmm_col '${SCMM_COL}' \
            --signac_file '${SIGNAC_FILE}' --signac_col '${SIGNAC_COL}' \
            --ctar_file '${CTAR_FILE}' --ctar_col '${CTAR_COL}' --ctar_col_z '${CTAR_COL_Z}' \
            --ctar_filt_file '${CTAR_FILT_FILE}' --ctar_filt_col '${CTAR_FILT_COL}' --ctar_filt_col_z '${CTAR_FILT_COL_Z}' \
            --dedup '${DEDUP}' \
            --agg_path '${AGG_DIR}' \
            --res_path '${MERGE_DIR}'")
echo "MERGE_JOB_ID=${MERGE_JOB_ID}"

echo "Submitting calculate_auerc_seeds.py (after merge) ..."
AURC_JOB_ID=$(submit \
  --dependency=afterok:${MERGE_JOB_ID} \
  -t 3-00:00:00 --ntasks-per-node=64 -J "auerc_${DATASET_NAME}" \
  -A bio240051p \
  -o "${LOG_DIR}/auerc_${DATASET_NAME}-%j.out" \
  -e "${LOG_DIR}/auerc_${DATASET_NAME}-%j.err" \
  --wrap "source ~/.bashrc && conda activate ${CONDA_ENV} && \
          mkdir -p '${AURC_DIR}' && \
          python ${SCRIPTS_DIR}/calculate_auerc_seeds.py \
            --merge_path '${MERGE_DIR}' \
            --res_path '${AURC_DIR}' \
            --dataset_name '${DATASET_NAME}' \
            --n_bootstrap '${N_BOOTSTRAP}' \
            --eqtl_pip_thres_list "0.5" \
            --eqtl_filter_mode "all_only" \
            --reference_method '${REFERENCE_METHOD}' \
            --method_cols '${METHOD_COLS}'")
echo "AURC_JOB_ID=${AURC_JOB_ID}"

echo "Submitted jobs:"
echo "  make_union_bed: ${UNION_JOB_ID}"
echo "  intersect:      ${INTERSECT_JOB_ID}"
echo "  aggregate:      ${AGG_JOB_ID}"
echo "  merge:          ${MERGE_JOB_ID}"
echo "  auerc:          ${AURC_JOB_ID}"