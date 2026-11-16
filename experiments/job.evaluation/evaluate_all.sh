#!/usr/bin/env bash
set -euo pipefail

# bash /projects/zhanglab/users/ana/cts_peak_gene/experiments/job.evaluation/evaluate_all.sh

# Slurm settings
PARTITIONS="${PARTITIONS:-mzhang,statgen}"
LOG_DIR="${LOG_DIR:-/projects/zhanglab/users/ana/logs}"
CONDA_ENV="${CONDA_ENV:-ctar}"

FILTER_TAG="${FILTER_TAG:-unfiltered}"
REPO_ROOT="${REPO_ROOT:-/projects/zhanglab/users/ana/cts_peak_gene}"
SCRIPTS_DIR="${SCRIPTS_DIR:-${REPO_ROOT}/experiments/job.evaluation}"
BEDTOOLS_BIN="${BEDTOOLS_BIN:-/projects/zhanglab/users/ana/bedtools2/bin/intersectBed}"

DEDUP="${DEDUP:-min}"
REFERENCE_METHOD="${REFERENCE_METHOD:-ctar}"
N_BOOTSTRAP="${N_BOOTSTRAP:-1000}"
FDR_CORRECTION="${FDR_CORRECTION:-true}"
USE_QVAL="${USE_QVAL:-true}"

# one Slurm array task per merge file (= per eval .bed in EVAL_DIR)
AURC_CPUS="${AURC_CPUS:-16}"

# Optionally rebuild the per-dataset union_links bed before its chains
RUN_UNION="${RUN_UNION:-true}"

# Rerun only the metrics step (calculate_auerc_seeds.py) on the existing merge files;
# skips union/intersect/aggregate/merge and their dependencies
ONLY_AUERC="${ONLY_AUERC:-false}"

FDR_FLAG=""
[[ "${FDR_CORRECTION}" == "true" ]] && FDR_FLAG="--fdr_correction"
USE_QVAL_FLAG=""
[[ "${USE_QVAL}" == "true" ]] && USE_QVAL_FLAG="--use_qval"

read -ra DATASETS <<< "${DATASETS_STR:-bmmc pbmc pbmc_male_chrom_ctrl pbmc_male_chrom_x hdma_spleen shareseq neat brain_10k brain}"
EVALS=(gwas_pops crispr_k562 gtex_tissues)

mkdir -p "${LOG_DIR}"

submit() {
  sbatch --parsable -p "${PARTITIONS}" "$@"
}

submit_union() {
  local DATASET="$1"

  local METHODS_CSV="${SCRIPTS_DIR}/methods_manifests/${DATASET}_methods.csv"
  local BASE="/projects/zhanglab/users/ana"
  local UNION_DIR="${BASE}/bedtools2/ana_bedfiles/validation/union_links/${FILTER_TAG}"

  submit \
    -t 1-00:00:00 --mem 128Gb -J "union_${DATASET}" \
    -o "${LOG_DIR}/union_${DATASET}-%j.out" \
    -e "${LOG_DIR}/union_${DATASET}-%j.err" \
    --wrap "source ~/.bashrc && conda activate ${CONDA_ENV} && \
            mkdir -p '${UNION_DIR}' && \
            python ${SCRIPTS_DIR}/make_union_bed.py \
              --dataset_name '${DATASET}' \
              --bed_path '${UNION_DIR}' \
              --methods_csv '${METHODS_CSV}'"
}

submit_chain() {
  local DATASET="$1"
  local EVAL="$2"
  local UNION_DEP="${3:-}"

  local METHODS_CSV="${SCRIPTS_DIR}/methods_manifests/${DATASET}_methods.csv"
  local METHOD_COLS
  METHOD_COLS=$(python3 -c "
import csv, sys
with open('${METHODS_CSV}') as f:
    rows = list(csv.DictReader(f))
names = [r['method_name'].strip() for r in rows if r.get('method_name','').strip()]
if not names:
    sys.exit('ERROR: no method_name values found in ${METHODS_CSV}')
print(','.join(names))
")

  local BASE="/projects/zhanglab/users/ana"
  local UNION_DIR="${BASE}/bedtools2/ana_bedfiles/validation/union_links/${FILTER_TAG}"
  local EVAL_DIR="${BASE}/bedtools2/ana_bedfiles/validation/tissue_max/${EVAL}"
  local OVERLAP_DIR="${BASE}/bedtools2/ana_bedfiles/validation/overlap/${DATASET}/${FILTER_TAG}/${EVAL}"
  local AGG_DIR="${BASE}/multiome/validation/overlap_${FILTER_TAG}/${DATASET}/${EVAL}"
  local MERGE_DIR="${BASE}/multiome/validation/evaluation_${FILTER_TAG}/${DATASET}/${EVAL}"
  local AURC_DIR="${BASE}/multiome/validation/tables/metrics_jitter_no_dedup_${FILTER_TAG}_qvalue_added_data/${DATASET}"

  echo "=== Submitting chain: dataset=${DATASET}, eval=${EVAL} ==="
  echo "  METHOD_COLS: ${METHOD_COLS}"

  local AURC_DEP_FLAG=""
  if [[ "${ONLY_AUERC}" != "true" ]]; then
  local INTERSECT_DEP_FLAG=""
  [[ -n "${UNION_DEP}" ]] && INTERSECT_DEP_FLAG="--dependency=${UNION_DEP}"

  local INTERSECT_JOB_ID
  INTERSECT_JOB_ID=$(submit \
    ${INTERSECT_DEP_FLAG} \
    -t 0-02:00:00 --mem 128Gb -J "intersect_${DATASET}_${EVAL}" \
    -o "${LOG_DIR}/intersect_${DATASET}_${EVAL}-%j.out" \
    -e "${LOG_DIR}/intersect_${DATASET}_${EVAL}-%j.err" \
    --wrap "mkdir -p '${OVERLAP_DIR}' && \
            bash ${SCRIPTS_DIR}/run_bedtools_intersect.sh \
             --dataset-name '${DATASET}' \
             --eval-dir '${EVAL_DIR}' \
             --union-dir '${UNION_DIR}' \
             --overlap-dir '${OVERLAP_DIR}' \
             --bedtools-bin '${BEDTOOLS_BIN}'")
  echo "  INTERSECT_JOB_ID=${INTERSECT_JOB_ID}"

  local AGG_JOB_ID
  AGG_JOB_ID=$(submit \
    --dependency=afterok:${INTERSECT_JOB_ID} \
    -t 0-06:00:00 --mem 128Gb -J "aggregate_${DATASET}_${EVAL}" \
    -o "${LOG_DIR}/aggregate_${DATASET}_${EVAL}-%j.out" \
    -e "${LOG_DIR}/aggregate_${DATASET}_${EVAL}-%j.err" \
    --wrap "source ~/.bashrc && conda activate ${CONDA_ENV} && \
            mkdir -p '${AGG_DIR}' && \
            python ${SCRIPTS_DIR}/aggregate_overlaps.py \
              --bed_path '${OVERLAP_DIR}' \
              --res_path '${AGG_DIR}' \
              --dataset_name '${DATASET}'")
  echo "  AGG_JOB_ID=${AGG_JOB_ID}"

  local MERGE_JOB_ID
  MERGE_JOB_ID=$(submit \
    --dependency=afterok:${AGG_JOB_ID} \
    -t 1-00:00:00 --mem 128Gb -J "merge_${DATASET}_${EVAL}" \
    -o "${LOG_DIR}/merge_${DATASET}_${EVAL}-%j.out" \
    -e "${LOG_DIR}/merge_${DATASET}_${EVAL}-%j.err" \
    --wrap "source ~/.bashrc && conda activate ${CONDA_ENV} && \
            mkdir -p '${MERGE_DIR}' && \
            python ${SCRIPTS_DIR}/merge_methods.py \
              --dataset_name '${DATASET}' \
              --methods_csv '${METHODS_CSV}' \
              --dedup '${DEDUP}' \
              --agg_path '${AGG_DIR}' \
              --res_path '${MERGE_DIR}' \
              ${FDR_FLAG}")
  echo "  MERGE_JOB_ID=${MERGE_JOB_ID}"
  AURC_DEP_FLAG="--dependency=afterok:${MERGE_JOB_ID}"
  fi

  # One array task per eval .bed (= per merge file); each task uses AURC_CPUS joblib workers.
  local N_EVAL_FILES
  N_EVAL_FILES=$(find "${EVAL_DIR}" -maxdepth 1 -name '*.bed' 2>/dev/null | wc -l | tr -d '[:space:]')
  [[ "${N_EVAL_FILES}" -lt 1 ]] && N_EVAL_FILES=1
  local AURC_ARRAY_FLAG=""
  [[ "${N_EVAL_FILES}" -gt 1 ]] && AURC_ARRAY_FLAG="--array=0-$((N_EVAL_FILES - 1))"

  local AURC_JOB_ID
  AURC_JOB_ID=$(submit \
    ${AURC_DEP_FLAG} ${AURC_ARRAY_FLAG} \
    -c "${AURC_CPUS}" \
    -t 3-00:00:00 --mem 128Gb -J "auerc_${DATASET}_${EVAL}" \
    -o "${LOG_DIR}/auerc_${DATASET}_${EVAL}-%j.out" \
    -e "${LOG_DIR}/auerc_${DATASET}_${EVAL}-%j.err" \
    --wrap "source ~/.bashrc && conda activate ${CONDA_ENV} && \
            mkdir -p '${AURC_DIR}' && \
            python ${SCRIPTS_DIR}/calculate_auerc_seeds.py \
              --merge_path '${MERGE_DIR}' \
              --res_path '${AURC_DIR}' \
              --dataset_name '${DATASET}' \
              --n_bootstrap '${N_BOOTSTRAP}' \
              --n_jobs '${AURC_CPUS}' \
              --task_id \"\${SLURM_ARRAY_TASK_ID:-0}\" \
              --n_tasks '${N_EVAL_FILES}' \
              --eqtl_pip_thres_list '0.5' \
              --eqtl_filter_mode 'all_only' \
              --reference_method '${REFERENCE_METHOD}' \
              --method_cols '${METHOD_COLS}' \
              ${USE_QVAL_FLAG}")
  echo "  AURC_JOB_ID=${AURC_JOB_ID}"
}

for DATASET in "${DATASETS[@]}"; do
  UNION_DEP=""
  if [[ "${RUN_UNION}" == "true" && "${ONLY_AUERC}" != "true" ]]; then
    echo "=== Submitting union: dataset=${DATASET} ==="
    UNION_JOB_ID=$(submit_union "${DATASET}")
    echo "  UNION_JOB_ID=${UNION_JOB_ID}"
    UNION_DEP="afterok:${UNION_JOB_ID}"
  fi
  for EVAL in "${EVALS[@]}"; do
    submit_chain "${DATASET}" "${EVAL}" "${UNION_DEP}"
  done
done

echo ""
if [[ "${RUN_UNION}" == "true" ]]; then
  echo "Submitted ${#DATASETS[@]} union jobs (one per dataset)."
fi
echo "All 15 chains submitted (5 datasets x 3 evals)."