#!/usr/bin/env bash
set -euo pipefail

# Defaults (PSC)
DATASET_NAME="${DATASET_NAME:-neat}"
EVAL_DIR="${EVAL_DIR:-/projects/zhanglab/users/ana/bedtools2/ana_bedfiles/validation/tissue_max}"
UNION_DIR="${UNION_DIR:-/projects/zhanglab/users/ana/bedtools2/ana_bedfiles/validation/union_links}"
OVERLAP_DIR="${OVERLAP_DIR:-/projects/zhanglab/users/ana/bedtools2/ana_bedfiles/validation/overlap/${DATASET_NAME}}"
BEDTOOLS_BIN="${BEDTOOLS_BIN:-/projects/zhanglab/users/ana/bedtools2/bin/intersectBed}"

# Parse CLI overrides
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-name) DATASET_NAME="$2"; shift 2;;
    --eval-dir)     EVAL_DIR="$2"; shift 2;;
    --union-dir)    UNION_DIR="$2"; shift 2;;
    --overlap-dir)  OVERLAP_DIR="$2"; shift 2;;
    --bedtools-bin) BEDTOOLS_BIN="$2"; shift 2;;
    --) shift; break;;
    *) echo "Unknown argument: $1" >&2; exit 1;;
  esac
done

UNION_BED="${UNION_BED:-${UNION_DIR}/${DATASET_NAME}.bed}"
mkdir -p "${OVERLAP_DIR}"
module load bedtools

if [[ ! -s "${UNION_BED}" ]]; then
  echo "ERROR: UNION_BED not found or empty at ${UNION_BED}" >&2
  exit 2
fi

echo "[bedtools] Intersecting A in ${EVAL_DIR} with B=${UNION_BED} -> ${OVERLAP_DIR}"
shopt -s nullglob
for A_FILE in "${EVAL_DIR}"/*.bed; do
  base="$(basename "${A_FILE}" .bed)"
  out="${OVERLAP_DIR}/${base}_${DATASET_NAME}.bed"

  if [[ "${base}" == *gtex* || "${base}" == *onek1k* || "${base}" == *tenk10k* || "${base}" == *gwas* ]]; then
    "${BEDTOOLS_BIN}" -wo \
      -a "${A_FILE}" \
      -b "${UNION_BED}" \
      > "${out}"
  else
    "${BEDTOOLS_BIN}" -wo -f 0.5 -r \
      -a "${A_FILE}" \
      -b "${UNION_BED}" \
      > "${out}"
  fi
  echo "Wrote: ${out}"

done