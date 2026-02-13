#!/usr/bin/env bash
set -euo pipefail

# Defaults (match repo)
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

if [[ ! -x "${BEDTOOLS_BIN}" ]]; then
  echo "ERROR: BEDTOOLS_BIN not executable at ${BEDTOOLS_BIN}" >&2
  exit 1
fi

if [[ ! -s "${UNION_BED}" ]]; then
  echo "ERROR: UNION_BED not found or empty at ${UNION_BED}" >&2
  exit 2
fi

echo "[bedtools] Intersecting A in ${EVAL_DIR} with B=${UNION_BED} -> ${OVERLAP_DIR}"
shopt -s nullglob
for A_FILE in "${EVAL_DIR}"/*.bed; do
  base="$(basename "${A_FILE}" .bed)"

  # Always allow global truth sets
  if [[ "${base}" =~ ^(crispr|ctcf_chiapet|rnap2_chiapet|intact_hic)(_|$) ]]; then
    :
  # Allow only dataset-matched beds otherwise
  elif [[ "${base}" != *"_${DATASET_NAME}" ]]; then
    echo "[skip] ${base} does not match dataset ${DATASET_NAME}"
    continue
  fi

  out="${OVERLAP_DIR}/${base}_${DATASET_NAME}.bed"

  # GTEx: variant overlaps (no fraction requirement)
  if [[ "${base}" == gtex_* || "${base}" == onek1k_* ]]; then
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
