#!/usr/bin/env bash
set -euo pipefail

# Defaults (PSC)
ANALYSIS_BED="${ANALYSIS_BED:-/projects/zhanglab/users/ana/bedtools2/ana_bedfiles/analysis/ukbb_hg38.bed}"
UNION_BED="${UNION_BED:-/projects/zhanglab/users/ana/bedtools2/ana_bedfiles/analysis/multiome_union.bed}"
OVERLAP_BED="${OVERLAP_BED:-/projects/zhanglab/users/ana/bedtools2/ana_bedfiles/analysis/overlap/multiome_ukbb.bed}"
BEDTOOLS_BIN="${BEDTOOLS_BIN:-/projects/zhanglab/users/ana/bedtools2/bin/intersectBed}"

# Parse CLI overrides
while [[ $# -gt 0 ]]; do
  case "$1" in
    --analysis-bed)     ANALYSIS_BED="$2"; shift 2;;
    --union-bed)        UNION_BED="$2"; shift 2;;
    --overlap-bed)      OVERLAP_BED="$2"; shift 2;;
    --bedtools-bin) BEDTOOLS_BIN="$2"; shift 2;;
    --) shift; break;;
    *) echo "Unknown argument: $1" >&2; exit 1;;
  esac
done

mkdir -p "$(dirname "${OVERLAP_BED}")"
module load bedtools

if [[ ! -s "${UNION_BED}" ]]; then
  echo "ERROR: UNION_BED not found or empty at ${UNION_BED}" >&2
  exit 2
fi

base="$(basename "${ANALYSIS_BED}" .bed)"
if [[ "${base}" == *eqtl* || "${base}" == *gwas* || "${base}" == *ukbb* || "${base}" == *gtex* ]]; then
  "${BEDTOOLS_BIN}" -wo \
    -a "${ANALYSIS_BED}" \
    -b "${UNION_BED}" \
    > "${OVERLAP_BED}"
elif [[ "${base}" == *hichip* ]]; then
  "${BEDTOOLS_BIN}" -wo -F 0.5 \
    -a "${ANALYSIS_BED}" \
    -b "${UNION_BED}" \
    > "${OVERLAP_BED}"
else
  "${BEDTOOLS_BIN}" -wo -f 0.5 -r \
    -a "${ANALYSIS_BED}" \
    -b "${UNION_BED}" \
    > "${OVERLAP_BED}"
fi
echo "Wrote: ${OVERLAP_BED}"