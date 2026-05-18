#!/usr/bin/env bash
set -euo pipefail

# Defaults (PSC)
ANALYSIS_BED="${ANALYSIS_BED:-/ocean/projects/bio240051p/aprieto/bedfiles/analysis/ukbb_hg38.bed}"
UNION_BED="${UNION_BED:-/ocean/projects/bio240051p/aprieto/bedfiles/analysis/multiome_union.bed}"
OVERLAP_BED="${OVERLAP_BED:-/ocean/projects/bio240051p/aprieto/bedfiles/analysis/overlap/multiome_ukbb.bed}"

# Parse CLI overrides
while [[ $# -gt 0 ]]; do
  case "$1" in
    --analysis-bed)     ANALYSIS_BED="$2"; shift 2;;
    --union-bed)        UNION_BED="$2"; shift 2;;
    --overlap-bed)      OVERLAP_BED="$2"; shift 2;;
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
if [[ "${base}" == *eqtl* || "${base}" == *gwas* || "${base}" == *ukbb* ]]; then
  bedtools intersect -wo \
    -a "${ANALYSIS_BED}" \
    -b "${UNION_BED}" \
    > "${OVERLAP_BED}"
else
  bedtools intersect -wo -f 0.5 -r \
    -a "${ANALYSIS_BED}" \
    -b "${UNION_BED}" \
    > "${OVERLAP_BED}"
fi
echo "Wrote: ${OVERLAP_BED}"