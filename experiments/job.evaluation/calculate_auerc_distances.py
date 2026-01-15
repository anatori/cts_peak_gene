"""
calculate_auerc_distances.py

Distance-stratified AUERC evaluation.

Key features
- Uses dataset-specific truth files.
- Applies GTEx / ABC thresholds using prefix matching (gtex*, abc*).
- Computes peak–gene distance using *gene TSS* (Ensembl BioMart-derived table),
  with safe fallbacks to parse peak coordinates from a 'peak' string.
- Computes AUERC per distance bin and writes one CSV per (truth_source, bin).

Assumptions
- overlap_path contains CSVs produced by aggregate_overlaps.py.
- overlap_df includes method score columns + at least:
    - 'peak' and 'gene'
  AND ideally also:
    - 'chr','start','end' for peak coords.
  If chr/start/end missing, script tries to parse from overlap_df['peak'].

"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import ctar


def _infer_label_from_filename(path: str) -> str:
    """Extract truth label prefix from filename like gtex_bmmc_bmmc.csv -> gtex_bmmc."""
    base = os.path.basename(path)
    stem = base.rsplit(".", 1)[0]
    # remove trailing dataset suffix if present: *_<dataset>
    # (your aggregator writes e.g. {label}_{DATASET_NAME}.csv sometimes; keep prefix robust)
    return stem


def _parse_peak_coords_from_peak_col(df: pd.DataFrame, peak_col: str = "peak") -> pd.DataFrame:
    """
    Try to parse chr/start/end from df[peak_col]. Handles common encodings:
    - "chr1_123_456"
    - "chr1:123-456"
    - "chr1-123-456" (less common)
    """
    if peak_col not in df.columns:
        raise ValueError(f"Cannot parse peak coords: missing column '{peak_col}'.")

    s = df[peak_col].astype(str)

    # Normalize separators to underscore: chr1:123-456 -> chr1_123_456
    s_norm = (
        s.str.replace(":", "_", regex=False)
         .str.replace("-", "_", regex=False)
         .str.replace(r"\s+", "", regex=True)
    )

    parts = s_norm.str.split("_", expand=True)

    if parts.shape[1] < 3:
        raise ValueError(
            f"Peak parsing failed: expected at least 3 underscore-separated fields in '{peak_col}'. "
            f"Example values: {s.head(5).tolist()}"
        )

    out = df.copy()
    out["chr"] = parts[0]
    out["start"] = pd.to_numeric(parts[1], errors="coerce")
    out["end"] = pd.to_numeric(parts[2], errors="coerce")

    # If end missing (some encodings might be chr_pos), treat as 1bp
    out.loc[out["end"].isna() & out["start"].notna(), "end"] = out.loc[
        out["end"].isna() & out["start"].notna(), "start"
    ]

    return out


def main(args):
    overlap_path = args.overlap_path
    res_path = args.res_path
    dataset_name = args.dataset_name

    method_cols = [m.strip() for m in args.method_cols.split(",") if m.strip()]
    gold_col = args.gold_col
    reference_method = args.reference_method
    fillna = args.fillna
    n_bootstrap = args.n_bootstrap

    gtex_score_thres = args.gtex_score_thres
    abc_score_thres = args.abc_score_thres

    gene_tss_path = args.gene_tss_path
    peak_col = args.peak_col
    gene_col = args.gene_col

    # Distance binning
    # Accept either comma-separated ints (bp) or use defaults
    if args.distance_bins_bp:
        # Example: "0,50000,250000,1000000,5000000,inf"
        raw = [x.strip() for x in args.distance_bins_bp.split(",")]
        bins = []
        for x in raw:
            if x.lower() in ("inf", "infty", "infinite"):
                bins.append(np.inf)
            else:
                bins.append(int(x))
        if len(bins) < 3 or bins[0] != 0:
            raise ValueError("distance_bins_bp must start with 0 and have at least 3 edges.")
        dist_bins = bins
    else:
        dist_bins = [0, 50_000, 250_000, 1_000_000, 5_000_000, np.inf]

    if args.distance_bin_labels:
        dist_labels = [x.strip() for x in args.distance_bin_labels.split(",")]
        if len(dist_labels) != len(dist_bins) - 1:
            raise ValueError("distance_bin_labels must have length len(distance_bins_bp)-1.")
    else:
        dist_labels = ["0-50kb", "50-250kb", "250kb-1Mb", "1-5Mb", "5Mb+"][: (len(dist_bins) - 1)]

    os.makedirs(res_path, exist_ok=True)

    # Load gene TSS table once
    gene_df = pd.read_csv(gene_tss_path, sep=args.gene_tss_sep, compression=args.gene_tss_compression)
    required_gene_cols = {"gene", "gene_chr", "tss"}
    missing = required_gene_cols - set(gene_df.columns)
    if missing:
        raise ValueError(f"gene_tss_path is missing required columns: {sorted(missing)}")

    files = [f for f in os.listdir(overlap_path) if os.path.isfile(os.path.join(overlap_path, f))]
    files = [f for f in files if f.endswith(".csv")]
    files.sort()

    print("Using overlap files:", files)
    print("With methods:", method_cols)
    print("Distance bins:", dist_bins)
    print("Distance labels:", dist_labels)

    for fname in files:
        fpath = os.path.join(overlap_path, fname)
        overlap_df = pd.read_csv(fpath)

        # Infer label from filename and normalize to prefix checks
        label_full = _infer_label_from_filename(fpath)  # e.g. "gtex_bmmc_bmmc" or "abc_neat_neat"
        # truth source prefix
        # prefer startswith checks
        is_gtex = label_full.startswith("gtex")
        is_abc = label_full.startswith("abc")
        is_hic = any(label_full.startswith(x) for x in ["ctcf_chiapet", "rnap2_chiapet", "intact_hic"])

        # If this is HIC truth, skip for now
        if is_hic:
            print(f"Skipping Hi-C truth set for now: {label_full}")
            continue

        # Create binary label column (as in your calculate_auerc.py)
        if is_gtex:
            overlap_df["label"] = overlap_df["score"] >= gtex_score_thres
        elif is_abc:
            overlap_df["label"] = overlap_df["score"] >= abc_score_thres
        else:
            # For crispr and other truth sets, assume 'score' is already boolean-ish or 0/1
            overlap_df["label"] = overlap_df["score"].astype(bool) if overlap_df["score"].dtype != bool else overlap_df["score"]

        # Ensure peak coords exist; parse if missing
        if not {"chr", "start", "end"}.issubset(overlap_df.columns):
            try:
                overlap_df = _parse_peak_coords_from_peak_col(overlap_df, peak_col=peak_col)
            except Exception as e:
                raise RuntimeError(
                    f"Could not compute distances for {fname}: missing chr/start/end and failed to parse from '{peak_col}'. "
                    f"Error: {e}"
                )

        # Add TSS distance
        overlap_df = ctar.gentruth.add_tss_distances(overlap_df, gene_df, gene_col=gene_col)

        # Keep cis only for distance-stratified evaluation (recommended)
        if args.require_same_chr:
            overlap_df = overlap_df[overlap_df["same_chr"]].copy()

        # Bin distances
        overlap_df["dist_bin"] = pd.cut(
            overlap_df["abs_distance"],
            bins=dist_bins,
            labels=dist_labels,
            include_lowest=True,
            right=True
        )

        # Optional: write a quick summary
        if args.print_bin_summary:
            tmp = overlap_df.groupby("dist_bin", observed=True)["label"].agg(["count", "sum"])
            print(f"\n[{label_full}] distance-bin summary:")
            print(tmp)
            print("")

        # Compute AUERC per bin
        for b in dist_labels:
            sub = overlap_df[overlap_df["dist_bin"] == b].copy()
            n = len(sub)
            n_pos = int(sub["label"].sum()) if "label" in sub.columns else 0

            if n < args.min_bin_n or n_pos < args.min_bin_pos:
                continue

            print(f"Computing AUERC for {label_full} @ {b} (n={n}, pos={n_pos})...")

            # Reuse existing bootstrap table function
            res_df = ctar.simu.compute_bootstrap_table(
                sub,
                method_cols,
                gold_col="label",
                n_bootstrap=n_bootstrap,
                fillna=fillna,
                reference_method=reference_method,
                extrapolate=False,
                weighted=True,
                ascending=True
            )
            res_df["truth_set"] = label_full
            res_df["dataset_name"] = dataset_name
            res_df["dist_bin"] = b
            res_df["n_items"] = n
            res_df["n_pos"] = n_pos

            out_file = os.path.join(res_path, f"{label_full}_{dataset_name}_{b}_auerc.csv")
            res_df.to_csv(out_file, index=False)

        # Optional: also compute overall AUERC (no stratification)
        if args.compute_overall:
            print(f"Computing overall AUERC for {label_full} (no distance stratification)...")
            res_df = ctar.simu.compute_bootstrap_table(
                overlap_df,
                method_cols,
                gold_col="label",
                n_bootstrap=n_bootstrap,
                fillna=fillna,
                reference_method=reference_method,
                extrapolate=False,
                weighted=True,
                ascending=True
            )
            res_df["truth_set"] = label_full
            res_df["dataset_name"] = dataset_name
            res_df["dist_bin"] = "ALL"
            res_df["n_items"] = len(overlap_df)
            res_df["n_pos"] = int(overlap_df["label"].sum())

            out_file = os.path.join(res_path, f"{label_full}_{dataset_name}_ALL_auerc.csv")
            res_df.to_csv(out_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--overlap_path", type=str, default="/projects/zhanglab/users/ana/multiome/validation/neat",
                        help="Path containing aggregated overlap CSVs (output of aggregate_overlaps.py).")
    parser.add_argument("--res_path", type=str, default="/projects/zhanglab/users/ana/multiome/validation/tables/neat",
                        help="Output directory for AUERC results.")
    parser.add_argument("--dataset_name", type=str, default="neat")

    parser.add_argument("--method_cols", type=str, default="scent,scmm,signac,ctar_z,ctar,ctar_filt_z,ctar_filt")
    parser.add_argument("--gold_col", type=str, default="label", help="Gold column name (script writes 'label').")
    parser.add_argument("--reference_method", type=str, default="ctar")
    parser.add_argument("--fillna", type=bool, default=True)
    parser.add_argument("--n_bootstrap", type=int, default=1000)

    parser.add_argument("--gtex_score_thres", type=float, default=0.5)
    parser.add_argument("--abc_score_thres", type=float, default=0.2)

    # Distance computation inputs
    parser.add_argument("--gene_tss_path", type=str, required=True,
                        help="Path to a gene TSS table with columns gene,gene_chr,tss (tsv/tsv.gz).")
    parser.add_argument("--gene_tss_sep", type=str, default="\t")
    parser.add_argument("--gene_tss_compression", type=str, default="infer",
                        help="pandas compression arg: 'infer', 'gzip', or None.")
    parser.add_argument("--peak_col", type=str, default="peak",
                        help="Column name containing peak identifier/coords.")
    parser.add_argument("--gene_col", type=str, default="gene",
                        help="Column name containing Ensembl gene id (optionally with version).")

    # Binning controls
    parser.add_argument("--distance_bins_bp", type=str, default="",
                        help="Comma-separated bin edges in bp starting with 0. Example: 0,50000,250000,1000000,5000000,inf")
    parser.add_argument("--distance_bin_labels", type=str, default="",
                        help="Comma-separated labels for bins (len = len(edges)-1).")
    parser.add_argument("--require_same_chr", action="store_true",
                        help="If set, restrict to same-chromosome peak-gene pairs for distance evaluation.")
    parser.add_argument("--min_bin_n", type=int, default=100,
                        help="Minimum number of items required in a distance bin to compute AUERC.")
    parser.add_argument("--min_bin_pos", type=int, default=10,
                        help="Minimum number of positives required in a distance bin to compute AUERC.")
    parser.add_argument("--print_bin_summary", action="store_true",
                        help="Print counts/positives per distance bin for each truth set.")
    parser.add_argument("--compute_overall", action="store_true",
                        help="Also compute overall (ALL distances) AUERC per truth set.")

    args = parser.parse_args()
    main(args)