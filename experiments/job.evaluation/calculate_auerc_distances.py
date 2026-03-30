import argparse
import os

import ctar
import numpy as np
import pandas as pd

"""
Distance-stratified AUERC evaluation using the same seed-averaged bootstrap
structure as calculate_auerc_seeds.py.
"""


def _parse_bool(value):
    return str(value).lower() in {"1", "true", "t", "yes", "y"}


def _parse_label(path: str, dataset_name: str) -> str:
    return os.path.basename(path).removesuffix(".csv").removesuffix(f"_{dataset_name}")


def _parse_peak_coords_from_peak_col(df: pd.DataFrame, peak_col: str = "peak") -> pd.DataFrame:
    if peak_col not in df.columns:
        raise ValueError(f"Cannot parse peak coords: missing column '{peak_col}'.")

    s = df[peak_col].astype(str)
    s_norm = (
        s.str.replace(":", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace(r"\s+", "", regex=True)
    )
    parts = s_norm.str.split("_", expand=True)

    if parts.shape[1] < 3:
        raise ValueError(
            f"Peak parsing failed for '{peak_col}'. Example values: {s.head(5).tolist()}"
        )

    out = df.copy()
    out["chr"] = parts[0]
    out["start"] = pd.to_numeric(parts[1], errors="coerce")
    out["end"] = pd.to_numeric(parts[2], errors="coerce")
    out.loc[out["end"].isna() & out["start"].notna(), "end"] = out.loc[
        out["end"].isna() & out["start"].notna(), "start"
    ]
    return out


def _parse_distance_bins(raw_bins: str):
    if raw_bins:
        bins = []
        for item in [x.strip() for x in raw_bins.split(",") if x.strip()]:
            if item.lower() in {"inf", "infty", "infinite"}:
                bins.append(np.inf)
            else:
                bins.append(int(item))
        if len(bins) < 3 or bins[0] != 0:
            raise ValueError("distance_bins_bp must start with 0 and have at least 3 edges.")
        return bins
    return [0, 50_000, 250_000, 1_000_000, 5_000_000, np.inf]


def main(args):
    MERGE_PATH = args.merge_path
    RES_PATH = args.res_path
    DATASET_NAME = args.dataset_name

    METHOD_COLS = [m.strip() for m in args.method_cols.split(",") if m.strip()]
    GOLD_COL = args.gold_col
    REFERENCE_METHOD = args.reference_method
    FILLNA = _parse_bool(args.fillna)
    N_BOOTSTRAP = int(args.n_bootstrap)

    GTEX_SCORE_THRES = float(args.gtex_score_thres)
    ABC_SCORE_THRES = float(args.abc_score_thres)

    GENCODE_GTF = args.gencode_gtf
    PEAK_COL = args.peak_col
    GENE_COL = args.gene_col

    DIST_BINS = _parse_distance_bins(args.distance_bins_bp)
    if args.distance_bin_labels:
        DIST_LABELS = [x.strip() for x in args.distance_bin_labels.split(",") if x.strip()]
        if len(DIST_LABELS) != len(DIST_BINS) - 1:
            raise ValueError("distance_bin_labels must have length len(distance_bins_bp)-1.")
    else:
        DIST_LABELS = ["0-50kb", "50-250kb", "250kb-1Mb", "1-5Mb", "5Mb+"][: len(DIST_BINS) - 1]

    os.makedirs(RES_PATH, exist_ok=True)

    gene_df = ctar.gentruth.load_gene_tss_from_gencode(GENCODE_GTF)

    files = [
        f for f in os.listdir(MERGE_PATH)
        if os.path.isfile(f"{MERGE_PATH}/{f}") and f.endswith(".csv")
    ]
    files.sort()

    print("Using overlap files:", files, flush=True)
    print("With methods:", METHOD_COLS, flush=True)
    print("Distance bins:", DIST_BINS, flush=True)
    print("Distance labels:", DIST_LABELS, flush=True)
    print("Fillna:", FILLNA, "Bootstrap:", N_BOOTSTRAP, flush=True)

    for file in files:
        overlap_df0 = pd.read_csv(f"{MERGE_PATH}/{file}")
        label = _parse_label(file, DATASET_NAME)

        if label in ["ctcf_chiapet", "rnap2_chiapet", "intact_hic"]:
            print(f"Skipping Hi-C truth set for now: {label}", flush=True)
            continue

        overlap_df = overlap_df0.copy()

        if label.startswith("gtex") or label.startswith("onek1k"):
            overlap_df["label"] = overlap_df["score"] >= GTEX_SCORE_THRES
        elif label.startswith("abc"):
            overlap_df["label"] = overlap_df["score"] >= ABC_SCORE_THRES
        elif label.startswith("crispr"):
            overlap_df["label"] = overlap_df["score"]
        else:
            raise ValueError(f"Unrecognized dataset label prefix for file {file} -> label={label}")

        overlap_df["label"] = overlap_df["label"].astype(bool)
        overlap_df[GENE_COL] = ctar.gentruth.strip_gene_version(overlap_df[GENE_COL])

        if not {"chr", "start", "end"}.issubset(overlap_df.columns):
            overlap_df = _parse_peak_coords_from_peak_col(overlap_df, peak_col=PEAK_COL)

        overlap_df = ctar.gentruth.add_tss_distances(overlap_df, gene_df, gene_col=GENE_COL)

        if args.require_same_chr:
            overlap_df = overlap_df.loc[overlap_df["same_chr"]].copy()

        overlap_df["dist_bin"] = pd.cut(
            overlap_df["abs_distance"],
            bins=DIST_BINS,
            labels=DIST_LABELS,
            include_lowest=True,
            right=True,
        )
        overlap_df["pg_pair"] = overlap_df["peak"] + ";" + overlap_df["gene"]

        if args.print_bin_summary:
            summary = overlap_df.groupby("dist_bin", observed=True)["label"].agg(["count", "sum"])
            print(f"\n[{label}] distance-bin summary:", flush=True)
            print(summary, flush=True)
            print("", flush=True)

        run_bins = DIST_LABELS.copy()
        if args.compute_overall:
            run_bins.append("ALL")

        for dist_bin in run_bins:
            if dist_bin == "ALL":
                sub = overlap_df.copy()
            else:
                sub = overlap_df.loc[overlap_df["dist_bin"] == dist_bin].copy()

            n_total = int(sub.shape[0])
            n_pos = int(sub["label"].sum()) if n_total > 0 else 0
            n_neg = int(n_total - n_pos)

            if n_total < int(args.min_bin_n) or n_pos < int(args.min_bin_pos):
                print(
                    f"Skipping {label} @ {dist_bin} (n_total={n_total}, n_pos={n_pos}, n_neg={n_neg})",
                    flush=True,
                )
                continue

            run_tag = f"{label}_{dist_bin}"
            print(
                f"Computing metrics for {run_tag}... "
                f"(n_total={n_total}, n_pos={n_pos}, n_neg={n_neg})",
                flush=True,
            )

            metric_specs = ctar.metrics.build_default_metric_specs(
                sub,
                gold_col="label",
                pvals_smaller_is_better=True,
                early_R=0.2,
            )

            res_df = ctar.metrics.compute_bootstrap_table_seed_avg(
                all_df=sub,
                methods=METHOD_COLS,
                metrics_list=metric_specs,
                gold_col=GOLD_COL,
                reference_method=REFERENCE_METHOD,
                n_bootstrap=N_BOOTSTRAP,
                fillna=FILLNA,
                jitter_amount=1e-12,
                seeds=list(range(10)),
                handle_dup="consensus",
                dup_key_cols=["pg_pair"],
                tie="zero",
                stratified_bootstrap=True,
            )

            res_df["dataset"] = label
            res_df["dataset_name"] = DATASET_NAME
            res_df["run_tag"] = run_tag
            res_df["dist_bin"] = dist_bin
            res_df["require_same_chr"] = bool(args.require_same_chr)
            res_df["n_total"] = n_total
            res_df["n_pos"] = n_pos
            res_df["n_neg"] = n_neg

            out_file = f"{RES_PATH}/{run_tag}_{DATASET_NAME}_seed_avg.csv"
            print(res_df.head(), flush=True)
            res_df.to_csv(out_file, index=False)
            print("Wrote:", out_file, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--merge_path",
        type=str,
        default="/projects/zhanglab/users/ana/multiome/validation/neat",
        help="Path containing merged evaluation CSVs.",
    )
    parser.add_argument(
        "--overlap_path",
        dest="merge_path",
        type=str,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--res_path",
        type=str,
        default="/projects/zhanglab/users/ana/multiome/validation/tables/neat",
        help="Output directory for distance-stratified AUERC results.",
    )
    parser.add_argument("--dataset_name", type=str, default="neat")

    parser.add_argument(
        "--method_cols",
        type=str,
        default="scent,scmm,signac,ctar_z,ctar,ctar_filt_z,ctar_filt",
    )
    parser.add_argument("--gold_col", type=str, default="label")
    parser.add_argument("--reference_method", type=str, default="ctar")
    parser.add_argument("--fillna", type=str, default=True)
    parser.add_argument("--n_bootstrap", type=str, default="1000")

    parser.add_argument("--gtex_score_thres", type=str, default="0.5")
    parser.add_argument("--abc_score_thres", type=str, default="0.2")

    parser.add_argument(
        "--gencode_gtf",
        type=str,
        required=True,
        help="Path to a GENCODE GTF or GTF.GZ annotation file.",
    )
    parser.add_argument(
        "--gene_tss_path",
        dest="gencode_gtf",
        type=str,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--peak_col", type=str, default="peak")
    parser.add_argument("--gene_col", type=str, default="gene")

    parser.add_argument(
        "--distance_bins_bp",
        type=str,
        default="",
        help="Comma-separated bin edges in bp starting with 0.",
    )
    parser.add_argument(
        "--distance_bin_labels",
        type=str,
        default="",
        help="Comma-separated labels for bins (len = len(edges)-1).",
    )
    parser.add_argument("--require_same_chr", action="store_true")
    parser.add_argument("--min_bin_n", type=int, default=100)
    parser.add_argument("--min_bin_pos", type=int, default=10)
    parser.add_argument("--print_bin_summary", action="store_true")
    parser.add_argument("--compute_overall", action="store_true")

    args = parser.parse_args()
    main(args)
