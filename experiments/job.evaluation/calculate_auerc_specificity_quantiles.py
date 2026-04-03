import argparse
import os

import ctar
import numpy as np
import pandas as pd

"""
Celltype-specificity-stratified AUERC evaluation using the same seed-averaged
bootstrap structure as calculate_auerc_distances.py.
"""


def _parse_bool(value):
    return str(value).lower() in {"1", "true", "t", "yes", "y"}


def _parse_label(path: str, dataset_name: str) -> str:
    return os.path.basename(path).removesuffix(".csv").removesuffix(f"_{dataset_name}")


def _parse_quantiles(raw_quantiles: str):
    if not raw_quantiles:
        return [0.0, 0.25, 0.5, 0.75, 1.0]

    quantiles = [float(x.strip()) for x in raw_quantiles.split(",") if x.strip()]
    if len(quantiles) < 3:
        raise ValueError("specificity_quantiles must contain at least 3 values.")
    if quantiles[0] != 0.0 or quantiles[-1] != 1.0:
        raise ValueError("specificity_quantiles must start at 0 and end at 1.")
    if any(q2 <= q1 for q1, q2 in zip(quantiles[:-1], quantiles[1:])):
        raise ValueError("specificity_quantiles must be strictly increasing.")
    return quantiles


def _default_quantile_labels(quantiles):
    labels = []
    for q1, q2 in zip(quantiles[:-1], quantiles[1:]):
        labels.append(f"q{int(round(q1 * 100)):02d}-{int(round(q2 * 100)):02d}")
    return labels


def _attach_specificity(
    df: pd.DataFrame,
    truth_spec_df: pd.DataFrame | None,
    gene_spec_df: pd.DataFrame | None,
    peak_spec_df: pd.DataFrame | None,
    specificity_col: str,
    gene_col: str,
    peak_col: str,
    gene_spec_col: str,
    peak_spec_col: str,
    merge_truth_id_col: str,
    merge_truth_gene_col: str,
    truth_id_col: str,
    truth_gene_col: str,
    truth_specificity_col: str,
    truth_duplicate_strategy: str,
    force_recompute: bool,
    verbose: bool = True,
) -> pd.DataFrame:
    if (specificity_col in df.columns) and not force_recompute:
        return df

    if (gene_spec_df is not None) or (peak_spec_df is not None):
        if (gene_spec_df is None) or (peak_spec_df is None):
            raise ValueError(
                "Both --gene_specificity_path and --peak_specificity_path are required "
                "when computing link specificity."
            )
        out = ctar.celltype_specificity.attach_link_specificity(
            df.copy(),
            gene_spec_df=gene_spec_df,
            peak_spec_df=peak_spec_df,
            gene_col=gene_col,
            peak_col=peak_col,
            gene_spec_col=gene_spec_col,
            peak_spec_col=peak_spec_col,
        )
        if verbose:
            n_rows = len(out)
            n_gene = int(out["gene_specificity"].notna().sum()) if "gene_specificity" in out.columns else 0
            n_peak = int(out["peak_specificity"].notna().sum()) if "peak_specificity" in out.columns else 0
            n_both = (
                int((out["gene_specificity"].notna() & out["peak_specificity"].notna()).sum())
                if {"gene_specificity", "peak_specificity"}.issubset(out.columns)
                else 0
            )
            n_unique_gene_input = int(df[gene_col].nunique(dropna=True)) if gene_col in df.columns else 0
            n_unique_peak_input = int(df[peak_col].nunique(dropna=True)) if peak_col in df.columns else 0
            n_unique_gene_mapped = (
                int(out.loc[out["gene_specificity"].notna(), gene_col].nunique(dropna=True))
                if "gene_specificity" in out.columns and gene_col in out.columns
                else 0
            )
            n_unique_peak_mapped = (
                int(out.loc[out["peak_specificity"].notna(), peak_col].nunique(dropna=True))
                if "peak_specificity" in out.columns and peak_col in out.columns
                else 0
            )
            print(
                "Link-specificity mapping summary: "
                f"rows_with_gene_spec={n_gene}/{n_rows}, "
                f"rows_with_peak_spec={n_peak}/{n_rows}, "
                f"rows_with_both={n_both}/{n_rows}",
                flush=True,
            )
            print(
                "Unique feature mapping summary: "
                f"genes_mapped={n_unique_gene_mapped}/{n_unique_gene_input}, "
                f"peaks_mapped={n_unique_peak_mapped}/{n_unique_peak_input}",
                flush=True,
            )
    else:
        if truth_spec_df is None:
            raise ValueError(
                f"Specificity column '{specificity_col}' is not already present, so either "
                "--truth_specificity_path or both --gene_specificity_path and "
                "--peak_specificity_path are required."
            )

        out = ctar.celltype_specificity.attach_ground_truth_specificity(
            df.copy(),
            truth_spec_df=truth_spec_df,
            link_key_cols=(merge_truth_id_col, merge_truth_gene_col),
            truth_key_cols=(truth_id_col, truth_gene_col),
            truth_spec_col=truth_specificity_col,
            out_col=specificity_col,
            duplicate_strategy=truth_duplicate_strategy,
        )
    if specificity_col not in out.columns:
        raise ValueError(
            f"Expected specificity column '{specificity_col}' after attaching specificity, "
            f"but only found: {list(out.columns)}"
        )
    return out


def _assign_quantile_bins(
    df: pd.DataFrame,
    specificity_col: str,
    quantiles,
    quantile_labels,
    rank_method: str = "average",
) -> pd.DataFrame:
    out = df.copy()
    scores = pd.to_numeric(out[specificity_col], errors="coerce")
    valid = scores.notna()
    if valid.sum() == 0:
        raise ValueError(f"No valid values found in specificity column '{specificity_col}'.")

    percentiles = scores.loc[valid].rank(method=rank_method, pct=True)
    percentiles = percentiles.clip(lower=0.0, upper=1.0)

    # Include the minimum ranked observation in the first bin.
    percentiles = percentiles.mask(percentiles == 0.0, np.nextafter(0.0, 1.0))

    out["specificity_quantile"] = np.nan
    out.loc[valid, "specificity_quantile"] = percentiles
    out["specificity_bin"] = pd.cut(
        out["specificity_quantile"],
        bins=quantiles,
        labels=quantile_labels,
        include_lowest=True,
        right=True,
    )
    return out


def main(args):
    merge_path = args.merge_path
    res_path = args.res_path
    dataset_name = args.dataset_name

    method_cols = [m.strip() for m in args.method_cols.split(",") if m.strip()]
    gold_col = args.gold_col
    reference_method = args.reference_method
    fillna = _parse_bool(args.fillna)
    n_bootstrap = int(args.n_bootstrap)

    gtex_score_thres = float(args.gtex_score_thres)
    abc_score_thres = float(args.abc_score_thres)

    specificity_col = args.specificity_col
    force_recompute_specificity = bool(args.force_recompute_specificity)
    truth_specificity_path = args.truth_specificity_path
    gene_specificity_path = args.gene_specificity_path
    peak_specificity_path = args.peak_specificity_path
    gene_spec_col = args.gene_spec_col
    peak_spec_col = args.peak_spec_col
    merge_truth_id_col = args.merge_truth_id_col
    merge_truth_gene_col = args.merge_truth_gene_col
    truth_id_col = args.truth_id_col
    truth_gene_col = args.truth_gene_col
    truth_specificity_col = args.truth_specificity_col
    truth_duplicate_strategy = args.truth_duplicate_strategy
    peak_col = args.peak_col
    gene_col = args.gene_col

    quantiles = _parse_quantiles(args.specificity_quantiles)
    if args.specificity_quantile_labels:
        quantile_labels = [x.strip() for x in args.specificity_quantile_labels.split(",") if x.strip()]
        if len(quantile_labels) != len(quantiles) - 1:
            raise ValueError("specificity_quantile_labels must have length len(specificity_quantiles)-1.")
    else:
        quantile_labels = _default_quantile_labels(quantiles)

    os.makedirs(res_path, exist_ok=True)

    files = [
        f for f in os.listdir(merge_path)
        if os.path.isfile(f"{merge_path}/{f}") and f.endswith(".csv")
    ]
    files.sort()

    print("Using overlap files:", files, flush=True)
    print("With methods:", method_cols, flush=True)
    print("Specificity quantiles:", quantiles, flush=True)
    print("Specificity labels:", quantile_labels, flush=True)
    print("Specificity column:", specificity_col, flush=True)
    print("Truth specificity path:", truth_specificity_path, flush=True)
    print("Gene specificity path:", gene_specificity_path, flush=True)
    print("Peak specificity path:", peak_specificity_path, flush=True)
    print("Merge truth key cols:", [merge_truth_id_col, merge_truth_gene_col], flush=True)
    print("Truth table key cols:", [truth_id_col, truth_gene_col], flush=True)
    print("Fillna:", fillna, "Bootstrap:", n_bootstrap, flush=True)

    truth_spec_df = pd.read_csv(truth_specificity_path) if truth_specificity_path else None
    gene_spec_df = pd.read_csv(gene_specificity_path) if gene_specificity_path else None
    peak_spec_df = pd.read_csv(peak_specificity_path) if peak_specificity_path else None

    for file in files:
        overlap_df0 = pd.read_csv(f"{merge_path}/{file}")
        label = _parse_label(file, dataset_name)

        if label in ["ctcf_chiapet", "rnap2_chiapet", "intact_hic"]:
            print(f"Skipping Hi-C truth set for now: {label}", flush=True)
            continue

        overlap_df = overlap_df0.copy()

        if ("gtex" in label) or ("onek1k" in label) or ("tenk10k" in label):
            overlap_df["label"] = overlap_df["score"] >= gtex_score_thres
        elif label.startswith("abc"):
            overlap_df["label"] = overlap_df["score"] >= abc_score_thres
        elif label.startswith("crispr"):
            overlap_df["label"] = overlap_df["score"]
        else:
            raise ValueError(f"Unrecognized dataset label prefix for file {file} -> label={label}")

        overlap_df["label"] = overlap_df["label"].astype(bool)
        overlap_df = _attach_specificity(
            overlap_df,
            truth_spec_df=truth_spec_df,
            gene_spec_df=gene_spec_df,
            peak_spec_df=peak_spec_df,
            specificity_col=specificity_col,
            gene_col=gene_col,
            peak_col=peak_col,
            gene_spec_col=gene_spec_col,
            peak_spec_col=peak_spec_col,
            merge_truth_id_col=merge_truth_id_col,
            merge_truth_gene_col=merge_truth_gene_col,
            truth_id_col=truth_id_col,
            truth_gene_col=truth_gene_col,
            truth_specificity_col=truth_specificity_col,
            truth_duplicate_strategy=truth_duplicate_strategy,
            force_recompute=force_recompute_specificity,
            verbose=True,
        )
        n_specificity = int(overlap_df[specificity_col].notna().sum())
        print(
            f"[{label}] mapped {n_specificity}/{len(overlap_df)} rows to specificity",
            flush=True,
        )
        overlap_df = _assign_quantile_bins(
            overlap_df,
            specificity_col=specificity_col,
            quantiles=quantiles,
            quantile_labels=quantile_labels,
            rank_method=args.quantile_rank_method,
        )
        overlap_df["pg_pair"] = overlap_df[peak_col].astype(str) + ";" + overlap_df[gene_col].astype(str)

        if args.print_bin_summary:
            summary = overlap_df.groupby("specificity_bin", observed=True)["label"].agg(["count", "sum"])
            print(f"\n[{label}] specificity-bin summary:", flush=True)
            print(summary, flush=True)
            print("", flush=True)

        run_bins = quantile_labels.copy()
        if args.compute_overall:
            run_bins.append("ALL")

        for specificity_bin in run_bins:
            if specificity_bin == "ALL":
                sub = overlap_df.copy()
            else:
                sub = overlap_df.loc[overlap_df["specificity_bin"] == specificity_bin].copy()

            if args.drop_missing_method_scores:
                sub = sub.loc[sub[method_cols].notna().all(axis=1)].copy()

            n_total = int(sub.shape[0])
            n_pos = int(sub["label"].sum()) if n_total > 0 else 0
            n_neg = int(n_total - n_pos)

            if n_total < int(args.min_bin_n) or n_pos < int(args.min_bin_pos):
                print(
                    f"Skipping {label} @ {specificity_bin} "
                    f"(n_total={n_total}, n_pos={n_pos}, n_neg={n_neg})",
                    flush=True,
                )
                continue

            run_tag = f"{label}_{specificity_bin}"
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
                methods=method_cols,
                metrics_list=metric_specs,
                gold_col=gold_col,
                reference_method=reference_method,
                n_bootstrap=n_bootstrap,
                fillna=fillna,
                jitter_amount=1e-12,
                seeds=list(range(10)),
                handle_dup="consensus",
                dup_key_cols=["pg_pair"],
                tie="zero",
                stratified_bootstrap=True,
            )

            res_df["dataset"] = label
            res_df["dataset_name"] = dataset_name
            res_df["run_tag"] = run_tag
            res_df["specificity_bin"] = specificity_bin
            res_df["specificity_col"] = specificity_col
            res_df["n_total"] = n_total
            res_df["n_pos"] = n_pos
            res_df["n_neg"] = n_neg

            out_file = f"{res_path}/{run_tag}_{dataset_name}_seed_avg.csv"
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
        help="Output directory for specificity-stratified AUERC results.",
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

    parser.add_argument("--peak_col", type=str, default="peak")
    parser.add_argument("--gene_col", type=str, default="gene")
    parser.add_argument(
        "--truth_specificity_path",
        type=str,
        default="",
        help="CSV containing truth specificity annotations.",
    )
    parser.add_argument("--specificity_col", type=str, default="specificity")
    parser.add_argument(
        "--gene_specificity_path",
        type=str,
        default="",
        help="CSV containing gene specificity annotations with columns including feature,specificity.",
    )
    parser.add_argument(
        "--peak_specificity_path",
        type=str,
        default="",
        help="CSV containing peak specificity annotations with columns including feature,specificity.",
    )
    parser.add_argument("--gene_spec_col", type=str, default="specificity")
    parser.add_argument("--peak_spec_col", type=str, default="specificity")
    parser.add_argument("--merge_truth_id_col", type=str, default="tenk10k_id")
    parser.add_argument("--merge_truth_gene_col", type=str, default="gt_gene")
    parser.add_argument("--truth_id_col", type=str, default="tenk10k_id")
    parser.add_argument("--truth_gene_col", type=str, default="gt_gene")
    parser.add_argument("--truth_specificity_col", type=str, default="specificity")
    parser.add_argument(
        "--truth_duplicate_strategy",
        type=str,
        default="error",
        choices=["error", "first", "mean"],
    )
    parser.add_argument("--force_recompute_specificity", action="store_true")

    parser.add_argument(
        "--specificity_quantiles",
        type=str,
        default="0,0.25,0.5,0.75,1",
        help="Comma-separated quantile edges starting at 0 and ending at 1.",
    )
    parser.add_argument(
        "--specificity_quantile_labels",
        type=str,
        default="",
        help="Comma-separated labels for quantile bins (len = len(edges)-1).",
    )
    parser.add_argument(
        "--quantile_rank_method",
        type=str,
        default="average",
        choices=["average", "min", "max", "first", "dense"],
        help="Pandas rank method used before assigning quantile bins.",
    )

    parser.add_argument("--min_bin_n", type=int, default=100)
    parser.add_argument("--min_bin_pos", type=int, default=10)
    parser.add_argument("--print_bin_summary", action="store_true")
    parser.add_argument("--compute_overall", action="store_true")
    parser.add_argument(
        "--drop_missing_method_scores",
        action="store_true",
        help="Within each bin, require all method columns to be non-null before evaluation.",
    )

    args = parser.parse_args()
    main(args)
