import argparse
import math
import os

import ctar
import numpy as np
import pandas as pd

"""
Ground-truth count-stratified AUERC evaluation using the same seed-averaged
bootstrap structure as calculate_auerc_specificity_quantiles.py.
"""


def _parse_bool(value):
    return str(value).lower() in {"1", "true", "t", "yes", "y"}


def _parse_label(path: str, dataset_name: str) -> str:
    return os.path.basename(path).removesuffix(".csv").removesuffix(f"_{dataset_name}")


def _parse_count_bin_specs(raw_bins: str):
    if not raw_bins:
        raw_bins = "1-2,3-4,5+"

    specs = []
    for raw_item in [x.strip() for x in raw_bins.split(",") if x.strip()]:
        if raw_item.endswith("+"):
            lower = int(raw_item[:-1])
            upper = math.inf
            label = raw_item
        elif "-" in raw_item:
            left, right = raw_item.split("-", 1)
            lower = int(left)
            upper = int(right)
            label = raw_item
        else:
            lower = int(raw_item)
            upper = int(raw_item)
            label = raw_item

        if lower < 0:
            raise ValueError(f"Count bins must be nonnegative: {raw_item}")
        if upper != math.inf and upper < lower:
            raise ValueError(f"Invalid count bin with upper < lower: {raw_item}")

        specs.append((lower, upper, label))

    return specs


def _check_count_bin_overlap(bin_specs):
    for i, (lower_i, upper_i, label_i) in enumerate(bin_specs):
        for lower_j, upper_j, label_j in bin_specs[i + 1:]:
            overlap = max(lower_i, lower_j) <= min(upper_i, upper_j)
            if overlap:
                raise ValueError(f"Overlapping count bins are not allowed: {label_i} vs {label_j}")


def _attach_counts(
    df: pd.DataFrame,
    truth_count_df: pd.DataFrame | None,
    count_col: str,
    merge_truth_id_col: str,
    merge_truth_gene_col: str,
    truth_id_col: str,
    truth_gene_col: str,
    truth_count_col: str,
    truth_duplicate_strategy: str,
    force_recompute: bool,
) -> pd.DataFrame:
    if (count_col in df.columns) and not force_recompute:
        return df

    if truth_count_df is None:
        raise ValueError(
            f"Count column '{count_col}' is not already present, so --truth_count_path is required."
        )

    out = ctar.celltype_specificity.attach_ground_truth_specificity(
        df.copy(),
        truth_spec_df=truth_count_df,
        link_key_cols=(merge_truth_id_col, merge_truth_gene_col),
        truth_key_cols=(truth_id_col, truth_gene_col),
        truth_spec_col=truth_count_col,
        out_col=count_col,
        duplicate_strategy=truth_duplicate_strategy,
    )
    if count_col not in out.columns:
        raise ValueError(
            f"Expected count column '{count_col}' after attach_ground_truth_specificity, "
            f"but only found: {list(out.columns)}"
        )

    out[count_col] = pd.to_numeric(out[count_col], errors="coerce")
    return out


def _assign_count_bins(df: pd.DataFrame, count_col: str, bin_specs) -> pd.DataFrame:
    out = df.copy()
    counts = pd.to_numeric(out[count_col], errors="coerce")
    out[count_col] = counts
    out["count_bin"] = pd.Series(pd.NA, index=out.index, dtype="object")

    for lower, upper, label in bin_specs:
        if upper == math.inf:
            mask = counts >= lower
        else:
            mask = (counts >= lower) & (counts <= upper)
        out.loc[mask, "count_bin"] = label

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

    count_col = args.count_col
    force_recompute_count = bool(args.force_recompute_count)
    truth_count_path = args.truth_count_path
    merge_truth_id_col = args.merge_truth_id_col
    merge_truth_gene_col = args.merge_truth_gene_col
    truth_id_col = args.truth_id_col
    truth_gene_col = args.truth_gene_col
    truth_count_col = args.truth_count_col
    truth_duplicate_strategy = args.truth_duplicate_strategy
    peak_col = args.peak_col
    gene_col = args.gene_col

    count_bin_specs = _parse_count_bin_specs(args.count_bins)
    _check_count_bin_overlap(count_bin_specs)
    count_bin_labels = [label for _, _, label in count_bin_specs]

    os.makedirs(res_path, exist_ok=True)

    files = [
        f for f in os.listdir(merge_path)
        if os.path.isfile(f"{merge_path}/{f}") and f.endswith(".csv")
    ]
    files.sort()

    print("Using overlap files:", files, flush=True)
    print("With methods:", method_cols, flush=True)
    print("Count bins:", count_bin_specs, flush=True)
    print("Count column:", count_col, flush=True)
    print("Truth count path:", truth_count_path, flush=True)
    print("Merge truth key cols:", [merge_truth_id_col, merge_truth_gene_col], flush=True)
    print("Truth table key cols:", [truth_id_col, truth_gene_col], flush=True)
    print("Fillna:", fillna, "Bootstrap:", n_bootstrap, flush=True)

    truth_count_df = pd.read_csv(truth_count_path) if truth_count_path else None

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
        overlap_df = _attach_counts(
            overlap_df,
            truth_count_df=truth_count_df,
            count_col=count_col,
            merge_truth_id_col=merge_truth_id_col,
            merge_truth_gene_col=merge_truth_gene_col,
            truth_id_col=truth_id_col,
            truth_gene_col=truth_gene_col,
            truth_count_col=truth_count_col,
            truth_duplicate_strategy=truth_duplicate_strategy,
            force_recompute=force_recompute_count,
        )
        n_count = int(overlap_df[count_col].notna().sum())
        print(
            f"[{label}] mapped {n_count}/{len(overlap_df)} rows to truth counts",
            flush=True,
        )

        overlap_df = _assign_count_bins(
            overlap_df,
            count_col=count_col,
            bin_specs=count_bin_specs,
        )
        overlap_df["pg_pair"] = overlap_df[peak_col].astype(str) + ";" + overlap_df[gene_col].astype(str)

        if args.print_bin_summary:
            summary = overlap_df.groupby("count_bin", observed=True)["label"].agg(["count", "sum"])
            print(f"\n[{label}] count-bin summary:", flush=True)
            print(summary, flush=True)
            print("", flush=True)

        run_bins = count_bin_labels.copy()
        if args.compute_overall:
            run_bins.append("ALL")

        for count_bin in run_bins:
            if count_bin == "ALL":
                sub = overlap_df.copy()
            else:
                sub = overlap_df.loc[overlap_df["count_bin"] == count_bin].copy()

            if args.drop_missing_method_scores:
                sub = sub.loc[sub[method_cols].notna().all(axis=1)].copy()

            n_total = int(sub.shape[0])
            n_pos = int(sub["label"].sum()) if n_total > 0 else 0
            n_neg = int(n_total - n_pos)

            if n_total < int(args.min_bin_n) or n_pos < int(args.min_bin_pos):
                print(
                    f"Skipping {label} @ {count_bin} "
                    f"(n_total={n_total}, n_pos={n_pos}, n_neg={n_neg})",
                    flush=True,
                )
                continue

            run_tag = f"{label}_{count_bin}"
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
            res_df["count_bin"] = count_bin
            res_df["count_col"] = count_col
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
        help="Output directory for count-stratified AUERC results.",
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
        "--truth_count_path",
        type=str,
        default="",
        help="CSV containing truth count annotations.",
    )
    parser.add_argument("--count_col", type=str, default="count")
    parser.add_argument("--merge_truth_id_col", type=str, default="tenk10k_id")
    parser.add_argument("--merge_truth_gene_col", type=str, default="gt_gene")
    parser.add_argument("--truth_id_col", type=str, default="tenk10k_id")
    parser.add_argument("--truth_gene_col", type=str, default="gt_gene")
    parser.add_argument("--truth_count_col", type=str, default="count")
    parser.add_argument(
        "--truth_duplicate_strategy",
        type=str,
        default="error",
        choices=["error", "first", "mean"],
    )
    parser.add_argument("--force_recompute_count", action="store_true")

    parser.add_argument(
        "--count_bins",
        type=str,
        default="1-2,3-4,5+",
        help="Comma-separated count bins such as '1-2,3-4,5+'.",
    )

    parser.add_argument("--peak_col", type=str, default="peak")
    parser.add_argument("--gene_col", type=str, default="gene")
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
