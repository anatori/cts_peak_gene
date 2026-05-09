import argparse
import csv
import os
from typing import List, Optional, Tuple
import pandas as pd
from statsmodels.stats.multitest import multipletests
import ctar


def _opt_none(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if s.lower() in {"", "none", "null", "na"}:
        return None
    return s


def _compute_bh_qvalues(pvals: pd.Series) -> pd.Series:
    mask = pvals.notna()
    qvals = pvals.copy()
    if mask.any():
        _, qcorrected, _, _ = multipletests(pvals[mask].values, method="fdr_bh")
        qvals[mask] = qcorrected
    return qvals


def _aggregate_duplicates(
    df: pd.DataFrame,
    pval_col: str,
    qval_col: str,
    how: str,
) -> pd.DataFrame:
    if how == "none":
        print("Warning: no deduplication!")
        return df[["id", "peak", "gene", "dataset", pval_col, qval_col]].copy()

    if how not in {"min", "max", "mean", "median"}:
        raise ValueError(f"Unsupported dedup method: {how}")

    tmp = df[["id", "peak", "gene", "dataset", pval_col, qval_col]].copy()
    tmp[pval_col] = pd.to_numeric(tmp[pval_col], errors="coerce")
    tmp[qval_col] = pd.to_numeric(tmp[qval_col], errors="coerce")

    if how in {"min", "max"}:
        # Select the row with the chosen p-value extreme; q-value comes from that same row
        row_idx = (
            tmp.groupby("id", sort=False)[pval_col].idxmin()
            if how == "min"
            else tmp.groupby("id", sort=False)[pval_col].idxmax()
        )
        out = tmp.loc[row_idx.values, ["id", "peak", "gene", "dataset", pval_col, qval_col]].reset_index(drop=True)
    else:
        # mean/median: independently aggregate both columns
        agg_p = tmp.groupby("id", sort=False)[pval_col].agg(how)
        agg_q = tmp.groupby("id", sort=False)[qval_col].agg(how)
        # dataset is constant within each id (id encodes dataset), so take first
        dataset = tmp.groupby("id", sort=False)["dataset"].first()
        peak_gene_ds = agg_p.index.to_series().str.split(";", n=2, expand=True)
        out = pd.DataFrame({
            "id": agg_p.index,
            "peak": peak_gene_ds[0].values,
            "gene": peak_gene_ds[1].values,
            "dataset": dataset.values,
            pval_col: agg_p.values,
            qval_col: agg_q.values,
        })

    return out


def load_method_scores(
    file_path: str,
    method_name: str,
    dataset: str,
    pval_source_col: str,
    fdr_source_col: Optional[str],
    dedup: str = "min",
) -> pd.DataFrame:
    """Returns DataFrame with columns: id, peak, gene, dataset, {method_name}_pval, {method_name}_qval"""
    if not file_path or not os.path.isfile(file_path):
        raise FileNotFoundError(f"Missing method file: {file_path}")

    df = pd.read_csv(file_path, sep=None, engine="python")
    df["peak"] = ctar.data_loader.canonicalize_peak(df["peak"])
    df["gene"] = df["gene"].astype(str).str.strip()
    df["dataset"] = dataset
    df["id"] = df["peak"].astype(str) + ";" + df["gene"].astype(str) + ";" + dataset

    pval_col = f"{method_name}_{dataset}_pval"
    qval_col = f"{method_name}_{dataset}_qval"

    if pval_source_col not in df.columns:
        raise ValueError(f"File '{file_path}' missing p-value column '{pval_source_col}'")
    df[pval_col] = pd.to_numeric(df[pval_source_col], errors="coerce")

    if fdr_source_col is not None:
        if fdr_source_col not in df.columns:
            raise ValueError(f"File '{file_path}' missing fdr column '{fdr_source_col}'")
        df[qval_col] = pd.to_numeric(df[fdr_source_col], errors="coerce")
    else:
        print(f"[{method_name}_{dataset}] no fdr_col — computing BH q-values from '{pval_source_col}'")
        df[qval_col] = _compute_bh_qvalues(df[pval_col])

    n_raw = len(df)
    n_unique = df["id"].nunique()
    print(f"[{method_name}_{dataset}] raw_rows={n_raw} unique_ids={n_unique} dup_rows={n_raw - n_unique} dedup={dedup}")

    out = _aggregate_duplicates(df, pval_col=pval_col, qval_col=qval_col, how=dedup)
    print(f"[{method_name}_{dataset}] out_rows={len(out)}")
    return out


def merge_method_tables(method_tables: List[pd.DataFrame]) -> pd.DataFrame:
    if not method_tables:
        raise ValueError("No method tables to merge.")
    merged = method_tables[0].copy()
    for t in method_tables[1:]:
        merged = merged.merge(t, on=["id", "peak", "gene", "dataset"], how="outer", validate="one_to_one")
    print(f"[union scores] rows={len(merged)} methods={len(method_tables)} cols={len(merged.columns)}")
    return merged


def _read_methods_csv(methods_csv: str) -> List[Tuple[str, str, str, str, Optional[str]]]:
    """
    Parse methods manifest CSV.
    Returns list of (file_path, dataset_name, method_name, pval_col, fdr_col) in row order.
    """
    if not os.path.isfile(methods_csv):
        raise FileNotFoundError(f"Methods CSV not found: {methods_csv}")
    with open(methods_csv, newline="") as f:
        reader = csv.DictReader(f)
        required = {"file_path", "dataset_name", "method_name", "pval_col"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"methods_csv must have columns: {required}. Got: {reader.fieldnames}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"methods_csv has no data rows: {methods_csv}")
    return [
        (
            r["file_path"].strip(),
            r["dataset_name"].strip(),
            r["method_name"].strip(),
            r["pval_col"].strip(),
            _opt_none(r.get("fdr_col", "")),
        )
        for r in rows
    ]


def load_overlap_bed(overlap_bed: str) -> pd.DataFrame:
    """
    Parse bedtools intersect -wo output (ukbb_hg38 x multiome_union).
    A cols (4): chr_a  start_a  end_a  snp_id
    B cols (4): chr_b  start_b  end_b  peak;gene;dataset
    Final col:  overlap_len
    """
    col_names = [
        "chr_a", "start_a", "end_a", "snp_id",
        "chr_b", "start_b", "end_b", "peak_gene_dataset",
        "overlap_len",
    ]
    df = pd.read_csv(overlap_bed, sep="\t", header=None, names=col_names)

    split = df["peak_gene_dataset"].str.split(";", n=2, expand=True)
    df["peak"] = ctar.data_loader.canonicalize_peak(split[0])
    df["gene"] = split[1].str.strip()
    df["dataset"] = split[2].str.strip() if split.shape[1] > 2 else ""
    df["id"] = df["peak"].astype(str) + ";" + df["gene"].astype(str) + ";" + df["dataset"].astype(str)
    df = df.drop(columns=["peak_gene_dataset"])

    print(
        f"[overlap bed] rows={len(df)} "
        f"unique_snps={df['snp_id'].nunique()} "
        f"unique_peak_gene={df['id'].nunique()}"
    )
    return df


def main(args):
    overlap_df = load_overlap_bed(args.overlap_bed)

    raw_sources = _read_methods_csv(args.methods_csv)
    method_sources = []
    for fp, dataset, name, col, fdr_col in raw_sources:
        if not fp:
            print(f"[skip] {name}/{dataset}: empty file_path"); continue
        if not os.path.isfile(fp):
            print(f"[skip] {name}/{dataset}: file not found ({fp})"); continue
        if not col:
            print(f"[skip] {name}/{dataset}: empty pval_col"); continue
        method_sources.append((fp, dataset, name, col, fdr_col))

    if not method_sources:
        raise RuntimeError("No method sources configured (all rows invalid).")

    print(f"Loading {len(method_sources)} method/dataset files with dedup={args.dedup}...")
    method_tables = [
        load_method_scores(
            fp, method_name=name, dataset=dataset,
            pval_source_col=col, fdr_source_col=fdr_col, dedup=args.dedup,
        )
        for fp, dataset, name, col, fdr_col in method_sources
    ]

    scores_df = merge_method_tables(method_tables)

    n_before = len(overlap_df)
    merged = overlap_df.merge(scores_df.drop(columns=["peak", "gene", "dataset"]), on="id", how="left")
    if len(merged) != n_before:
        raise RuntimeError(f"Row count changed after method merge: {n_before} -> {len(merged)}")

    score_cols = [c for c in scores_df.columns if c not in {"id", "peak", "gene", "dataset"}]
    hit_report = {c: float(merged[c].notna().mean()) for c in score_cols}
    print(f"[method merge] rows={len(merged)} score_hit_rate={hit_report}")

    if args.snp_annot_file:
        annot_df = pd.read_csv(args.snp_annot_file, sep=None, engine="python")
        if args.snp_annot_id_col not in annot_df.columns:
            raise ValueError(
                f"snp_annot_id_col '{args.snp_annot_id_col}' not found in annotation file"
            )
        annot_cols = [c.strip() for c in args.snp_annot_cols.split(",")] if args.snp_annot_cols else []
        keep_cols = [args.snp_annot_id_col] + [c for c in annot_cols if c != args.snp_annot_id_col]
        missing = [c for c in keep_cols if c not in annot_df.columns]
        if missing:
            raise ValueError(f"Annotation file missing columns: {missing}")
        annot_sub = annot_df[keep_cols].rename(columns={args.snp_annot_id_col: "snp_id"})
        merged = merged.merge(annot_sub, on="snp_id", how="left")
        print(f"[snp annot merge] rows={len(merged)} annot_cols={annot_cols}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    merged.to_csv(args.out_path, index=False)
    print(f"Done. Wrote {len(merged)} rows to: {args.out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overlap_bed", type=str, required=True,
        help="bedtools intersect -wo output (ukbb_hg38 x multiome_union). "
             "Expected 9 cols: chr_a start_a end_a snp_id chr_b start_b end_b peak;gene;dataset overlap_len",
    )
    parser.add_argument(
        "--methods_csv", type=str, required=True,
        help="CSV with columns: file_path, dataset_name, method_name, pval_col, fdr_col. "
             "If fdr_col is empty/none, BH q-values are computed from pval_col pre-merge. "
             "Rows sharing a method_name are concatenated across datasets before inter-method merging.",
    )
    parser.add_argument("--out_path", type=str, required=True, help="Output CSV path.")
    parser.add_argument(
        "--snp_annot_file", type=str, default=None,
        help="File mapping snp_id to additional annotations.",
    )
    parser.add_argument(
        "--snp_annot_id_col", type=str, default="snp_id",
        help="Column in snp_annot_file that holds the SNP ID.",
    )
    parser.add_argument(
        "--snp_annot_cols", type=str, default=None,
        help="Comma-separated column names to pull from snp_annot_file.",
    )
    parser.add_argument(
        "--dedup", type=str, default="min",
        choices=["min", "max", "mean", "median", "none"],
        help="How to aggregate duplicate (peak,gene) rows within each method file. "
             "For min/max, the q-value is taken from the same row as the selected p-value.",
    )
    args = parser.parse_args()
    main(args)
