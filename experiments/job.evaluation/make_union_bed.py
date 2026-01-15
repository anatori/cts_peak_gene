import argparse
import os
import pandas as pd
from collections import OrderedDict
import re


peak_regex = re.compile(r"^(?:chr)?(?P<chr>[0-9XYMT]+)[:\-](?P<start>\d+)-(?P<end>\d+)$", re.IGNORECASE)

def canonicalize_peak(s: pd.Series) -> pd.Series:
    """
    Normalize peak strings to 'chr{chr}:{start}-{end}'.
    Handles '1:100-200', 'chr1-100-200', 'chr1:100-200', etc.
    Strips whitespace.
    """
    s = s.astype(str).str.strip()
    def _canon(x: str) -> str:
        m = peak_regex.match(x)
        if not m:
            return x
        chrom = m.group("chr").upper()
        if chrom in {"X", "Y", "MT"}:
            chrom = chrom.lower()
        return f"chr{chrom}:{m.group('start')}-{m.group('end')}"
    return s.map(_canon)


def load_indexed_series(file_path: str, metric_specs: list[tuple[str, str]]) -> dict[str, pd.Series]:
    """
    Load a file and return a dict of Series indexed by canonical 'peak;gene'.
    If duplicates exist for the same 'peak;gene', reduce by taking max.
    """
    if not os.path.isfile(file_path):
        return {}

    df = pd.read_csv(file_path, sep=None, engine='python')

    # Ensure required keys exist
    for required in ["peak", "gene"]:
        if required not in df.columns:
            raise ValueError(f"File '{file_path}' is missing required column '{required}'")

    # Canonicalize peak and clean gene strings
    df["peak"] = canonicalize_peak(df["peak"])
    df["gene"] = df["gene"].astype(str).str.strip()

    # Build index "peak;gene"
    idx = df["peak"].astype(str) + ";" + df["gene"].astype(str)

    out: dict[str, pd.Series] = {}
    for out_label, source_col in metric_specs:
        if source_col in df.columns:
            s = df[source_col].copy()
            s.index = idx
            s.name = out_label

            # Deduplicate by taking max per 'peak;gene'
            if s.index.has_duplicates:
                # coerce to numeric to ensure proper max behavior for pvals/z-scores
                s_numeric = pd.to_numeric(s, errors="coerce")
                s = s_numeric.groupby(level=0).max()
                s.name = out_label

            out[out_label] = s

    return out


def _opt_none(x: str | None) -> str | None:
    """
    Return None for empty/none-like strings, otherwise the original string.
    """
    if x is None:
        return None
    s = str(x).strip().lower()
    return None if s in {"", "none", "null", "na"} else x


def main(args):

    SCENT_FILE = args.scent_file
    SCMM_FILE = args.scmm_file
    SIGNAC_FILE = args.signac_file
    CTAR_FILE = args.ctar_file
    CTAR_FILT_FILE = args.ctar_filt_file

    SCENT_COL = _opt_none(args.scent_col)
    SCMM_COL = _opt_none(args.scmm_col)
    SIGNAC_COL = _opt_none(args.signac_col)
    CTAR_COL_Z = _opt_none(args.ctar_col_z)
    CTAR_COL = _opt_none(args.ctar_col)
    CTAR_FILT_COL_Z = _opt_none(args.ctar_filt_col_z)
    CTAR_FILT_COL = _opt_none(args.ctar_filt_col)

    BED_PATH = args.bed_path
    DATASET_NAME = args.dataset_name

    if args.method_cols:
        METHOD_COLS = [m.strip() for m in args.method_cols.split(",") if m.strip()]
    else:
        METHOD_COLS = ["scent", "scmm", "signac", "ctar_z", "ctar", "ctar_filt_z", "ctar_filt"]

    # Build source configs only with provided columns
    source_configs: list[tuple[str, list[tuple[str, str]]]] = []

    def add_source(file_path: str | None, pairs: list[tuple[str, str | None]]):
        if not file_path:
            return
        specs = [(label, col) for (label, col) in pairs if col]  # keep only non-empty columns
        if specs:
            source_configs.append((file_path, specs))
    add_source(SCENT_FILE, [("scent", SCENT_COL)])
    add_source(SCMM_FILE, [("scmm", SCMM_COL)])
    add_source(SIGNAC_FILE, [("signac", SIGNAC_COL)])
    add_source(CTAR_FILE, [("ctar_z", CTAR_COL_Z), ("ctar", CTAR_COL)])
    add_source(CTAR_FILT_FILE, [("ctar_filt_z", CTAR_FILT_COL_Z), ("ctar_filt", CTAR_FILT_COL)])

    # Build dict of Series
    dfs: "OrderedDict[str, pd.Series]" = OrderedDict()
    per_label_stats = []

    for file_path, specs in source_configs:
        try:
            series_dict = load_indexed_series(file_path, specs)
        except Exception as e:
            raise RuntimeError(f"Error processing '{file_path}': {e}") from e

        for label, _ in specs:
            if label in series_dict:
                s = series_dict[label]
                dfs[label] = s
                per_label_stats.append(
                    {
                        "label": label,
                        "rows": len(s),
                        "unique_index": s.index.nunique(),
                        "duplicates": len(s) - s.index.nunique(),
                    }
                )

    if not dfs:
        raise RuntimeError("No input series were loaded. Check file paths and column names.")

    print("Per-source series stats after canonicalization:")
    for stat in per_label_stats:
        print(
            f"  {stat['label']}: rows={stat['rows']}, unique_index={stat['unique_index']}, "
            f"duplicates={stat['duplicates']}"
        )

    # Concat (outer join)
    links_df = pd.concat(dfs, axis=1)

    total_rows = len(links_df)
    unique_rows = links_df.index.nunique()
    print(f"After concat: total_rows={total_rows}, unique_index_rows={unique_rows}")

    # Recover peak and gene from index
    idx_split = links_df.index.to_series().str.split(";", n=1, expand=True)
    links_df["peak"] = idx_split[0].values
    links_df["gene"] = idx_split[1].values

    # Parse chr, start, end from canonical peak
    peak_parts = links_df["peak"].str.extract(r"^(?P<chr>chr(?:[0-9XYMT]+)):(?P<start>\d+)-(?P<end>\d+)$", flags=re.IGNORECASE)
    if peak_parts.isnull().any().any():
        bad = links_df.loc[peak_parts.isnull().any(axis=1), "peak"].unique()[:5]
        raise RuntimeError(f"Some peaks are malformed and could not be parsed, e.g.: {bad}")
    links_df[["chr", "start", "end"]] = peak_parts[["chr", "start", "end"]]
    links_df["chr"] = links_df["chr"].astype(str)
    links_df[["start", "end"]] = links_df[["start", "end"]].astype(int)

    # Build final_col in canonical metric order, only including present metrics
    present_metric_cols = [c for c in METHOD_COLS if c in links_df.columns]
    missing = [c for c in METHOD_COLS if c not in links_df.columns]
    if missing:
        print(f"Warning! Missing metrics in merged dataframe: {missing}")
    final_components = present_metric_cols + ["peak", "gene"]
    links_df["final_col"] = links_df[final_components].astype(str).agg(";".join, axis=1)

    os.makedirs(BED_PATH, exist_ok=True)
    out_path = os.path.join(BED_PATH, f"{DATASET_NAME}.bed")
    links_df[["chr", "start", "end", "final_col"]].to_csv(out_path, header=False, index=False, sep="\t")

    print(f"final_col contains metrics (in order): {present_metric_cols}")
    print(f"Wrote {len(links_df)} records to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--scent_file", type=str, default="/projects/zhanglab/users/ana/multiome/results/scent/myscent_neatseq.txt")
    parser.add_argument("--scmm_file", type=str, default="/projects/zhanglab/users/ana/multiome/results/scmultimap/scmultimap_neat_cis.csv")
    parser.add_argument("--signac_file", type=str, default="/projects/zhanglab/users/ana/multiome/results/signac/signac_neatseq_links.csv")
    parser.add_argument("--ctar_file", type=str, default="/projects/zhanglab/users/ana/multiome/results/ctar/final_eval/neat/neat_unfiltered_5.5.5.5.1000/cis_links_df.csv")
    parser.add_argument("--ctar_filt_file", type=str, default="/projects/zhanglab/users/ana/multiome/results/ctar/final_eval/neat/neat_filtered_5.5.5.5.1000/cis_links_df.csv")

    parser.add_argument("--scent_col", type=str, default="boot_basic_p")
    parser.add_argument("--scmm_col", type=str, default="pval")
    parser.add_argument("--signac_col", type=str, default="pvalue")
    parser.add_argument("--ctar_col_z", type=str, default="5.5.5.5.1000_mcpval_z")
    parser.add_argument("--ctar_col", type=str, default="5.5.5.5.1000_mcpval")
    parser.add_argument("--ctar_filt_col_z", type=str, default="5.5.5.5.1000_mcpval_z")
    parser.add_argument("--ctar_filt_col", type=str, default="5.5.5.5.1000_mcpval")

    parser.add_argument("--bed_path", type=str, default="/projects/zhanglab/users/ana/bedtools2/ana_bedfiles/validation/union_links")
    parser.add_argument("--dataset_name", type=str, default="neat")

    parser.add_argument("--method_cols", type=str, default=None,
                        help="Comma-separated list of metric column names to order final_col, e.g. 'scent,scmm,signac,ctar_filt_z,ctar_filt'")

    args = parser.parse_args()
    main(args)