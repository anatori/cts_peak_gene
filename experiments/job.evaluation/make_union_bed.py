import argparse
import csv
import os
import re

import pandas as pd
from ctar.data_loader import canonicalize_peak


peak_regex = re.compile(r"^(?:chr)?(?P<chr>[0-9XYMT]+)[:\-](?P<start>\d+)-(?P<end>\d+)$", re.IGNORECASE)


def load_pg_df(file_path: str) -> pd.DataFrame:
    """
    Load a file, canonicalize peaks.
    Return dataframe with columns canonical peak, gene only.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path, sep=None, engine='python')

    for required in ["peak", "gene"]:
        if required not in df.columns:
            raise ValueError(f"File '{file_path}' is missing required column '{required}'")

    df["peak"] = canonicalize_peak(df["peak"])
    df["gene"] = df["gene"].astype(str).str.strip()
    df["idx"] = df["peak"].astype(str) + ";" + df["gene"].astype(str)

    return df[["peak", "gene", "idx"]]


def read_methods_csv(methods_csv: str):
    """
    Read methods manifest CSV (file_path, score_col, method_name).
    Returns list of unique file paths, preserving first-occurrence order.
    A file may appear more than once (different score columns) but is only unioned once.
    """
    if not os.path.isfile(methods_csv):
        raise FileNotFoundError(f"Methods CSV not found: {methods_csv}")
    with open(methods_csv, newline="") as f:
        reader = csv.DictReader(f)
        required = {"file_path", "score_col", "method_name"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"methods_csv must have columns: {required}. Got: {reader.fieldnames}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"methods_csv has no data rows: {methods_csv}")
    seen: dict = {}
    for row in rows:
        fp = row["file_path"].strip()
        if fp and fp not in seen:
            seen[fp] = True
    return list(seen.keys())


def _resolve_file_paths(args) -> list:
    """
    Returns list of unique file_paths from --methods_csv (old) or --config_file+--celltype (new).
    """
    if args.methods_csv:
        return read_methods_csv(args.methods_csv)
    if not os.path.isfile(args.config_file):
        raise FileNotFoundError(f"Config file not found: {args.config_file}")
    with open(args.config_file, newline="") as f:
        reader = csv.DictReader(f)
        required = {"file_path", "score_col", "method_name"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"config_file must have columns: {required}. Got: {reader.fieldnames}"
            )
        rows = list(reader)
    if args.celltype:
        rows = [r for r in rows if r.get("celltype", "").strip() == args.celltype]
        if not rows:
            raise ValueError(
                f"No rows found for celltype={args.celltype!r} in {args.config_file}"
            )
    seen: dict = {}
    for row in rows:
        fp = row["file_path"].strip()
        if fp and fp not in seen:
            seen[fp] = True
    return list(seen.keys())


def main(args):
    BED_PATH = args.bed_path
    DATASET_NAME = args.dataset_name

    source_files = _resolve_file_paths(args)
    src_label = args.methods_csv or args.config_file
    print(f"Loaded {len(source_files)} unique input files from {src_label}")

    pg_set = set()
    per_label_stats = []

    for file_path in source_files:
        try:
            df = load_pg_df(file_path)
        except Exception as e:
            raise RuntimeError(f"Error processing '{file_path}': {e}") from e

        pg_set.update(zip(df['peak'], df['gene'], df['idx']))
        per_label_stats.append(
            {
                "file": os.path.basename(file_path),
                "rows": len(df),
                "unique_index": df.index.nunique(),
                "duplicates": len(df) - df.index.nunique(),
            }
        )

    if not pg_set:
        raise RuntimeError("No input dataframes were loaded. Check file paths.")

    print("Per-source df stats after canonicalization:")
    for stat in per_label_stats:
        print(
            f"  {stat['file']}: rows={stat['rows']}, unique_index={stat['unique_index']}, "
            f"duplicates={stat['duplicates']}"
        )

    links_df = pd.DataFrame(list(pg_set), columns=["peak", "gene", "idx"])
    total_rows = len(links_df)
    unique_rows = links_df.index.nunique()
    print(f"After concat: total_rows={total_rows}, unique_index_rows={unique_rows}")

    peak_parts = links_df["peak"].str.extract(r"^(?P<chr>chr(?:[0-9XYMT]+)):(?P<start>\d+)-(?P<end>\d+)$", flags=re.IGNORECASE)
    if peak_parts.isnull().any().any():
        bad = links_df.loc[peak_parts.isnull().any(axis=1), "peak"].unique()[:5]
        raise RuntimeError(f"Some peaks are malformed and could not be parsed, e.g.: {bad}")
    links_df[["chr", "start", "end"]] = peak_parts[["chr", "start", "end"]]
    links_df["chr"] = links_df["chr"].astype(str)
    links_df[["start", "end"]] = links_df[["start", "end"]].astype(int)

    os.makedirs(BED_PATH, exist_ok=True)
    out_path = os.path.join(BED_PATH, f"{DATASET_NAME}.bed")
    links_df[["chr", "start", "end", "idx"]].to_csv(out_path, header=False, index=False, sep="\t")
    print(f"Wrote {len(links_df)} records to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--methods_csv",
        type=str,
        default=None,
        help="CSV with columns: file_path, score_col, method_name. "
             "Mutually exclusive with --config_file.",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Celltype config CSV with columns: celltype, file_path, score_col, method_name[, eval_dir]. "
             "Use with --celltype to filter rows by cell type.",
    )
    parser.add_argument(
        "--celltype",
        type=str,
        default=None,
        help="Filter --config_file rows to this cell type. No-op without --config_file.",
    )
    parser.add_argument(
        "--bed_path",
        type=str,
        default="/ocean/projects/bio240051p/aprieto/bedfiles/validation/union_links",
    )
    parser.add_argument("--dataset_name", type=str, default="pbmc")

    args = parser.parse_args()
    if not args.methods_csv and not args.config_file:
        parser.error("Provide --methods_csv or --config_file (with optional --celltype).")
    main(args)
