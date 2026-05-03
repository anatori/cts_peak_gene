import argparse
import os
import pandas as pd
from collections import OrderedDict
from typing import Optional
from ctar.data_loader import canonicalize_peak
import re


peak_regex = re.compile(r"^(?:chr)?(?P<chr>[0-9XYMT]+)[:\-](?P<start>\d+)-(?P<end>\d+)$", re.IGNORECASE)


def load_pg_df(file_path: str) -> pd.DataFrame:
    """
    Load a file, canonicalize peaks. 
    Return dataframe with columns canonical peak, gene only.
    """

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path, sep=None, engine='python')

    # Ensure required keys exist
    for required in ["peak", "gene"]:
        if required not in df.columns:
            raise ValueError(f"File '{file_path}' is missing required column '{required}'")

    # Canonicalize peak and clean gene strings
    df["peak"] = canonicalize_peak(df["peak"])
    df["gene"] = df["gene"].astype(str).str.strip()
    df["idx"] = df["peak"].astype(str) + ";" + df["gene"].astype(str)

    return df[["peak","gene","idx"]]


def main(args):

    SCENT_FILE = args.scent_file
    SCMM_FILE = args.scmm_file
    SIGNAC_FILE = args.signac_file
    CTAR_FILE = args.ctar_file
    CTAR_FILT_FILE = args.ctar_filt_file

    BED_PATH = args.bed_path
    DATASET_NAME = args.dataset_name

    # Load each file (skip if empty string)
    source_files = [
        f for f in [SCENT_FILE, SCMM_FILE, SIGNAC_FILE, CTAR_FILE, CTAR_FILT_FILE] 
        if f and f.strip()
    ]
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

    # Concat (outer join)
    links_df = pd.DataFrame(list(pg_set), columns=["peak", "gene", "idx"])
    total_rows = len(links_df)
    unique_rows = links_df.index.nunique()
    print(f"After concat: total_rows={total_rows}, unique_index_rows={unique_rows}")

    # Parse chr, start, end from canonical peak
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

    parser.add_argument("--scent_file", type=str, default="/projects/zhanglab/users/ana/multiome/results/scent/myscent_neatseq.txt")
    parser.add_argument("--scmm_file", type=str, default="/projects/zhanglab/users/ana/multiome/results/scmultimap/scmultimap_neat_cis.csv")
    parser.add_argument("--signac_file", type=str, default="/projects/zhanglab/users/ana/multiome/results/signac/signac_neatseq_links.csv")
    parser.add_argument("--ctar_file", type=str, default="/projects/zhanglab/users/ana/multiome/results/ctar/final_eval/neat/neat_unfiltered_5.5.5.5.1000/cis_links_df.csv")
    parser.add_argument("--ctar_filt_file", type=str, default="/projects/zhanglab/users/ana/multiome/results/ctar/final_eval/neat/neat_filtered_5.5.5.5.1000/cis_links_df.csv")

    parser.add_argument("--bed_path", type=str, default="/projects/zhanglab/users/ana/bedtools2/ana_bedfiles/validation/union_links/no_score")
    parser.add_argument("--dataset_name", type=str, default="neat")

    args = parser.parse_args()
    main(args)