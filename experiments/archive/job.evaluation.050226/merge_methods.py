import argparse
import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
import ctar


def _opt_none(x: Optional[str]) -> Optional[str]:
	if x is None:
		return None
	s = str(x).strip()
	if s.lower() in {"", "none", "null", "na"}:
		return None
	return s


def _aggregate_duplicates(
	df: pd.DataFrame,
	id_col: str,
	score_col: str,
	how: str,
) -> pd.DataFrame:
	"""
	Collapse duplicate ids to one row (id -> score) using a chosen aggregation.
	Keeps id, peak, gene, score.
	"""
	if how == "none":
		print("Warning: no deduplication!")
		return df[["id", "peak", "gene", score_col]].copy()

	if how not in {"min", "max", "mean", "median"}:
		raise ValueError(f"Unsupported dedup method: {how}")

	tmp = df[["id", "peak", "gene", score_col]].copy()
	tmp[score_col] = pd.to_numeric(tmp[score_col], errors="coerce")

	# Keep peak gene consistent per id
	agg_score = tmp.groupby("id", sort=False)[score_col].agg(how)

	# Rebuild peak gene from id
	peak_gene = agg_score.index.to_series().str.split(";", n=1, expand=True)
	out = pd.DataFrame(
		{
			"id": agg_score.index,
			"peak": peak_gene[0].values,
			"gene": peak_gene[1].values,
			score_col: agg_score.values,
		}
	)
	return out


def load_method_scores(
	file_path: str,
	out_label: str,
	source_col: str,
	dedup: str = "min",
) -> pd.DataFrame:
	"""
	Returns a DataFrame with columns: id, peak, gene, <out_label>
	One row per (peak,gene) after dedup unless dedup='none'.
	"""
	if not file_path or not os.path.isfile(file_path):
		raise FileNotFoundError(f"Missing method file: {file_path}")

	df = pd.read_csv(file_path, sep=None, engine="python")

	df["peak"] = ctar.data_loader.canonicalize_peak(df["peak"])
	df["gene"] = df["gene"].astype(str).str.strip()
	df["id"] = df["peak"].astype(str) + ';' + df["gene"].astype(str)
	df["peak"] = ctar.data_loader.canonicalize_peak(df["peak"])

	if source_col not in df.columns:
		raise ValueError(f"File '{file_path}' missing score column '{source_col}'")

	# Copy score into standardized column name
	df[out_label] = df[source_col]

	# Deduplicate safely
	out = _aggregate_duplicates(df, id_col="id", score_col=out_label, how=dedup)

	# Print stats
	n_raw = len(df)
	n_unique = df["id"].nunique()
	n_out = len(out)
	dup = n_raw - n_unique
	print(f"[{out_label}] raw_rows={n_raw} unique_ids={n_unique} dup_rows={dup} out_rows={n_out} dedup={dedup}")

	return out


def merge_method_tables(method_tables: List[pd.DataFrame]) -> pd.DataFrame:
	"""
	Outer-merge all method score tables on id/peak/gene.
	Produces one row per id.
	"""
	if not method_tables:
		raise ValueError("No method tables to merge.")

	merged = method_tables[0].copy()
	for t in method_tables[1:]:
		# Validate 1:1 on id to prevent row explosions
		merged = merged.merge(
			t,
			on=["id", "peak", "gene"],
			how="outer",
			validate="one_to_one",
		)

	print(f"[union scores] rows={len(merged)} methods={len(method_tables)} cols={len(merged.columns)}")
	return merged


def merge_scores_onto_eval(
	eval_path: str,
	scores_df: pd.DataFrame,
	out_path: str,
) -> None:
	eval_df = pd.read_csv(eval_path, sep=None, engine="python")
	n_before = len(eval_df)

	# LEFT join scores onto eval; keep eval row count unchanged
	merged = eval_df.merge(
		scores_df,
		on=["id", "peak", "gene"],
		how="left",
		validate="many_to_one", # eval may have duplicates; scores_df must be unique
	)

	if len(merged) != n_before:
		raise RuntimeError(f"Row count changed after merge for {eval_path}: {n_before} -> {len(merged)}")

	# Print hit rates per method col
	method_cols = [c for c in scores_df.columns if c not in {"id", "peak", "gene"}]
	hit_report = {c: float(merged[c].notna().mean()) for c in method_cols}
	print(f"[merge {os.path.basename(eval_path)}] rows={n_before} score_hit_rate={hit_report}")

	merged.to_csv(out_path, index=False)


def main(args):
	AGG_PATH = args.agg_path
	RES_PATH = args.res_path

	# Parse columns
	SCENT_COL = _opt_none(args.scent_col)
	SCMM_COL = _opt_none(args.scmm_col)
	SIGNAC_COL = _opt_none(args.signac_col)
	CTAR_COL_Z = _opt_none(args.ctar_col_z)
	CTAR_COL = _opt_none(args.ctar_col)
	CTAR_FILT_COL_Z = _opt_none(args.ctar_filt_col_z)
	CTAR_FILT_COL = _opt_none(args.ctar_filt_col)

	# Build method sources
	method_sources: List[Tuple[str, str, str]] = []  # (file, out_label, source_col)

	def add(file_path: Optional[str], out_label: str, col: Optional[str]):
		if file_path is None:
			print(f"[skip] {out_label}: no file provided")
			return
		if not os.path.isfile(file_path):
			print(f"[skip] {out_label}: file not found ({file_path})")
			return
		if col is None:
			print(f"[skip] {out_label}: no score column specified")
			return
		method_sources.append((file_path, out_label, col))

	add(args.scent_file, "scent", SCENT_COL)
	add(args.scmm_file, "scmm", SCMM_COL)
	add(args.signac_file, "signac", SIGNAC_COL)
	add(args.ctar_file, "ctar_z", CTAR_COL_Z)
	add(args.ctar_file, "ctar", CTAR_COL)
	add(args.ctar_filt_file, "ctar_filt_z", CTAR_FILT_COL_Z)
	add(args.ctar_filt_file, "ctar_filt", CTAR_FILT_COL)

	if not method_sources:
		raise RuntimeError("No method sources configured (all columns are None).")

	# Load + dedup each method table
	print(f"Loading using {args.dedup}...")
	method_tables = []
	for fp, out_label, col in method_sources:
		method_tables.append(load_method_scores(fp, out_label=out_label, source_col=col, dedup=args.dedup))

	# Merge into union score table
	scores_df = merge_method_tables(method_tables)

	# Merge onto each eval file
	eval_files = [f'{AGG_PATH}/{f}' for f in os.listdir(AGG_PATH) if os.path.isfile(f'{AGG_PATH}/{f}')]
	if not eval_files:
		raise RuntimeError(f"No eval files found in {AGG_PATH}")

	print(f"Found {len(eval_files)} eval files in {AGG_PATH}")
	for eval_fp in eval_files:
		label = os.path.basename(eval_fp).rsplit(".", maxsplit=1)[0]
		out_fp = os.path.join(RES_PATH, f"{label}.csv")
		merge_scores_onto_eval(eval_fp, scores_df=scores_df, out_path=out_fp)

	print(f"Done. Wrote merged eval files to: {RES_PATH}")



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

	parser.add_argument("--agg_path", type=str, default='/projects/zhanglab/users/ana/multiome/validation/overlap/neat')
	parser.add_argument("--res_path", type=str, default='/projects/zhanglab/users/ana/multiome/validation/evaluation/neat')
	parser.add_argument("--dataset_name", type=str, default='neat')

	# Dedup strategy
	parser.add_argument(
		"--dedup",
		type=str,
		default="min",
		choices=["min", "max", "mean", "median", "none"],
		help="How to aggregate duplicate (peak,gene) rows within each method file before merging.",
	)

	args = parser.parse_args()
	main(args)
