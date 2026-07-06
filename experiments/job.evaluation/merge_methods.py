import argparse
import csv
import os
from typing import List, Optional, Tuple
import numpy as np
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

	agg_score = tmp.groupby("id", sort=False)[score_col].agg(how)

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
	fdr_col: Optional[str] = None,
	fdr_correction: bool = False,
	dedup: str = "min",
) -> pd.DataFrame:
	"""
	Returns a DataFrame with columns: id, peak, gene, <out_label>[, <out_label>_qval]
	One row per (peak,gene) after dedup unless dedup='none'.

	If fdr_col is given and found in the file, it is deduplicated and carried as
	<out_label>_qval. If fdr_col is None and fdr_correction is True, BH correction
	is applied to the deduplicated p-values of this method's own universe.
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

	df[out_label] = df[source_col]

	out = _aggregate_duplicates(df, id_col="id", score_col=out_label, how=dedup)

	n_raw = len(df)
	n_unique = df["id"].nunique()
	n_out = len(out)
	dup = n_raw - n_unique
	print(f"[{out_label}] raw_rows={n_raw} unique_ids={n_unique} dup_rows={dup} out_rows={n_out} dedup={dedup}")

	qval_col = f"{out_label}_qval"
	if fdr_col is not None:
		if fdr_col not in df.columns:
			print(f"[{out_label}] WARNING: fdr col '{fdr_col}' not found in {file_path} — skipping")
		else:
			df[qval_col] = df[fdr_col]
			qval_agg = _aggregate_duplicates(df, id_col="id", score_col=qval_col, how=dedup)
			out = out.merge(qval_agg[["id", qval_col]], on="id", how="left")
			print(f"[{out_label}] using pre-existing fdr column '{fdr_col}'")
	elif fdr_correction:
		pvals = pd.to_numeric(out[out_label], errors="coerce").values
		nan_mask = np.isnan(pvals)
		q = np.full(len(pvals), np.nan)
		if (~nan_mask).sum() > 0:
			_, bh_q, _, _ = multipletests(pvals[~nan_mask], method="fdr_bh")
			q[~nan_mask] = np.minimum(bh_q, 1.0)
		out[qval_col] = q
		print(f"[{out_label}] computed BH q-values on {(~nan_mask).sum()} deduplicated pairs")

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

	merged = eval_df.merge(
		scores_df,
		on=["id", "peak", "gene"],
		how="left",
		validate="many_to_one",
	)

	if len(merged) != n_before:
		raise RuntimeError(f"Row count changed after merge for {eval_path}: {n_before} -> {len(merged)}")

	method_cols = [c for c in scores_df.columns if c not in {"id", "peak", "gene"}]
	hit_report = {c: float(merged[c].notna().mean()) for c in method_cols}
	print(f"[merge {os.path.basename(eval_path)}] rows={n_before} score_hit_rate={hit_report}")

	merged.to_csv(out_path, index=False)


def _read_methods_csv(methods_csv: str) -> List[Tuple[str, str, str, Optional[str]]]:
	"""
	Parse methods manifest CSV.
	Returns list of (file_path, method_name, score_col, fdr_col) in row order.
	fdr_col is None if the 'fdr_col' column is absent or blank.
	Row order determines METHOD_COLS order in downstream evaluation.
	"""
	if not os.path.isfile(methods_csv):
		raise FileNotFoundError(f"Methods CSV not found: {methods_csv}")
	with open(methods_csv, newline="") as f:
		reader = csv.DictReader(f)
		required = {"file_path", "score_col", "method_name"}
		stripped_fieldnames = [f.strip() for f in (reader.fieldnames or [])]
		if not required.issubset(set(stripped_fieldnames)):
			raise ValueError(
				f"methods_csv must have columns: {required}. Got: {reader.fieldnames}"
			)
		rows = [{k.strip(): v for k, v in row.items()} for row in reader]
	if not rows:
		raise ValueError(f"methods_csv has no data rows: {methods_csv}")
	return [
		(
			r["file_path"].strip(),
			r["method_name"].strip(),
			r["score_col"].strip(),
			_opt_none(r.get("fdr_col", "")),
		)
		for r in rows
	]


def _resolve_method_rows(args) -> List[Tuple[str, str, str, Optional[str]]]:
	"""
	Returns list of (file_path, method_name, score_col, fdr_col) from --methods_csv (old)
	or --config_file+--celltype (new).
	"""
	if args.methods_csv:
		return _read_methods_csv(args.methods_csv)
	if not os.path.isfile(args.config_file):
		raise FileNotFoundError(f"Config file not found: {args.config_file}")
	with open(args.config_file, newline="") as f:
		reader = csv.DictReader(f)
		required = {"file_path", "score_col", "method_name"}
		stripped_fieldnames = [f.strip() for f in (reader.fieldnames or [])]
		if not required.issubset(set(stripped_fieldnames)):
			raise ValueError(
				f"config_file must have columns: {required}. Got: {reader.fieldnames}"
			)
		rows = [{k.strip(): v for k, v in row.items()} for row in reader]
	if args.celltype:
		rows = [r for r in rows if r.get("celltype", "").strip() == args.celltype]
		if not rows:
			raise ValueError(
				f"No rows found for celltype={args.celltype!r} in {args.config_file}"
			)
	return [
		(
			r["file_path"].strip(),
			r["method_name"].strip(),
			r["score_col"].strip(),
			_opt_none(r.get("fdr_col", "")),
		)
		for r in rows
	]


def main(args):
	AGG_PATH = args.agg_path
	RES_PATH = args.res_path
	FDR_CORRECTION = args.fdr_correction

	raw_sources = _resolve_method_rows(args)

	method_sources: List[Tuple[str, str, str, Optional[str]]] = []
	for fp, name, col, fdr_col in raw_sources:
		if not fp:
			print(f"[skip] {name}: empty file_path")
			continue
		if not os.path.isfile(fp):
			print(f"[skip] {name}: file not found ({fp})")
			continue
		if not col:
			print(f"[skip] {name}: empty score_col")
			continue
		method_sources.append((fp, name, col, fdr_col))

	if not method_sources:
		raise RuntimeError("No method sources configured (all rows invalid).")

	print(f"Loading using {args.dedup}...")
	method_tables = []
	for fp, out_label, col, fdr_col in method_sources:
		method_tables.append(load_method_scores(
			fp, out_label=out_label, source_col=col,
			fdr_col=fdr_col, fdr_correction=FDR_CORRECTION,
			dedup=args.dedup,
		))

	scores_df = merge_method_tables(method_tables)

	eval_files = [f'{AGG_PATH}/{f}' for f in os.listdir(AGG_PATH) if os.path.isfile(f'{AGG_PATH}/{f}')]
	if not eval_files:
		raise RuntimeError(f"No eval files found in {AGG_PATH}")

	print(f"Found {len(eval_files)} eval files in {AGG_PATH}")
	os.makedirs(RES_PATH, exist_ok=True)
	for eval_fp in eval_files:
		label = os.path.basename(eval_fp).rsplit(".", maxsplit=1)[0]
		out_fp = os.path.join(RES_PATH, f"{label}.csv")
		merge_scores_onto_eval(eval_fp, scores_df=scores_df, out_path=out_fp)

	print(f"Done. Wrote merged eval files to: {RES_PATH}")


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
		"--agg_path",
		type=str,
		default="/ocean/projects/bio240051p/aprieto/multiome/validation/overlap/pbmc",
	)
	parser.add_argument(
		"--res_path",
		type=str,
		default="/ocean/projects/bio240051p/aprieto/multiome/validation/evaluation/pbmc",
	)
	parser.add_argument("--dataset_name", type=str, default="pbmc")

	parser.add_argument(
		"--dedup",
		type=str,
		default="min",
		choices=["min", "max", "mean", "median", "none"],
		help="How to aggregate duplicate (peak,gene) rows within each method file before merging.",
	)
	parser.add_argument(
		"--fdr_correction",
		action="store_true",
		default=False,
		help="Compute BH q-values for methods with no pre-existing fdr column in the manifest.",
	)

	args = parser.parse_args()
	if not args.methods_csv and not args.config_file:
		parser.error("Provide --methods_csv or --config_file (with optional --celltype).")
	main(args)