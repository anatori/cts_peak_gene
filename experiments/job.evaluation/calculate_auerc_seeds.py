import argparse
import os
import pandas as pd
import ctar
import numpy as np

"""
TODO
----
- add support for hic thresholds
"""


def _parse_float_list(s: str):
    if s is None:
        return []
    parts = [p.strip() for p in str(s).split(",") if p.strip() != ""]
    return [float(p) for p in parts]

def main(args):

    MERGE_PATH = args.merge_path
    ABC_SCORE_THRES = float(args.abc_score_thres)
    METHOD_COLS = [m.strip() for m in args.method_cols.split(",")]
    GOLD_COL = args.gold_col
    REFERENCE_METHOD = args.reference_method
    FILLNA = str(args.fillna).lower() in {"1", "true", "t", "yes", "y"}
    N_BOOTSTRAP = int(args.n_bootstrap)
    RES_PATH = args.res_path
    DATASET_NAME = args.dataset_name
    HANDLE_DUP = args.handle_dup
    USE_QVAL = args.use_qval

    EQTL_PIP_THRES_LIST = _parse_float_list(args.eqtl_pip_thres_list)
    if len(EQTL_PIP_THRES_LIST) == 0:
        # backward compatible fallback to old single threshold arg
        EQTL_PIP_THRES_LIST = [float(args.eqtl_score_thres)]

    # negative threshold
    EQTL_NEG_THRES = float(args.eqtl_neg_thres)

    os.makedirs(RES_PATH, exist_ok=True)
    files = [f for f in os.listdir(MERGE_PATH) if os.path.isfile(f"{MERGE_PATH}/{f}")]
    print("Using merge files:", files, flush=True)
    print("With methods:", METHOD_COLS, flush=True)
    print("GTEx/oneK1K pip thresholds:", EQTL_PIP_THRES_LIST, flush=True)
    print("Neg threshold:", EQTL_NEG_THRES, flush=True)
    print("Fillna:", FILLNA, "Bootstrap:", N_BOOTSTRAP, "Dedup:", HANDLE_DUP, flush=True)


    for file in files:
        merge_df0 = pd.read_csv(f"{MERGE_PATH}/{file}")

        if USE_QVAL:
            resolved_methods = []
            for m in METHOD_COLS:
                qcol = f"{m}_qval"
                if qcol not in merge_df0.columns:
                    raise ValueError(
                        f"--use_qval is set but '{qcol}' not found in {file}. "
                        f"Run merge_methods.py with --fdr_correction or add an 'fdr' column to the manifest."
                    )
                resolved_methods.append(qcol)
            ref_qcol = f"{REFERENCE_METHOD}_qval"
            if ref_qcol not in merge_df0.columns:
                raise ValueError(
                    f"--use_qval is set but reference method column '{ref_qcol}' not found in {file}."
                )
            ref_resolved = ref_qcol
        else:
            resolved_methods = METHOD_COLS
            ref_resolved = REFERENCE_METHOD

        label = os.path.basename(file).removesuffix('.csv').removesuffix(f'_{DATASET_NAME}')

        # skipping hic for now
        if label in ["ctcf_chiapet", "rnap2_chiapet", "intact_hic"]:
            continue

        is_eqtl = (label.startswith("gtex") or label.startswith("onek1k"))

        # for gtex/onek1k run modes for EACH pip threshold
        if is_eqtl:
            run_plan = []
            for pip_thres in EQTL_PIP_THRES_LIST:
                if args.eqtl_filter_mode in ("both", "all_only"):
                    run_plan.append((pip_thres, "all"))
                if args.eqtl_filter_mode in ("both", "strict_only"):
                    run_plan.append((pip_thres, f"str_pos{pip_thres}_neg{EQTL_NEG_THRES}"))
        else:
            run_plan = [(None, "all")]

        for pip_thres, filter_mode in run_plan:
            merge_df = merge_df0.copy()

            # build labels (and optionally filter)
            if is_eqtl:
                # label based on pip_thres
                merge_df["label"] = merge_df["score"] >= float(pip_thres)

                # filtering: keep only sure pos OR sure neg
                if filter_mode != "all":
                    keep = (merge_df["score"] >= float(pip_thres)) | (merge_df["score"] <= EQTL_NEG_THRES)
                    merge_df = merge_df.loc[keep].copy()
                    merge_df["label"] = merge_df["score"] >= float(pip_thres)

            elif label.startswith("abc"):
                merge_df["label"] = merge_df["score"] >= ABC_SCORE_THRES

            elif label.startswith("crispr"):
                merge_df["label"] = merge_df["score"]

            elif label.startswith("gwas"):
                merge_df["label"] = merge_df["score"]

            else:
                raise ValueError(f"Unrecognized dataset label prefix for file {file} -> label={label}")

            # Ensure clean boolean labels
            merge_df["label"] = merge_df["label"].astype(bool)

            # Counts for reporting
            n_pos = int(merge_df["label"].sum())
            n_total = int(merge_df.shape[0])
            n_neg = int(n_total - n_pos)

            # Tag for this run
            run_tag = f"{label}"
            if is_eqtl:
                run_tag += f"_pip{pip_thres}_{filter_mode}"

            print(f"Computing metrics for {run_tag}... (n_total={n_total}, n_pos={n_pos}, n_neg={n_neg})")

            merge_df["pg_pair"] = merge_df["peak"] + ";" + merge_df["gene"]
            metric_specs = ctar.metrics.build_default_metric_specs(
                merge_df,
                gold_col="label",
                pvals_smaller_is_better=True,
                early_R=0.2
            )

            res_df = ctar.metrics.compute_bootstrap_table_seed_avg(
                all_df=merge_df,
                methods=resolved_methods,
                metrics_list=metric_specs,
                gold_col=GOLD_COL,
                reference_method=ref_resolved,
                n_bootstrap=N_BOOTSTRAP,
                fillna=FILLNA,
                jitter_amount=1e-12,
                seeds=list(range(10)),
                handle_dup=HANDLE_DUP,
                dup_key_cols=["pg_pair"],
                stratified_bootstrap=True,
                tie="zero",
            )

            # annotate output
            res_df["dataset"] = label
            res_df["dataset_name"] = DATASET_NAME
            res_df["run_tag"] = run_tag
            res_df["filter_mode"] = filter_mode
            res_df["n_total"] = n_total
            res_df["n_pos"] = n_pos
            res_df["n_neg"] = n_neg
            res_df["pip_thres"] = float(pip_thres) if is_eqtl else np.nan

            # save per-run
            out_file = f"{RES_PATH}/{run_tag}_{DATASET_NAME}_seed_avg.csv"
            print(res_df.head())
            res_df.to_csv(out_file, index=False)
            print("Wrote:", out_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--merge_path", type=str,
                        default="/projects/zhanglab/users/ana/multiome/validation/evaluation/neat")
    parser.add_argument("--res_path", type=str,
                        default="/projects/zhanglab/users/ana/multiome/validation/evaluation/tables/metrics_jitter/neat")
    parser.add_argument("--dataset_name", type=str, default="neat")

    parser.add_argument("--method_cols", type=str, default="scent,scmm,signac,ctar_filt_z,ctar_filt")
    parser.add_argument("--gold_col", type=str, default="label")
    parser.add_argument("--reference_method", type=str, default="ctar_filt")
    parser.add_argument("--fillna", type=str, default=True)
    parser.add_argument("--n_bootstrap", type=str, default="1000")
    
    parser.add_argument("--handle_dup",
        type=str,
        default=None,
        choices=[None, 'any', 'consensus', 'proportion'],
        help="Handle deduplication of evaluation pairs."
    )
    parser.add_argument(
        "--use_qval",
        action="store_true",
        default=False,
        help="Use {method}_qval columns for evaluation. Raises an error if any expected _qval column is missing.",
    )

    parser.add_argument("--eqtl_filter_mode",
        type=str,
        default="both",
        choices=["both", "all_only", "strict_only"],
        help="For eQTL datasets: run both modes, only 'all', or only strict pos/neg filtering."
    )

    parser.add_argument("--eqtl_score_thres", type=str, default="0.5")
    parser.add_argument("--abc_score_thres", type=str, default="0.2")

    parser.add_argument("--eqtl_pip_thres_list", type=str, default="0.2,0.5")
    parser.add_argument("--eqtl_neg_thres", type=str, default="0.01")

    args = parser.parse_args()
    main(args)