import argparse
import os
import pandas as pd
import ctar

"""
TODO
----
- add support for hic thresholds
"""

def main(args):

	AGG_PATH = args.agg_path
	GTEX_SCORE_THRES = float(args.gtex_score_thres)
	ABC_SCORE_THRES = float(args.abc_score_thres)
	METHOD_COLS = [m.strip() for m in args.method_cols.split(",")]
	GOLD_COL = args.gold_col
	REFERENCE_METHOD = args.reference_method
	FILLNA = bool(args.fillna)
	N_BOOTSTRAP = int(args.n_bootstrap)
	RES_PATH = args.res_path
	DATASET_NAME = args.dataset_name

	os.makedirs(RES_PATH,exist_ok=True)
	files = [f for f in os.listdir(AGG_PATH) if os.path.isfile(f'{AGG_PATH}/{f}')]
	print('Using agg files:',files)
	print('With methods:',METHOD_COLS)

	for file in files:
		agg_df = pd.read_csv(f'{AGG_PATH}/{file}')
		label = os.path.basename(file).rsplit('.',maxsplit=1)[0].rsplit('_',maxsplit=1)[0]
		if label.startswith("gtex") or label.startswith("onek1k"):
		    agg_df['label'] = agg_df.score >= GTEX_SCORE_THRES
		if label.startswith("abc"):
		    agg_df['label'] = agg_df.score >= ABC_SCORE_THRES
		if label.startswith("crispr"):
		    agg_df['label'] = agg_df.score

		# skipping hic for now
		if label in ['ctcf_chiapet','rnap2_chiapet','intact_hic']:
			continue

		# computing auerc
		print(f'Computing AUERC for {label}...')
		agg_df['pg_pair'] = agg_df['peak'] + ';' + agg_df['gene']
		res_df = ctar.simu.compute_bootstrap_table(agg_df, METHOD_COLS, 
	                                  gold_col=GOLD_COL, 
	                                  handle_dup='consensus',
	                                  dup_key_cols=['pg_pair'],
	                                  tie='zero',
	                                  n_bootstrap=N_BOOTSTRAP, 
	                                  fillna=FILLNA, 
	                                  reference_method=REFERENCE_METHOD, 
	                                  extrapolate=True, 
	                                  weighted=True,
	                                  ascending=True)
		res_df.to_csv(f'{RES_PATH}/{label}_{DATASET_NAME}_auerc.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--agg_path", type=str, default='/projects/zhanglab/users/ana/multiome/validation/bmmc')
    parser.add_argument("--res_path", type=str, default='/projects/zhanglab/users/ana/multiome/validation/tables/bmmc')
    parser.add_argument("--dataset_name", type=str, default='bmmc')

    parser.add_argument("--method_cols", type=str, default='scent,scmm,signac,ctar_filt_z,ctar_filt')
    parser.add_argument("--gold_col", type=str, default='label')
    parser.add_argument("--reference_method", type=str, default='ctar_filt')
    parser.add_argument("--fillna", type=str, default='True')
    parser.add_argument("--n_bootstrap", type=str, default='1000')

    parser.add_argument("--gtex_score_thres", type=str, default='0.5')
    parser.add_argument("--abc_score_thres", type=str, default='0.2')

    args = parser.parse_args()
    main(args)