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

	OVERLAP_PATH = args.overlap_path
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
	files = [f for f in os.listdir(OVERLAP_PATH) if os.path.isfile(f'{OVERLAP_PATH}/{f}')]
	print('Using overlap files:',files)
	print('With methods:',METHOD_COLS)

	for file in files:
		overlap_df = pd.read_csv(f'{OVERLAP_PATH}/{file}',index_col=0)
		label = os.path.basename(file).rsplit('.',maxsplit=1)[0].rsplit('_',maxsplit=1)[0]
		if label.startswith("gtex"):
		    overlap_df['label'] = overlap_df.score >= GTEX_SCORE_THRES
		if label.startswith("abc"):
		    overlap_df['label'] = overlap_df.score >= ABC_SCORE_THRES

		# skipping hic for now
		if label in ['ctcf_chiapet','rnap2_chiapet','intact_hic']:
			continue

		# computing auerc
		print(f'Computing AUERC for {label}...')
		res_df = ctar.simu.compute_bootstrap_table(overlap_df, METHOD_COLS, 
	                                  gold_col = GOLD_COL, 
	                                  n_bootstrap = N_BOOTSTRAP, 
	                                  fillna = FILLNA, 
	                                  reference_method=REFERENCE_METHOD, 
	                                  extrapolate=False, 
	                                  weighted=True,
	                                  ascending=True)
		res_df.to_csv(f'{RES_PATH}/{label}_{DATASET_NAME}_auerc.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--overlap_path", type=str, default='/projects/zhanglab/users/ana/multiome/validation/neat')
    parser.add_argument("--res_path", type=str, default='/projects/zhanglab/users/ana/multiome/validation/tables/neat')
    parser.add_argument("--dataset_name", type=str, default='neat')

    parser.add_argument("--method_cols", type=str, default='scent,scmm,signac,ctar_z,ctar,ctar_filt_z,ctar_filt')
    parser.add_argument("--gold_col", type=str, default='score')
    parser.add_argument("--reference_method", type=str, default='ctar')
    parser.add_argument("--fillna", type=str, default='True')
    parser.add_argument("--n_bootstrap", type=str, default='1000')

    parser.add_argument("--gtex_score_thres", type=str, default='0.5')
    parser.add_argument("--abc_score_thres", type=str, default='0.2')

    args = parser.parse_args()
    main(args)