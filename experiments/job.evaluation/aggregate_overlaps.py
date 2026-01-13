import argparse
import os
import pandas as pd
import ctar


def main(args):

    BED_PATH = args.bed_path
    RES_PATH = args.res_path
    DATASET_NAME = args.dataset_name
    METHOD_COLS = args.method_cols
    CAND_COLS = METHOD_COLS + ['peak','gene']

    cols_dict = {
        'gtex':['gtex_id','score','gt_gene'],
        'abc':['abc_id','exp_id','assay_type','score','gt_gene'],
        'crispr':['crispr_id','score','gt_gene'],
        'ctcf_chiapet':['3d_id','exp_id','score','gt_gene'],
        'rnap2_chiapet':['3d_id','exp_id','score','gt_gene'],
        'intact_hic':['3d_id','exp_id','pval','score','gt_gene']
    }

    eval_df_dict = dict.fromkeys(cols_dict.keys(), [])
    for file in os.listdir(BED_PATH):
        label = [k for k in cols_dict.keys() if k in file][0]
        score_type = bool if label == 'crispr' else float
        eval_df_dict[label] = ctar.data_loader.load_validation_intersect_bed(f'{BED_PATH}/{label}_{DATASET_NAME}.bed',
                                                                             cols_dict[label],
                                                                             candidate_cols=CAND_COLS,
                                                                             candidate_methods_cols=METHOD_COLS,
                                                                             score_type=score_type)
        eval_df_dict[label].to_csv(f'{RES_PATH}/{file.split('.')[0]}.csv',index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--bed_path", type=str, default='/projects/zhanglab/users/ana/bedtools2/ana_bedfiles/validation/overlap/neat')
    parser.add_argument("--res_path", type=str, default='/projects/zhanglab/users/ana/multiome/validation/neat')
    parser.add_argument("--dataset_name", type=str, default='neat')
    parser.add_argument("--method_cols", type=list, default=['scent','scmm','signac','ctar','ctar_z','ctar_filt_z','ctar_filt'])

    args = parser.parse_args()
    main(args)