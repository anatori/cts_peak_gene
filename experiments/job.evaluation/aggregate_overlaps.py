import argparse
import os
import pandas as pd
import ctar


def main(args):

    BED_PATH = args.bed_path
    RES_PATH = args.res_path
    DATASET_NAME = args.dataset_name
    GENE_MISMATCH = args.gene_mismatch
    TAG = args.tag

    cols_dict = {
        'onek1k':['onek1k_id','score','gt_gene','celltype'],
        'tenk10k':['tenk10k_id','score','gt_gene','celltype'],
        'gtex':['gtex_id','score','gt_gene'],
        'abc':['abc_id','exp_id','assay_type','score','gt_gene'],
        'crispr':['crispr_id','score','gt_gene'],
        'ctcf_chiapet':['3d_id','exp_id','score','gt_gene'],
        'rnap2_chiapet':['3d_id','exp_id','score','gt_gene'],
        'intact_hic':['3d_id','exp_id','pval','score','gt_gene'],
        'gwas':['gwas_id','score','gt_gene'],
    }

    eval_df_dict = dict.fromkeys(cols_dict.keys(), [])
    for file in os.listdir(BED_PATH):
        if os.path.isdir(f'{BED_PATH}/{file}'):
            continue
        label = [k for k in cols_dict.keys() if k in file][0]
        filename = file.split('.')[0]
        print(f'{filename}, {label}')
        score_type = bool if label in ('crispr','gwas') else float
        eval_df_dict[filename] = ctar.data_loader.load_validation_intersect_bed(f'{BED_PATH}/{file}',
                                                                             cols_dict[label],
                                                                             score_type=score_type,
                                                                             gene_mismatch=GENE_MISMATCH)
        if eval_df_dict[filename] is None:
            print(f'Failed to read {file}. Skipping...')
            continue

        out_filename = f'{filename}_{TAG}' if TAG else filename
        eval_df_dict[filename].to_csv(f'{RES_PATH}/{out_filename}.csv',index=False)

    print(f'Files written to {RES_PATH}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--bed_path", type=str, default='/projects/zhanglab/users/ana/bedtools2/ana_bedfiles/validation/overlap/pbmc/no_score')
    parser.add_argument("--res_path", type=str, default='/projects/zhanglab/users/ana/multiome/validation/overlap/pbmc')
    parser.add_argument("--dataset_name", type=str, default='pbmc')
    parser.add_argument("--gene_mismatch", type=str, default='drop', choices=['drop', 'zero'],
                        help="How to handle gene mismatches: 'drop' removes them (default), 'zero' keeps them with score=0.")
    parser.add_argument("--tag", type=str, default='',
                        help="If set, appended as '_{tag}' to each output CSV's basename, "
                             "to distinguish runs (e.g. different gene_mismatch modes) writing to the same res_path.")

    args = parser.parse_args()
    main(args)