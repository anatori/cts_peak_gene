import argparse
import os
import pandas as pd
import ctar


def main(args):

    AGG_PATH = args.agg_path
    DATASET_NAME = args.dataset_name
    NEW_FILE = args.new_file
    ORIGINAL_COL = args.original_col
    NEW_COL = args.new_col

    new_df = pd.read_csv(NEW_FILE,index_col=0)
    new_df = new_df.rename(columns={ORIGINAL_COL:NEW_COL})
    print('new_df',new_df.shape)

    eval_df_dict = {}
    for file in os.listdir(f'{AGG_PATH}'):
        if os.path.isdir(f'{AGG_PATH}/{file}'):
            continue
        filename = file.removesuffix('.csv')
        eval_df_dict[filename] = pd.read_csv(f'{AGG_PATH}/{file}',index_col=0)
        print(filename, eval_df_dict[filename].shape)
        eval_df_dict[filename] = eval_df_dict[filename].merge(new_df[[NEW_COL,'peak','gene']],how='left')
        print(filename, eval_df_dict[filename].shape)
        eval_df_dict[filename].to_csv(f'{AGG_PATH}/{file}',index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--agg_path", type=str, default='/projects/zhanglab/users/ana/multiome/validation/pbmc')
    parser.add_argument("--new_file", type=str, default='/projects/zhanglab/users/ana/multiome/results/ctar/final_eval/pbmc/pbmc_filtered_5.5.5.5.10000/cis_links_df.csv')
    parser.add_argument("--original_col", type=str, default='5.5.5.5.10000_mcpval')
    parser.add_argument("--new_col", type=str, default='ctar_filt_10k')
    parser.add_argument("--dataset_name", type=str, default='neat')

    args = parser.parse_args()
    main(args)