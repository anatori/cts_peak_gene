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
        eval_df_dict[filename] = pd.read_csv(f'{AGG_PATH}/{file}')
        print(filename, eval_df_dict[filename].shape)
        eval_df_dict[filename] = eval_df_dict[filename].merge(new_df[[NEW_COL,'peak','gene']],how='left')
        print(filename, eval_df_dict[filename].shape)
        print(eval_df_dict[filename].head())
        eval_df_dict[filename].to_csv(f'{AGG_PATH}/{file}',index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--agg_path", type=str, default='/projects/zhanglab/users/ana/multiome/validation/neat')
    parser.add_argument("--new_file", type=str, default='/projects/zhanglab/users/ana/multiome/results/scmultimap/scmm_ctrl/res_df/scmultimap_neat.csv')
    parser.add_argument("--original_col", type=str, default='scmm_mcpval')
    parser.add_argument("--new_col", type=str, default='scmm_mc')
    parser.add_argument("--dataset_name", type=str, default='neat')

    args = parser.parse_args()
    main(args)