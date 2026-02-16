import argparse
import os
import pandas as pd
import ctar


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


def main(args):

    MERGE_PATH = args.merge_path
    DATASET_NAME = args.dataset_name
    NEW_FILE = args.new_file
    ORIGINAL_COL = args.original_col
    NEW_COL = args.new_col

    new_df = pd.read_csv(NEW_FILE,index_col=0)
    new_df = new_df.rename(columns={ORIGINAL_COL:NEW_COL})
    print('new_df shape before deduplication:',new_df.shape)
    
    # Deduplication
    new_df['id'] = new_df['peak'].astype(str) + ';' + new_df['gene'].astype(str)
    new_df = _aggregate_duplicates(new_df, id_col='id', score_col=NEW_COL, how=args.dedup)
    print(f'new_df shape after {args.dedup} deduplication',new_df.shape)

    eval_df_dict = {}
    for file in os.listdir(f'{MERGE_PATH}'):
        if os.path.isdir(f'{MERGE_PATH}/{file}'):
            continue
        filename = file.removesuffix('.csv')
        eval_df_dict[filename] = pd.read_csv(f'{MERGE_PATH}/{file}')
        if NEW_COL in eval_df_dict[filename].columns:
            print(f'{NEW_COL} exists in {filename} columns. Dropping..')
            eval_df_dict[filename] = eval_df_dict[filename].drop(columns=NEW_COL)
        print(f'{filename} shape before merge:', eval_df_dict[filename].shape)
        eval_df_dict[filename] = eval_df_dict[filename].merge(new_df[[NEW_COL,'peak','gene']],how='left')
        print(f'{filename} shape after merge:', eval_df_dict[filename].shape)
        print(eval_df_dict[filename].head())
        print(f'Saving merged file to {MERGE_PATH}/{file} ...')
        eval_df_dict[filename].to_csv(f'{MERGE_PATH}/{file}',index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--merge_path", type=str, default='/projects/zhanglab/users/ana/multiome/validation/evaluation/neat')
    parser.add_argument("--new_file", type=str, default='/projects/zhanglab/users/ana/multiome/results/scmultimap/scmm_ctrl/res_df/scmultimap_neat.csv')
    parser.add_argument("--original_col", type=str, default='scmm_mcpval')
    parser.add_argument("--new_col", type=str, default='scmm_mc')
    parser.add_argument("--dedup", type=str, default='min')
    parser.add_argument("--dataset_name", type=str, default='neat')

    args = parser.parse_args()
    main(args)