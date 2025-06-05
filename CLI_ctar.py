import numpy as np
import anndata as ad
import muon as mu
import sys
import ctar
import os
import pandas as pd
from tqdm import tqdm
import pybedtools

import time
import argparse




"""
Job description
----------------

create_ctrl : create control peaks or peak-gene pairs
    - Input : --job | --multiome_file | --links_file | --target_path | --genome_file | --bin_config | --bin_type | --pybedtools_path 
    - Output : an array of control peaks or a folder containing several arrays of peak-gene pairs listed as ATAC and RNA indices of shape (2, n_ctrl), per bin

compute_corr : compute pearson correlation for control pairs
    - Input : --job | --multiome_file | --batch_size | --batch | --links_file | --target_path | --bin_config | --bin_type
    - Output : a folder containing several arrays of control peak-gene pair correlations of shape (n_ctrl,) or (n_batch, n_ctrl)
    
TODO
----
- add poisson regression

"""


def main(args):
    sys_start_time = time.time()

    ###########################################################################################
    ######                                    Parse Options                              ######
    ###########################################################################################

    JOB = args.job
    MULTIOME_FILE = args.multiome_file
    BATCH_SIZE = int(args.batch_size)
    BATCH_ID = int(args.batch_id)
    BATCH_OFFSET = int(args.batch_offset)
    LINKS_FILE = args.links_file
    TARGET_PATH = args.target_path
    GENOME_FILE = args.genome_file
    BIN_CONFIG = args.binning_config
    BIN_TYPE = args.binning_type
    PYBEDTOOLS_PATH = args.pybedtools_path

    BATCH = BATCH_ID + BATCH_OFFSET

    # Parse and check arguments
    LEGAL_JOB_LIST = [
        "create_ctrl",
        "compute_corr",
    ]
    err_msg = "# CLI_ldspec: --job=%s not supported" % JOB
    assert JOB is not None, "--job required"
    assert JOB in LEGAL_JOB_LIST, err_msg

    if JOB in [
        "create_ctrl",
        "compute_corr",
    ]:
        assert MULTIOME_FILE is not None, "--multiome_file required for --job=%s" % JOB
        assert LINKS_FILE is not None, "--links_file required for --job=%s" % JOB
        assert TARGET_PATH is not None, "--target_path required for --job=%s" % JOB
    if JOB in ["create_ctrl"]:
        assert BIN_CONFIG is not None, (
            "--binning_config required for --job=%s" % JOB
        )
    if JOB in ["compute_corr"]:
        assert BATCH is not None, "--batch required for --job=%s" % JOB

    # Print input options
    header = ctar.util.get_cli_head()
    header += "Call: CLI_ldspec.py \\\n"
    header += "--job %s\\\n" % JOB
    header += "--multiome_file %s\\\n" % MULTIOME_FILE
    header += "--batch_size %s\\\n" % BATCH_SIZE
    header += "--batch_id %s\\\n" % BATCH_ID
    header += "--batch_offset %s\\\n" % BATCH_OFFSET
    header += "--batch %s\\\n" % BATCH
    header += "--links_file %s\\\n" % LINKS_FILE
    header += "--target_path %s\\\n" % TARGET_PATH
    header += "--genome_file %s\\\n" % GENOME_FILE
    header += "--binning_config %s\\\n" % BIN_CONFIG
    header += "--binning_type %s\\\n" % BIN_TYPE
    header += "--pybedtools_path %s\\\n" % PYBEDTOOLS_PATH
    print(header)

    ###########################################################################################
    ######                                  Data Loading                                 ######
    ###########################################################################################
    
    # Load --multiome_file
    if JOB in [
        "create_ctrl",
        "compute_corr",
    ]:
        print("# Loading --multiome_file")
        if MULTIOME_FILE.endswith('.h5mu'):
            adata_rna = mu.read(MULTIOME_FILE)
            adata_atac = adata_rna.mod['atac']
            adata_rna = adata_rna.mod['rna']
        else:
            adata_rna = ad.read_h5ad(MULTIOME_FILE)
            adata_atac = adata_rna[:,adata_rna.var.feature_types == 'ATAC'].copy()
            adata_rna = adata_rna[:,adata_rna.var.feature_types == 'GEX'].copy()
        adata_rna.var['gene'] = adata_rna.var.index
        adata_atac.var['peak'] = adata_atac.var.index

        print("# Loading --links_file")
        if LINKS_FILE.endswith('.tsv'):
            sep='\t'
        elif LINKS_FILE.endswith('.csv'):
            sep=','
        else:
            raise ValueError("LINKS_FILE must be .tsv or .csv")
        eval_df = pd.read_csv(LINKS_FILE, sep=sep, index_col=0)
        # Convert to ENSEMBL gene ID
        if eval_df.gene.str.startswith('ENSG').all():
            adata_rna.var['gene'] = adata_rna.var.gene_id
            adata_rna.var.index = adata_rna.var.gene_id
        links_arr = eval_df[['peak','gene']].values.T

        # Setting bin config
        # Order: MEAN.GC.MEAN.VAR
        n_atac_mean, n_atac_gc, n_rna_mean, n_rna_var, n_ctrl = BIN_CONFIG.split('.')
        n_atac_mean, n_atac_gc, n_ctrl = map(int, [n_atac_mean, n_atac_gc, n_ctrl])

        # Setting corr destination path
        corr_path = os.path.join(TARGET_PATH, 'ctrl_corr')
        # Setting peak source path
        ctrl_path = os.path.join(TARGET_PATH, 'ctrl_peaks')

    # Load --pybedtools_path
    if JOB in ["create_ctrl"]:
        print("# Setting --pybedtools_path")
        pybedtools.helpers.set_bedtools_path(PYBEDTOOLS_PATH)

        # Setting atac bin type
        if (n_atac_gc == 1) and (n_atac_mean > 1):
            atac_type = 'mean'
            atac_bins = n_atac_mean # only depends on mean
        if BIN_TYPE == 'cholesky':
            atac_type = 'chol_logsum_gc'
            atac_bins = [n_atac_mean, n_atac_gc]
        else:
            atac_type = 'mean_gc'
            atac_bins = [n_atac_mean, n_atac_gc]

    ###########################################################################################
    ######                                  Computation                                  ######
    ###########################################################################################
    
    if JOB == "create_ctrl":
        print("# Running --job create_ctrl")

        if os.path.exists(corr_path) is False:
            os.makedirs(corr_path)
        if os.path.exists(ctrl_path) is False:
            os.makedirs(ctrl_path)

        if f'{n_rna_mean}.{n_rna_var}' == 'inf.inf':
            ctrl_dest_file = os.path.join(ctrl_path, f'ctrl_peaks_{BIN_CONFIG}.npy')
            if os.path.exists(ctrl_dest_file):
                print(f"# {ctrl_dest_file} already exists.")
            else:
                ctrl_peaks = ctar.method.create_ctrl_peaks(
                    adata_atac, type=atac_type, num_bins=atac_bins, b=n_ctrl,
                    peak_col='peak', layer='counts', genome_file=GENOME_FILE
                )
                np.save(ctrl_dest_file, ctrl_peaks)

        else:
            
            final_ctrl_path = os.path.join(ctrl_path, f'ctrl_links_{BIN_CONFIG}')
            if os.path.exists(final_ctrl_path) is False:
                os.makedirs(final_ctrl_path)

            n_rna_mean, n_rna_var = map(int, [n_rna_mean, n_rna_var])
            ctrl_dic, atac_bins_df, rna_bins_df = ctar.method.create_ctrl_pairs(
                adata_atac, adata_rna, atac_bins=atac_bins, rna_bins=[n_rna_mean, n_rna_var],
                atac_type=atac_type, rna_type='mean_var', b=n_ctrl,
                atac_layer='counts', rna_layer='counts', genome_file=GENOME_FILE
            )

            atac_bins_df.index = atac_bins_df.peak
            rna_bins_df.index = rna_bins_df.gene

            eval_df[f'atac_{atac_type}_bin_{n_atac_mean}.{n_atac_gc}'] = eval_df.peak.map(atac_bins_df[f'{atac_type}_bin'].to_dict())
            eval_df[f'rna_mean_var_bin_{n_rna_mean}.{n_rna_var}'] = eval_df.gene.map(rna_bins_df['mean_var_bin'].to_dict())
            eval_df[f'combined_bin_{BIN_CONFIG}'] = eval_df[f'atac_{atac_type}_bin_{n_atac_mean}.{n_atac_gc}'].astype(str) + '_' + eval_df[f'rna_mean_var_bin_{n_rna_mean}.{n_rna_var}'].astype(str)
            eval_df.to_csv(LINKS_FILE, sep=sep)

            unique_bins =  eval_df[f'combined_bin_{BIN_CONFIG}'].unique()
            if len(os.listdir(final_ctrl_path)) < len(unique_bins):
                for key in ctrl_dic:
                    if key in unique_bins:
                        np.save(os.path.join(final_ctrl_path, f'ctrl_{key}.npy'), ctrl_dic[key])
            else:
                print(f"# {final_ctrl_path} is already complete.")
            
            final_corr_path = os.path.join(corr_path, f'ctrl_corr_{BIN_CONFIG}')
            if os.path.exists(final_corr_path) is False:
                os.makedirs(final_corr_path)

    if JOB == "compute_corr":
        print("# Running --job compute_corr")

        start, end = BATCH * BATCH_SIZE, (BATCH + 1) * BATCH_SIZE
        final_corr_path = os.path.join(corr_path, f'ctrl_corr_{BIN_CONFIG}')

        if os.path.exists(final_corr_path) is False:
            os.makedirs(final_corr_path)

        if f'{n_rna_mean}.{n_rna_var}' == 'inf.inf':

            ctrl_peaks = np.load(os.path.join(ctrl_path, f'ctrl_peaks_{BIN_CONFIG}.npy'))
            adata_atac.varm['ctrl_peaks'] = ctrl_peaks[:, :n_ctrl]

            last_batch = eval_df.shape[0] // BATCH_SIZE
            if BATCH == last_batch: end = eval_df.shape[0]

            links = links_arr[:, start:end]
            ctrl_peaks = adata_atac[:, links[0]].varm['ctrl_peaks']
            rna_data = adata_rna[:, links[1]].layers['counts']

            ctrl_pearsonr = [
                ctar.method.pearson_corr_sparse(adata_atac[:, ctrl_peaks[i]].layers['counts'], rna_data[:, [i]])
                for i in range(end - start)
            ]

            if len(ctrl_pearsonr) > 0:
                print(f"# Saving pearsonr_ctrl_{start}_{end}.npy to {final_corr_path}")
                np.save(os.path.join(final_corr_path, f'pearsonr_ctrl_{start}_{end}.npy'), np.array(ctrl_pearsonr))

            curr_n_files = len(os.listdir(final_corr_path))
            if curr_n_files == (last_batch + 1):
                print(f"# All {curr_n_files} batches complete. Consolidating null")
                full_ctrl_pearsonr = ctar.data_loader.consolidate_null(final_corr_path + '/', startswith = 'pearsonr_ctrl_', b=last_batch+1)
                np.save(os.path.join(corr_path, f'ctrl_corr_{BIN_CONFIG}.npy'), full_ctrl_pearsonr)

        else:

            ctrl_files = os.listdir(os.path.join(ctrl_path, f'ctrl_links_{BIN_CONFIG}'))
            if len(ctrl_files) < end: end = len(ctrl_files)
            ctrl_files = ctrl_files[start:end]

            for ctrl_file in ctrl_files:
                ctrl_links = np.load(os.path.join(ctrl_path, f'ctrl_links_{BIN_CONFIG}', ctrl_file))[:, :n_ctrl]

                atac_data = adata_atac[:, ctrl_links[:, 0]].layers['counts']
                rna_data = adata_rna[:, ctrl_links[:, 1]].layers['counts']
                ctrl_pearsonr = ctar.method.pearson_corr_sparse(atac_data, rna_data)
                
                print(f"# Saving pearsonr_{os.path.basename(ctrl_file)} to {final_corr_path}")
                np.save(os.path.join(final_corr_path, f'pearsonr_{os.path.basename(ctrl_file)}'), np.array(ctrl_pearsonr))





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ctar")

    parser.add_argument("--job", type=str, required=True, help="create_ctrl, compute_pr")
    parser.add_argument("--multiome_file", type=str, required=True, default=None)
    parser.add_argument("--batch_size", type=str, required=False, default='100')
    parser.add_argument("--batch_id", type=str, required=False, default='0')
    parser.add_argument("--batch_offset", type=str, required=False, default='0', help="offset for batch calculation")
    parser.add_argument("--links_file", type=str, required=True, default=None)
    parser.add_argument("--target_path", type=str, required=True, default=None)
    parser.add_argument(
        "--genome_file", type=str, required=False, default=None, help="GRCh38.p14.genome.fa.bgz reference"
    )
    parser.add_argument(
        "--binning_config",
        type=str,
        required=True,
        default=None,
        help="{# ATAC mean bins}.{# ATAC GC bins}.{# RNA mean bins}.{# RNA var bins}.{# Sampled controls per bin}",
    )
    parser.add_argument("--binning_type", type=str, required=False, default=None)
    parser.add_argument("--pybedtools_path", type=str, required=False, default=None)

    args = parser.parse_args()
    main(args)