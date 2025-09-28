import numpy as np
import anndata as ad
import muon as mu
import sys
import ctar
import os
import pandas as pd
import pybedtools
import pickle

import time
import argparse




"""
Job description
----------------

compute_ctar : find cis pairs within 500kb of annotated peak and genes
    - Input : --job | --job_id | --multiome_file | --batch_size | --target_path | --binning_config | --bin_type
    - Output : additional columns added to links_file pertaining to computed pvalues
    
TODO
----
- Rewrite job description

"""


def main(args):
    sys_start_time = time.time()

    ###########################################################################################
    ######                                    Parse Options                              ######
    ###########################################################################################

    JOB = args.job
    JOB_ID = args.job_id
    MULTIOME_FILE = args.multiome_file
    BATCH_SIZE = int(args.batch_size)
    TARGET_PATH = args.target_path
    RESULTS_PATH = args.results_path
    GENOME_FILE = args.genome_file
    BIN_CONFIG = args.binning_config
    BIN_TYPE = args.binning_type
    PYBEDTOOLS_PATH = args.pybedtools_path
    COVAR_FILE = args.covar_file
    ARRAY_IDX = args.array_idx

    # Parse and check arguments
    LEGAL_JOB_LIST = [
        "compute_ctar",
        "compute_cis_only",
        "compute_ctrl_only",
    ]
    err_msg = "# CLI_ctar: --job=%s not supported" % JOB
    assert JOB is not None, "--job required"
    assert JOB in LEGAL_JOB_LIST, err_msg

    if JOB in [
        "compute_ctar"
        "compute_cis_only",
        "compute_ctrl_only",
    ]:
        assert MULTIOME_FILE is not None, "--multiome_file required for --job=%s" % JOB
        assert TARGET_PATH is not None, "--target_path required for --job=%s" % JOB
    
    if JOB in [
        "compute_ctar"
        "compute_ctrl_only",
    ]:
        assert BIN_TYPE is not None, "--binning_type required for --job=%s" % JOB

    if JOB in ["compute_ctrl_only"]:
        assert RESULTS_PATH is not None, "--results_path required for --job=%s" % JOB

    # Print input options
    header = ctar.util.get_cli_head()
    header += "Call: CLI_ctar.py \\\n"
    header += "--job %s\\\n" % JOB
    header += "--job_id %s\\\n" % JOB_ID
    header += "--multiome_file %s\\\n" % MULTIOME_FILE
    header += "--batch_size %s\\\n" % BATCH_SIZE
    header += "--target_path %s\\\n" % TARGET_PATH
    header += "--results_path %s\\\n" % RESULTS_PATH
    header += "--genome_file %s\\\n" % GENOME_FILE
    header += "--binning_config %s\\\n" % BIN_CONFIG
    header += "--binning_type %s\\\n" % BIN_TYPE
    header += "--pybedtools_path %s\\\n" % PYBEDTOOLS_PATH
    header += "--covar_file %s\\\n" % COVAR_FILE
    header += "--array_idx %s\\\n" % ARRAY_IDX
    print(header)

    ###########################################################################################
    ######                                  Data Loading                                 ######
    ###########################################################################################
    
    if JOB in [
        "compute_ctar",
        "compute_cis_only",
        "compute_ctrl_only"
    ]:

        # Setting bin config
        # Order: MEAN.GC.MEAN.VAR
        n_atac_mean, n_atac_gc, n_rna_mean, n_rna_var, n_ctrl = BIN_CONFIG.split('.')
        n_atac_mean, n_atac_gc, n_rna_mean, n_rna_var, n_ctrl = map(int, [n_atac_mean, 
            n_atac_gc,
            n_rna_mean,
            n_rna_var, 
            n_ctrl]
        )

        print("# Loading --multiome_file")
        if MULTIOME_FILE.endswith('.h5mu'):
            adata_rna = mu.read(MULTIOME_FILE)
            adata_atac = adata_rna.mod['atac']
            adata_rna = adata_rna.mod['rna']
        else:
            adata_rna = ad.read_h5ad(MULTIOME_FILE)
            adata_atac = adata_rna[:,adata_rna.var.feature_types == 'ATAC'].copy()
            adata_rna = adata_rna[:,adata_rna.var.feature_types == 'GEX'].copy()

        if adata_rna.var.index.name is not None:
            adata_rna.var.index.name = None
        if adata_atac.var.index.name is not None:
            adata_atac.var.index.name = None

        adata_rna.var['gene'] = adata_rna.var.index
        adata_atac.var['peak'] = adata_atac.var.index

        try:
            adata_rna.layers['counts']
            adata_atac.layers['counts']
            pass
        except:
            print("# Assuming counts are found in .X attribute...")
            adata_rna.layers['counts'] = adata_rna.X
            adata_atac.layers['counts'] = adata_atac.X

        # Preferentially use gene_id
        if 'gene_id' in adata_rna.var.columns:
            adata_rna.var['gene'] = adata_rna.var.gene_id
            adata_rna.var.index = adata_rna.var.gene_id 

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
        rna_type = 'mean_var'

        print("# Setting --pybedtools_path")
        pybedtools.helpers.set_bedtools_path(PYBEDTOOLS_PATH)

        if COVAR_FILE:
            print("# Loading --covar_file")
            covar_mat = np.load(COVAR_FILE)

    if JOB in ["compute_ctrl_only"]:

        print("# Loading --results_path")
        csv_file = os.path.join(RESULTS_PATH, 'cis_links_df.csv')
        if not os.path.isfile(csv_file):
            raise FileNotFoundError(f"Expected cis_links_df.csv at {csv_file}, but it was not found.")
        cis_links_df = pd.read_csv(csv_file)

        if ARRAY_IDX is not None:
            ARRAY_IDX = int(ARRAY_IDX)


    ###########################################################################################
    ######                                  Computation                                  ######
    ###########################################################################################


    if JOB == "compute_ctar":

        print("# Running --job compute_ctar")

        adata_rna.var = ctar.data_loader.get_gene_coords(adata_rna.var)
        cis_links_df = ctar.data_loader.peak_to_gene(adata_atac.var, adata_rna.var, split_peaks=True)

        ctrl_links_dic, atac_bins_df, rna_bins_df = ctar.method.create_ctrl_pairs(
            adata_atac, adata_rna, atac_bins=[n_atac_mean,n_atac_gc], rna_bins=[n_rna_mean,n_rna_var],
            atac_type=atac_type, rna_type=rna_type, b=n_ctrl,
            atac_layer='counts', rna_layer='counts', genome_file=GENOME_FILE
        )

        cis_links_df = ctar.data_loader.combine_peak_gene_bins(cis_links_df, atac_bins_df, rna_bins_df, 
            atac_bins=[n_atac_mean,n_atac_gc], 
            rna_bins=[n_rna_mean,n_rna_var]
        )

        cis_links_dic, cis_idx_dic = ctar.data_loader.groupby_combined_bins(cis_links_df, 
            combined_bin_col=f'combined_bin_{BIN_CONFIG.rsplit('.',1)[0]}', 
            return_dic=True
        )

        rna_sparse = adata_rna.layers['counts']
        atac_sparse = adata_atac.layers['counts']

        print('# Starting cis-links IRLS...')
        start_time = time.time()

        cis_coeff_dic = ctar.parallel.multiprocess_poisson_irls_chunked(
                links_dict=cis_links_dic,
                atac_sparse=atac_sparse,
                rna_sparse=rna_sparse,
                batch_size=BATCH_SIZE,
                scheduler="threads",
            )
        print('# Cis-links IRLS time = %0.2fs' % (time.time() - start_time))

        print('# Starting control links IRLS...')
        start_time = time.time()

        ctrl_coeff_dic = ctar.parallel.multiprocess_poisson_irls_chunked(
            links_dict=ctrl_links_dic,
            atac_sparse=atac_sparse,
            rna_sparse=rna_sparse,
            batch_size=BATCH_SIZE,
            scheduler="threads",
        )
        print('# Control links IRLS time = %0.2fs' % (time.time() - start_time))

        mcpval_dic, ppval_dic = ctar.method.binned_mcpval(cis_coeff_dic, ctrl_coeff_dic)

        cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, mcpval_dic, col_name=f'{BIN_CONFIG}_mcpval')
        cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, mcpval_dic, col_name=f'{BIN_CONFIG}_ppval')
        cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, cis_coeff_dic, col_name='poissonb')

        results_folder = f'{TARGET_PATH}/{JOB_ID}_results/'
        print(f'# Saving files to {results_folder}')
        os.makedirs(results_folder, exist_ok=True)
        with open(f'{results_folder}cis_coeff_dic.pkl', 'wb') as f:
            pickle.dump(cis_coeff_dic, f)
        with open(f'{results_folder}cis_idx_dic.pkl', 'wb') as f:
            pickle.dump(cis_idx_dic, f)

        with open(f'{results_folder}ctrl_coeff_dic.pkl', 'wb') as f:
            pickle.dump(ctrl_coeff_dic, f)
        with open(f'{results_folder}ctrl_links_dic.pkl', 'wb') as f:
            pickle.dump(ctrl_links_dic, f)

        cis_links_df.to_csv(f'{results_folder}cis_links_df.csv')


    if JOB == "compute_cis_only":

        print("# Running --job compute_cis_only")

        adata_rna.var = ctar.data_loader.get_gene_coords(adata_rna.var)
        cis_links_df = ctar.data_loader.peak_to_gene(adata_atac.var, adata_rna.var, split_peaks=True)

        ctrl_links_dic, atac_bins_df, rna_bins_df = ctar.method.create_ctrl_pairs(
            adata_atac, adata_rna, atac_bins=[n_atac_mean,n_atac_gc], rna_bins=[n_rna_mean,n_rna_var],
            atac_type=atac_type, rna_type=rna_type, b=n_ctrl,
            atac_layer='counts', rna_layer='counts', genome_file=GENOME_FILE
        )

        cis_links_df = ctar.data_loader.combine_peak_gene_bins(cis_links_df, atac_bins_df, rna_bins_df, 
            atac_bins=[n_atac_mean,n_atac_gc], 
            rna_bins=[n_rna_mean,n_rna_var]
        )

        cis_links_dic, cis_idx_dic = ctar.data_loader.groupby_combined_bins(cis_links_df, 
            combined_bin_col=f'combined_bin_{BIN_CONFIG.rsplit('.',1)[0]}', 
            return_dic=True
        )

        rna_sparse = adata_rna.layers['counts']
        atac_sparse = adata_atac.layers['counts']

        print('# Starting cis-links IRLS...')
        start_time = time.time()

        if COVAR_FILE:
            cis_coeff_dic = ctar.parallel.multiprocess_poisson_irls_multivar(
                    links_dict=cis_links_dic,
                    atac_sparse=atac_sparse,
                    rna_sparse=rna_sparse,
                    covar_mat=covar_mat,
                    batch_size=BATCH_SIZE,
                    scheduler="threads",
                )
        else:
            cis_coeff_dic = ctar.parallel.multiprocess_poisson_irls(
                    links_dict=cis_links_dic,
                    atac_sparse=atac_sparse,
                    rna_sparse=rna_sparse,
                    batch_size=BATCH_SIZE,
                    scheduler="threads",
                    flag_float32=False,
                    tol=1e-6,
                )
        print('# Cis-links IRLS time = %0.2fs' % (time.time() - start_time))

        cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, cis_coeff_dic, col_name='poissonb')

        results_folder = f'{TARGET_PATH}/{JOB_ID}_results/'
        print(f'# Saving files to {results_folder}')
        os.makedirs(results_folder, exist_ok=True)

        with open(f'{results_folder}cis_coeff_dic.pkl', 'wb') as f:
            pickle.dump(cis_coeff_dic, f)
        with open(f'{results_folder}cis_idx_dic.pkl', 'wb') as f:
            pickle.dump(cis_idx_dic, f)

        cis_links_df.to_csv(f'{results_folder}cis_links_df.csv')


    if JOB == "compute_ctrl_only":

        print("# Running --job compute_ctrl_only")

        if ARRAY_IDX is None:

            ctrl_links_dic, atac_bins_df, rna_bins_df = ctar.method.create_ctrl_pairs(
                adata_atac, adata_rna, atac_bins=[n_atac_mean,n_atac_gc], rna_bins=[n_rna_mean,n_rna_var],
                atac_type=atac_type, rna_type=rna_type, b=n_ctrl,
                atac_layer='counts', rna_layer='counts', genome_file=GENOME_FILE
            )

            cis_links_df = ctar.data_loader.combine_peak_gene_bins(cis_links_df, atac_bins_df, rna_bins_df, 
                atac_bins=[n_atac_mean,n_atac_gc], 
                rna_bins=[n_rna_mean,n_rna_var]
            )

            cis_links_dic, cis_idx_dic = ctar.data_loader.groupby_combined_bins(cis_links_df, 
                combined_bin_col=f'combined_bin_{BIN_CONFIG.rsplit('.',1)[0]}', 
                return_dic=True
            )

            file_suffix = ''

        else:

            file_path = f'{TARGET_PATH}/ctrl_links_dic.pkl'
            with open(file_path, 'rb') as file:
                ctrl_links_dic = pickle.load(file)
            ctrl_links_dic = dict(sorted(ctrl_links_dic.items()))
            ctrl_links_dic = dict(list(ctrl_links_dic.items())[ARRAY_IDX*BATCH_SIZE : (ARRAY_IDX+1)*BATCH_SIZE])
            print('# Running %d controls...' % (len(list(ctrl_links_dic.keys()))))

            file_suffix = f'_{ARRAY_IDX}'

        rna_sparse = adata_rna.layers['counts']
        atac_sparse = adata_atac.layers['counts']

        print('# Starting control links IRLS...')
        print(f'# ATAC and RNA dtype: {atac_sparse.dtype}, {rna_sparse.dtype}')
        start_time = time.time()

        if COVAR_FILE:
            ctrl_coeff_dic = ctar.parallel.multiprocess_poisson_irls_multivar(
                    links_dict=ctrl_links_dic,
                    atac_sparse=atac_sparse,
                    rna_sparse=rna_sparse,
                    covar_mat=covar_mat,
                    batch_size=BATCH_SIZE,
                    scheduler="threads",                
                )
        else:
            ctrl_coeff_dic = ctar.parallel.multiprocess_poisson_irls(
                    links_dict=ctrl_links_dic,
                    atac_sparse=atac_sparse,
                    rna_sparse=rna_sparse,
                    batch_size=BATCH_SIZE,
                    scheduler="threads",
                    flag_float32=False,
                    tol=1e-6,
                )
        print('# Control links IRLS time = %0.2fs' % (time.time() - start_time))

        if ARRAY_IDX is None:

            cis_coeff_dic = ctar.data_loader.map_df_to_dic(cis_links_df, 
                keys_col=f'combined_bin_{BIN_CONFIG.rsplit('.',1)[0]}',
                values_col='poissonb')

            mcpval_dic, ppval_dic = ctar.method.binned_mcpval(cis_coeff_dic, ctrl_coeff_dic)

            cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, mcpval_dic, col_name=f'{BIN_CONFIG}_mcpval')
            cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, ppval_dic, col_name=f'{BIN_CONFIG}_ppval')

            results_folder = f'{TARGET_PATH}/{JOB_ID}_results/'
            
        else:
            results_folder = f'{TARGET_PATH}/'

        print(f'# Saving files to {results_folder}')
        os.makedirs(results_folder, exist_ok=True)

        with open(f'{results_folder}ctrl_coeff_dic{file_suffix}.pkl', 'wb') as f:
            pickle.dump(ctrl_coeff_dic, f)

        if ARRAY_IDX is None:
            with open(f'{results_folder}ctrl_links_dic.pkl', 'wb') as f:
                pickle.dump(ctrl_links_dic, f)

            cis_links_df.to_csv(f'{results_folder}cis_links_df.csv')





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ctar")

    parser.add_argument("--job", type=str, required=True, help="compute_cis_pairs")
    parser.add_argument("--job_id", type=str, required=False, default='1', help='job id')
    parser.add_argument("--multiome_file", type=str, required=False, default=None)
    parser.add_argument("--batch_size", type=str, required=False, default='100')
    parser.add_argument("--links_file", type=str, required=False, default=None)
    parser.add_argument("--target_path", type=str, required=False, default=None)
    parser.add_argument("--results_path", type=str, required=False, default=None)
    parser.add_argument(
        "--genome_file", type=str, required=False, default=None, help="GRCh38.p14.genome.fa.bgz reference"
    )
    parser.add_argument("--covar_file", type=str, required=False, default=None)
    parser.add_argument(
        "--binning_config",
        type=str,
        required=False,
        default=None,
        help="{# ATAC mean bins}.{# ATAC GC bins}.{# RNA mean bins}.{# RNA var bins}.{# Sampled controls per bin}",
    )
    parser.add_argument("--binning_type", type=str, required=False, default='mean_var', help='mean_var, cholesky')
    parser.add_argument("--pybedtools_path", type=str, required=False, default=None)
    parser.add_argument("--array_idx", type=str, required=False, default=None)

    args = parser.parse_args()
    main(args)