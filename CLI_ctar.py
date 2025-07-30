import numpy as np
import anndata as ad
import muon as mu
import sys
import ctar
import os
import pandas as pd
from tqdm import tqdm
import pybedtools
import psutil

import time
import argparse

from dask.distributed import Client, LocalCluster




"""
Job description
----------------

create_ctrl : create control peaks or peak-gene pairs
    - Input : --job | --multiome_file | --links_file | --target_path | --genome_file | --bin_config | --bin_type | --pybedtools_path | --method 
    - Output : an array of control peaks or a folder containing several arrays of peak-gene pairs listed as ATAC and RNA indices of shape (2, n_ctrl), per bin

compute_corr : compute pearson correlation for control pairs
    - Input : --job | --multiome_file | --batch_size | --batch | --links_file | --target_path | --bin_config | --bin_type
    - Output : a folder containing several arrays of control peak-gene pair correlations of shape (n_ctrl,) or (n_batch, n_ctrl)

compute_pr : compute poisson regression coeff for control pairs
    - Input : --job | --multiome_file | --batch_size | --batch | --links_file | --target_path | --bin_config | --bin_type
    - Output : a folder containing several arrays of control peak-gene pair coefficients of shape (n_ctrl,) or (n_batch, n_ctrl)

compute_pval : compute zscore pval, pooled pval, mc pval based on control pairs
    - Input : --job | --links_file | --target_path | --bin_config | --bin_type | --method
    - Output : additional columns added to links_file pertaining to computed pvalues

compute_cis_pairs : find cis pairs within 500kb of annotated peak and genes
    - Input : --job | --links_file | --method
    - Output : additional columns added to links_file pertaining to computed pvalues
    
TODO
----
- test compute_cis_pairs

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
    METHOD = args.method

    BATCH = BATCH_ID + BATCH_OFFSET

    # Parse and check arguments
    LEGAL_JOB_LIST = [
        "create_ctrl",
        "compute_corr",
        "compute_pr",
        "compute_pval",
        "compute_cis_pairs"
    ]
    err_msg = "# CLI_ctar: --job=%s not supported" % JOB
    assert JOB is not None, "--job required"
    assert JOB in LEGAL_JOB_LIST, err_msg

    if JOB in [
        "create_ctrl",
        "compute_corr",
        "compute_pr",
        "compute_pval"
    ]:
        assert LINKS_FILE is not None, "--links_file required for --job=%s" % JOB
        assert TARGET_PATH is not None, "--target_path required for --job=%s" % JOB
        assert BIN_CONFIG is not None, "--binning_config required for --job=%s" % JOB
    if JOB in [
        "create_ctrl",
        "compute_corr",
        "compute_pr",
        "compute_cis_pairs",
    ]:
        assert MULTIOME_FILE is not None, "--multiome_file required for --job=%s" % JOB
    if JOB in ["create_ctrl","compute_pval"]:
        assert METHOD is not None, "--method required for --job=%s" % JOB
        assert BIN_TYPE is not None, "--binning_type required for --job=%s" % BIN_TYPE
    if JOB in ["compute_corr","compute_pr"]:
        assert BATCH_ID is not None, "--batch_id required for --job=%s" % JOB

    # Print input options
    header = ctar.util.get_cli_head()
    header += "Call: CLI_ctar.py \\\n"
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
    header += "--method %s\\\n" % METHOD
    print(header)

    ###########################################################################################
    ######                                  Data Loading                                 ######
    ###########################################################################################
    
    if JOB in [
        "create_ctrl",
        "compute_corr",
        "compute_pr",
        "compute_pval",
    ]:

        print("# Loading --links_file")
        if LINKS_FILE.endswith('.tsv'):
            sep='\t'
        elif LINKS_FILE.endswith('.csv'):
            sep=','
        else:
            raise TypeError("LINKS_FILE must be .tsv or .csv")
        eval_df = pd.read_csv(LINKS_FILE, sep=sep, index_col=0)

        # Setting bin config
        # Order: MEAN.GC.MEAN.VAR
        n_atac_mean, n_atac_gc, n_rna_mean, n_rna_var, n_ctrl = BIN_CONFIG.split('.')
        n_atac_mean, n_atac_gc, n_ctrl = map(int, [n_atac_mean, n_atac_gc, n_ctrl])

        # Setting corr destination path
        if METHOD == 'corr' or JOB == 'compute_corr':
            corr_path = os.path.join(TARGET_PATH, 'ctrl_corr')
        if METHOD == 'pr' or JOB == 'compute_pr':
            corr_path = os.path.join(TARGET_PATH, 'ctrl_poiss')
        
        # Setting peak source path
        ctrl_path = os.path.join(TARGET_PATH, 'ctrl_peaks')

    if JOB in [
        "create_ctrl",
        "compute_corr",
        "compute_pr",
        "compute_cis_pairs",
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

    if JOB in [
        "create_ctrl",
        "compute_corr",
        "compute_pr",
    ]:

        # Convert genes to ENSEMBL gene ID
        if eval_df.gene.str.startswith('ENSG').all():
            adata_rna.var['gene'] = adata_rna.var.gene_id
            adata_rna.var.index = adata_rna.var.gene_id
        links_arr = eval_df[['peak','gene']].values.T

    if JOB in [
        "create_ctrl",
        "compute_pval",
    ]:

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

    if JOB in ["create_ctrl"]:
        print("# Setting --pybedtools_path")
        pybedtools.helpers.set_bedtools_path(PYBEDTOOLS_PATH)

        if METHOD == 'corr':
            final_path_prefix = 'ctrl_corr_'
        if METHOD == 'pr':
            final_path_prefix = 'ctrl_poiss_'

    if JOB in ["compute_pval"]:

        try:
            if f'combined_bin_{BIN_CONFIG}' in eval_df.columns:
                pass

            elif (
                os.path.exists(os.path.join(TARGET_PATH, 'atac_bins_df.csv')) and
                os.path.exists(os.path.join(TARGET_PATH, 'rna_bins_df.csv'))
            ):
                existing_atac_df = pd.read_csv(os.path.join(TARGET_PATH, 'atac_bins_df.csv'))
                existing_rna_df = pd.read_csv(os.path.join(TARGET_PATH, 'rna_bins_df.csv'))

                if (
                    f'atac_{atac_type}_bin_{n_atac_mean}.{n_atac_gc}' in existing_atac_df.columns and
                    f'rna_{rna_type}_bin_{n_rna_mean}.{n_rna_var}' in existing_rna_df.columns
                ):
                    pass
                else:
                    raise FileNotFoundError("Required ATAC or RNA bin columns not found in respective CSVs.")
            else:
                raise FileNotFoundError("ATAC or RNA bin CSV files not found in control path.")
        except Exception as e:
            raise FileNotFoundError(
                f"Either eval_df must contain 'combined_bin_{BIN_CONFIG}', "
                f"or both individual ATAC and RNA bin columns must exist in their CSVs. Error: {str(e)}"
            )

        if METHOD == 'corr':
            corr_type = 'pearsonr'
            file_prefix = 'pearsonr_ctrl_'
        if METHOD == 'pr':
            corr_type = 'poissonb'
            file_prefix = 'poissonb_ctrl_'

        if corr_type in eval_df.columns:
            corr = eval_df[f'{corr_type}'].values
        else:
            raise FileNotFoundError(f"LINKS_FILE must contain {corr_type} in columns")

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
                atac_type=atac_type, rna_type=rna_type, b=n_ctrl,
                atac_layer='counts', rna_layer='counts', genome_file=GENOME_FILE
            )

            atac_bins_df.index = atac_bins_df.peak
            rna_bins_df.index = rna_bins_df.gene

            atac_bins_dict = atac_bins_df[f'{atac_type}_bin'].to_dict()
            rna_bins_dict = rna_bins_df['mean_var_bin'].to_dict()

            try:
                existing_atac_df = pd.read_csv(os.path.join(TARGET_PATH, 'atac_bins_df.csv'))
                existing_rna_df = pd.read_csv(os.path.join(TARGET_PATH, 'rna_bins_df.csv'))
                existing_atac_df[f'atac_{atac_type}_bin_{n_atac_mean}.{n_atac_gc}'] = existing_atac_df.peak.map(atac_bins_dict)
                existing_rna_df[f'rna_mean_var_bin_{n_rna_mean}.{n_rna_var}'] = existing_rna_df.gene.map(rna_bins_dict)
                
                existing_atac_df.to_csv(os.path.join(TARGET_PATH, 'atac_bins_df.csv'),index=None)
                existing_rna_df.to_csv(os.path.join(TARGET_PATH, 'rna_bins_df.csv'),index=None)

            except FileNotFoundError:
                print("No atac_bins_df.csv or rna_bins_df.csv file found; refraining from adding binning columns")

            eval_df[f'atac_{atac_type}_bin_{n_atac_mean}.{n_atac_gc}'] = eval_df.peak.map(atac_bins_dict)
            eval_df[f'rna_mean_var_bin_{n_rna_mean}.{n_rna_var}'] = eval_df.gene.map(rna_bins_dict)
            eval_df[f'combined_bin_{BIN_CONFIG}'] = eval_df[f'atac_{atac_type}_bin_{n_atac_mean}.{n_atac_gc}'].astype(str) + '_' + eval_df[f'rna_mean_var_bin_{n_rna_mean}.{n_rna_var}'].astype(str)
            eval_df.to_csv(LINKS_FILE, sep=sep)

            existing_corr = os.listdir(final_ctrl_path)
            if len(existing_corr) < len(ctrl_dic.keys()):
                for key in ctrl_dic:
                    filename = f'ctrl_{key}.npy'
                    if filename not in existing_corr:
                        np.save(os.path.join(final_ctrl_path, filename), ctrl_dic[key])
            else:
                print(f"# {final_ctrl_path} is already complete.")
            
            final_corr_path = os.path.join(corr_path, f'{final_path_prefix}{BIN_CONFIG}')
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
            n_ctrl_files = len(ctrl_files)
            if n_ctrl_files < end: end = n_ctrl_files
            ctrl_files = ctrl_files[start:end]

            for ctrl_file in ctrl_files:
                ctrl_links = np.load(os.path.join(ctrl_path, f'ctrl_links_{BIN_CONFIG}', ctrl_file))[:, :n_ctrl]

                atac_data = adata_atac.layers['counts'][:, ctrl_links[:, 0]]
                rna_data = adata_rna.layers['counts'][:, ctrl_links[:, 1]]
                ctrl_pearsonr = ctar.method.pearson_corr_sparse(atac_data, rna_data)
                
                print(f"# Saving pearsonr_{os.path.basename(ctrl_file)} to {final_corr_path}")
                np.save(os.path.join(final_corr_path, f'pearsonr_{os.path.basename(ctrl_file)}'), np.array(ctrl_pearsonr))

            # check for missing files
            if end == n_ctrl_files:
                print('# Last array job...')

                final_ctrl_path = os.path.join(ctrl_path, f'ctrl_links_{BIN_CONFIG}')
                missing_bins = ctar.data_loader.check_missing_bins(final_ctrl_path, final_corr_path)

                if len(missing_bins) > 0:
                    print(f"# Computing correlation for an additional {len(missing_bins)} ctrls.")

                for ctrl_file in missing_bins:
                    ctrl_links = np.load(os.path.join(ctrl_path, f'ctrl_links_{BIN_CONFIG}', ctrl_file))[:, :n_ctrl]

                    atac_data = adata_atac.layers['counts'][:, ctrl_links[:, 0]]
                    rna_data = adata_rna.layers['counts'][:, ctrl_links[:, 1]]
                    ctrl_pearsonr = ctar.method.pearson_corr_sparse(atac_data, rna_data)
                    
                    print(f"# Saving pearsonr_{os.path.basename(ctrl_file)} to {final_corr_path}")
                    np.save(os.path.join(final_corr_path, f'pearsonr_{os.path.basename(ctrl_file)}'), np.array(ctrl_pearsonr))

    if JOB == "compute_pr":
        print("# Running --job compute_pr")

        start, end = BATCH * BATCH_SIZE, (BATCH + 1) * BATCH_SIZE
        final_corr_path = os.path.join(corr_path, f'ctrl_poiss_{BIN_CONFIG}')
        dask_tmp_path = os.path.join(TARGET_PATH, 'dask_temp')

        if os.path.exists(final_corr_path) is False:
            os.makedirs(final_corr_path)
        if os.path.exists(dask_tmp_path) is False:
            os.makedirs(dask_tmp_path)

        n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

        atac_sparse = adata_atac.layers['counts']
        rna_sparse = adata_rna.layers['counts']

        delayed_tasks = []

        if f'{n_rna_mean}.{n_rna_var}' == 'inf.inf':

            # Adjust end for last batch
            last_batch = eval_df.shape[0] // BATCH_SIZE
            if BATCH == last_batch: end = eval_df.shape[0]

            ctrl_peaks = np.load(os.path.join(ctrl_path, f'ctrl_peaks_{BIN_CONFIG}.npy'), mmap_mode='r')

            ctrl_links_ls = []
            ctrl_labels_ls = []

            for i in range(start, end):
                # single peak index
                peak_i = adata_atac.var.index.get_loc(links_arr[0][i - start])
                gene_i = adata_rna.var.index.get_loc(links_arr[1][i - start])

                # only get row i from the ctrl_peaks matrix
                ctrl_peak_i = ctrl_peaks[peak_i, :n_ctrl]

                ctrl_links = np.vstack((
                    ctrl_peak_i,
                    np.full(n_ctrl, gene_i)
                )).T

                ctrl_links = np.ascontiguousarray(ctrl_links)

                ctrl_links_ls.append(ctrl_links)
                ctrl_labels_ls.append(f'ctrl_{i}.npy')

            proc = psutil.Process()
            print(f"# [MEM BEFORE DASK] RSS: {proc.memory_info().rss / 1024**2:.2f} MB")
            print(f"# Total ctrl_links size: {sum(arr.nbytes for arr in ctrl_links_ls) / 1024**2:.2f} MB")

            start_time = time.time()

            ctar.parallel.parallel_poisson_bins_dask_sequential(
                ctrl_links_ls=ctrl_links_ls,
                ctrl_labels_ls=ctrl_labels_ls,
                atac_sparse=atac_sparse,
                rna_sparse=rna_sparse,
                out_path=final_corr_path,
                max_iter=100,
                tol=1e-3,
                n_workers=n_jobs,
                local_directory=dask_tmp_path,
            )

            print('# Sequential regression_time = %0.2fs' % (time.time() - start_time))

            curr_n_files = len(os.listdir(final_corr_path))
            if curr_n_files == (last_batch + 1):
                print(f"# All {curr_n_files} batches complete. Consolidating null")
                full_ctrl_poissonb = ctar.data_loader.consolidate_individual_nulls(final_corr_path + '/', startswith = 'poissonb_ctrl_', b=last_batch+1)
                np.save(os.path.join(corr_path, f'ctrl_poiss_{BIN_CONFIG}.npy'), full_ctrl_poissonb)

        else:

            ctrl_files = os.listdir(os.path.join(ctrl_path, f'ctrl_links_{BIN_CONFIG}'))
            n_ctrl_files = len(ctrl_files)
            if n_ctrl_files < end: end = n_ctrl_files
            ctrl_files = ctrl_files[start:end]

            ctrl_links_ls = []
            file_name_ls = []

            for ctrl_file in ctrl_files:
                ctrl_links = np.load(os.path.join(ctrl_path, f'ctrl_links_{BIN_CONFIG}', ctrl_file))[:n_ctrl]
                ctrl_links_ls.append(ctrl_links)
                file_name_ls.append(os.path.basename(ctrl_file))

            proc = psutil.Process()
            print(f"[MEM BEFORE DASK] RSS: {proc.memory_info().rss / 1024**2:.2f} MB")
            print(f"# Total ctrl_links size: {sum(arr.nbytes for arr in ctrl_links_ls) / 1024**2:.2f} MB")

            start_time = time.time()

            ctar.parallel.parallel_poisson_bins_dask_sequential(
                ctrl_links_ls=ctrl_links_ls,
                ctrl_labels_ls=file_name_ls,
                atac_sparse=atac_sparse,
                rna_sparse=rna_sparse,
                out_path=final_corr_path,
                max_iter=100,
                tol=1e-3,
                n_workers=n_jobs,
                local_directory=dask_tmp_path,
            )

            print('# Sequential regression_time = %0.2fs' % (time.time() - start_time))

            # If last batch, check for missing files and process them
            if end == n_ctrl_files:
                print('# Last array job...')

                final_ctrl_path = os.path.join(ctrl_path, f'ctrl_links_{BIN_CONFIG}')
                missing_bins = ctar.data_loader.check_missing_bins(final_ctrl_path, final_corr_path)

                if len(missing_bins) > 0:
                    print(f"# Computing correlation for an additional {len(missing_bins)} ctrls.")

                for ctrl_file in missing_bins:
                    ctrl_links = np.load(os.path.join(final_ctrl_path, ctrl_file))[:, :n_ctrl]

                    atac_data = adata_atac.layers['counts'][:, ctrl_links[:, 0]]
                    rna_data = adata_rna.layers['counts'][:, ctrl_links[:, 1]]

                    ctrl_poissonb = ctar.method.poisson_irls_delayed_loop(
                        atac_data, rna_data, tol=1e-3
                    )

                    print(f"# Saving poissonb_{os.path.basename(ctrl_file)} to {final_corr_path}")
                    np.save(
                        os.path.join(final_corr_path, f'poissonb_{os.path.basename(ctrl_file)}'),
                        ctrl_poissonb[:,1]
                    )


    if JOB == "compute_cis_pairs":
        print("# Running --job compute_cis_pairs")

        adata_rna.var = ctar.data_loader.get_gene_coords(adata_rna.var)
        cis_pairs_df = ctar.data_loader.peak_to_gene(adata_atac.var, adata_rna.var, split_peaks=True)

        adata_atac.var['ind'] = range(len(adata_atac.var))
        adata_rna.var['ind'] = range(len(adata_rna.var))

        cis_peak_inds = adata_atac[:,cis_pairs_df.peak].var['ind'].values
        cis_gene_inds = adata_rna[:,cis_pairs_df.gene].var['ind'].values

        atac_data = adata_atac.layers['counts']
        rna_data = adata_rna.layers['counts']

        corr = []
        for i in tqdm(range((len(eval_df) // BATCH_SIZE) + 1)):

            atac_subset = atac_data[:,cis_peak_inds[i*BATCH_SIZE:(i+1)*BATCH_SIZE]]
            rna_subset = rna_data[:,cis_gene_inds[i*BATCH_SIZE:(i+1)*BATCH_SIZE]]

            if METHOD == 'corr':
                pearsonr = ctar.method.pearson_corr_sparse(atac_subset,rna_subset)[0]
                corr.append(pearsonr)
            if METHOD == 'poiss':
                poissonb = ctar.method.poisson_irls_delayed_loop(atac_subset,rna_subset)
                corr.append(poissonb[:,1])
            else:
                raise ValueError("Must choose correlation method (currently implemented: corr, poiss)")

        corr = np.concatenate(corr,axis=None)
        eval_df[f'{METHOD}'] = corr

        cis_pairs_df.to_csv('/projects/zhanglab/users/ana/multiome/simulations/bin_analysis/cis_pairs_df.csv')

    if JOB == "compute_pval":
        print("# Running --job compute_pval")

        if f'{n_rna_mean}.{n_rna_var}' == 'inf.inf':

            ctrl_corr = np.load(os.path.join(corr_path, f'{file_prefix}{BIN_CONFIG}.npy'))
            eval_df[f'{BIN_CONFIG}_mcpval'] = ctar.method.initial_mcpval(ctrl_corr,corr)
            eval_df[f'{BIN_CONFIG}_ppval'] = ctar.method.pooled_mcpval(ctrl_corr,corr)
            eval_df[f'{BIN_CONFIG}_zpval'] = ctar.method.zscore_pval(ctrl_corr,corr)[0]

        else:

            try:
                atac_col = f'atac_{atac_type}_bin_{n_atac_mean}.{n_atac_gc}'
                rna_col = f'rna_{rna_type}_bin_{n_rna_mean}.{n_rna_var}'
                combined_col = f'combined_bin_{BIN_CONFIG}'

                eval_df[combined_col] = eval_df[atac_col].astype(str) + '_' + eval_df[rna_col].astype(str)

            except KeyError:
                print("# Required bin columns missing in eval_df. Attempting to load and map from bin CSVs...")

                atac_bins_df = pd.read_csv(os.path.join(TARGET_PATH, 'atac_bins_df.csv'))
                rna_bins_df = pd.read_csv(os.path.join(TARGET_PATH, 'rna_bins_df.csv'))

                atac_bins_df.set_index('peak', inplace=True)
                rna_bins_df.set_index('gene', inplace=True)

                # bin column names
                atac_col = f'atac_{atac_type}_bin_{n_atac_mean}.{n_atac_gc}'
                rna_col = f'rna_{rna_type}_bin_{n_rna_mean}.{n_rna_var}'
                combined_col = f'combined_bin_{BIN_CONFIG}'

                # mapping
                atac_bin_dict = atac_bins_df[atac_col].to_dict()
                rna_bin_dict = rna_bins_df[rna_col].to_dict()
                eval_df[atac_col] = eval_df['peak'].map(atac_bin_dict)
                eval_df[rna_col] = eval_df['gene'].map(rna_bin_dict)

                # combined bin label
                eval_df[combined_col] = eval_df[atac_col].astype(str) + '_' + eval_df[rna_col].astype(str)

            pattern = r'[\d]+.[\d]+_[\d]+.[\d]+'
            folder_prefix = os.path.basename(corr_path)
            mcpval, ppval = ctar.method.binned_mcpval(os.path.join(corr_path, f'{folder_prefix}_{BIN_CONFIG}'),
                                                      eval_df, corr, 
                                                      pattern = pattern,
                                                      startswith=file_prefix,
                                                      bin_label = f'combined_bin_{BIN_CONFIG}')
            eval_df[f'{BIN_CONFIG}_mcpval'] = mcpval
            eval_df[f'{BIN_CONFIG}_ppval'] = ppval

        eval_df.to_csv(LINKS_FILE)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ctar")

    parser.add_argument("--job", type=str, required=True, help="create_ctrl, compute_corr, compute_pr, compute_pval, compute_cis_pairs")
    parser.add_argument("--multiome_file", type=str, required=False, default=None)
    parser.add_argument("--batch_size", type=str, required=False, default='100')
    parser.add_argument("--batch_id", type=str, required=False, default='0')
    parser.add_argument("--batch_offset", type=str, required=False, default='0', help="offset for batch calculation")
    parser.add_argument("--links_file", type=str, required=False, default=None)
    parser.add_argument("--target_path", type=str, required=False, default=None)
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
    parser.add_argument("--binning_type", type=str, required=False, default='mean_var', help='mean_var, cholesky')
    parser.add_argument("--pybedtools_path", type=str, required=False, default=None)
    parser.add_argument("--method", type=str, required=False, default=None,
        help="pr, corr")

    args = parser.parse_args()
    main(args)