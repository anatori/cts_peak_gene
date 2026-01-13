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

compute_ctar : end-to-end pipeline to compute cis peak-gene pairs, control pairs, and p-values
    - Input : --job | --job_id | --multiome_file | --batch_size | --target_path | --genome_file | --binning_config | --binning_type | --pybedtools_path | --covar_file
    - Output : cis_links_df.csv with p-values, control/cis coefficient dictionaries or arrays, and control peaks (if inf.inf mode)

generate_links : find cis pairs within 500kb of annotated peaks and genes, generate control peaks or control peak-gene pairs
    - Input :  --job | --job_id | --multiome_file | --batch_size | --target_path | --genome_file | --binning_config | --binning_type | --pybedtools_path
    - Output : cis_links_df.csv, control peaks array (if inf.inf mode) or control links dictionary (if binned mode), cis index dictionary (if binned mode)

generate_controls : create control peaks or control peak-gene pairs without generating cis-links
    - Input : --job | --job_id | --multiome_file | --batch_size | --target_path | --results_path | --genome_file | --binning_config | --binning_type | --pybedtools_path
    - Output :  control peaks array (if inf.inf mode) or control links dictionary (if binned mode)

compute_cis_only : compute poisson regression coefficients for cis peak-gene pairs
    - Input : --job | --job_id | --multiome_file | --batch_size | --target_path | --results_path | --genome_file | --binning_config | --binning_type | --pybedtools_path | --covar_file | --array_idx
    - Output : cis coefficient array (if inf.inf mode) or cis coefficient dictionary (if binned mode) of shape (n_links,) or (n_links, 2) with SE

compute_ctrl_only : compute poisson regression coefficients for control peak-gene pairs
    - Input : --job | --job_id | --multiome_file | --batch_size | --target_path | --results_path | --genome_file | --binning_config | --binning_type | --pybedtools_path | --covar_file | --array_idx
    - Output : control coefficient array (if inf.inf mode) or control coefficient dictionary (if binned mode) of shape (n_links, n_ctrl) or (n_links, n_ctrl, 2) with SE

compute_pval : compute monte carlo p-values, pooled p-values, and studentized p-values based on control pairs
    - Input :  --job | --results_path | --binning_config | --binning_type
    - Output : updated cis_links_df.csv with additional columns for mcpval, ppval, mcpval_z, ppval_z

TODO
----
- cleanup

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
        "generate_links",
        "generate_controls",
        "compute_pval",
    ]
    err_msg = "# CLI_ctar: --job=%s not supported" % JOB
    assert JOB is not None, "--job required"
    assert JOB in LEGAL_JOB_LIST, err_msg

    if JOB in [
        "compute_ctar",
        "compute_cis_only",
        "compute_ctrl_only",
        "generate_links",
        "generate_controls",
    ]:
        assert MULTIOME_FILE is not None, "--multiome_file required for --job=%s" % JOB
        assert TARGET_PATH is not None, "--target_path required for --job=%s" % JOB
    
    if JOB in [
        "compute_ctar",
        "compute_ctrl_only",
        "generate_controls",
    ]:
        assert BIN_TYPE is not None, "--binning_type required for --job=%s" % JOB

    if JOB in ["compute_ctrl_only","generate_controls","compute_pval"]:
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
        "compute_ctrl_only",
        "generate_links",
        "generate_controls",
        "compute_pval",
    ]: 

        # Setting bin config
        # Order:  MEAN.GC.MEAN.VAR
        n_atac_mean, n_atac_gc, n_rna_mean, n_rna_var, n_ctrl = BIN_CONFIG.split('.')
        n_atac_mean, n_atac_gc, n_ctrl = map(int, [n_atac_mean, n_atac_gc, n_ctrl])
        
        # Check if using inf.inf mode (no RNA binning)
        USE_INF_MODE = (f'{n_rna_mean}.{n_rna_var}' == 'inf.inf')
        
        if not USE_INF_MODE: 
            n_rna_mean, n_rna_var = map(int, [n_rna_mean, n_rna_var])

    if JOB in [
        "compute_ctar",
        "compute_cis_only",
        "compute_ctrl_only",
        "generate_links",
        "generate_controls",
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
        elif BIN_TYPE == 'cholesky':
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

        if ARRAY_IDX:
            ARRAY_IDX = int(ARRAY_IDX)

    if RESULTS_PATH and JOB in ["compute_cis_only", "compute_ctrl_only", "generate_controls", "compute_pval"]: 

        print("# Loading --results_path")
        csv_file = os.path.join(RESULTS_PATH, 'cis_links_df.csv')
        if not os.path.isfile(csv_file):
            raise FileNotFoundError(f"Expected cis_links_df.csv at {csv_file}, but it was not found.")
        cis_links_df = pd.read_csv(csv_file, index_col=0)

    if JOB in ["compute_ctrl_only", "compute_cis_only"]:

        n_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
        print(f'# SLURM allocated {n_cores} CPUs for this job')

    if JOB in ["compute_pval"]:
        
        print("# Loading cis coefficients...")
        cis_coeff_file = os.path.join(RESULTS_PATH, 'cis_coeff.npy' if USE_INF_MODE else 'cis_coeff_dic.pkl')
        
        if USE_INF_MODE:
            if not os.path.exists(cis_coeff_file):
                raise FileNotFoundError(f"cis_coeff.npy not found at {cis_coeff_file}")
            cis_coeff = np.load(cis_coeff_file)
            print(f"# Loaded cis_coeff.npy with shape {cis_coeff.shape}")
        else:
            if not os.path.exists(cis_coeff_file):
                raise FileNotFoundError(f"cis_coeff_dic.pkl not found at {cis_coeff_file}")
            with open(cis_coeff_file, 'rb') as f:
                cis_coeff_dic = pickle.load(f)
            print(f"# Loaded cis_coeff_dic with {len(cis_coeff_dic)} bins")

            cis_idx_file = os.path.join(RESULTS_PATH, 'cis_idx_dic.pkl')
            if not os.path.exists(cis_idx_file):
                raise FileNotFoundError(f"cis_idx_dic.pkl not found at {cis_idx_file}")
            with open(cis_idx_file, 'rb') as f:
                cis_idx_dic = pickle.load(f)
            print(f"# Loaded cis_idx_dic with {len(cis_idx_dic)} bins")
        
        print("# Loading control coefficients...")
        prefix = 'ctrl_coeff_' if USE_INF_MODE else 'ctrl_coeff_dic_'
        suffix = '.npy' if USE_INF_MODE else '.pkl'
        ctrl_coeff_files = [
            f for f in os.listdir(RESULTS_PATH) 
            if f.startswith(prefix) and f.endswith(suffix)
        ]
        print(f"# Found {len(ctrl_coeff_files)} ctrl coefficient files")
        
        if USE_INF_MODE: 
            if ctrl_coeff_files:
                print("# Consolidating array outputs...")
                ctrl_coeff = ctar.data_loader.consolidate_null_npy(
                    path=RESULTS_PATH,
                    startswith=prefix,
                    b=n_ctrl,
                    remove_empty=True,
                    print_missing=False
                )
            else:
                ctrl_coeff_file = os.path.join(RESULTS_PATH, 'ctrl_coeff.npy')
                if not os.path.exists(ctrl_coeff_file):
                    raise FileNotFoundError(f"No ctrl_coeff files found in {RESULTS_PATH}")
                ctrl_coeff = np.load(ctrl_coeff_file)
                print(f"# Loaded single ctrl_coeff.npy with shape {ctrl_coeff.shape}")
        
        else:
            if ctrl_coeff_files:
                print("# Consolidating dictionary outputs...")
                ctrl_coeff_dic = ctar.data_loader.consolidate_null_dic(
                    path=RESULTS_PATH,
                    startswith=prefix,
                    remove_empty=True,
                    print_missing=False
                )
            else:
                ctrl_coeff_file = os.path.join(RESULTS_PATH, 'ctrl_coeff_dic.pkl')
                if not os.path.exists(ctrl_coeff_file):
                    raise FileNotFoundError(f"No ctrl_coeff_dic files found in {RESULTS_PATH}")
                with open(ctrl_coeff_file, 'rb') as f:
                    ctrl_coeff_dic = pickle.load(f)
                print(f"# Loaded single ctrl_coeff_dic.pkl with {len(ctrl_coeff_dic)} bins")


    ###########################################################################################
    ######                                  Computation                                  ######
    ###########################################################################################


    if JOB == "compute_ctar":

        print("# Running --job compute_ctar")

        adata_rna.var = ctar.data_loader.get_gene_coords(adata_rna.var)
        cis_links_df = ctar.data_loader.peak_to_gene(adata_atac.var, adata_rna.var, split_peaks=True)

        if USE_INF_MODE: 
            print("# Using inf.inf mode - creating control peaks only")
            
            # Create control peaks (not peak-gene pairs)
            ctrl_peaks = ctar.method.create_ctrl_peaks(
                adata_atac, type=atac_type, num_bins=atac_bins, b=n_ctrl,
                peak_col='peak', layer='counts', genome_file=GENOME_FILE
            )
            
            # Get cis links indices
            adata_atac.var['atac_idx'] = range(len(adata_atac.var))
            adata_rna.var['rna_idx'] = range(len(adata_rna.var))
            cis_links_df['atac_idx'] = cis_links_df['peak'].map(adata_atac.var['atac_idx'])
            cis_links_df['rna_idx'] = cis_links_df['gene'].map(adata_rna.var['rna_idx'])
            
            # Create cis links dictionary (one entry per link)
            cis_links_dic = {}
            for idx, row in cis_links_df.iterrows():
                cis_links_dic[idx] = np.array([[row['atac_idx'], row['rna_idx']]])
            
            # Create control links dictionary (each cis-link has n_ctrl control peaks paired with same gene)
            ctrl_links_dic = {}
            for idx, row in cis_links_df.iterrows():
                atac_idx = row['atac_idx']
                rna_idx = row['rna_idx']
                ctrl_peak_indices = ctrl_peaks[atac_idx, :n_ctrl]
                ctrl_links_dic[idx] = np.column_stack([ctrl_peak_indices, np.full(n_ctrl, rna_idx)])
            
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
                    scheduler="processes",
                    n_workers=n_cores,
                )
            else:
                cis_coeff_dic = ctar.parallel.multiprocess_poisson_irls(
                    links_dict=cis_links_dic,
                    atac_sparse=atac_sparse,
                    rna_sparse=rna_sparse,
                    batch_size=BATCH_SIZE,
                    scheduler="processes",
                    n_workers=n_cores,
                    flag_se=True,
                )
            print('# Cis-links IRLS time = %0.2fs' % (time.time() - start_time))

            print('# Starting control links IRLS...')
            start_time = time.time()

            if COVAR_FILE:
                ctrl_coeff_dic = ctar.parallel.multiprocess_poisson_irls_multivar(
                    links_dict=ctrl_links_dic,
                    atac_sparse=atac_sparse,
                    rna_sparse=rna_sparse,
                    covar_mat=covar_mat,
                    batch_size=BATCH_SIZE,
                    scheduler="processes",
                    n_workers=n_cores,
                )
            else:
                ctrl_coeff_dic = ctar.parallel.multiprocess_poisson_irls(
                    links_dict=ctrl_links_dic,
                    atac_sparse=atac_sparse,
                    rna_sparse=rna_sparse,
                    batch_size=BATCH_SIZE,
                    scheduler="processes",
                    n_workers=n_cores,
                    flag_se=True,
                )
            print('# Control links IRLS time = %0.2fs' % (time.time() - start_time))
            
            # Extract coefficients from dictionaries
            cis_coeff = np.array([cis_coeff_dic[idx][0] for idx in sorted(cis_coeff_dic.keys())])
            ctrl_coeff = np.array([ctrl_coeff_dic[idx] for idx in sorted(ctrl_coeff_dic.keys())])
            
            # Compute p-values
            if cis_coeff.ndim == 2 and cis_coeff.shape[1] == 2:
                # We have [SE, coeff]
                cis_links_df[f'{BIN_CONFIG}_mcpval'] = ctar.method.initial_mcpval(ctrl_coeff[: , : , 1], cis_coeff[:, 1])
                cis_links_df[f'{BIN_CONFIG}_ppval'] = ctar.method.pooled_mcpval(ctrl_coeff[:, :, 1], cis_coeff[: , 1])
                cis_links_df['poissonb'] = cis_coeff[:, 1]
                cis_links_df['poissonb_se'] = cis_coeff[: , 0]
            else:
                cis_links_df[f'{BIN_CONFIG}_mcpval'] = ctar.method.initial_mcpval(ctrl_coeff, cis_coeff)
                cis_links_df[f'{BIN_CONFIG}_ppval'] = ctar.method.pooled_mcpval(ctrl_coeff, cis_coeff)
                cis_links_df['poissonb'] = cis_coeff

            results_folder = f'{TARGET_PATH}/{JOB_ID}_results'
            print(f'# Saving files to {results_folder}')
            os.makedirs(results_folder, exist_ok=True)
            
            np.save(f'{results_folder}/ctrl_peaks.npy', ctrl_peaks)
            np.save(f'{results_folder}/cis_coeff.npy', cis_coeff)
            np.save(f'{results_folder}/ctrl_coeff.npy', ctrl_coeff)
            cis_links_df.to_csv(f'{results_folder}/cis_links_df.csv')
            
        else:
            # Standard binned mode
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
                combined_bin_col=f'combined_bin_{BIN_CONFIG.rsplit(".",1)[0]}', 
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
                    scheduler="processes",
                    n_workers=n_cores,
                )
            else:
                cis_coeff_dic = ctar.parallel.multiprocess_poisson_irls_chunked(
                    links_dict=cis_links_dic,
                    atac_sparse=atac_sparse,
                    rna_sparse=rna_sparse,
                    batch_size=BATCH_SIZE,
                    scheduler="processes",
                    n_workers=n_cores,
                )
            print('# Cis-links IRLS time = %0.2fs' % (time.time() - start_time))

            print('# Starting control links IRLS...')
            start_time = time.time()

            if COVAR_FILE:
                ctrl_coeff_dic = ctar.parallel.multiprocess_poisson_irls_multivar(
                    links_dict=ctrl_links_dic,
                    atac_sparse=atac_sparse,
                    rna_sparse=rna_sparse,
                    covar_mat=covar_mat,
                    batch_size=BATCH_SIZE,
                    scheduler="processes",
                    n_workers=n_cores,
                )
            else:
                ctrl_coeff_dic = ctar.parallel.multiprocess_poisson_irls_chunked(
                    links_dict=ctrl_links_dic,
                    atac_sparse=atac_sparse,
                    rna_sparse=rna_sparse,
                    batch_size=BATCH_SIZE,
                    scheduler="processes",
                    n_workers=n_cores,
                )
            print('# Control links IRLS time = %0.2fs' % (time.time() - start_time))

            mcpval_dic, ppval_dic = ctar.method.binned_mcpval(cis_coeff_dic, ctrl_coeff_dic)

            cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, mcpval_dic, col_name=f'{BIN_CONFIG}_mcpval')
            cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, ppval_dic, col_name=f'{BIN_CONFIG}_ppval')
            cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, cis_coeff_dic, col_name='poissonb')

            results_folder = f'{TARGET_PATH}/{JOB_ID}_results'
            print(f'# Saving files to {results_folder}')
            os.makedirs(results_folder, exist_ok=True)
            
            with open(f'{results_folder}/cis_coeff_dic.pkl', 'wb') as f:
                pickle.dump(cis_coeff_dic, f)
            with open(f'{results_folder}/cis_idx_dic.pkl', 'wb') as f:
                pickle.dump(cis_idx_dic, f)
            with open(f'{results_folder}/ctrl_coeff_dic.pkl', 'wb') as f:
                pickle.dump(ctrl_coeff_dic, f)
            with open(f'{results_folder}/ctrl_links_dic.pkl', 'wb') as f:
                pickle.dump(ctrl_links_dic, f)

            cis_links_df.to_csv(f'{results_folder}/cis_links_df.csv')


    if JOB == "generate_links":

        print("# Running --job generate_links")

        adata_rna.var = ctar.data_loader.get_gene_coords(adata_rna.var)
        cis_links_df = ctar.data_loader.peak_to_gene(adata_atac.var, adata_rna.var, split_peaks=True)

        if USE_INF_MODE:
            print("# Using inf.inf mode - creating control peaks only")
            ctrl_peaks = ctar.method.create_ctrl_peaks(
                adata_atac, type=atac_type, num_bins=atac_bins, b=n_ctrl,
                peak_col='peak', layer='counts', genome_file=GENOME_FILE
            )
            
            results_folder = f'{TARGET_PATH}/{JOB_ID}_results'
            print(f'# Saving files to {results_folder}')
            os.makedirs(results_folder, exist_ok=True)
            
            np.save(f'{results_folder}/ctrl_peaks.npy', ctrl_peaks)
            cis_links_df.to_csv(f'{results_folder}/cis_links_df.csv')
            
        else:
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
                combined_bin_col=f'combined_bin_{BIN_CONFIG.rsplit(".",1)[0]}', 
                return_dic=True
            )

            results_folder = f'{TARGET_PATH}/{JOB_ID}_results'
            print(f'# Saving files to {results_folder}')
            os.makedirs(results_folder, exist_ok=True)

            with open(f'{results_folder}/cis_idx_dic.pkl', 'wb') as f:
                pickle.dump(cis_idx_dic, f)
            with open(f'{results_folder}/ctrl_links_dic.pkl', 'wb') as f:
                pickle.dump(ctrl_links_dic, f)
            cis_links_df.to_csv(f'{results_folder}/cis_links_df.csv')


    if JOB == "generate_controls":

        print("# Running --job generate_controls")

        if USE_INF_MODE: 
            print("# Using inf.inf mode - creating control peaks only")
            ctrl_peaks = ctar.method.create_ctrl_peaks(
                adata_atac, type=atac_type, num_bins=atac_bins, b=n_ctrl,
                peak_col='peak', layer='counts', genome_file=GENOME_FILE
            )
            
            results_folder = f'{TARGET_PATH}/{JOB_ID}_results'
            print(f'# Saving files to {results_folder}')
            os.makedirs(results_folder, exist_ok=True)
            
            np.save(f'{results_folder}/ctrl_peaks.npy', ctrl_peaks)
            
        else:
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
                combined_bin_col=f'combined_bin_{BIN_CONFIG.rsplit(".",1)[0]}', 
                return_dic=True
            )

            results_folder = f'{TARGET_PATH}/{JOB_ID}_results'
            print(f'# Saving files to {results_folder}')
            os.makedirs(results_folder, exist_ok=True)

            with open(f'{results_folder}/ctrl_links_dic.pkl', 'wb') as f:
                pickle.dump(ctrl_links_dic, f)
                
        cis_links_df.to_csv(f'{results_folder}/cis_links_df.csv')


    if JOB == "compute_cis_only":

        print("# Running --job compute_cis_only")

        if not RESULTS_PATH: 

            adata_rna.var = ctar.data_loader.get_gene_coords(adata_rna.var)
            cis_links_df = ctar.data_loader.peak_to_gene(adata_atac.var, adata_rna.var, split_peaks=True)

            if not USE_INF_MODE: 
                ctrl_links_dic, atac_bins_df, rna_bins_df = ctar.method.create_ctrl_pairs(
                    adata_atac, adata_rna, atac_bins=[n_atac_mean,n_atac_gc], rna_bins=[n_rna_mean,n_rna_var],
                    atac_type=atac_type, rna_type=rna_type, b=n_ctrl,
                    atac_layer='counts', rna_layer='counts', genome_file=GENOME_FILE
                )

                cis_links_df = ctar.data_loader.combine_peak_gene_bins(cis_links_df, atac_bins_df, rna_bins_df, 
                    atac_bins=[n_atac_mean,n_atac_gc], 
                    rna_bins=[n_rna_mean,n_rna_var]
                )

        file_suffix = ''

        if USE_INF_MODE:

            # For inf.inf mode, compute all cis links
            adata_atac.var['atac_idx'] = range(len(adata_atac.var))
            adata_rna.var['rna_idx'] = range(len(adata_rna.var))
            cis_links_df['atac_idx'] = cis_links_df['peak'].map(adata_atac.var['atac_idx'])
            cis_links_df['rna_idx'] = cis_links_df['gene'].map(adata_rna.var['rna_idx'])
            
            # Create cis links dictionary
            cis_links_dic = {}
            for idx, row in cis_links_df.iterrows():
                cis_links_dic[idx] = np.array([[row['atac_idx'], row['rna_idx']]])
            
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
                    scheduler="processes",
                    n_workers=n_cores,
                )
            else:
                cis_coeff_dic = ctar.parallel.multiprocess_poisson_irls(
                    links_dict=cis_links_dic,
                    atac_sparse=atac_sparse,
                    rna_sparse=rna_sparse,
                    batch_size=BATCH_SIZE,
                    scheduler="processes",
                    n_workers=n_cores,
                    flag_se=True,
                )
            print('# Cis-links IRLS time = %0.2fs' % (time.time() - start_time))
            
            # Convert dictionary to array
            cis_coeff = np.array([cis_coeff_dic[idx][0] for idx in sorted(cis_coeff_dic.keys())])
            
        else:
            
            # Standard binned mode
            cis_links_dic, cis_idx_dic = ctar.data_loader.groupby_combined_bins(cis_links_df, 
                combined_bin_col=f'combined_bin_{BIN_CONFIG.rsplit(".",1)[0]}', 
                return_dic=True
            )

            if ARRAY_IDX is not None: 

                with open(f'{TARGET_PATH}/cis_links_dic.pkl', 'rb') as file:
                    cis_links_dic = pickle.load(file)
                with open(f'{TARGET_PATH}/cis_idx_dic.pkl', 'rb') as file:
                    cis_idx_dic = pickle.load(file)

                cis_links_dic = dict(sorted(cis_links_dic.items()))
                cis_links_dic = dict(list(cis_links_dic.items())[ARRAY_IDX*BATCH_SIZE :  (ARRAY_IDX+1)*BATCH_SIZE])
                cis_idx_dic  = {key :  cis_idx_dic[key]  for key in cis_links_dic.keys()}
                print('# Running %d controls...' % (len(list(cis_links_dic.keys()))))

                file_suffix = f'_{ARRAY_IDX}'

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
                    scheduler="processes",
                    n_workers=n_cores,
                )
            else:
                cis_coeff_dic = ctar.parallel.multiprocess_poisson_irls(
                    links_dict=cis_links_dic,
                    atac_sparse=atac_sparse,
                    rna_sparse=rna_sparse,
                    batch_size=BATCH_SIZE,
                    scheduler="processes",
                    n_workers=n_cores,
                    flag_se=True,
                )
            print('# Cis-links IRLS time = %0.2fs' % (time.time() - start_time))

        results_folder = f'{TARGET_PATH}/{JOB_ID}_results'
        print(f'# Saving files to {results_folder}')
        os.makedirs(results_folder, exist_ok=True)

        if USE_INF_MODE: 
            np.save(f'{results_folder}/cis_coeff{file_suffix}.npy', cis_coeff)
        else:
            with open(f'{results_folder}/cis_coeff_dic{file_suffix}.pkl', 'wb') as f:
                pickle.dump(cis_coeff_dic, f)
            with open(f'{results_folder}/cis_idx_dic{file_suffix}.pkl', 'wb') as f:
                pickle.dump(cis_idx_dic, f)
            with open(f'{results_folder}/cis_links_dic{file_suffix}.pkl', 'wb') as f:
                pickle.dump(cis_links_dic, f)
                
        if not RESULTS_PATH:
            cis_links_df.to_csv(f'{results_folder}/cis_links_df.csv')


    if JOB == "compute_ctrl_only":

        print("# Running --job compute_ctrl_only")

        if USE_INF_MODE:

            # Load or create control peaks
            if RESULTS_PATH:
                ctrl_peaks_file = os.path.join(RESULTS_PATH, 'ctrl_peaks.npy')
                if os.path.exists(ctrl_peaks_file):
                    ctrl_peaks = np.load(ctrl_peaks_file)
                else:
                    ctrl_peaks = ctar.method.create_ctrl_peaks(
                        adata_atac, type=atac_type, num_bins=atac_bins, b=n_ctrl,
                        peak_col='peak', layer='counts', genome_file=GENOME_FILE
                    )
            else:
                ctrl_peaks = ctar.method.create_ctrl_peaks(
                    adata_atac, type=atac_type, num_bins=atac_bins, b=n_ctrl,
                    peak_col='peak', layer='counts', genome_file=GENOME_FILE
                )
            
            # Get cis links indices
            adata_atac.var['atac_idx'] = range(len(adata_atac.var))
            adata_rna.var['rna_idx'] = range(len(adata_rna.var))
            cis_links_df['atac_idx'] = cis_links_df['peak'].map(adata_atac.var['atac_idx'])
            cis_links_df['rna_idx'] = cis_links_df['gene'].map(adata_rna.var['rna_idx'])
            
            # Determine batch range
            if ARRAY_IDX is not None: 
                start_idx = ARRAY_IDX * BATCH_SIZE
                end_idx = min((ARRAY_IDX + 1) * BATCH_SIZE, len(cis_links_df))
                cis_links_subset = cis_links_df.iloc[start_idx:end_idx]
                file_suffix = f'_{ARRAY_IDX}'
            else: 
                cis_links_subset = cis_links_df
                file_suffix = ''
            
            print(f'# Computing controls for {len(cis_links_subset)} links...')
            
            # Create control links dictionary
            ctrl_links_dic = {}
            for idx, row in cis_links_subset.iterrows():
                atac_idx = row['atac_idx']
                rna_idx = row['rna_idx']
                ctrl_peak_indices = ctrl_peaks[atac_idx, : n_ctrl]
                ctrl_links_dic[idx] = np.column_stack([ctrl_peak_indices, np.full(n_ctrl, rna_idx)])
            
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
                    batch_size=(BATCH_SIZE//10)+1,
                    scheduler="processes",
                    n_workers=n_cores,
                )
            else:
                ctrl_coeff_dic = ctar.parallel.multiprocess_poisson_irls(
                    links_dict=ctrl_links_dic,
                    atac_sparse=atac_sparse,
                    rna_sparse=rna_sparse,
                    batch_size=(BATCH_SIZE//10)+1,
                    scheduler="processes",
                    n_workers=n_cores,
                    flag_se=True,
                )
            print('# Control links IRLS time = %0.2fs' % (time.time() - start_time))
            
            # Convert dictionary to array
            ctrl_coeff = np.array([ctrl_coeff_dic[idx] for idx in sorted(ctrl_coeff_dic.keys())])
            
        else:
            
            # Standard binned mode
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
                    combined_bin_col=f'combined_bin_{BIN_CONFIG.rsplit(".",1)[0]}', 
                    return_dic=True
                )

                file_suffix = ''

            else:

                file_path = f'{RESULTS_PATH}/ctrl_links_dic.pkl'
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
                    batch_size=(BATCH_SIZE//10)+1,
                    n_workers=n_cores,
                    scheduler="processes",                
                )
            else:
                ctrl_coeff_dic = ctar.parallel.multiprocess_poisson_irls(
                    links_dict=ctrl_links_dic,
                    atac_sparse=atac_sparse,
                    rna_sparse=rna_sparse,
                    batch_size=(BATCH_SIZE//10)+1,
                    scheduler="processes",
                    n_workers=n_cores,
                    flag_se=True,
                )
            print('# Control links IRLS time = %0.2fs' % (time.time() - start_time))

        if RESULTS_PATH: 
            results_folder = RESULTS_PATH
        elif ARRAY_IDX is None:
            results_folder = f'{TARGET_PATH}/{JOB_ID}_results'
        else:
            results_folder = f'{TARGET_PATH}'

        print(f'# Saving files to {results_folder}')
        os.makedirs(results_folder, exist_ok=True)

        if USE_INF_MODE:
            np.save(f'{results_folder}/ctrl_coeff{file_suffix}.npy', ctrl_coeff)
        else:
            with open(f'{results_folder}/ctrl_coeff_dic{file_suffix}.pkl', 'wb') as f:
                pickle.dump(ctrl_coeff_dic, f)

            if ARRAY_IDX is None:
                with open(f'{results_folder}ctrl_links_dic.pkl', 'wb') as f:
                    pickle.dump(ctrl_links_dic, f)
                cis_links_df.to_csv(f'{results_folder}cis_links_df.csv')


    if JOB == "compute_pval":
        
        print("# Running --job compute_pval")
        
        MAX_VAL = 100  # To clip studentized values
        
        if USE_INF_MODE:
            print("# Computing p-values for inf.inf mode")
            
            # Check if we have SE (3D array with shape (n_links, n_ctrl, 2))
            if ctrl_coeff.ndim == 3 and ctrl_coeff.shape[2] == 2:
                print("# Found SE in control coefficients - computing both regular and studentized p-values")
                
                ctrl_se = ctrl_coeff[:, :, 0]
                ctrl_beta = ctrl_coeff[:, : , 1]
                
                # Regular p-values
                cis_links_df[f'{BIN_CONFIG}_mcpval'] = ctar.method.initial_mcpval(ctrl_beta, cis_coeff[: , 1])
                cis_links_df[f'{BIN_CONFIG}_ppval'] = ctar.method.pooled_mcpval(ctrl_beta, cis_coeff[:, 1])
                
                # Studentized p-values
                cis_studentized = np.clip(cis_coeff[:, 1] / cis_coeff[:, 0], -MAX_VAL, MAX_VAL)
                ctrl_studentized = np.clip(ctrl_beta / ctrl_se, -MAX_VAL, MAX_VAL)
                
                cis_links_df[f'{BIN_CONFIG}_mcpval_z'] = ctar.method.initial_mcpval(ctrl_studentized, cis_studentized)
                cis_links_df[f'{BIN_CONFIG}_ppval_z'] = ctar.method.pooled_mcpval(ctrl_studentized, cis_studentized)
                
                cis_links_df['poissonb'] = cis_coeff[:, 1]
                cis_links_df['poissonb_se'] = cis_coeff[:, 0]
                
            elif ctrl_coeff.ndim == 2: 
                print("# No SE found - computing only regular p-values")
                
                if cis_coeff.ndim == 1:
                    cis_beta = cis_coeff
                elif cis_coeff.ndim == 2 and cis_coeff.shape[1] == 2:
                    cis_beta = cis_coeff[:, 1]
                    cis_links_df['poissonb'] = cis_coeff[:, 1]
                    cis_links_df['poissonb_se'] = cis_coeff[:, 0]
                else:
                    cis_beta = cis_coeff
                
                cis_links_df[f'{BIN_CONFIG}_mcpval'] = ctar.method.initial_mcpval(ctrl_coeff, cis_beta)
                cis_links_df[f'{BIN_CONFIG}_ppval'] = ctar.method.pooled_mcpval(ctrl_coeff, cis_beta)
                
                if 'poissonb' not in cis_links_df.columns:
                    cis_links_df['poissonb'] = cis_beta
            
            else:
                raise ValueError(f"Unexpected ctrl_coeff shape: {ctrl_coeff.shape}")
        
        else:
            print("# Computing p-values for binned mode")
            
            # Determine combined bin column name
            bin_config_only = BIN_CONFIG.rsplit('.', 1)[0]
            combined_bin_col = f'combined_bin_{bin_config_only}'
            
            if combined_bin_col not in cis_links_df.columns:
                raise ValueError(f"Column {combined_bin_col} not found in cis_links_df")
            
            # Check if we have SE (2D arrays in dictionary)
            first_key = list(ctrl_coeff_dic.keys())[0]
            if ctrl_coeff_dic[first_key].ndim == 2 and ctrl_coeff_dic[first_key].shape[1] == 2:
                print("# Found SE in coefficients - computing both regular and studentized p-values")
                
                # Regular p-values
                mcpval_dic, ppval_dic = ctar.method.binned_mcpval(
                    {key: value[: , 1] for key, value in cis_coeff_dic.items()},
                    {key: value[: , 1] for key, value in ctrl_coeff_dic.items()},
                    b=n_ctrl,
                    flag_corrected=False
                )
                
                cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, mcpval_dic, col_name=f'{BIN_CONFIG}_mcpval')
                cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, ppval_dic, col_name=f'{BIN_CONFIG}_ppval')
                
                # Studentized p-values
                cis_coeff_dic_z = {}
                ctrl_coeff_dic_z = {}
                
                for key in cis_coeff_dic.keys():
                    cis_coeff_dic_z[key] = np.clip(
                        cis_coeff_dic[key][:, 1] / cis_coeff_dic[key][:, 0],
                        -MAX_VAL, MAX_VAL
                    )
                    ctrl_coeff_dic_z[key] = np.clip(
                        ctrl_coeff_dic[key][:, 1] / ctrl_coeff_dic[key][:, 0],
                        -MAX_VAL, MAX_VAL
                    )
                
                mcpval_dic_z, ppval_dic_z = ctar.method.binned_mcpval(
                    cis_coeff_dic_z,
                    ctrl_coeff_dic_z,
                    b=n_ctrl,
                    flag_corrected=False
                )
                
                cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, mcpval_dic_z, col_name=f'{BIN_CONFIG}_mcpval_z')
                cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, ppval_dic_z, col_name=f'{BIN_CONFIG}_ppval_z')
                
            else:
                print("# No SE found - computing only regular p-values")
                
                # Regular p-values
                mcpval_dic, ppval_dic = ctar.method.binned_mcpval(
                    cis_coeff_dic,
                    ctrl_coeff_dic,
                    b=n_ctrl,
                    flag_corrected=False
                )
                
                cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, mcpval_dic, col_name=f'{BIN_CONFIG}_mcpval')
                cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, ppval_dic, col_name=f'{BIN_CONFIG}_ppval')
        
        # Save updated dataframe
        output_file = os.path.join(RESULTS_PATH, 'cis_links_df.csv')
        print(f"# Saving updated cis_links_df to {output_file}")
        cis_links_df.to_csv(output_file)
        print(f"# Added columns: {[col for col in cis_links_df.columns if BIN_CONFIG in col]}")


    ###########################################################################################
    ######                                  Timing Summary                               ######
    ###########################################################################################
    
    sys_end_time = time.time()
    total_elapsed = sys_end_time - sys_start_time
    
    print("\n" + "="*80)
    print(f"# Job:  {JOB} completed successfully")
    print(f"# Total elapsed time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
    print(f"# Started at:    {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(sys_start_time))}")
    print(f"# Finished at:  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(sys_end_time))}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ctar")

    parser.add_argument("--job", type=str, required=True, help="compute_ctar, compute_cis_only, compute_ctrl_only, generate_links, generate_controls, compute_pval")
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
        help="{# ATAC mean bins}.{# ATAC GC bins}.{# RNA mean bins}.{# RNA var bins}.{# Sampled controls per bin}. Use 'inf.inf' for RNA bins to skip RNA binning",
    )
    parser.add_argument("--binning_type", type=str, required=False, default='mean_var', help='mean_var, cholesky')
    parser.add_argument("--pybedtools_path", type=str, required=False, default=None)
    parser.add_argument("--array_idx", type=str, required=False, default=None)

    args = parser.parse_args()
    main(args)