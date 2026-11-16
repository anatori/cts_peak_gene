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
import scanpy as sc




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

compute_spec : compute link specificity (cis and control links) and its monte carlo p-values from existing links/controls
    - Input : --job | --multiome_file | --results_path | --celltype_col | --celltype_file | --spec_stat | --spec_norm | --spec_filter_fdr | --spec_filter_qcol | --binning_config | --batch_size | --n_cores
    --spec_stat coexp_log1p (default) | --spec_stat joint
    - Output : cis_{tag}.npy, ctrl_{tag}.npy (inf.inf mode) or cis_{tag}_dic.pkl, ctrl_{tag}_dic.pkl (binned mode), with tag = spec_coexp
      (coexp_log1p) or spec[_mean] (joint); updated cis_links_df.csv with link_specificity_coexp | link_specificity_joint, mcpval/ppval/zpval_{tag},
      link_dominant_subtype_coexp (coexp_log1p), and {BIN_CONFIG}_qval_{tag}_filt = BH over mcpval_{tag} among links with association
      q < --spec_filter_fdr, NaN elsewhere

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
    DIC_BATCH_SIZE = int(args.dic_batch_size)
    TARGET_PATH = args.target_path
    RESULTS_PATH = args.results_path
    GENOME_FILE = args.genome_file
    BIN_CONFIG = args.binning_config
    BIN_TYPE = args.binning_type
    PYBEDTOOLS_PATH = args.pybedtools_path
    COVAR_FILE = args.covar_file
    ARRAY_IDX = args.array_idx
    FLAG_SE = (args.flag_se == 'True')
    FLAG_LL = (args.flag_ll == 'True')
    FLAG_CORR = (getattr(args, 'flag_corr', 'False') == 'True')
    FLAG_SPEC = (getattr(args, 'flag_spec', 'False') == 'True')
    CELLTYPE_COL = getattr(args, 'celltype_col', None) or None
    CELLTYPE_FILE = getattr(args, 'celltype_file', None) or None
    SPEC_NORM = getattr(args, 'spec_norm', None) or 'pseudobulk'
    SPEC_SUFFIX = '' if SPEC_NORM == 'pseudobulk' else f'_{SPEC_NORM}'  # 'mean' columns keep a _mean suffix
    SPEC_STAT = getattr(args, 'spec_stat', None) or 'coexp_log1p'
    SPEC_FILTER_FDR = float(getattr(args, 'spec_filter_fdr', None) or 0.2)
    SPEC_FILTER_QCOL = getattr(args, 'spec_filter_qcol', None) or None
    SPEC_TAG = f'spec{SPEC_SUFFIX}' if SPEC_STAT == 'joint' else 'spec_coexp'
    LINK_SPEC_COL = f'link_specificity_joint{SPEC_SUFFIX}' if SPEC_STAT == 'joint' else 'link_specificity_coexp'
    N_CORES = int(args.n_cores) if args.n_cores is not None else None
    RANDOM_SEED = int(args.random_seed) if args.random_seed is not None else None
    DONOR_COL = args.donor_col or None
    N_LENGTH_BINS = int(args.n_length_bins or '1')
    FDR_BIN_TYPE = getattr(args, 'fdr_bin_type', None) or None
    TSS_FILE = getattr(args, 'tss_file', None) or None
    RNA_LAYER = getattr(args, 'rna_layer', None) or None
    N_FDR_RNA_BINS = int(getattr(args, 'n_fdr_rna_bins', None) or '4')
    FDR_MIN_BIN_COUNT = int(getattr(args, 'fdr_min_bin_count', None) or '25000')
    FDR_DIST_BINS = (
        [float(edge) for edge in args.fdr_dist_bins.split(',')]
        if getattr(args, 'fdr_dist_bins', None) else None
    )
    FLAG_TFIDF = (args.flag_tfidf == 'True')
    FLAG_BIN = (args.flag_bin == 'True')
    assert not (FLAG_TFIDF and FLAG_BIN), "--flag_tfidf and --flag_bin are mutually exclusive ATAC transforms"

    # Parse and check arguments
    LEGAL_JOB_LIST = [
        "compute_ctar",
        "compute_cis_only",
        "compute_ctrl_only",
        "generate_links",
        "generate_controls",
        "compute_pval",
        "compute_spec",
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

    if JOB in ["compute_ctrl_only","generate_controls","compute_pval","compute_spec"]:
        assert RESULTS_PATH is not None, "--results_path required for --job=%s" % JOB

    if JOB == "compute_spec":
        assert MULTIOME_FILE is not None, "--multiome_file required for --job=%s" % JOB

    if FLAG_SPEC or JOB == "compute_spec":
        assert CELLTYPE_COL is not None, "--celltype_col required for --flag_spec / --job=compute_spec"
        assert SPEC_NORM in ("pseudobulk", "mean"), "--spec_norm must be 'pseudobulk' or 'mean'"
        assert SPEC_STAT in ("coexp_log1p", "joint"), "--spec_stat must be 'coexp_log1p' or 'joint'"

    # Print input options
    header = ctar.util.get_cli_head()
    header += "Call: CLI_ctar.py \\\n"
    header += "--job %s\\\n" % JOB
    header += "--job_id %s\\\n" % JOB_ID
    header += "--multiome_file %s\\\n" % MULTIOME_FILE
    header += "--batch_size %s\\\n" % BATCH_SIZE
    header += "--dic_batch_size %s\\\n" % DIC_BATCH_SIZE
    header += "--target_path %s\\\n" % TARGET_PATH
    header += "--results_path %s\\\n" % RESULTS_PATH
    header += "--genome_file %s\\\n" % GENOME_FILE
    header += "--binning_config %s\\\n" % BIN_CONFIG
    header += "--binning_type %s\\\n" % BIN_TYPE
    header += "--pybedtools_path %s\\\n" % PYBEDTOOLS_PATH
    header += "--covar_file %s\\\n" % COVAR_FILE
    header += "--array_idx %s\\\n" % ARRAY_IDX
    header += "--flag_se %s\\\n" % FLAG_SE
    header += "--flag_ll %s\\\n" % FLAG_LL
    header += "--flag_corr %s\\\n" % FLAG_CORR
    header += "--flag_spec %s\\\n" % FLAG_SPEC
    header += "--celltype_col %s\\\n" % CELLTYPE_COL
    header += "--celltype_file %s\\\n" % CELLTYPE_FILE
    header += "--spec_norm %s\\\n" % SPEC_NORM
    header += "--spec_stat %s\\\n" % SPEC_STAT
    header += "--spec_filter_fdr %s\\\n" % SPEC_FILTER_FDR
    header += "--spec_filter_qcol %s\\\n" % SPEC_FILTER_QCOL
    header += "--flag_tfidf %s\\\n" % FLAG_TFIDF
    header += "--flag_bin %s\\\n" % FLAG_BIN
    header += "--random_seed %s\\\n" % RANDOM_SEED
    header += "--donor_col %s\\\n" % DONOR_COL
    header += "--n_length_bins %s\\\n" % N_LENGTH_BINS
    header += "--fdr_bin_type %s\\\n" % FDR_BIN_TYPE
    header += "--tss_file %s\\\n" % TSS_FILE
    header += "--rna_layer %s\\\n" % RNA_LAYER
    header += "--n_fdr_rna_bins %s\\\n" % N_FDR_RNA_BINS
    header += "--fdr_min_bin_count %s\\\n" % FDR_MIN_BIN_COUNT
    header += "--fdr_dist_bins %s\\\n" % FDR_DIST_BINS
    print(header)

    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)
        print(f"# Setting random seed to {RANDOM_SEED}")

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
        "compute_spec",
    ]:

        # Setting bin config
        # Order:  MEAN.GC.MEAN.VAR
        n_atac_mean, n_atac_gc, n_rna_mean, n_rna_var, n_ctrl = BIN_CONFIG.split('.')
        n_atac_mean, n_atac_gc, n_ctrl = map(int, [n_atac_mean, n_atac_gc, n_ctrl])

        # Check if using inf.inf mode (no RNA binning)
        USE_INF_MODE = (f'{n_rna_mean}.{n_rna_var}' == 'inf.inf')
        if not USE_INF_MODE: 
            n_rna_mean, n_rna_var = map(int, [n_rna_mean, n_rna_var])
        print(f'# Setting n_atac_mean={n_atac_mean}, n_atac_gc={n_atac_gc}, n_rna_mean={n_rna_mean}, n_rna_var={n_rna_var}, n_length_bins={N_LENGTH_BINS}, n_ctrl={n_ctrl}')

    # compute_pval needs the multiome when it has to recompute RNA moments
    if JOB in [
        "compute_ctar",
        "compute_cis_only",
        "compute_ctrl_only",
        "generate_links",
        "generate_controls",
        "compute_spec",
    ] or (JOB == "compute_pval" and MULTIOME_FILE is not None): 

        print("# Loading --multiome_file")
        if MULTIOME_FILE.endswith('.h5mu'):
            adata_rna = mu.read(MULTIOME_FILE)
            adata_atac = adata_rna.mod['atac']
            adata_rna = adata_rna.mod['rna']
        else:
            adata_rna = ad.read_h5ad(MULTIOME_FILE)
            adata_atac = adata_rna[:,adata_rna.var.feature_types == 'ATAC'].copy()
            adata_rna = adata_rna[:,adata_rna.var.feature_types == 'GEX'].copy()

        print("# RNA file is shape",adata_rna.shape)
        print("# ATAC file is shape",adata_atac.shape)

        if CELLTYPE_FILE is not None:
            assert CELLTYPE_COL is not None, "--celltype_col must name the label column in --celltype_file"
            n_labeled = ctar.data_loader.attach_celltype_labels(adata_rna, CELLTYPE_FILE, CELLTYPE_COL)
            print(f"# Read '{CELLTYPE_COL}' from --celltype_file for {n_labeled} of {adata_rna.n_obs} cells")

        # Convert peak to chr#:###-### format
        if adata_atac.var_names.str.match(r"^[^-]+-\d+-\d+$").any():
            adata_atac.var_names = adata_atac.var_names.astype(str).str.replace(
                r"\b([^-]+)-(\d+)-(\d+)\b",
                r"\1:\2-\3",
                regex=True,
            )

        adata_rna.var.index.name = ''
        adata_atac.var.index.name = ''

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

        # Preferentially use ENSEMBL ID
        if not adata_rna.var.gene.str.startswith('ENSG').all():
            if 'gene_id' in adata_rna.var.columns:
                adata_rna.var['gene'] = adata_rna.var.gene_id
                adata_rna.var.index = adata_rna.var.gene_id
                print("# Using gene_id columns for RNA genes...")
            elif 'gene_ids' in adata_rna.var.columns:
                adata_rna.var['gene'] = adata_rna.var.gene_ids
                adata_rna.var.index = adata_rna.var.gene_ids
                print("# Using gene_ids columns for RNA genes...")
            assert adata_rna.var.gene.str.startswith('ENSG').all(), "Must use ENSEMBL IDs for genes"
            
        # Setting atac bin type
        if BIN_TYPE == 'cholesky':
            atac_type = 'chol_logsum_gc'
            atac_bins = [n_atac_mean, n_atac_gc]
        elif BIN_TYPE in ('mean_fano_bio', 'mean_fano_cvbio'):
            atac_type = 'mean_fano_bio'
            atac_bins = [n_atac_mean, n_atac_gc]
        else:
            atac_type = 'mean_gc'
            atac_bins = [n_atac_mean, n_atac_gc]
        print(f'# Setting atac bin type to {atac_type}...')
        rna_type = 'mean_cv_bio' if BIN_TYPE in ('mean_cv_bio', 'mean_fano_cvbio') else 'mean_var'
        print(f'# Setting rna bin type to {rna_type}...')

        if PYBEDTOOLS_PATH:
            print("# Setting --pybedtools_path")
            pybedtools.helpers.set_bedtools_path(PYBEDTOOLS_PATH)

        if COVAR_FILE:
            print("# Loading --covar_file")
            covar_mat = np.load(COVAR_FILE)
        elif DONOR_COL and not FLAG_CORR:
            assert DONOR_COL in adata_rna.obs.columns, f"--donor_col '{DONOR_COL}' not found in obs"
            print(f"# Building one-hot donor covariate matrix from '{DONOR_COL}'...")
            covar_mat = pd.get_dummies(adata_rna.obs[DONOR_COL], drop_first=True).to_numpy(dtype=np.float32)
            print(f"# covar_mat shape: {covar_mat.shape} ({covar_mat.shape[1]} donor dummies)")
        else:
            covar_mat = None

        if ARRAY_IDX:
            ARRAY_IDX = int(ARRAY_IDX)

        if RESULTS_PATH: 
            results_folder = RESULTS_PATH
        elif ARRAY_IDX is None:
            results_folder = f'{TARGET_PATH}/{JOB_ID}_results'
        else:
            results_folder = f'{TARGET_PATH}'
        print(f"# Setting results_folder to {results_folder}")

    if FLAG_CORR and JOB in ["compute_ctar", "compute_cis_only", "compute_ctrl_only"]:
        print("# Normalizing for Pearson correlation (normalize_total + log1p)...")
        adata_rna_tmp = adata_rna.copy()
        adata_atac_tmp = adata_atac.copy()
        # Normalize from the raw counts layer into .X, which is what rna_norm/atac_norm read.
        # (sc.pp.normalize_total(..., layer='counts') writes only to the layer, never to .X.)
        adata_rna_tmp.X = adata_rna_tmp.layers['counts'].copy()
        sc.pp.normalize_total(adata_rna_tmp)
        sc.pp.log1p(adata_rna_tmp)
        if FLAG_TFIDF:
            print('# Using TFIDF for ATAC...')
            mu.atac.pp.tfidf(adata_atac_tmp, from_layer='counts')
        elif FLAG_BIN:
            print('# Using binarized ATAC...')
            adata_atac_tmp.X = (adata_atac_tmp.layers['counts'] > 0).astype(np.float32)
        else:
            adata_atac_tmp.X = adata_atac_tmp.layers['counts'].copy()
            sc.pp.normalize_total(adata_atac_tmp)
            sc.pp.log1p(adata_atac_tmp)
        if DONOR_COL is not None:
            assert DONOR_COL in adata_rna.obs.columns, f"--donor_col '{DONOR_COL}' not found in obs"
            print(f"# Donor column '{DONOR_COL}' set — using within-donor Pearson correlation (sparse, no centering).")
            donor_labels = adata_rna.obs[DONOR_COL].values
        else:
            donor_labels = None
        rna_norm = adata_rna_tmp.X
        atac_norm = adata_atac_tmp.X
        del adata_rna_tmp, adata_atac_tmp
        print(f"# rna_norm: {rna_norm.shape}, atac_norm: {atac_norm.shape}")

    if RESULTS_PATH and JOB in ["compute_cis_only", "compute_ctrl_only", "generate_controls", "compute_pval", "compute_spec"]: 

        print("# Loading --results_path")
        csv_file = os.path.join(RESULTS_PATH, 'cis_links_df.csv')
        if not os.path.isfile(csv_file):
            raise FileNotFoundError(f"Expected cis_links_df.csv at {csv_file}, but it was not found.")
        cis_links_df = pd.read_csv(csv_file, index_col=0)

        # Allow custom peak-gene tables to be supplied for jobs that load the multiome object.
        if JOB in ["compute_cis_only", "compute_ctrl_only", "generate_controls"]:
            if {'peak', 'gene'}.issubset(cis_links_df.columns):
                cis_links_df = ctar.data_loader.attach_link_indices(cis_links_df, adata_atac, adata_rna)

                if not USE_INF_MODE:
                    cis_links_df = ctar.data_loader.ensure_combined_bin_column(
                        cis_links_df,
                        adata_atac,
                        adata_rna,
                        BIN_CONFIG,
                        atac_type,
                        rna_type,
                        n_atac_mean,
                        n_atac_gc,
                        n_rna_mean,
                        n_rna_var,
                        GENOME_FILE,
                        n_length_bins=N_LENGTH_BINS,
                    )

    if JOB in ["compute_ctar", "compute_ctrl_only", "compute_cis_only", "compute_spec"]:

        if N_CORES:
            n_cores = N_CORES
        else:
            n_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
        print(f'# SLURM allocated {n_cores} CPUs for this job')

    if JOB in ["compute_pval"]:
        
        print("# Loading cis coefficients...")
        if FLAG_CORR:
            cis_coeff_file = os.path.join(RESULTS_PATH, 'cis_corr.npy' if USE_INF_MODE else 'cis_corr_dic.pkl')
        else:
            cis_coeff_file = os.path.join(RESULTS_PATH, 'cis_coeff.npy' if USE_INF_MODE else 'cis_coeff_dic.pkl')

        if USE_INF_MODE:
            cis_prefix = 'cis_corr_' if FLAG_CORR else 'cis_coeff_'
            cis_chunk_files = [
                f for f in os.listdir(RESULTS_PATH)
                if f.startswith(cis_prefix) and f.endswith('.npy')
            ]
            if cis_chunk_files:
                print(f"# Found {len(cis_chunk_files)} cis coefficient chunks — consolidating...")
                cis_coeff, cis_coeff_ranges = ctar.data_loader.consolidate_ranged_npy(
                    RESULTS_PATH, startswith=cis_prefix, allow_pickle=False
                )
                cis_id_chunks = [
                    f for f in os.listdir(RESULTS_PATH)
                    if f.startswith('cis_link_ids_') and f.endswith('.npy')
                ]
                if cis_id_chunks:
                    cis_ids_consolidated, _ = ctar.data_loader.consolidate_ranged_npy(
                        RESULTS_PATH, startswith='cis_link_ids_', allow_pickle=True
                    )
                    np.save(os.path.join(RESULTS_PATH, 'cis_link_ids.npy'), cis_ids_consolidated, allow_pickle=True)
                    for fname in os.listdir(RESULTS_PATH):
                        if fname.startswith('cis_link_ids_') and fname.endswith('.npy'):
                            os.remove(os.path.join(RESULTS_PATH, fname))
                    print("# Saved cis_link_ids.npy and deleted chunks")
                single_fname = 'cis_corr.npy' if FLAG_CORR else 'cis_coeff.npy'
                np.save(os.path.join(RESULTS_PATH, single_fname), cis_coeff)
                for fname in cis_chunk_files:
                    os.remove(os.path.join(RESULTS_PATH, fname))
                print(f"# Saved {single_fname} and deleted {len(cis_chunk_files)} chunk(s)")
            else:
                if not os.path.exists(cis_coeff_file):
                    raise FileNotFoundError(f"{os.path.basename(cis_coeff_file)} not found at {cis_coeff_file}")
                cis_coeff = np.load(cis_coeff_file)
            print(f"# Loaded cis coefficients with shape {cis_coeff.shape}")
        else:
            cis_prefix = 'cis_corr_dic_' if FLAG_CORR else 'cis_coeff_dic_'
            cis_coeff_chunks = [
                f for f in os.listdir(RESULTS_PATH)
                if f.startswith(cis_prefix) and f.endswith('.pkl')
            ]
            if cis_coeff_chunks:
                print(f"# Found {len(cis_coeff_chunks)} cis coefficient dict chunks — consolidating...")
                cis_coeff_dic = ctar.data_loader.consolidate_null_dic(
                    path=RESULTS_PATH, startswith=cis_prefix, remove_empty=True, print_missing=False
                )
                cis_idx_dic = {}
                for chunk_fname in sorted(f for f in os.listdir(RESULTS_PATH) if f.startswith('cis_idx_dic_') and f.endswith('.pkl')):
                    with open(os.path.join(RESULTS_PATH, chunk_fname), 'rb') as f:
                        cis_idx_dic.update(pickle.load(f))
                single_fname = 'cis_corr_dic.pkl' if FLAG_CORR else 'cis_coeff_dic.pkl'
                with open(os.path.join(RESULTS_PATH, single_fname), 'wb') as f:
                    pickle.dump(cis_coeff_dic, f)
                with open(os.path.join(RESULTS_PATH, 'cis_idx_dic.pkl'), 'wb') as f:
                    pickle.dump(cis_idx_dic, f)
                for chunk_fname in os.listdir(RESULTS_PATH):
                    if (chunk_fname.startswith(cis_prefix) or chunk_fname.startswith('cis_idx_dic_') or chunk_fname.startswith('cis_links_dic_')) and chunk_fname.endswith('.pkl'):
                        os.remove(os.path.join(RESULTS_PATH, chunk_fname))
                print(f"# Saved {single_fname}, cis_idx_dic.pkl and deleted chunks")
            else:
                if not os.path.exists(cis_coeff_file):
                    raise FileNotFoundError(f"{os.path.basename(cis_coeff_file)} not found at {cis_coeff_file}")
                with open(cis_coeff_file, 'rb') as f:
                    cis_coeff_dic = pickle.load(f)
            print(f"# Loaded cis coefficients with {len(cis_coeff_dic)} bins")

            cis_idx_file = os.path.join(RESULTS_PATH, 'cis_idx_dic.pkl')
            if not os.path.exists(cis_idx_file):
                raise FileNotFoundError(f"cis_idx_dic.pkl not found at {cis_idx_file}")
            with open(cis_idx_file, 'rb') as f:
                cis_idx_dic = pickle.load(f)
            print(f"# Loaded cis_idx_dic with {len(cis_idx_dic)} bins")

        print("# Loading control coefficients...")
        if FLAG_CORR:
            prefix = 'ctrl_corr_' if USE_INF_MODE else 'ctrl_corr_dic_'
        else:
            prefix = 'ctrl_coeff_' if USE_INF_MODE else 'ctrl_coeff_dic_'
        suffix = '.npy' if USE_INF_MODE else '.pkl'
        ctrl_coeff_files = [
            f for f in os.listdir(RESULTS_PATH)
            if f.startswith(prefix) and f.endswith(suffix)
        ]
        print(f"# Found {len(ctrl_coeff_files)} ctrl coefficient files")

        if FDR_BIN_TYPE is not None:
            assert FDR_BIN_TYPE in ('dist_rmean_rvar'), (
                f"--fdr_bin_type '{FDR_BIN_TYPE}' is not implemented; "
                "supported: 'dist_rmean_rvar'."
            )

        if USE_INF_MODE:
            if ctrl_coeff_files:
                print("# Consolidating array outputs...")
                ctrl_coeff, coeff_ranges = ctar.data_loader.consolidate_ranged_npy(
                    RESULTS_PATH, startswith=prefix, allow_pickle=False
                )

                # load ids if present
                cis_ids_path = os.path.join(RESULTS_PATH, 'cis_link_ids.npy')
                if os.path.exists(cis_ids_path):
                    cis_ids = np.load(cis_ids_path, allow_pickle=True)
                    ctrl_ids, id_ranges = ctar.data_loader.consolidate_ranged_npy(
                        RESULTS_PATH, startswith="ctrl_link_ids_", allow_pickle=True)

                    assert len(coeff_ranges) == len(id_ranges), "Different number of ctrl coeff vs id chunks"
                    assert [r[:2] for r in coeff_ranges] == [r[:2] for r in id_ranges], "Coeff/id ranges mismatch"
                    assert ctrl_coeff.shape[0] == ctrl_ids.shape[0], "ctrl coeff rows != ctrl id rows"
                    assert cis_coeff.shape[0] == cis_ids.shape[0], "cis coeff rows != cis id rows"
                    assert ctrl_ids.shape[0] == cis_ids.shape[0], "ctrl links != cis links"
                    assert np.array_equal(ctrl_ids, cis_ids), "CIS/CTRL link order mismatch"

                    # enforce df order to match arrays
                    cis_links_df = cis_links_df.loc[cis_ids]

                    # save consolidated ids and delete chunks
                    np.save(os.path.join(RESULTS_PATH, 'ctrl_link_ids.npy'), ctrl_ids, allow_pickle=True)
                    for fname in os.listdir(RESULTS_PATH):
                        if fname.startswith('ctrl_link_ids_') and fname.endswith('.npy'):
                            os.remove(os.path.join(RESULTS_PATH, fname))
                    print("# Saved ctrl_link_ids.npy and deleted chunks")

                else:
                    print("WARNING: link id files not found; alignment not guaranteed.")

                if not FLAG_CORR:
                    assert ctrl_coeff.ndim == 3 and ctrl_coeff.shape[1] == n_ctrl, f"ctrl_coeff shape unexpected: {ctrl_coeff.shape}"
                assert cis_coeff.shape[0] == ctrl_coeff.shape[0], "cis/ctrl link count mismatch"

                # save consolidated ctrl coeff and delete chunks
                single_fname = 'ctrl_corr.npy' if FLAG_CORR else 'ctrl_coeff.npy'
                np.save(os.path.join(RESULTS_PATH, single_fname), ctrl_coeff)
                for fname in ctrl_coeff_files:
                    os.remove(os.path.join(RESULTS_PATH, fname))
                print(f"# Saved {single_fname} and deleted {len(ctrl_coeff_files)} chunk(s)")

            else:
                single_fname = 'ctrl_corr.npy' if FLAG_CORR else 'ctrl_coeff.npy'
                ctrl_coeff_file = os.path.join(RESULTS_PATH, single_fname)
                if not os.path.exists(ctrl_coeff_file):
                    raise FileNotFoundError(f"No ctrl files found in {RESULTS_PATH}")
                ctrl_coeff = np.load(ctrl_coeff_file)
                print(f"# Loaded single {single_fname} with shape {ctrl_coeff.shape}")

        else:
            if ctrl_coeff_files:
                print("# Consolidating dictionary outputs...")
                ctrl_coeff_dic = ctar.data_loader.consolidate_null_dic(
                    path=RESULTS_PATH,
                    startswith=prefix,
                    remove_empty=True,
                    print_missing=False
                )
                # save consolidated dict and delete chunks
                single_fname = 'ctrl_corr_dic.pkl' if FLAG_CORR else 'ctrl_coeff_dic.pkl'
                with open(os.path.join(RESULTS_PATH, single_fname), 'wb') as f:
                    pickle.dump(ctrl_coeff_dic, f)
                for fname in ctrl_coeff_files:
                    os.remove(os.path.join(RESULTS_PATH, fname))
                print(f"# Saved {single_fname} and deleted {len(ctrl_coeff_files)} chunk(s)")
            else:
                single_fname = 'ctrl_corr_dic.pkl' if FLAG_CORR else 'ctrl_coeff_dic.pkl'
                ctrl_coeff_file = os.path.join(RESULTS_PATH, single_fname)
                if not os.path.exists(ctrl_coeff_file):
                    raise FileNotFoundError(f"No ctrl dic files found in {RESULTS_PATH}")
                with open(ctrl_coeff_file, 'rb') as f:
                    ctrl_coeff_dic = pickle.load(f)
                print(f"# Loaded single {single_fname} with {len(ctrl_coeff_dic)} bins")


    ###########################################################################################
    ######                                  Computation                                  ######
    ###########################################################################################


    if JOB == "compute_ctar":

        print("# Running --job compute_ctar")

        adata_rna.var = ctar.data_loader.get_gene_coords(adata_rna.var, add_tss=True)
        cis_links_df = ctar.data_loader.peak_to_gene(adata_atac.var, adata_rna.var, clean=True, split_peaks=True)

        if USE_INF_MODE: 
            print("# Using inf.inf mode - creating control peaks only")
            
            # Create control peaks (not peak-gene pairs)
            ctrl_peaks = ctar.method.create_ctrl_peaks(
                adata_atac, type=atac_type, num_bins=atac_bins, b=n_ctrl,
                peak_col='peak', layer='counts', genome_file=GENOME_FILE,
                add_promoter_bin=False, genes_df=adata_rna.var,
            )
            
            # Get cis links indices
            cis_links_df = ctar.data_loader.attach_link_indices(cis_links_df, adata_atac, adata_rna)
            cis_links_array = cis_links_df[['atac_idx', 'rna_idx']].to_numpy()
            link_ids = cis_links_df.index.to_numpy()

            cis_links_dic = ctar.parallel.chunk_links_dict(cis_links_array, dic_batch_size=DIC_BATCH_SIZE)
            atac_idx_arr = cis_links_df['atac_idx'].to_numpy()
            rna_idx_arr = cis_links_df['rna_idx'].to_numpy()
            ctrl_peak_sub = ctrl_peaks[atac_idx_arr, :n_ctrl]
            ctrl_links_array = np.column_stack([
                ctrl_peak_sub.ravel(),
                np.repeat(rna_idx_arr, n_ctrl),
            ])

            ctrl_links_dic = ctar.parallel.chunk_links_dict(ctrl_links_array, dic_batch_size=DIC_BATCH_SIZE)
            rna_sparse = rna_norm if FLAG_CORR else adata_rna.layers['counts']
            atac_sparse = atac_norm if FLAG_CORR else adata_atac.layers['counts']

            n_links = len(cis_links_array)

            if FLAG_CORR:
                print('# Starting cis-links Pearson correlation...')
                start_time = time.time()
                cis_corr_dic = ctar.parallel.multiprocess_pearson_corr(
                    links_dict=cis_links_dic,
                    atac_sparse=atac_sparse,
                    rna_sparse=rna_sparse,
                    batch_size=BATCH_SIZE,
                    n_workers=n_cores,
                    donor_labels=donor_labels,
                )
                print('# Cis Pearson time = %0.2fs' % (time.time() - start_time))
                cis_coeff = np.concatenate([np.asarray(cis_corr_dic[k]) for k in sorted(cis_corr_dic)]).astype(np.float32)

                print('# Starting control links Pearson correlation...')
                start_time = time.time()
                ctrl_corr_dic = ctar.parallel.multiprocess_pearson_corr(
                    links_dict=ctrl_links_dic,
                    atac_sparse=atac_sparse,
                    rna_sparse=rna_sparse,
                    batch_size=BATCH_SIZE,
                    n_workers=n_cores,
                    donor_labels=donor_labels,
                )
                print('# Control Pearson time = %0.2fs' % (time.time() - start_time))
                ctrl_coeff = np.concatenate(
                    [np.asarray(ctrl_corr_dic[k]) for k in sorted(ctrl_corr_dic)]
                ).astype(np.float32).reshape(len(cis_links_df), n_ctrl)

                cis_links_df['corr'] = cis_coeff
                cis_links_df[f'{BIN_CONFIG}_mcpval'] = ctar.method.initial_mcpval(ctrl_coeff, cis_coeff)
                cis_links_df[f'{BIN_CONFIG}_ppval'] = ctar.method.pooled_mcpval(ctrl_coeff, cis_coeff)
                cis_links_df[f'{BIN_CONFIG}_zpval'] = ctar.method.zscore_pval(ctrl_coeff, cis_coeff)[0]

                if FLAG_SPEC:
                    print(f"# Computing {SPEC_STAT} link specificity ({SPEC_NORM} gene/peak cell-type tables)...")
                    rna_table, atac_table = ctar.celltype_specificity.celltype_tables(
                        adata_rna, adata_atac, CELLTYPE_COL, method=SPEC_NORM)
                    # reuse the --flag_corr normalize_total + log1p matrices (same median target) when ATAC is log1p too
                    reuse_log1p = FLAG_CORR and not (FLAG_TFIDF or FLAG_BIN)
                    coexp_ctx = (ctar.celltype_specificity.coexp_context(
                        adata_rna, adata_atac, CELLTYPE_COL,
                        log_rna=rna_norm if reuse_log1p else None, log_atac=atac_norm if reuse_log1p else None)
                                 if SPEC_STAT == 'coexp_log1p' else None)
                    cis_spec, cis_spec_dom = ctar.parallel.multiprocess_link_specificity(
                        cis_links_array, SPEC_STAT, rna_table, atac_table, coexp_ctx, batch_size=BATCH_SIZE, n_workers=n_cores)
                    cis_spec = cis_spec.astype(np.float32)
                    ctrl_spec = ctar.parallel.multiprocess_link_specificity(
                        ctrl_links_array, SPEC_STAT, rna_table, atac_table, coexp_ctx, batch_size=BATCH_SIZE, n_workers=n_cores,
                    )[0].astype(np.float32).reshape(len(cis_spec), n_ctrl)

                    cis_links_df[LINK_SPEC_COL] = cis_spec
                    cis_links_df[f'{BIN_CONFIG}_mcpval_{SPEC_TAG}'] = ctar.method.initial_mcpval(ctrl_spec, cis_spec)
                    cis_links_df[f'{BIN_CONFIG}_ppval_{SPEC_TAG}'] = ctar.method.pooled_mcpval(ctrl_spec, cis_spec)
                    cis_links_df[f'{BIN_CONFIG}_zpval_{SPEC_TAG}'] = ctar.method.zscore_pval(ctrl_spec, cis_spec)[0]

                    os.makedirs(results_folder, exist_ok=True)
                    np.save(f'{results_folder}/cis_{SPEC_TAG}.npy', cis_spec)
                    np.save(f'{results_folder}/ctrl_{SPEC_TAG}.npy', ctrl_spec)

                    # Marginal specificity (gene only for inf case)
                    gene_spec, peak_spec = ctar.celltype_specificity.marginal_specificity(rna_table, atac_table)
                    cis_peak_spec = peak_spec[cis_links_array[:, 0]].astype(np.float32)
                    ctrl_peak_spec = peak_spec[ctrl_links_array[:, 0]].astype(np.float32).reshape(len(cis_spec), n_ctrl)
                    cis_links_df[f'gene_specificity{SPEC_SUFFIX}'] = gene_spec[cis_links_array[:, 1]].astype(np.float32)
                    cis_links_df[f'peak_specificity{SPEC_SUFFIX}'] = cis_peak_spec
                    cis_links_df[f'{BIN_CONFIG}_mcpval_peak_spec{SPEC_SUFFIX}'] = ctar.method.initial_mcpval(ctrl_peak_spec, cis_peak_spec)
                    np.save(f'{results_folder}/gene_spec{SPEC_SUFFIX}.npy', gene_spec)
                    np.save(f'{results_folder}/peak_spec{SPEC_SUFFIX}.npy', peak_spec)

                    # Dominant cell type of the gene, the peak and the joint profile
                    gene_dominant, peak_dominant, link_dominant, dominant_concordant = ctar.celltype_specificity.dominant_celltypes(
                        rna_table, atac_table, cis_links_array[:, 1], cis_links_array[:, 0])
                    cis_links_df[f'gene_dominant_subtype{SPEC_SUFFIX}'] = gene_dominant
                    cis_links_df[f'peak_dominant_subtype{SPEC_SUFFIX}'] = peak_dominant
                    cis_links_df[f'link_dominant_subtype{SPEC_SUFFIX}'] = link_dominant
                    if cis_spec_dom is not None:
                        cis_links_df['link_dominant_subtype_coexp'] = np.where(
                            np.asarray(cis_spec_dom) >= 0, coexp_ctx['celltypes'][np.maximum(cis_spec_dom, 0)], None)
                    cis_links_df = ctar.method.add_filtered_spec_qval(
                        cis_links_df, f'{BIN_CONFIG}_mcpval_{SPEC_TAG}', f'{BIN_CONFIG}_qval_{SPEC_TAG}_filt',
                        SPEC_STAT, SPEC_FILTER_FDR, SPEC_FILTER_QCOL)
                    cis_links_df[f'dominant_subtype_concordant{SPEC_SUFFIX}'] = dominant_concordant

                print(f'# Saving files to {results_folder}')
                os.makedirs(results_folder, exist_ok=True)
                np.save(f'{results_folder}/ctrl_peaks.npy', ctrl_peaks)
                np.save(f'{results_folder}/cis_corr.npy', cis_coeff)
                np.save(f'{results_folder}/ctrl_corr.npy', ctrl_coeff)
                np.save(f'{results_folder}/cis_link_ids.npy', link_ids, allow_pickle=True)
                cis_links_df.to_csv(f'{results_folder}/cis_links_df.csv')

            else:
                print('# Starting cis-links IRLS...')
                start_time = time.time()

                if covar_mat is not None:
                    cis_coeff_dic = ctar.parallel.multiprocess_poisson_irls_multivar(
                        links_dict=cis_links_dic,
                        atac_sparse=atac_sparse,
                        rna_sparse=rna_sparse,
                        covar_mat=covar_mat,
                        batch_size=BATCH_SIZE,
                        scheduler="processes",
                        n_workers=n_cores,
                        flag_se=FLAG_SE,
                        flag_ll=FLAG_LL
                    )
                else:
                    cis_coeff_dic = ctar.parallel.multiprocess_poisson_irls(
                        links_dict=cis_links_dic,
                        atac_sparse=atac_sparse,
                        rna_sparse=rna_sparse,
                        batch_size=BATCH_SIZE,
                        scheduler="processes",
                        n_workers=n_cores,
                        flag_se=FLAG_SE,
                        flag_ll=FLAG_LL
                    )
                print('# Cis-links IRLS time = %0.2fs' % (time.time() - start_time))

                cis_coeff = np.concatenate(
                    [np.asarray(cis_coeff_dic[k]) for k in sorted(cis_coeff_dic)]
                ).astype(np.float32)

                print('# Starting control links IRLS...')
                start_time = time.time()

                if covar_mat is not None:
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
                        batch_size=(BATCH_SIZE//10)+1,
                        scheduler="processes",
                        n_workers=n_cores,
                        flag_se=FLAG_SE,
                        flag_ll=FLAG_LL,
                    )
                print('# Control links IRLS time = %0.2fs' % (time.time() - start_time))

                first_ctrl_key = list(ctrl_coeff_dic.keys())[0]
                first_ctrl_val = np.asarray(ctrl_coeff_dic[first_ctrl_key])

                ctrl_coeff = np.concatenate(
                    [np.asarray(ctrl_coeff_dic[k]) for k in sorted(ctrl_coeff_dic)]
                ).astype(np.float32)
                if first_ctrl_val.ndim == 2:
                    ctrl_coeff = ctrl_coeff.reshape(len(cis_links_df), n_ctrl, first_ctrl_val.shape[1])
                else:
                    ctrl_coeff = ctrl_coeff.reshape(len(cis_links_df), n_ctrl)

                MAX_VAL = 100
                if ctrl_coeff.ndim == 3 and ctrl_coeff.shape[2] == 2:
                    ctrl_se = ctrl_coeff[:, :, 0]
                    ctrl_beta = ctrl_coeff[:, :, 1]

                    cis_links_df[f'{BIN_CONFIG}_mcpval'] = ctar.method.initial_mcpval(ctrl_beta, cis_coeff[:, 1])
                    cis_links_df[f'{BIN_CONFIG}_ppval'] = ctar.method.pooled_mcpval(ctrl_beta, cis_coeff[:, 1])
                    cis_links_df[f'{BIN_CONFIG}_zpval'] = ctar.method.zscore_pval(ctrl_beta, cis_coeff[:, 1])[0]

                    cis_studentized = np.clip(cis_coeff[:, 1] / cis_coeff[:, 0], -MAX_VAL, MAX_VAL)
                    ctrl_studentized = np.clip(ctrl_beta / ctrl_se, -MAX_VAL, MAX_VAL)

                    cis_links_df[f'{BIN_CONFIG}_mcpval_z'] = ctar.method.initial_mcpval(ctrl_studentized, cis_studentized)
                    cis_links_df[f'{BIN_CONFIG}_ppval_z'] = ctar.method.pooled_mcpval(ctrl_studentized, cis_studentized)
                    cis_links_df[f'{BIN_CONFIG}_zpval_z'] = ctar.method.zscore_pval(ctrl_studentized, cis_studentized)[0]

                    cis_links_df['poissonb'] = cis_coeff[:, 1]
                    cis_links_df['poissonb_se'] = cis_coeff[:, 0]
                else:
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
                    cis_links_df[f'{BIN_CONFIG}_zpval'] = ctar.method.zscore_pval(ctrl_coeff, cis_beta)[0]

                    if 'poissonb' not in cis_links_df.columns:
                        cis_links_df['poissonb'] = cis_beta

                if FLAG_SPEC:
                    print(f"# Computing {SPEC_STAT} link specificity ({SPEC_NORM} gene/peak cell-type tables)...")
                    rna_table, atac_table = ctar.celltype_specificity.celltype_tables(
                        adata_rna, adata_atac, CELLTYPE_COL, method=SPEC_NORM)
                    # reuse the --flag_corr normalize_total + log1p matrices (same median target) when ATAC is log1p too
                    reuse_log1p = FLAG_CORR and not (FLAG_TFIDF or FLAG_BIN)
                    coexp_ctx = (ctar.celltype_specificity.coexp_context(
                        adata_rna, adata_atac, CELLTYPE_COL,
                        log_rna=rna_norm if reuse_log1p else None, log_atac=atac_norm if reuse_log1p else None)
                                 if SPEC_STAT == 'coexp_log1p' else None)
                    cis_spec, cis_spec_dom = ctar.parallel.multiprocess_link_specificity(
                        cis_links_array, SPEC_STAT, rna_table, atac_table, coexp_ctx, batch_size=BATCH_SIZE, n_workers=n_cores)
                    cis_spec = cis_spec.astype(np.float32)
                    ctrl_spec = ctar.parallel.multiprocess_link_specificity(
                        ctrl_links_array, SPEC_STAT, rna_table, atac_table, coexp_ctx, batch_size=BATCH_SIZE, n_workers=n_cores,
                    )[0].astype(np.float32).reshape(len(cis_spec), n_ctrl)

                    cis_links_df[LINK_SPEC_COL] = cis_spec
                    cis_links_df[f'{BIN_CONFIG}_mcpval_{SPEC_TAG}'] = ctar.method.initial_mcpval(ctrl_spec, cis_spec)
                    cis_links_df[f'{BIN_CONFIG}_ppval_{SPEC_TAG}'] = ctar.method.pooled_mcpval(ctrl_spec, cis_spec)
                    cis_links_df[f'{BIN_CONFIG}_zpval_{SPEC_TAG}'] = ctar.method.zscore_pval(ctrl_spec, cis_spec)[0]

                    os.makedirs(results_folder, exist_ok=True)
                    np.save(f'{results_folder}/cis_{SPEC_TAG}.npy', cis_spec)
                    np.save(f'{results_folder}/ctrl_{SPEC_TAG}.npy', ctrl_spec)

                    # Marginal specificity (gene only for inf case)
                    gene_spec, peak_spec = ctar.celltype_specificity.marginal_specificity(rna_table, atac_table)
                    cis_peak_spec = peak_spec[cis_links_array[:, 0]].astype(np.float32)
                    ctrl_peak_spec = peak_spec[ctrl_links_array[:, 0]].astype(np.float32).reshape(len(cis_spec), n_ctrl)
                    cis_links_df[f'gene_specificity{SPEC_SUFFIX}'] = gene_spec[cis_links_array[:, 1]].astype(np.float32)
                    cis_links_df[f'peak_specificity{SPEC_SUFFIX}'] = cis_peak_spec
                    cis_links_df[f'{BIN_CONFIG}_mcpval_peak_spec{SPEC_SUFFIX}'] = ctar.method.initial_mcpval(ctrl_peak_spec, cis_peak_spec)
                    np.save(f'{results_folder}/gene_spec{SPEC_SUFFIX}.npy', gene_spec)
                    np.save(f'{results_folder}/peak_spec{SPEC_SUFFIX}.npy', peak_spec)

                    # Dominant cell type of the gene, the peak and the joint profile
                    gene_dominant, peak_dominant, link_dominant, dominant_concordant = ctar.celltype_specificity.dominant_celltypes(
                        rna_table, atac_table, cis_links_array[:, 1], cis_links_array[:, 0])
                    cis_links_df[f'gene_dominant_subtype{SPEC_SUFFIX}'] = gene_dominant
                    cis_links_df[f'peak_dominant_subtype{SPEC_SUFFIX}'] = peak_dominant
                    cis_links_df[f'link_dominant_subtype{SPEC_SUFFIX}'] = link_dominant
                    if cis_spec_dom is not None:
                        cis_links_df['link_dominant_subtype_coexp'] = np.where(
                            np.asarray(cis_spec_dom) >= 0, coexp_ctx['celltypes'][np.maximum(cis_spec_dom, 0)], None)
                    cis_links_df = ctar.method.add_filtered_spec_qval(
                        cis_links_df, f'{BIN_CONFIG}_mcpval_{SPEC_TAG}', f'{BIN_CONFIG}_qval_{SPEC_TAG}_filt',
                        SPEC_STAT, SPEC_FILTER_FDR, SPEC_FILTER_QCOL)
                    cis_links_df[f'dominant_subtype_concordant{SPEC_SUFFIX}'] = dominant_concordant

                print(f'# Saving files to {results_folder}')
                os.makedirs(results_folder, exist_ok=True)

                np.save(f'{results_folder}/ctrl_peaks.npy', ctrl_peaks)
                np.save(f'{results_folder}/cis_coeff.npy', cis_coeff)
                np.save(f'{results_folder}/ctrl_coeff.npy', ctrl_coeff)
                np.save(f'{results_folder}/cis_link_ids.npy', link_ids, allow_pickle=True)
                cis_links_df.to_csv(f'{results_folder}/cis_links_df.csv')
            
        else:
            # Standard binned mode
            ctrl_links_dic, atac_bins_df, rna_bins_df = ctar.method.create_ctrl_pairs(
                adata_atac, adata_rna, atac_bins=[n_atac_mean,n_atac_gc], rna_bins=[n_rna_mean,n_rna_var],
                atac_type=atac_type, rna_type=rna_type, b=n_ctrl,
                atac_layer='counts', rna_layer='counts', genome_file=GENOME_FILE,
                n_length_bins=N_LENGTH_BINS,
            )

            cis_links_df = ctar.data_loader.combine_peak_gene_bins(cis_links_df, atac_bins_df, rna_bins_df,
                atac_type=atac_type, rna_type=rna_type,
                atac_bins=[n_atac_mean,n_atac_gc],
                rna_bins=[n_rna_mean,n_rna_var]
            )

            cis_links_dic, cis_idx_dic = ctar.data_loader.groupby_combined_bins(cis_links_df,
                combined_bin_col=f'combined_bin_{BIN_CONFIG.rsplit(".",1)[0]}',
                return_dic=True
            )

            rna_sparse = rna_norm if FLAG_CORR else adata_rna.layers['counts']
            atac_sparse = atac_norm if FLAG_CORR else adata_atac.layers['counts']

            if FLAG_CORR:
                print('# Starting cis-links Pearson correlation...')
                start_time = time.time()
                cis_coeff_dic = ctar.parallel.multiprocess_pearson_corr(
                    links_dict=cis_links_dic,
                    atac_sparse=atac_sparse,
                    rna_sparse=rna_sparse,
                    batch_size=BATCH_SIZE,
                    n_workers=n_cores,
                    donor_labels=donor_labels,
                )
                print('# Cis Pearson time = %0.2fs' % (time.time() - start_time))

                print('# Starting control links Pearson correlation...')
                start_time = time.time()
                ctrl_coeff_dic = ctar.parallel.multiprocess_pearson_corr(
                    links_dict=ctrl_links_dic,
                    atac_sparse=atac_sparse,
                    rna_sparse=rna_sparse,
                    batch_size=BATCH_SIZE,
                    n_workers=n_cores,
                    donor_labels=donor_labels,
                )
                print('# Control Pearson time = %0.2fs' % (time.time() - start_time))

                mcpval_dic, ppval_dic, zpval_dic = ctar.method.binned_mcpval(cis_coeff_dic, ctrl_coeff_dic, b=n_ctrl)
                cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, mcpval_dic, col_name=f'{BIN_CONFIG}_mcpval')
                cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, ppval_dic, col_name=f'{BIN_CONFIG}_ppval')
                cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, zpval_dic, col_name=f'{BIN_CONFIG}_zpval')
                cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, cis_coeff_dic, col_name='corr')

                if FLAG_SPEC:
                    print(f"# Computing {SPEC_STAT} link specificity ({SPEC_NORM} gene/peak cell-type tables)...")
                    rna_table, atac_table = ctar.celltype_specificity.celltype_tables(
                        adata_rna, adata_atac, CELLTYPE_COL, method=SPEC_NORM)
                    # reuse the --flag_corr normalize_total + log1p matrices (same median target) when ATAC is log1p too
                    reuse_log1p = FLAG_CORR and not (FLAG_TFIDF or FLAG_BIN)
                    coexp_ctx = (ctar.celltype_specificity.coexp_context(
                        adata_rna, adata_atac, CELLTYPE_COL,
                        log_rna=rna_norm if reuse_log1p else None, log_atac=atac_norm if reuse_log1p else None)
                                 if SPEC_STAT == 'coexp_log1p' else None)
                    cis_spec_dic, cis_spec_dom_dic = ctar.parallel.multiprocess_link_specificity(
                        cis_links_dic, SPEC_STAT, rna_table, atac_table, coexp_ctx, batch_size=BATCH_SIZE, n_workers=n_cores)
                    cis_spec_dom = None if cis_spec_dom_dic is None else ctar.data_loader.map_dic_to_df(
                        cis_links_df[[]].copy(), cis_idx_dic, cis_spec_dom_dic, col_name='code')['code'].fillna(-1).astype(int).to_numpy()
                    ctrl_spec_dic = ctar.parallel.multiprocess_link_specificity(
                        ctrl_links_dic, SPEC_STAT, rna_table, atac_table, coexp_ctx, batch_size=BATCH_SIZE, n_workers=n_cores)[0]
                    mcpval_spec_dic, ppval_spec_dic, zpval_spec_dic = ctar.method.binned_mcpval(cis_spec_dic, ctrl_spec_dic, b=n_ctrl)

                    cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, cis_spec_dic, col_name=LINK_SPEC_COL)
                    cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, mcpval_spec_dic, col_name=f'{BIN_CONFIG}_mcpval_{SPEC_TAG}')
                    cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, ppval_spec_dic, col_name=f'{BIN_CONFIG}_ppval_{SPEC_TAG}')
                    cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, zpval_spec_dic, col_name=f'{BIN_CONFIG}_zpval_{SPEC_TAG}')

                    os.makedirs(results_folder, exist_ok=True)
                    with open(f'{results_folder}/cis_{SPEC_TAG}_dic.pkl', 'wb') as f:
                        pickle.dump(cis_spec_dic, f)
                    with open(f'{results_folder}/ctrl_{SPEC_TAG}_dic.pkl', 'wb') as f:
                        pickle.dump(ctrl_spec_dic, f)

                    # Marginal specificity
                    gene_spec, peak_spec = ctar.celltype_specificity.marginal_specificity(rna_table, atac_table)
                    for feature, feature_spec, pair_column in (('gene', gene_spec, 1), ('peak', peak_spec, 0)):
                        cis_feature_dic = {key: feature_spec[pairs[:, pair_column]] for key, pairs in cis_links_dic.items()}
                        ctrl_feature_dic = {key: feature_spec[pairs[:, pair_column]] for key, pairs in ctrl_links_dic.items()}
                        mcpval_feature_dic, _, _ = ctar.method.binned_mcpval(cis_feature_dic, ctrl_feature_dic, b=n_ctrl)
                        cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, cis_feature_dic, col_name=f'{feature}_specificity{SPEC_SUFFIX}')
                        cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, mcpval_feature_dic, col_name=f'{BIN_CONFIG}_mcpval_{feature}_spec{SPEC_SUFFIX}')
                    np.save(f'{results_folder}/gene_spec{SPEC_SUFFIX}.npy', gene_spec)
                    np.save(f'{results_folder}/peak_spec{SPEC_SUFFIX}.npy', peak_spec)

                    # Row-aligned (atac_idx, rna_idx) pairs rebuilt from the bins; -1 marks rows that are not in a bin
                    cis_links_array = np.full((len(cis_links_df), 2), -1, dtype=np.int64)
                    for key, idx in cis_idx_dic.items():
                        cis_links_array[np.asarray(idx)] = cis_links_dic[key]

                    # Dominant cell type of the gene, the peak and the joint profile
                    gene_dominant, peak_dominant, link_dominant, dominant_concordant = ctar.celltype_specificity.dominant_celltypes(
                        rna_table, atac_table, cis_links_array[:, 1], cis_links_array[:, 0])
                    cis_links_df[f'gene_dominant_subtype{SPEC_SUFFIX}'] = gene_dominant
                    cis_links_df[f'peak_dominant_subtype{SPEC_SUFFIX}'] = peak_dominant
                    cis_links_df[f'link_dominant_subtype{SPEC_SUFFIX}'] = link_dominant
                    if cis_spec_dom is not None:
                        cis_links_df['link_dominant_subtype_coexp'] = np.where(
                            np.asarray(cis_spec_dom) >= 0, coexp_ctx['celltypes'][np.maximum(cis_spec_dom, 0)], None)
                    cis_links_df = ctar.method.add_filtered_spec_qval(
                        cis_links_df, f'{BIN_CONFIG}_mcpval_{SPEC_TAG}', f'{BIN_CONFIG}_qval_{SPEC_TAG}_filt',
                        SPEC_STAT, SPEC_FILTER_FDR, SPEC_FILTER_QCOL)
                    cis_links_df[f'dominant_subtype_concordant{SPEC_SUFFIX}'] = dominant_concordant

                print(f'# Saving files to {results_folder}')
                os.makedirs(results_folder, exist_ok=True)
                with open(f'{results_folder}/cis_corr_dic.pkl', 'wb') as f:
                    pickle.dump(cis_coeff_dic, f)
                with open(f'{results_folder}/cis_idx_dic.pkl', 'wb') as f:
                    pickle.dump(cis_idx_dic, f)
                with open(f'{results_folder}/ctrl_corr_dic.pkl', 'wb') as f:
                    pickle.dump(ctrl_coeff_dic, f)
                with open(f'{results_folder}/ctrl_links_dic.pkl', 'wb') as f:
                    pickle.dump(ctrl_links_dic, f)
                cis_links_df.to_csv(f'{results_folder}/cis_links_df.csv')

            else:
                print('# Starting cis-links IRLS...')
                start_time = time.time()

                if covar_mat is not None:
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
                        flag_se=FLAG_SE,
                        flag_ll=FLAG_LL,
                    )
                print('# Cis-links IRLS time = %0.2fs' % (time.time() - start_time))

                print('# Starting control links IRLS...')
                start_time = time.time()

                if covar_mat is not None:
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
                        batch_size=(BATCH_SIZE//10)+1,
                        scheduler="processes",
                        n_workers=n_cores,
                        flag_se=FLAG_SE,
                        flag_ll=FLAG_LL,
                    )
                print('# Control links IRLS time = %0.2fs' % (time.time() - start_time))

                first_key = list(ctrl_coeff_dic.keys())[0]
                if ctrl_coeff_dic[first_key].ndim == 2 and ctrl_coeff_dic[first_key].shape[1] == 2:
                    mcpval_dic, ppval_dic, zpval_dic = ctar.method.binned_mcpval(
                        {key: value[:, 1] for key, value in cis_coeff_dic.items()},
                        {key: value[:, 1] for key, value in ctrl_coeff_dic.items()},
                        b=n_ctrl,
                    )

                    cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, mcpval_dic, col_name=f'{BIN_CONFIG}_mcpval')
                    cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, ppval_dic, col_name=f'{BIN_CONFIG}_ppval')
                    cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, zpval_dic, col_name=f'{BIN_CONFIG}_zpval')
                    cis_links_df = ctar.data_loader.map_dic_to_df(
                        cis_links_df,
                        cis_idx_dic,
                        {key: value[:, 1] for key, value in cis_coeff_dic.items()},
                        col_name='poissonb'
                    )
                    cis_links_df = ctar.data_loader.map_dic_to_df(
                        cis_links_df,
                        cis_idx_dic,
                        {key: value[:, 0] for key, value in cis_coeff_dic.items()},
                        col_name='poissonb_se'
                    )

                    MAX_VAL = 100
                    cis_coeff_dic_z = {}
                    ctrl_coeff_dic_z = {}
                    for key in cis_coeff_dic.keys():
                        cis_coeff_dic_z[key] = np.clip(cis_coeff_dic[key][:, 1] / cis_coeff_dic[key][:, 0], -MAX_VAL, MAX_VAL)
                        ctrl_coeff_dic_z[key] = np.clip(ctrl_coeff_dic[key][:, 1] / ctrl_coeff_dic[key][:, 0], -MAX_VAL, MAX_VAL)

                    mcpval_dic_z, ppval_dic_z, zpval_dic_z = ctar.method.binned_mcpval(
                        cis_coeff_dic_z,
                        ctrl_coeff_dic_z,
                        b=n_ctrl,
                    )

                    cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, mcpval_dic_z, col_name=f'{BIN_CONFIG}_mcpval_z')
                    cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, ppval_dic_z, col_name=f'{BIN_CONFIG}_ppval_z')
                    cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, zpval_dic_z, col_name=f'{BIN_CONFIG}_zpval_z')
                else:
                    mcpval_dic, ppval_dic, zpval_dic = ctar.method.binned_mcpval(cis_coeff_dic, ctrl_coeff_dic, b=n_ctrl)

                    cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, mcpval_dic, col_name=f'{BIN_CONFIG}_mcpval')
                    cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, ppval_dic, col_name=f'{BIN_CONFIG}_ppval')
                    cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, zpval_dic, col_name=f'{BIN_CONFIG}_zpval')
                    cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, cis_coeff_dic, col_name='poissonb')

                if FLAG_SPEC:
                    print(f"# Computing {SPEC_STAT} link specificity ({SPEC_NORM} gene/peak cell-type tables)...")
                    rna_table, atac_table = ctar.celltype_specificity.celltype_tables(
                        adata_rna, adata_atac, CELLTYPE_COL, method=SPEC_NORM)
                    # reuse the --flag_corr normalize_total + log1p matrices (same median target) when ATAC is log1p too
                    reuse_log1p = FLAG_CORR and not (FLAG_TFIDF or FLAG_BIN)
                    coexp_ctx = (ctar.celltype_specificity.coexp_context(
                        adata_rna, adata_atac, CELLTYPE_COL,
                        log_rna=rna_norm if reuse_log1p else None, log_atac=atac_norm if reuse_log1p else None)
                                 if SPEC_STAT == 'coexp_log1p' else None)
                    cis_spec_dic, cis_spec_dom_dic = ctar.parallel.multiprocess_link_specificity(
                        cis_links_dic, SPEC_STAT, rna_table, atac_table, coexp_ctx, batch_size=BATCH_SIZE, n_workers=n_cores)
                    cis_spec_dom = None if cis_spec_dom_dic is None else ctar.data_loader.map_dic_to_df(
                        cis_links_df[[]].copy(), cis_idx_dic, cis_spec_dom_dic, col_name='code')['code'].fillna(-1).astype(int).to_numpy()
                    ctrl_spec_dic = ctar.parallel.multiprocess_link_specificity(
                        ctrl_links_dic, SPEC_STAT, rna_table, atac_table, coexp_ctx, batch_size=BATCH_SIZE, n_workers=n_cores)[0]
                    mcpval_spec_dic, ppval_spec_dic, zpval_spec_dic = ctar.method.binned_mcpval(cis_spec_dic, ctrl_spec_dic, b=n_ctrl)

                    cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, cis_spec_dic, col_name=LINK_SPEC_COL)
                    cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, mcpval_spec_dic, col_name=f'{BIN_CONFIG}_mcpval_{SPEC_TAG}')
                    cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, ppval_spec_dic, col_name=f'{BIN_CONFIG}_ppval_{SPEC_TAG}')
                    cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, zpval_spec_dic, col_name=f'{BIN_CONFIG}_zpval_{SPEC_TAG}')

                    os.makedirs(results_folder, exist_ok=True)
                    with open(f'{results_folder}/cis_{SPEC_TAG}_dic.pkl', 'wb') as f:
                        pickle.dump(cis_spec_dic, f)
                    with open(f'{results_folder}/ctrl_{SPEC_TAG}_dic.pkl', 'wb') as f:
                        pickle.dump(ctrl_spec_dic, f)

                    # Marginal specificity
                    gene_spec, peak_spec = ctar.celltype_specificity.marginal_specificity(rna_table, atac_table)
                    for feature, feature_spec, pair_column in (('gene', gene_spec, 1), ('peak', peak_spec, 0)):
                        cis_feature_dic = {key: feature_spec[pairs[:, pair_column]] for key, pairs in cis_links_dic.items()}
                        ctrl_feature_dic = {key: feature_spec[pairs[:, pair_column]] for key, pairs in ctrl_links_dic.items()}
                        mcpval_feature_dic, _, _ = ctar.method.binned_mcpval(cis_feature_dic, ctrl_feature_dic, b=n_ctrl)
                        cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, cis_feature_dic, col_name=f'{feature}_specificity{SPEC_SUFFIX}')
                        cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, mcpval_feature_dic, col_name=f'{BIN_CONFIG}_mcpval_{feature}_spec{SPEC_SUFFIX}')
                    np.save(f'{results_folder}/gene_spec{SPEC_SUFFIX}.npy', gene_spec)
                    np.save(f'{results_folder}/peak_spec{SPEC_SUFFIX}.npy', peak_spec)

                    # Row-aligned (atac_idx, rna_idx) pairs rebuilt from the bins; -1 marks rows that are not in a bin
                    cis_links_array = np.full((len(cis_links_df), 2), -1, dtype=np.int64)
                    for key, idx in cis_idx_dic.items():
                        cis_links_array[np.asarray(idx)] = cis_links_dic[key]

                    # Dominant cell type of the gene, the peak and the joint profile
                    gene_dominant, peak_dominant, link_dominant, dominant_concordant = ctar.celltype_specificity.dominant_celltypes(
                        rna_table, atac_table, cis_links_array[:, 1], cis_links_array[:, 0])
                    cis_links_df[f'gene_dominant_subtype{SPEC_SUFFIX}'] = gene_dominant
                    cis_links_df[f'peak_dominant_subtype{SPEC_SUFFIX}'] = peak_dominant
                    cis_links_df[f'link_dominant_subtype{SPEC_SUFFIX}'] = link_dominant
                    if cis_spec_dom is not None:
                        cis_links_df['link_dominant_subtype_coexp'] = np.where(
                            np.asarray(cis_spec_dom) >= 0, coexp_ctx['celltypes'][np.maximum(cis_spec_dom, 0)], None)
                    cis_links_df = ctar.method.add_filtered_spec_qval(
                        cis_links_df, f'{BIN_CONFIG}_mcpval_{SPEC_TAG}', f'{BIN_CONFIG}_qval_{SPEC_TAG}_filt',
                        SPEC_STAT, SPEC_FILTER_FDR, SPEC_FILTER_QCOL)
                    cis_links_df[f'dominant_subtype_concordant{SPEC_SUFFIX}'] = dominant_concordant

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

        adata_rna.var = ctar.data_loader.get_gene_coords(adata_rna.var, add_tss=True)
        cis_links_df = ctar.data_loader.peak_to_gene(adata_atac.var, adata_rna.var, clean=True, split_peaks=True)

        if USE_INF_MODE:
            print("# Using inf.inf mode - creating control peaks only")
            print(f"# {atac_type},{atac_bins},{n_ctrl},{adata_atac.shape},{adata_rna.var.shape}")
            ctrl_peaks = ctar.method.create_ctrl_peaks(
                adata_atac, type=atac_type, num_bins=atac_bins, b=n_ctrl,
                peak_col='peak', layer='counts', genome_file=GENOME_FILE,
                add_promoter_bin=False, genes_df=adata_rna.var,
            )
            
            print(f'# Saving files to {results_folder}')
            os.makedirs(results_folder, exist_ok=True)
            
            np.save(f'{results_folder}/ctrl_peaks.npy', ctrl_peaks)
            cis_links_df.to_csv(f'{results_folder}/cis_links_df.csv')
            
        else:
            ctrl_links_dic, atac_bins_df, rna_bins_df = ctar.method.create_ctrl_pairs(
                adata_atac, adata_rna, atac_bins=[n_atac_mean,n_atac_gc], rna_bins=[n_rna_mean,n_rna_var],
                atac_type=atac_type, rna_type=rna_type, b=n_ctrl,
                atac_layer='counts', rna_layer='counts', genome_file=GENOME_FILE,
                n_length_bins=N_LENGTH_BINS,
            )

            cis_links_df = ctar.data_loader.combine_peak_gene_bins(cis_links_df, atac_bins_df, rna_bins_df,
                atac_type=atac_type, rna_type=rna_type,
                atac_bins=[n_atac_mean,n_atac_gc],
                rna_bins=[n_rna_mean,n_rna_var]
            )

            cis_links_dic, cis_idx_dic = ctar.data_loader.groupby_combined_bins(cis_links_df,
                combined_bin_col=f'combined_bin_{BIN_CONFIG.rsplit(".",1)[0]}',
                return_dic=True
            )

            print(f'# Saving files to {results_folder}')
            os.makedirs(results_folder, exist_ok=True)

            with open(f'{results_folder}/cis_links_dic.pkl', 'wb') as f:
                pickle.dump(cis_links_dic, f)
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
                peak_col='peak', layer='counts', genome_file=GENOME_FILE,
                add_promoter_bin=False, genes_df=adata_rna.var,
            )
            
            print(f'# Saving files to {results_folder}')
            os.makedirs(results_folder, exist_ok=True)
            
            np.save(f'{results_folder}/ctrl_peaks.npy', ctrl_peaks)
            
        else:
            if TSS_FILE is not None:
                print(f"# Loading gene chromosomes from {TSS_FILE} (skipping BioMart)...")
                gene_tss_df = ctar.gentruth.load_gene_tss_from_gencode(TSS_FILE)
                chr_map = gene_tss_df.set_index('gene')['gene_chr']
                adata_rna.var['chr'] = adata_rna.var['gene'].map(chr_map)
            else:
                adata_rna.var = ctar.data_loader.get_gene_coords(adata_rna.var, add_tss=False)
            ctrl_links_dic, atac_bins_df, rna_bins_df = ctar.method.create_ctrl_pairs(
                adata_atac, adata_rna, atac_bins=[n_atac_mean,n_atac_gc], rna_bins=[n_rna_mean,n_rna_var],
                atac_type=atac_type, rna_type=rna_type, b=n_ctrl,
                atac_layer='counts', rna_layer='counts', genome_file=GENOME_FILE,
                n_length_bins=N_LENGTH_BINS,
            )

            cis_links_df = ctar.data_loader.combine_peak_gene_bins(cis_links_df, atac_bins_df, rna_bins_df,
                atac_type=atac_type, rna_type=rna_type,
                atac_bins=[n_atac_mean,n_atac_gc],
                rna_bins=[n_rna_mean,n_rna_var]
            )

            cis_links_dic, cis_idx_dic = ctar.data_loader.groupby_combined_bins(cis_links_df,
                combined_bin_col=f'combined_bin_{BIN_CONFIG.rsplit(".",1)[0]}',
                return_dic=True
            )

            print(f'# Saving files to {results_folder}')
            os.makedirs(results_folder, exist_ok=True)

            with open(f'{results_folder}/ctrl_links_dic.pkl', 'wb') as f:
                pickle.dump(ctrl_links_dic, f)

        cis_links_df.to_csv(f'{results_folder}/cis_links_df.csv')


    if JOB == "compute_cis_only":

        print("# Running --job compute_cis_only")

        if not RESULTS_PATH: 

            adata_rna.var = ctar.data_loader.get_gene_coords(adata_rna.var, add_tss=True)
            cis_links_df = ctar.data_loader.peak_to_gene(adata_atac.var, adata_rna.var, clean=True, split_peaks=True)

            if not USE_INF_MODE:
                ctrl_links_dic, atac_bins_df, rna_bins_df = ctar.method.create_ctrl_pairs(
                    adata_atac, adata_rna, atac_bins=[n_atac_mean,n_atac_gc], rna_bins=[n_rna_mean,n_rna_var],
                    atac_type=atac_type, rna_type=rna_type, b=n_ctrl,
                    atac_layer='counts', rna_layer='counts', genome_file=GENOME_FILE,
                    n_length_bins=N_LENGTH_BINS,
                )

                cis_links_df = ctar.data_loader.combine_peak_gene_bins(cis_links_df, atac_bins_df, rna_bins_df,
                    atac_type=atac_type, rna_type=rna_type,
                    atac_bins=[n_atac_mean,n_atac_gc],
                    rna_bins=[n_rna_mean,n_rna_var]
                )

        file_suffix = ''

        if USE_INF_MODE:

            # For inf.inf mode, compute all cis links
            cis_links_df = ctar.data_loader.attach_link_indices(cis_links_df, adata_atac, adata_rna)

            if ARRAY_IDX is not None:
                start_idx = ARRAY_IDX * BATCH_SIZE
                end_idx = min((ARRAY_IDX + 1) * BATCH_SIZE, len(cis_links_df))
                cis_links_df = cis_links_df.iloc[start_idx:end_idx]
                file_suffix = f'_{start_idx}_{end_idx}'

            cis_links_array = cis_links_df[['atac_idx', 'rna_idx']].to_numpy()
            link_ids = cis_links_df.index.to_numpy()

            # Create cis links dictionary
            cis_links_dic = ctar.parallel.chunk_links_dict(cis_links_array, dic_batch_size=DIC_BATCH_SIZE)
            rna_sparse = rna_norm if FLAG_CORR else adata_rna.layers['counts']
            atac_sparse = atac_norm if FLAG_CORR else adata_atac.layers['counts']

            n_links = len(cis_links_array)

            if FLAG_CORR:
                print('# Starting cis-links Pearson correlation...')
                start_time = time.time()
                cis_corr_dic = ctar.parallel.multiprocess_pearson_corr(
                    links_dict=cis_links_dic,
                    atac_sparse=atac_sparse,
                    rna_sparse=rna_sparse,
                    batch_size=BATCH_SIZE,
                    n_workers=n_cores,
                    donor_labels=donor_labels,
                )
                print('# Cis Pearson time = %0.2fs' % (time.time() - start_time))
                cis_coeff = np.concatenate([np.asarray(cis_corr_dic[k]) for k in sorted(cis_corr_dic)]).astype(np.float32)
                cis_links_df['corr'] = cis_coeff
            else:
                print('# Starting cis-links IRLS...')
                start_time = time.time()

                if covar_mat is not None:
                    cis_coeff_dic = ctar.parallel.multiprocess_poisson_irls_multivar(
                        links_dict=cis_links_dic,
                        atac_sparse=atac_sparse,
                        rna_sparse=rna_sparse,
                        covar_mat=covar_mat,
                        batch_size=BATCH_SIZE,
                        scheduler="processes",
                        n_workers=n_cores,
                        flag_se=FLAG_SE,
                    )
                else:
                    cis_coeff_dic = ctar.parallel.multiprocess_poisson_irls(
                        links_dict=cis_links_dic,
                        atac_sparse=atac_sparse,
                        rna_sparse=rna_sparse,
                        batch_size=BATCH_SIZE,
                        scheduler="processes",
                        n_workers=n_cores,
                        flag_se=FLAG_SE,
                        flag_ll=FLAG_LL
                    )
                print('# Cis-links IRLS time = %0.2fs' % (time.time() - start_time))

                cis_coeff = np.concatenate(
                    [np.asarray(cis_coeff_dic[k]) for k in sorted(cis_coeff_dic)]
                ).astype(np.float32)
                cis_links_df['poissonb'] = cis_coeff[:, 1]
                cis_links_df['poissonb_se'] = cis_coeff[:, 0]

        else:

            # Standard binned mode
            cis_links_df = ctar.data_loader.ensure_combined_bin_column(
                cis_links_df,
                adata_atac,
                adata_rna,
                BIN_CONFIG,
                atac_type,
                rna_type,
                n_atac_mean,
                n_atac_gc,
                n_rna_mean,
                n_rna_var,
                GENOME_FILE,
                n_length_bins=N_LENGTH_BINS,
            )

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
                print('# Running %d links...' % (len(list(cis_links_dic.keys()))))

                file_suffix = f'_{ARRAY_IDX}'

            rna_sparse = rna_norm if FLAG_CORR else adata_rna.layers['counts']
            atac_sparse = atac_norm if FLAG_CORR else adata_atac.layers['counts']

            if FLAG_CORR:
                print('# Starting cis-links Pearson correlation...')
                start_time = time.time()
                cis_corr_dic = ctar.parallel.multiprocess_pearson_corr(
                    links_dict=cis_links_dic,
                    atac_sparse=atac_sparse,
                    rna_sparse=rna_sparse,
                    batch_size=BATCH_SIZE,
                    n_workers=n_cores,
                    donor_labels=donor_labels,
                )
                print('# Cis Pearson time = %0.2fs' % (time.time() - start_time))
                cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, cis_corr_dic, col_name='corr')
            else:
                print('# Starting cis-links IRLS...')
                start_time = time.time()

                if covar_mat is not None:
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
                        flag_se=FLAG_SE,
                        flag_ll=FLAG_LL
                    )
                print('# Cis-links IRLS time = %0.2fs' % (time.time() - start_time))

                first_key = list(cis_coeff_dic.keys())[0]
                if cis_coeff_dic[first_key].ndim == 2 and cis_coeff_dic[first_key].shape[1] == 2:
                    cis_links_df = ctar.data_loader.map_dic_to_df(
                        cis_links_df, cis_idx_dic,
                        {k: v[:, 1] for k, v in cis_coeff_dic.items()},
                        col_name='poissonb'
                    )
                    cis_links_df = ctar.data_loader.map_dic_to_df(
                        cis_links_df, cis_idx_dic,
                        {k: v[:, 0] for k, v in cis_coeff_dic.items()},
                        col_name='poissonb_se'
                    )
                else:
                    cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, cis_coeff_dic, col_name='poissonb')

        print(f'# Saving files to {results_folder}')
        os.makedirs(results_folder, exist_ok=True)

        if USE_INF_MODE:
            if FLAG_CORR:
                np.save(f'{results_folder}/cis_corr{file_suffix}.npy', cis_coeff)
            else:
                np.save(f'{results_folder}/cis_coeff{file_suffix}.npy', cis_coeff)
            np.save(f'{results_folder}/cis_link_ids{file_suffix}.npy', link_ids, allow_pickle=True)
            if ARRAY_IDX is None:
                cis_links_df.to_csv(f'{results_folder}/cis_links_df.csv')
        else:
            if FLAG_CORR:
                with open(f'{results_folder}/cis_corr_dic{file_suffix}.pkl', 'wb') as f:
                    pickle.dump(cis_corr_dic, f)
            else:
                with open(f'{results_folder}/cis_coeff_dic{file_suffix}.pkl', 'wb') as f:
                    pickle.dump(cis_coeff_dic, f)
            with open(f'{results_folder}/cis_idx_dic{file_suffix}.pkl', 'wb') as f:
                pickle.dump(cis_idx_dic, f)
            with open(f'{results_folder}/cis_links_dic{file_suffix}.pkl', 'wb') as f:
                pickle.dump(cis_links_dic, f)
            if ARRAY_IDX is None:
                cis_links_df.to_csv(f'{results_folder}/cis_links_df.csv')

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
                        peak_col='peak', layer='counts', genome_file=GENOME_FILE,
                        add_promoter_bin=False, genes_df=adata_rna.var,
                    )
            else:
                ctrl_peaks = ctar.method.create_ctrl_peaks(
                    adata_atac, type=atac_type, num_bins=atac_bins, b=n_ctrl,
                    peak_col='peak', layer='counts', genome_file=GENOME_FILE,
                    add_promoter_bin=False, genes_df=adata_rna.var,
                )
            
            # Get cis links indices
            cis_links_df = ctar.data_loader.attach_link_indices(cis_links_df, adata_atac, adata_rna)
            
            # Determine batch range
            if ARRAY_IDX is not None: 
                start_idx = ARRAY_IDX * BATCH_SIZE
                end_idx = min((ARRAY_IDX + 1) * BATCH_SIZE, len(cis_links_df))
                cis_links_subset = cis_links_df.iloc[start_idx:end_idx]
                file_suffix = f'_{start_idx}_{end_idx}'
            else: 
                cis_links_subset = cis_links_df
                file_suffix = ''
            link_ids = cis_links_subset.index.to_numpy()
            
            print(f'# Computing controls for {len(cis_links_subset)} links...')
            
            # Create control links dictionary
            atac_idx_arr = cis_links_subset['atac_idx'].to_numpy()
            rna_idx_arr = cis_links_subset['rna_idx'].to_numpy()
            ctrl_peak_sub = ctrl_peaks[atac_idx_arr, :n_ctrl]
            ctrl_links_array = np.column_stack([
                ctrl_peak_sub.ravel(),
                np.repeat(rna_idx_arr, n_ctrl),
            ])
            print(f'# ctrl_links_array.shape: {ctrl_links_array.shape}')

            # Chunk the control links array into batches
            ctrl_links_dic = ctar.parallel.chunk_links_dict(ctrl_links_array, dic_batch_size=DIC_BATCH_SIZE)
            rna_sparse = rna_norm if FLAG_CORR else adata_rna.layers['counts']
            atac_sparse = atac_norm if FLAG_CORR else adata_atac.layers['counts']

            n_links = len(cis_links_subset)
            n_rows = n_links * n_ctrl
            assert ctrl_links_array.shape[0] == n_rows, "ctrl_links_array and cis_links_subset shape mismatch"

            print(f'# ATAC and RNA dtype: {atac_sparse.dtype}, {rna_sparse.dtype}')

            if FLAG_CORR:
                print('# Starting control links Pearson correlation...')
                start_time = time.time()
                ctrl_corr_dic = ctar.parallel.multiprocess_pearson_corr(
                    links_dict=ctrl_links_dic,
                    atac_sparse=atac_sparse,
                    rna_sparse=rna_sparse,
                    batch_size=BATCH_SIZE,
                    n_workers=n_cores,
                    donor_labels=donor_labels,
                )
                print('# Control Pearson time = %0.2fs' % (time.time() - start_time))
                ctrl_coeff = np.concatenate(
                    [np.asarray(ctrl_corr_dic[k]) for k in sorted(ctrl_corr_dic)]
                ).astype(np.float32).reshape(n_links, n_ctrl)
            else:
                print('# Starting control links IRLS...')
                start_time = time.time()

                if covar_mat is not None:
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
                        flag_se=FLAG_SE,
                        flag_ll=FLAG_LL
                    )
                print('# Control links IRLS time = %0.2fs' % (time.time() - start_time))

                ctrl_coeff = np.concatenate(
                    [np.asarray(ctrl_coeff_dic[k]) for k in sorted(ctrl_coeff_dic)]
                ).astype(np.float32).reshape(n_links, n_ctrl, 2)

        else:
            
            # Standard binned mode
            if ARRAY_IDX is None: 

                print(f'# Computing control pairs for {BIN_CONFIG}...')
                adata_rna.var = ctar.data_loader.get_gene_coords(adata_rna.var, add_tss=False)
                ctrl_links_dic, atac_bins_df, rna_bins_df = ctar.method.create_ctrl_pairs(
                    adata_atac, adata_rna, atac_bins=[n_atac_mean,n_atac_gc], rna_bins=[n_rna_mean,n_rna_var],
                    atac_type=atac_type, rna_type=rna_type, b=n_ctrl,
                    atac_layer='counts', rna_layer='counts', genome_file=GENOME_FILE,
                    n_length_bins=N_LENGTH_BINS,
                )

                cis_links_df = ctar.data_loader.combine_peak_gene_bins(cis_links_df, atac_bins_df, rna_bins_df,
                    atac_type=atac_type, rna_type=rna_type,
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

                file_suffix = f'_{ARRAY_IDX}'

            rna_sparse = rna_norm if FLAG_CORR else adata_rna.layers['counts']
            atac_sparse = atac_norm if FLAG_CORR else adata_atac.layers['counts']

            print('# Running %d controls...' % (len(list(ctrl_links_dic.keys()))))
            print(f'# ATAC and RNA dtype: {atac_sparse.dtype}, {rna_sparse.dtype}')

            if FLAG_CORR:
                print('# Starting control links Pearson correlation...')
                start_time = time.time()
                ctrl_coeff_dic = ctar.parallel.multiprocess_pearson_corr(
                    links_dict=ctrl_links_dic,
                    atac_sparse=atac_sparse,
                    rna_sparse=rna_sparse,
                    batch_size=BATCH_SIZE,
                    n_workers=n_cores,
                    donor_labels=donor_labels,
                )
                print('# Control Pearson time = %0.2fs' % (time.time() - start_time))
            else:
                print('# Starting control links IRLS...')
                start_time = time.time()

                if covar_mat is not None:
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
                        flag_se=FLAG_SE,
                        flag_ll=FLAG_LL
                    )
                print('# Control links IRLS time = %0.2fs' % (time.time() - start_time))

        print(f'# Saving files to {results_folder}')
        os.makedirs(results_folder, exist_ok=True)

        if USE_INF_MODE:
            np.save(f'{results_folder}/ctrl_link_ids{file_suffix}.npy', link_ids, allow_pickle=True)
            if FLAG_CORR:
                np.save(f'{results_folder}/ctrl_corr{file_suffix}.npy', ctrl_coeff)
            else:
                np.save(f'{results_folder}/ctrl_coeff{file_suffix}.npy', ctrl_coeff)
        else:
            if FLAG_CORR:
                with open(f'{results_folder}/ctrl_corr_dic{file_suffix}.pkl', 'wb') as f:
                    pickle.dump(ctrl_coeff_dic, f)
                print(f'# Saved file(s) ./ctrl_corr_dic{file_suffix}.pkl')
            else:
                with open(f'{results_folder}/ctrl_coeff_dic{file_suffix}.pkl', 'wb') as f:
                    pickle.dump(ctrl_coeff_dic, f)
                print(f'# Saved file(s) ./ctrl_coeff_dic{file_suffix}.pkl')

            if ARRAY_IDX is None:
                with open(f'{results_folder}/ctrl_links_dic.pkl', 'wb') as f:
                    pickle.dump(ctrl_links_dic, f)
                cis_links_df.to_csv(f'{results_folder}/cis_links_df.csv')
                print(f'# Saved file(s) ./ctrl_links_dic.pkl, ./cis_links_df.csv')


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
                cis_links_df[f'{BIN_CONFIG}_zpval'] = ctar.method.zscore_pval(ctrl_beta, cis_coeff[:, 1])[0]

                # Studentized p-values
                cis_studentized = np.clip(cis_coeff[:, 1] / cis_coeff[:, 0], -MAX_VAL, MAX_VAL)
                ctrl_studentized = np.clip(ctrl_beta / ctrl_se, -MAX_VAL, MAX_VAL)
                
                cis_links_df[f'{BIN_CONFIG}_mcpval_z'] = ctar.method.initial_mcpval(ctrl_studentized, cis_studentized)
                cis_links_df[f'{BIN_CONFIG}_ppval_z'] = ctar.method.pooled_mcpval(ctrl_studentized, cis_studentized)
                cis_links_df[f'{BIN_CONFIG}_zpval_z'] = ctar.method.zscore_pval(ctrl_studentized, cis_studentized)[0]

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
                cis_links_df[f'{BIN_CONFIG}_zpval'] = ctar.method.zscore_pval(ctrl_coeff, cis_beta)[0]

                stat_col_inf = 'corr' if FLAG_CORR else 'poissonb'
                if stat_col_inf not in cis_links_df.columns:
                    cis_links_df[stat_col_inf] = cis_beta

            else:
                raise ValueError(f"Unexpected ctrl_coeff shape: {ctrl_coeff.shape}")

            # FDR (inf.inf mode)
            print("# Computing empirical FDR (inf.inf mode)...")
            if cis_coeff.ndim == 2:
                cis_fdr_stat = cis_coeff[:, 1]
                ctrl_fdr_stat = ctrl_coeff[:, :, 1].flatten() if ctrl_coeff.ndim == 3 else ctrl_coeff.flatten()
            else:
                cis_fdr_stat = cis_coeff
                ctrl_fdr_stat = ctrl_coeff.flatten()
            q_values, _, _, _ = ctar.method.empirical_fdr(cis_fdr_stat, ctrl_fdr_stat)
            cis_links_df['q_value'] = q_values

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
                mcpval_dic, ppval_dic, zpval_dic = ctar.method.binned_mcpval(
                    {key: value[:, 1] for key, value in cis_coeff_dic.items()},
                    {key: value[:, 1] for key, value in ctrl_coeff_dic.items()},
                    b=n_ctrl,
                )

                cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, mcpval_dic, col_name=f'{BIN_CONFIG}_mcpval')
                cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, ppval_dic, col_name=f'{BIN_CONFIG}_ppval')
                cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, zpval_dic, col_name=f'{BIN_CONFIG}_zpval')
                cis_links_df = ctar.data_loader.map_dic_to_df(
                    cis_links_df, cis_idx_dic,
                    {k: v[:, 1] for k, v in cis_coeff_dic.items()},
                    col_name='poissonb'
                )
                cis_links_df = ctar.data_loader.map_dic_to_df(
                    cis_links_df, cis_idx_dic,
                    {k: v[:, 0] for k, v in cis_coeff_dic.items()},
                    col_name='poissonb_se'
                )

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

                mcpval_dic_z, ppval_dic_z, zpval_dic_z  = ctar.method.binned_mcpval(
                    cis_coeff_dic_z,
                    ctrl_coeff_dic_z,
                    b=n_ctrl,
                )

                cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, mcpval_dic_z, col_name=f'{BIN_CONFIG}_mcpval_z')
                cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, ppval_dic_z, col_name=f'{BIN_CONFIG}_ppval_z')
                cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, zpval_dic_z, col_name=f'{BIN_CONFIG}_zpval_z')

            else:
                print("# No SE found - computing only regular p-values")

                mcpval_dic, ppval_dic, zpval_dic = ctar.method.binned_mcpval(
                    cis_coeff_dic,
                    ctrl_coeff_dic,
                    b=n_ctrl,
                )

                cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, mcpval_dic, col_name=f'{BIN_CONFIG}_mcpval')
                cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, ppval_dic, col_name=f'{BIN_CONFIG}_ppval')
                cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, zpval_dic, col_name=f'{BIN_CONFIG}_zpval')

                stat_col_bin = 'corr' if FLAG_CORR else 'poissonb'
                if stat_col_bin not in cis_links_df.columns:
                    cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, cis_coeff_dic, col_name=stat_col_bin)

            # FDR
            print("# Computing stratified empirical FDR...")
            stat_col_bin = 'corr' if FLAG_CORR else 'poissonb'
            first_key = list(ctrl_coeff_dic.keys())[0]
            if ctrl_coeff_dic[first_key].ndim == 2 and ctrl_coeff_dic[first_key].shape[1] == 2:
                ctrl_fdr_dic = {k: v[:, 1] for k, v in ctrl_coeff_dic.items()}
            else:
                ctrl_fdr_dic = ctrl_coeff_dic
            cis_links_df = ctar.method.binned_empirical_fdr(
                cis_links_df, ctrl_fdr_dic,
                stat_col=stat_col_bin,
                ctrl_bin_col=combined_bin_col,
                pi0=None,
                pval_col=f'{BIN_CONFIG}_mcpval',
                fdr_bin_col=None,
                qval_col='q_value',
            )

            if FDR_BIN_TYPE == 'dist_rmean_rvar':
                # Fixed TSS-distance bins, then nested rna_mean / rna_var quantile bins.
                gene_df = None
                if TSS_FILE is not None:
                    print(f"# Loading TSS annotations from {TSS_FILE}...")
                    gene_df = ctar.gentruth.load_gene_tss_from_gencode(TSS_FILE)
                    cis_links_df = cis_links_df.drop(columns=['abs_distance'], errors='ignore')
                else:
                    assert 'abs_distance' in cis_links_df.columns, (
                        "--fdr_bin_type dist_rmean_rvar needs --tss_file (GENCODE GTF) "
                        "or an 'abs_distance' column in cis_links_df."
                    )

                rna_adata = None
                if not {'rna_mean', 'rna_var'}.issubset(cis_links_df.columns):
                    assert MULTIOME_FILE is not None, (
                        "--fdr_bin_type dist_rmean_rvar needs 'rna_mean'/'rna_var' in cis_links_df "
                        "or --multiome_file to compute them."
                    )
                    print(f"# Computing RNA moments from --multiome_file (layer={RNA_LAYER}).")
                    rna_adata = adata_rna

                dist_bin_kwargs = {} if FDR_DIST_BINS is None else {'dist_bins': FDR_DIST_BINS}
                cis_links_df = ctar.method.dist_rmean_rvar_sfdr(
                    cis_links_df, ctrl_fdr_dic,
                    gene_df=gene_df,
                    gene_col='gene',
                    rna_adata=rna_adata,
                    rna_layer=RNA_LAYER,
                    rna_gene_col='gene',
                    min_bin_count=FDR_MIN_BIN_COUNT,
                    n_bins=N_FDR_RNA_BINS,
                    stat_col=stat_col_bin,
                    ctrl_bin_col=combined_bin_col,
                    pi0=None,
                    pval_col=f'{BIN_CONFIG}_mcpval',
                    qval_col='q_value_fixed_dist_rmean_rvar',
                    **dist_bin_kwargs,
                )
                print('# Added q_value_fixed_dist_rmean_rvar.')

        # Two-stage specificity FDR (association q < --spec_filter_fdr), if a specificity p-value is already in the table
        cis_links_df = ctar.method.add_filtered_spec_qval(cis_links_df, f'{BIN_CONFIG}_mcpval_{SPEC_TAG}',
                                              f'{BIN_CONFIG}_qval_{SPEC_TAG}_filt', SPEC_STAT, SPEC_FILTER_FDR, SPEC_FILTER_QCOL)

        # Save updated dataframe
        output_file = os.path.join(RESULTS_PATH, 'cis_links_df.csv')
        print(f"# Saving updated cis_links_df to {output_file}")
        cis_links_df.to_csv(output_file)
        print(f"# Added columns: {[col for col in cis_links_df.columns if BIN_CONFIG in col or col.startswith('q_value')]}")


    if JOB == "compute_spec":

        print("# Running --job compute_spec")

        # Index links on a copy so rows dropped for missing features stay in the saved table
        links_df = ctar.data_loader.attach_link_indices(cis_links_df.copy(), adata_atac, adata_rna)
        links_pairs = links_df[['atac_idx', 'rna_idx']].to_numpy()

        print(f"# Computing {SPEC_STAT} link specificity ({SPEC_NORM} gene/peak cell-type tables)...")
        rna_table, atac_table = ctar.celltype_specificity.celltype_tables(
            adata_rna, adata_atac, CELLTYPE_COL, method=SPEC_NORM)
        coexp_ctx = (ctar.celltype_specificity.coexp_context(adata_rna, adata_atac, CELLTYPE_COL)
                     if SPEC_STAT == 'coexp_log1p' else None)

        if USE_INF_MODE:
            ctrl_peaks_file = os.path.join(RESULTS_PATH, 'ctrl_peaks.npy')
            if not os.path.exists(ctrl_peaks_file):
                raise FileNotFoundError(f"ctrl_peaks.npy not found at {ctrl_peaks_file}")
            ctrl_peaks = np.load(ctrl_peaks_file)

            ctrl_links_array = np.column_stack([
                ctrl_peaks[links_pairs[:, 0], :n_ctrl].ravel(),
                np.repeat(links_pairs[:, 1], n_ctrl),
            ])
            cis_spec, cis_spec_dom = ctar.parallel.multiprocess_link_specificity(
                links_pairs, SPEC_STAT, rna_table, atac_table, coexp_ctx, batch_size=BATCH_SIZE, n_workers=n_cores)
            cis_spec = cis_spec.astype(np.float32)
            ctrl_spec = ctar.parallel.multiprocess_link_specificity(
                ctrl_links_array, SPEC_STAT, rna_table, atac_table, coexp_ctx, batch_size=BATCH_SIZE, n_workers=n_cores,
            )[0].astype(np.float32).reshape(len(cis_spec), n_ctrl)

            links_df[LINK_SPEC_COL] = cis_spec
            links_df[f'{BIN_CONFIG}_mcpval_{SPEC_TAG}'] = ctar.method.initial_mcpval(ctrl_spec, cis_spec)
            links_df[f'{BIN_CONFIG}_ppval_{SPEC_TAG}'] = ctar.method.pooled_mcpval(ctrl_spec, cis_spec)
            links_df[f'{BIN_CONFIG}_zpval_{SPEC_TAG}'] = ctar.method.zscore_pval(ctrl_spec, cis_spec)[0]

            os.makedirs(results_folder, exist_ok=True)
            np.save(f'{results_folder}/cis_{SPEC_TAG}.npy', cis_spec)
            np.save(f'{results_folder}/ctrl_{SPEC_TAG}.npy', ctrl_spec)

            # Marginal specificity
            gene_spec, peak_spec = ctar.celltype_specificity.marginal_specificity(rna_table, atac_table)
            cis_peak_spec = peak_spec[links_pairs[:, 0]].astype(np.float32)
            ctrl_peak_spec = peak_spec[ctrl_links_array[:, 0]].astype(np.float32).reshape(len(cis_spec), n_ctrl)
            links_df[f'gene_specificity{SPEC_SUFFIX}'] = gene_spec[links_pairs[:, 1]].astype(np.float32)
            links_df[f'peak_specificity{SPEC_SUFFIX}'] = cis_peak_spec
            links_df[f'{BIN_CONFIG}_mcpval_peak_spec{SPEC_SUFFIX}'] = ctar.method.initial_mcpval(ctrl_peak_spec, cis_peak_spec)
            np.save(f'{results_folder}/gene_spec{SPEC_SUFFIX}.npy', gene_spec)
            np.save(f'{results_folder}/peak_spec{SPEC_SUFFIX}.npy', peak_spec)

            # Dominant cell type of the gene, the peak and the joint profile
            gene_dominant, peak_dominant, link_dominant, dominant_concordant = ctar.celltype_specificity.dominant_celltypes(
                rna_table, atac_table, links_pairs[:, 1], links_pairs[:, 0])
            links_df[f'gene_dominant_subtype{SPEC_SUFFIX}'] = gene_dominant
            links_df[f'peak_dominant_subtype{SPEC_SUFFIX}'] = peak_dominant
            links_df[f'link_dominant_subtype{SPEC_SUFFIX}'] = link_dominant
            if cis_spec_dom is not None:
                links_df['link_dominant_subtype_coexp'] = np.where(
                    np.asarray(cis_spec_dom) >= 0, coexp_ctx['celltypes'][np.maximum(cis_spec_dom, 0)], None)
            links_df[f'dominant_subtype_concordant{SPEC_SUFFIX}'] = dominant_concordant

        else:
            cis_idx_file = os.path.join(RESULTS_PATH, 'cis_idx_dic.pkl')
            ctrl_links_file = os.path.join(RESULTS_PATH, 'ctrl_links_dic.pkl')
            for fname in (cis_idx_file, ctrl_links_file):
                if not os.path.exists(fname):
                    raise FileNotFoundError(f"{os.path.basename(fname)} not found at {fname}")
            with open(cis_idx_file, 'rb') as f:
                cis_idx_dic = pickle.load(f)
            with open(ctrl_links_file, 'rb') as f:
                ctrl_links_dic = pickle.load(f)

            assert len(links_df) == len(cis_links_df), "Links dropped for missing features; cis_idx_dic positions would misalign"
            cis_links_dic = {key: links_pairs[np.asarray(idx)] for key, idx in cis_idx_dic.items()}
            cis_spec_dic, cis_spec_dom_dic = ctar.parallel.multiprocess_link_specificity(
                cis_links_dic, SPEC_STAT, rna_table, atac_table, coexp_ctx, batch_size=BATCH_SIZE, n_workers=n_cores)
            cis_spec_dom = None if cis_spec_dom_dic is None else ctar.data_loader.map_dic_to_df(
                        links_df[[]].copy(), cis_idx_dic, cis_spec_dom_dic, col_name='code')['code'].fillna(-1).astype(int).to_numpy()
            ctrl_spec_dic = ctar.parallel.multiprocess_link_specificity(
                ctrl_links_dic, SPEC_STAT, rna_table, atac_table, coexp_ctx, batch_size=BATCH_SIZE, n_workers=n_cores)[0]
            mcpval_spec_dic, ppval_spec_dic, zpval_spec_dic = ctar.method.binned_mcpval(cis_spec_dic, ctrl_spec_dic, b=n_ctrl)

            links_df = ctar.data_loader.map_dic_to_df(links_df, cis_idx_dic, cis_spec_dic, col_name=LINK_SPEC_COL)
            links_df = ctar.data_loader.map_dic_to_df(links_df, cis_idx_dic, mcpval_spec_dic, col_name=f'{BIN_CONFIG}_mcpval_{SPEC_TAG}')
            links_df = ctar.data_loader.map_dic_to_df(links_df, cis_idx_dic, ppval_spec_dic, col_name=f'{BIN_CONFIG}_ppval_{SPEC_TAG}')
            links_df = ctar.data_loader.map_dic_to_df(links_df, cis_idx_dic, zpval_spec_dic, col_name=f'{BIN_CONFIG}_zpval_{SPEC_TAG}')

            os.makedirs(results_folder, exist_ok=True)
            with open(f'{results_folder}/cis_{SPEC_TAG}_dic.pkl', 'wb') as f:
                pickle.dump(cis_spec_dic, f)
            with open(f'{results_folder}/ctrl_{SPEC_TAG}_dic.pkl', 'wb') as f:
                pickle.dump(ctrl_spec_dic, f)

            # Marginal specificity
            gene_spec, peak_spec = ctar.celltype_specificity.marginal_specificity(rna_table, atac_table)
            for feature, feature_spec, pair_column in (('gene', gene_spec, 1), ('peak', peak_spec, 0)):
                cis_feature_dic = {key: feature_spec[pairs[:, pair_column]] for key, pairs in cis_links_dic.items()}
                ctrl_feature_dic = {key: feature_spec[pairs[:, pair_column]] for key, pairs in ctrl_links_dic.items()}
                mcpval_feature_dic, _, _ = ctar.method.binned_mcpval(cis_feature_dic, ctrl_feature_dic, b=n_ctrl)
                links_df = ctar.data_loader.map_dic_to_df(links_df, cis_idx_dic, cis_feature_dic, col_name=f'{feature}_specificity{SPEC_SUFFIX}')
                links_df = ctar.data_loader.map_dic_to_df(links_df, cis_idx_dic, mcpval_feature_dic, col_name=f'{BIN_CONFIG}_mcpval_{feature}_spec{SPEC_SUFFIX}')
            np.save(f'{results_folder}/gene_spec{SPEC_SUFFIX}.npy', gene_spec)
            np.save(f'{results_folder}/peak_spec{SPEC_SUFFIX}.npy', peak_spec)

            # Dominant cell type of the gene, the peak and the joint profile
            gene_dominant, peak_dominant, link_dominant, dominant_concordant = ctar.celltype_specificity.dominant_celltypes(
                rna_table, atac_table, links_pairs[:, 1], links_pairs[:, 0])
            links_df[f'gene_dominant_subtype{SPEC_SUFFIX}'] = gene_dominant
            links_df[f'peak_dominant_subtype{SPEC_SUFFIX}'] = peak_dominant
            links_df[f'link_dominant_subtype{SPEC_SUFFIX}'] = link_dominant
            if cis_spec_dom is not None:
                links_df['link_dominant_subtype_coexp'] = np.where(
                    np.asarray(cis_spec_dom) >= 0, coexp_ctx['celltypes'][np.maximum(cis_spec_dom, 0)], None)
            links_df[f'dominant_subtype_concordant{SPEC_SUFFIX}'] = dominant_concordant

        spec_cols = [LINK_SPEC_COL] + [f'{BIN_CONFIG}_{p}pval_{SPEC_TAG}' for p in ('mc', 'p', 'z')]
        spec_cols += [f'gene_specificity{SPEC_SUFFIX}', f'peak_specificity{SPEC_SUFFIX}', f'{BIN_CONFIG}_mcpval_peak_spec{SPEC_SUFFIX}']
        if not USE_INF_MODE:
            spec_cols.append(f'{BIN_CONFIG}_mcpval_gene_spec{SPEC_SUFFIX}')
        spec_cols.append(f'dominant_subtype_concordant{SPEC_SUFFIX}')
        dominant_cols = [f'{kind}_dominant_subtype{SPEC_SUFFIX}' for kind in ('gene', 'peak', 'link')]
        if SPEC_STAT == 'coexp_log1p':
            dominant_cols.append('link_dominant_subtype_coexp')
        for col in spec_cols:
            cis_links_df[col] = np.nan
            cis_links_df.loc[links_df.index, col] = links_df[col]
        for col in dominant_cols:
            cis_links_df[col] = pd.Series(None, index=cis_links_df.index, dtype=object)
            cis_links_df.loc[links_df.index, col] = links_df[col]
        filtered_col = f'{BIN_CONFIG}_qval_{SPEC_TAG}_filt'
        cis_links_df = ctar.method.add_filtered_spec_qval(cis_links_df, f'{BIN_CONFIG}_mcpval_{SPEC_TAG}', filtered_col,
                                              SPEC_STAT, SPEC_FILTER_FDR, SPEC_FILTER_QCOL)

        output_file = os.path.join(RESULTS_PATH, 'cis_links_df.csv')
        print(f"# Saving updated cis_links_df to {output_file}")
        cis_links_df.to_csv(output_file)
        print(f"# Added columns: {spec_cols + dominant_cols + ([filtered_col] if filtered_col in cis_links_df else [])}")


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

    parser.add_argument("--job", type=str, required=True, help="compute_ctar, compute_cis_only, compute_ctrl_only, generate_links, generate_controls, compute_pval, compute_spec")
    parser.add_argument("--job_id", type=str, required=False, default='1', help='job id')
    parser.add_argument("--multiome_file", type=str, required=False, default=None)
    parser.add_argument("--batch_size", type=str, required=False, default='100')
    parser.add_argument("--dic_batch_size", type=str, required=False, default='512', help="Rows per links_dict chunk in inf.inf mode (generate_links/compute_cis_only/compute_ctrl_only).")
    parser.add_argument("--links_file", type=str, required=False, default=None)
    parser.add_argument("--target_path", type=str, required=False, default=None)
    parser.add_argument("--results_path", type=str, required=False, default=None)
    parser.add_argument("--flag_ll", type=str, required=False, default='False')
    parser.add_argument("--flag_se", type=str, required=False, default='False')
    parser.add_argument("--flag_corr", type=str, required=False, default='False')
    parser.add_argument("--flag_spec", type=str, required=False, default='False', help="Compute joint entropy specificity and its monte carlo p-values (compute_ctar). Requires --celltype_col.")
    parser.add_argument("--spec_stat", type=str, required=False, default='coexp_log1p', choices=['coexp_log1p', 'joint'], help="Link specificity statistic: 'coexp_log1p' (default; pseudobulk of co-expression, the cell-type mean of the per-cell product log1p(normalized gene) * log1p(normalized peak), normalize_total to the median depth as in --flag_corr) or 'joint' (entropy of the product of the --spec_norm gene and peak cell-type tables).")
    parser.add_argument("--spec_filter_fdr", type=float, required=False, default=0.2, help="Association FDR threshold for the two-stage specificity FDR column {binning_config}_qval_{spec tag}_filt: BH over the specificity mcpval among links with association q < this value, NaN elsewhere (default 0.2).")
    parser.add_argument("--spec_filter_qcol", type=str, required=False, default=None, help="Association q-value column for --spec_filter_fdr (default: q_value_fixed_dist_rmean_rvar if present, else q_value).")
    parser.add_argument("--spec_norm", type=str, required=False, default='pseudobulk', choices=['pseudobulk', 'mean'], help="Cell-type tables for joint entropy specificity: 'pseudobulk' (summed counts / cell-type total + library-size-scaled pseudocount) or 'mean' (normalize_total per-cell-type means).")
    parser.add_argument("--celltype_col", type=str, required=False, default=None, help="obs column with cell-type labels for joint entropy specificity (--flag_spec / --job compute_spec); with --celltype_file, the label column in that file.")
    parser.add_argument("--celltype_file", type=str, required=False, default=None, help="Optional CSV (first column = cell barcode) holding the --celltype_col labels, for multiome files without an obs label column. Cells missing from the file are dropped from specificity.")
    parser.add_argument("--flag_tfidf", type=str, required=False, default='False')
    parser.add_argument("--flag_bin", type=str, required=False, default='False', help="Binarize ATAC counts (0/1) instead of log1p or TFIDF. Mutually exclusive with --flag_tfidf.")
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
    parser.add_argument("--binning_type", type=str, required=False, default='mean_var', help='mean_var, cholesky, mean_cv_bio, mean_fano_bio, mean_fano_cvbio')
    parser.add_argument("--pybedtools_path", type=str, required=False, default=None)
    parser.add_argument("--array_idx", type=str, required=False, default=None)
    parser.add_argument("--n_cores", type=str, required=False, default=None)
    parser.add_argument("--random_seed", type=str, required=False, default=None, help="Random seed for reproducible control link generation")
    parser.add_argument("--donor_col", type=str, required=False, default=None, help="obs column for donor centering before Pearson correlation (requires --flag_corr True)")
    parser.add_argument("--n_length_bins", type=str, required=False, default='1', help="Number of peak-length quantile bins added to ATAC binning (1 = off, 5 = recommended)")
    parser.add_argument("--fdr_bin_type", type=str, required=False, default=None, help="FDR stratification type for compute_pval: 'dist_rmean_rvar' (fixed distance bins x nested rna_mean/rna_var bins).")
    parser.add_argument("--tss_file", type=str, required=False, default=None, help="GENCODE GTF(.gz) used to compute TSS distances when --fdr_bin_type dist_rmean_rvar is used and abs_distance is missing.")
    parser.add_argument("--rna_layer", type=str, required=False, default='counts', help="AnnData layer used to recompute rna_mean/rna_var for FDR binning. Empty string uses .X.")
    parser.add_argument("--n_fdr_rna_bins", type=str, required=False, default='4', help="Number of nested quantile bins for rna_mean and rna_var when --fdr_bin_type dist_rmean_rvar is used.")
    parser.add_argument("--fdr_min_bin_count", type=str, required=False, default='25000', help="Minimum pairs per fixed distance bin when --fdr_bin_type dist_rmean_rvar is used.")
    parser.add_argument("--fdr_dist_bins", type=str, required=False, default=None, help="Comma-separated fixed distance bin edges (e.g. '-inf,1000,5000,inf') for --fdr_bin_type dist_rmean_rvar. Defaults to the function's built-in edges.")

    args = parser.parse_args()
    main(args)
