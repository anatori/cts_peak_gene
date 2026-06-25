import sys
import os
import argparse
import pickle
import math
import gc
import warnings
from pathlib import Path
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import scipy as sp
from scipy import sparse

import scanpy as sc
import anndata as ad
import muon as mu

from dask import delayed, compute
from dask.diagnostics import ProgressBar

from scdrs.pp import category2dummy, reg_out

import ctar.method
from ctar.method import pearson_corr_sparse
from ctar.parallel import batched_iterable, preprocess_batch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Covariate-corrected partial Pearson correlation for cis peak-gene links.'
    )
    parser.add_argument('--multiome_file', required=True)
    parser.add_argument('--ctar_output_dir', required=True)
    parser.add_argument('--cov_cols', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--n_workers', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=50)
    return parser.parse_args()


def build_cov_matrix(adata_rna, cov_cols):
    """
    Build (n_cells, n_cov) covariate matrix from adata.obs columns.
    Categorical columns are dummy-encoded using scDRS category2dummy.
    Fills missing values with column means. Adds a constant term if not already spanned.
    """
    n_cells = adata_rna.n_obs
    df_cov = adata_rna.obs[cov_cols].copy()
    df_cov = category2dummy(df_cov, verbose=True)
    df_cov.fillna(df_cov.mean(), inplace=True)
    v_resid = reg_out(np.ones(n_cells), df_cov.values)
    if (v_resid ** 2).mean() > 0.01:
        df_cov['CONST'] = 1
    return df_cov.values.astype(np.float64)


def compute_beta(X_sparse, cov_mat):
    """
    Compute OLS regression coefficients: beta = (C^T C)^{-1} C^T X.
    Uses sparse matmul.

    Returns
    -------
    beta : np.ndarray, shape (n_cov, n_features)
    """
    n = X_sparse.shape[0]
    CTC = cov_mat.T @ cov_mat / n # (n_cov, n_cov)
    CTX = X_sparse.T.dot(cov_mat).T / n # (n_cov, n_features)
    return np.linalg.solve(CTC, CTX)


def _pearson_corr_cov_worker(links_re, atac_sub, rna_sub,
                              cov_mat, beta_atac_sub, beta_rna_sub):
    """
    Dask worker: extract needed columns from batch submatrices, apply covariate
    correction in-place, return per-link Pearson correlations.

    Only batch-sized dense slices live in memory — the correction is applied
    lazily here rather than materializing a full corrected matrix upfront.

    Parameters
    ----------
    links_re : (n_links, 2) int array with local (remapped) indices
    atac_sub : (n_cells, n_unique_atac) sparse — unique ATAC cols for this batch
    rna_sub  : (n_cells, n_unique_rna) sparse
    cov_mat  : (n_cells, n_cov) dense
    beta_atac_sub : (n_cov, n_unique_atac) — OLS betas for this batch's peaks
    beta_rna_sub  : (n_cov, n_unique_rna)
    """
    atac = atac_sub[:, links_re[:, 0]].toarray() # (n_cells, n_links)
    rna  = rna_sub[:,  links_re[:, 1]].toarray()
    atac -= cov_mat @ beta_atac_sub[:, links_re[:, 0]] # subtract fitted values
    rna  -= cov_mat @ beta_rna_sub[:,  links_re[:, 1]]
    return pearson_corr_sparse(rna, atac).flatten() # (n_links,)


def multiprocess_pearson_corr_cov(
    links_dict,
    atac_sparse,
    rna_sparse,
    cov_mat,
    beta_atac,
    beta_rna,
    batch_size=50,
    n_workers=None,
):
    """
    Parallel covariate-corrected Pearson correlation using Dask delayed.

    Same as ctar.parallel.multiprocess_pearson_corr but adds covariate
    correction inside each worker.

    Parameters
    ----------
    links_dict  : dict, bin_key -> (n_links_in_bin, 2) int array [atac_idx, rna_idx]
    atac_sparse : (n_cells, n_peaks) sparse
    rna_sparse  : (n_cells, n_genes) sparse
    cov_mat     : (n_cells, n_cov) dense
    beta_atac   : (n_cov, n_peaks) dense  — precomputed OLS betas for ATAC
    beta_rna    : (n_cov, n_genes) dense  — precomputed OLS betas for RNA

    Returns
    -------
    dict, bin_key -> (n_links_in_bin,) float32 correlation array
    """
    if n_workers is None:
        n_workers = max(cpu_count() - 1, 1)

    results_dict = {}
    n_total = math.ceil(len(links_dict) / batch_size)

    with ProgressBar():
        for batch_idx, batch in enumerate(
            batched_iterable(links_dict.items(), batch_size), start=1
        ):
            print(f'# Pearson cov batch {batch_idx} / {n_total}', flush=True)
            tasks, keys = [], []

            for bin_key, link_array in batch:
                # preprocess_batch extracts unique-column submatrices and remaps indices
                links_re, atac_sub, rna_sub = preprocess_batch(
                    link_array, atac_sparse, rna_sparse
                )

                # Pull only the beta columns needed for this batch's unique peaks/genes
                needed_atac = np.unique(link_array[:, 0])
                needed_rna  = np.unique(link_array[:, 1])
                beta_atac_sub = beta_atac[:, needed_atac] # (n_cov, n_unique_atac)
                beta_rna_sub  = beta_rna[:,  needed_rna] # (n_cov, n_unique_rna)

                tasks.append(delayed(_pearson_corr_cov_worker)(
                    links_re, atac_sub, rna_sub,
                    cov_mat, beta_atac_sub, beta_rna_sub,
                ))
                keys.append(bin_key)
                del atac_sub, rna_sub, links_re, beta_atac_sub, beta_rna_sub

            results = compute(*tasks, num_workers=n_workers)
            for k, r in zip(keys, results):
                results_dict[k] = r
            del tasks, results
            gc.collect()

    return results_dict


def main():
    args = parse_args()
    cov_cols = [c.strip() for c in args.cov_cols.split(',')]
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    ctar_dir = Path(args.ctar_output_dir)
    n_workers = args.n_workers if args.n_workers is not None else max(cpu_count() - 1, 1)

    # 1. Load data
    print('# Loading --multiome_file')
    if args.multiome_file.endswith('.h5mu'):
        mdata      = mu.read(args.multiome_file)
        adata_atac = mdata.mod['atac']
        adata_rna  = mdata.mod['rna']
    else:
        adata_full = ad.read_h5ad(args.multiome_file)
        adata_atac = adata_full[:, adata_full.var.feature_types == 'ATAC'].copy()
        adata_rna  = adata_full[:, adata_full.var.feature_types == 'GEX'].copy()

    print(f'# RNA: {adata_rna.shape},  ATAC: {adata_atac.shape}')

    if adata_atac.var_names.str.match(r'^[^-]+-\d+-\d+$').any():
        adata_atac.var_names = adata_atac.var_names.astype(str).str.replace(
            r'\b([^-]+)-(\d+)-(\d+)\b', r'\1:\2-\3', regex=True
        )

    adata_rna.var.index.name  = ''
    adata_atac.var.index.name = ''
    adata_rna.var['gene']  = adata_rna.var.index
    adata_atac.var['peak'] = adata_atac.var.index

    try:
        _ = adata_rna.layers['counts']
        _ = adata_atac.layers['counts']
    except KeyError:
        print('# Assuming counts are in .X')
        adata_rna.layers['counts']  = adata_rna.X
        adata_atac.layers['counts'] = adata_atac.X

    if not adata_rna.var.gene.str.startswith('ENSG').all():
        if 'gene_id' in adata_rna.var.columns:
            adata_rna.var['gene'] = adata_rna.var.gene_id
            adata_rna.var.index   = adata_rna.var.gene_id
            print('# Using gene_id for RNA genes')
        elif 'gene_ids' in adata_rna.var.columns:
            adata_rna.var['gene'] = adata_rna.var.gene_ids
            adata_rna.var.index   = adata_rna.var.gene_ids
            print('# Using gene_ids for RNA genes')
        assert adata_rna.var.gene.str.startswith('ENSG').all(), \
            'Must use ENSEMBL IDs for genes'

    # 2. Normalize
    print('# Normalizing (normalize_total + log1p)...')
    adata_rna_tmp  = adata_rna.copy()
    adata_atac_tmp = adata_atac.copy()
    sc.pp.normalize_total(adata_rna_tmp,  layer='counts')
    sc.pp.log1p(adata_rna_tmp)
    sc.pp.normalize_total(adata_atac_tmp, layer='counts')
    sc.pp.log1p(adata_atac_tmp)
    rna_norm  = adata_rna_tmp.X
    atac_norm = adata_atac_tmp.X
    del adata_rna_tmp, adata_atac_tmp
    print(f'# rna_norm: {rna_norm.shape},  atac_norm: {atac_norm.shape}')

    # 3. Build covariate matrix
    print(f'# Building covariate matrix from: {cov_cols}')
    cov_mat = build_cov_matrix(adata_rna, cov_cols)
    print(f'# cov_mat: {cov_mat.shape}')

    # 4. Compute OLS betas
    print('# Computing OLS betas (RNA)...')
    beta_rna  = compute_beta(rna_norm,  cov_mat)       # (n_cov, n_genes)
    print('# Computing OLS betas (ATAC)...')
    beta_atac = compute_beta(atac_norm, cov_mat)       # (n_cov, n_peaks)
    print(f'# beta_rna: {beta_rna.shape},  beta_atac: {beta_atac.shape}')

    # 5. Load existing link structures
    print('# Loading ctar link structures...')
    cis_links_df   = pd.read_csv(ctar_dir / 'cis_links_df.csv',  index_col=0)
    cis_links_dic  = pickle.load(open(ctar_dir / 'cis_links_dic.pkl',  'rb'))
    cis_idx_dic    = pickle.load(open(ctar_dir / 'cis_idx_dic.pkl',    'rb'))
    ctrl_links_dic = pickle.load(open(ctar_dir / 'ctrl_links_dic.pkl', 'rb'))

    first_bin = next(iter(cis_idx_dic))
    n_ctrl = len(ctrl_links_dic[first_bin]) // len(cis_idx_dic[first_bin])
    print(f'# n_links={len(cis_links_df)}, n_ctrl={n_ctrl}')

    # 6. Cis correlations
    print('# Computing cis covariate-corrected correlations...')
    cis_corr_dic = multiprocess_pearson_corr_cov(
        cis_links_dic, atac_norm, rna_norm, cov_mat, beta_atac, beta_rna,
        batch_size=args.batch_size, n_workers=n_workers,
    )
    cis_coeff = np.empty(len(cis_links_df), dtype=np.float32)
    for bin_key, idxs in cis_idx_dic.items():
        cis_coeff[np.array(idxs)] = cis_corr_dic[bin_key]

    # 7. Control correlations
    print('# Computing control covariate-corrected correlations...')
    ctrl_corr_dic = multiprocess_pearson_corr_cov(
        ctrl_links_dic, atac_norm, rna_norm, cov_mat, beta_atac, beta_rna,
        batch_size=args.batch_size, n_workers=n_workers,
    )
    ctrl_coeff = np.empty((len(cis_links_df), n_ctrl), dtype=np.float32)
    for bin_key, idxs in cis_idx_dic.items():
        idxs = np.array(idxs)
        n_in_bin = len(idxs)
        ctrl_coeff[idxs] = ctrl_corr_dic[bin_key].reshape(n_in_bin, n_ctrl)

    # 8. P-values
    print('# Computing p-values...')
    cis_links_df['cov_corr']   = cis_coeff
    cis_links_df['cov_mcpval'] = ctar.method.initial_mcpval(ctrl_coeff, cis_coeff)
    cis_links_df['cov_zpval']  = ctar.method.zscore_pval(ctrl_coeff, cis_coeff)[0]
    cis_links_df['cov_ppval']  = ctar.method.pooled_mcpval(ctrl_coeff, cis_coeff)

    # 9. Save
    out_csv  = output_path / 'cis_links_cov_corr.csv'
    out_ctrl = output_path / 'ctrl_cov_corr.npy'
    cis_links_df.to_csv(out_csv)
    np.save(str(out_ctrl), ctrl_coeff)
    print(f'# Done. Saved: {out_csv}')
    print(f'#       Saved: {out_ctrl}')


if __name__ == '__main__':
    main()