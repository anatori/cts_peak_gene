import numpy as np
import pandas as pd 
import statsmodels.api as sm
import statsmodels.stats as stats
import scipy as sp
import scdrs
import math
import warnings
import random
from tqdm import tqdm
from Bio.SeqUtils import gc_fraction
import anndata as ad
import scanpy as sc
import muon as mu
import os
import re

from numba import njit, prange

######################### regression methods #########################


def pearson_corr_sparse(mat_X, mat_Y, var_filter=False):
    """Pairwise Pearson's correlation between columns in mat_X and mat_Y. Note that this will run
    much faster if given a csc_matrix rather than csr_matrix.

    Parameters
    ----------
    mat_X : np.ndarray
        First matrix of shape (N,M).
    mat_Y : np.ndarray
        Second matrix of shape (N,M).
        Assumes mat_X and mat_Y are aligned.
    var_filter : boolean
        Dictates whether to filter out columns with little to no variance.

    Returns
    -------
    mat_corr : np.ndarray
        Correlation array of shape (M,).
    no_var : np.ndarray
        A boolean mask where False represents excluded columns of mat_X and mat_Y of shape (N,M).
        
    """

    # Reshape
    if len(mat_X.shape) == 1:
        mat_X = mat_X.reshape([-1, 1])
    if len(mat_Y.shape) == 1:
        mat_Y = mat_Y.reshape([-1, 1])

    # Convert to sparse matrix if not already sparse
    if sp.sparse.issparse(mat_X) is False:
        mat_X = sp.sparse.csr_matrix(mat_X)
    if sp.sparse.issparse(mat_Y) is False:
        mat_Y = sp.sparse.csr_matrix(mat_Y)
    
    # Compute v_mean,v_var
    v_X_mean, v_X_var = scdrs.pp._get_mean_var(mat_X, axis=0)
    v_Y_mean, v_Y_var = scdrs.pp._get_mean_var(mat_Y, axis=0) 
    
    no_var = (v_X_var <= 1e-6) | (v_Y_var <= 1e-6)
    
    # This section removes columns with little to no variance.
    if var_filter and np.any(no_var):

        mat_X, mat_Y = mat_X[:,~no_var], mat_Y[:,~no_var]
        v_X_mean, v_X_var = v_X_mean[~no_var], v_X_var[~no_var]
        v_Y_mean, v_Y_var = v_Y_mean[~no_var], v_Y_var[~no_var]
        
    v_X_sd = np.sqrt(v_X_var.clip(1e-8))
    v_Y_sd = np.sqrt(v_Y_var.clip(1e-8))
    
    # Adjusted for column pairwise correlation only
    mat_corr = mat_X.multiply(mat_Y).mean(axis=0)
    mat_corr = mat_corr - v_X_mean * v_Y_mean
    mat_corr = mat_corr / v_X_sd / v_Y_sd

    mat_corr = np.array(mat_corr, dtype=np.float32)

    if (mat_X.shape[1] == 1) | (mat_Y.shape[1] == 1):
        return mat_corr.reshape([-1])
    if var_filter:
        return mat_corr, ~no_var

    return mat_corr


def poisson_irls_loop(mat_x_full, mat_y_full, links=None, max_iter=100, tol=1e-3, ridge=False, flag_float32=True):
    """ Run poisson_irls_single over all link pairs sequentially on worker.

    Parameters
    ----------
    mat_x_full : sp.sparse.csc
        Full sparse ATAC matrix (cells x features)
    mat_y_full : sp.sparse.csc
        Full sparse RNA matrix (cells x features)
    links : np.array
        Array of shape (n_links, 2), where each row is (x_idx, y_idx)

    Returns
    -------
    Returns : np.ndarray
        Array of shape (n_pairs), computing 
    """

    if flag_float32:
        dtype = np.float32

    n = mat_x_full.shape[1]
    if links is None: 
        links = np.stack([np.arange(n)] * 2, axis=1)

    n_links = links.shape[0]
    results = np.zeros((n_links, 2), dtype=dtype)

    # Convert to float32 if needed once
    if mat_x_full.dtype != dtype:
        mat_x_full = mat_x_full.astype(dtype)
    if mat_y_full.dtype != dtype:
        mat_y_full = mat_y_full.astype(dtype)

    for i, (x_idx, y_idx) in enumerate(links):
        x_i = mat_x_full[:, x_idx].toarray().ravel()
        y_i = mat_y_full[:, y_idx].toarray().ravel()

        if ridge:
            results[i, :] = poisson_irls_single_ridge(x_i, y_i, max_iter=max_iter, tol=tol)
        else:
            results[i, :] = poisson_irls_single(x_i, y_i, max_iter=max_iter, tol=tol, flag_float32=flag_float32)
    
    return results


def poisson_irls_single(x, y, max_iter=100, tol=1e-3, flag_float32=True):
    """
    Run IRLS Poisson regression on a single feature pair.

    Parameters
    ----------
    x : np.array
        Single paired ATAC feature with shape (n_cells,)
    y : np.array
        Single paired RNA feature with shape (n_cells,)

    Returns
    -------
    Returns : np.ndarray
        Array of shape (2,) [beta0, beta1] 
    """

    if flag_float32:

        dtype = np.float32
    
        beta0 = np.float32(0.0)
        beta1 = np.float32(0.0)

        MAX_EXP = np.float32(80.0)
        MAX_VAL = np.float32(1e10)
        MIN_VAL = np.float32(1e-8)

    else:

        dtype = np.float64

        beta0 = 0.0
        beta1 = 0.0

        MAX_EXP = 80.0
        MAX_VAL = 1e10
        MIN_VAL = 1e-8

    for _ in range(max_iter):
        eta = beta0 + x * beta1
        eta = np.clip(eta, -MAX_EXP, MAX_EXP)

        w = np.exp(eta)
        w = np.minimum(w, MAX_VAL)

        sum_w = np.maximum(np.sum(w), MIN_VAL)

        z = eta + (y - w) / np.maximum(w, MIN_VAL)

        xw = x * w
        xwz = x * w * z
        xxw = x * x * w

        sum_x = np.sum(xw)
        sum_y = np.sum(w * z)

        denom = np.maximum(np.sum(xxw) - (sum_x ** 2) / sum_w, MIN_VAL)

        beta1_new = (np.sum(xwz) - sum_x * sum_y / sum_w) / denom
        beta0_new = (sum_y - beta1_new * sum_x) / sum_w

        if abs(beta1_new - beta1) < tol and abs(beta0_new - beta0) < tol:
            break

        beta0, beta1 = beta0_new, beta1_new

    
    if flag_float32:

        return np.array([beta0, beta1], dtype=dtype)

    else:

        return np.array([beta0, beta1], dtype=dtype)


def poisson_irls_single_ridge(x, y, max_iter=100, tol=1e-3, lambda_reg=1.0):
    """
    Run IRLS Poisson regression with Ridge regularization on a single feature pair.

    Parameters
    ----------
    x : np.array
        Single paired ATAC feature with shape (n_cells,)
    y : np.array
        Single paired RNA feature with shape (n_cells,)
    lambda_reg : float
        Regularization parameter for Ridge (L2) regularization. Larger values lead to stronger regularization.

    Returns
    -------
    Returns : np.ndarray
        Array of shape (2,) [beta0, beta1] 
    """
    
    beta0 = np.float32(0.0)
    beta1 = np.float32(0.0)

    MAX_EXP = np.float32(80.0)
    MAX_VAL = np.float32(1e10)
    MIN_VAL = np.float32(1e-8)

    for _ in range(max_iter):
        eta = beta0 + x * beta1
        eta = np.clip(eta, -MAX_EXP, MAX_EXP)

        w = np.exp(eta)
        w = np.minimum(w, MAX_VAL)

        sum_w = np.maximum(np.sum(w), MIN_VAL)

        z = eta + (y - w) / np.maximum(w, MIN_VAL)

        xw = x * w
        xwz = x * w * z
        xxw = x * x * w

        sum_x = np.sum(xw)
        sum_y = np.sum(w * z)

        denom = np.maximum(np.sum(xxw) - (sum_x ** 2) / sum_w + lambda_reg, MIN_VAL)

        # update with regularization term
        beta1_new = (np.sum(xwz) - sum_x * sum_y / sum_w) / denom
        beta0_new = (sum_y - beta1_new * sum_x) / sum_w

        # convergence check
        if abs(beta1_new - beta1) < tol and abs(beta0_new - beta0) < tol:
            break

        beta0, beta1 = beta0_new, beta1_new

    return np.array([beta0, beta1], dtype=np.float32)


def poisson_irls_loop_multi(mat_x_full, mat_y_full, X_multi, links=None, max_iter=100, tol=1e-3, ridge=False, lambda_reg=0.0):
    """
    Run Poisson IRLS (multi-covariate) over all link pairs sequentially on worker.

    Parameters
    ----------
    mat_x_full : sp.sparse.csc
        Full sparse ATAC matrix (cells x features)
    mat_y_full : sp.sparse.csc
        Full sparse RNA matrix (cells x features)
    X_multi : np.array, shape (n_cells, n_covariates)
        Additional covariates to include for every link.
    links : np.array
        Array of shape (n_links, 2), where each row is (x_idx, y_idx)
    ridge : bool
        Whether to use Ridge regularization.

    Returns
    -------
    results : np.ndarray
        Array of shape (n_links, n_covariates + 1), with intercept in column 0
    """

    n = mat_x_full.shape[1]
    if links is None:
        links = np.stack([np.arange(n)] * 2, axis=1)

    n_links = links.shape[0]
    n_cov = X_multi.shape[1]
    results = np.zeros(n_links, dtype=np.float32)

    if mat_x_full.dtype != np.float32:
        mat_x_full = mat_x_full.astype(np.float32)
    if mat_y_full.dtype != np.float32:
        mat_y_full = mat_y_full.astype(np.float32)
    if X_multi.dtype != np.float32:
        X_multi = X_multi.astype(np.float32)

    for i, (x_idx, y_idx) in enumerate(links):
        y_i = mat_y_full[:, y_idx].toarray().ravel()

        # adding covariates
        x_i = mat_x_full[:, x_idx].toarray().ravel()[:, None]
        X_i = np.hstack([x_i, X_multi])

        if ridge:
            beta0, betas = poisson_irls_multi_ridge(X_i, y_i, max_iter=max_iter, tol=tol, lambda_reg=lambda_reg)
        else:
            beta0, betas = poisson_irls_multi(X_i, y_i, max_iter=max_iter, tol=tol)

        results[i] = betas[0]

    return results


def poisson_irls_multi(X, y, max_iter=100, tol=1e-3):
    """
    Run IRLS Poisson regression (multi-covariate, no regularization).

    Parameters
    ----------
    X : np.array, shape (n_cells, n_covariates)
        Covariate/features matrix
    y : np.array, shape (n_cells,)
        Poisson counts

    Returns
    -------
    beta0 : float
        Intercept
    betas : np.array, shape (n_covariates,)
        Coefficients
    """
    n, p = X.shape
    beta0 = 0.0
    betas = np.zeros(p, dtype=np.float32)

    MAX_EXP, MAX_VAL, MIN_VAL = 80.0, 1e10, 1e-8
    I = np.eye(p, dtype=np.float32)

    for _ in range(max_iter):
        eta = beta0 + X @ betas
        eta = np.clip(eta, -MAX_EXP, MAX_EXP)

        mu = np.exp(eta).astype(np.float32)
        mu = np.clip(mu, MIN_VAL, MAX_VAL)

        W = mu
        z = eta + (y - mu) / np.maximum(mu, MIN_VAL)

        # compute weighted gram and rhs
        WX = X * W[:, None]
        XtWX = np.dot(X.T, WX)
        XtWz = np.dot(X.T, W * z)

        sum_w = np.sum(W)
        sum_y = np.dot(W, z)
        sum_x = np.dot(X.T, W)

        XtWX_adj = XtWX - np.outer(sum_x, sum_x) / max(sum_w, MIN_VAL)
        XtWz_adj = XtWz - sum_x * sum_y / max(sum_w, MIN_VAL)

        # add small jitter to diagonal to ensure positive definiteness
        XtWX_jit = XtWX_adj + MIN_VAL * I

        try:
            c, low = sp.linalg.cho_factor(XtWX_jit, lower=True, check_finite=False)
            betas_new = sp.linalg.cho_solve((c, low), XtWz_adj, check_finite=False)
        except np.linalg.LinAlgError:
            betas_new = np.linalg.solve(XtWX_jit, XtWz_adj)
        
        beta0_new = (sum_y - betas_new @ sum_x) / max(sum_w, MIN_VAL)

        if np.allclose(betas_new, betas, atol=tol) and abs(beta0_new - beta0) < tol:
            break

        betas[:] = betas_new
        beta0 = beta0_new

    return beta0, betas


def poisson_irls_multi_ridge(X, y, max_iter=100, tol=1e-3, lambda_reg=1.0):
    """
    Run IRLS Poisson regression (multi-covariate, with Ridge regularization).

    Parameters
    ----------
    X : np.array, shape (n_cells, n_covariates)
        Covariate/features matrix
    y : np.array, shape (n_cells,)
        Poisson counts
    lambda_reg : float
        Ridge penalty strength (L2)

    Returns
    -------
    beta0 : float
        Intercept
    betas : np.array, shape (n_covariates,)
        Coefficients
    """
    n, p = X.shape
    beta0 = 0.0
    betas = np.zeros(p, dtype=np.float32)

    MAX_EXP, MAX_VAL, MIN_VAL = 80.0, 1e10, 1e-8

    I = np.eye(p, dtype=np.float32)
    I[0, 0] = 0.0

    for _ in range(max_iter):
        # linear predictor
        eta = beta0 + X @ betas
        eta = np.clip(eta, -MAX_EXP, MAX_EXP)

        # mean response
        mu = np.exp(eta)
        mu = np.minimum(mu, MAX_VAL)

        # working weights and response
        W = mu
        z = eta + (y - mu) / np.maximum(mu, MIN_VAL)

        # weighted gram matrix and rhs
        WX = X * W[:, None]
        XtWX = np.dot(X.T, WX)
        XtWz = np.dot(X.T, W * z)

        # center adjustment for intercept
        sum_w = np.sum(W)
        sum_y = np.dot(W, z)
        sum_x = np.dot(X.T, W)

        XtWX_adj = XtWX - np.outer(sum_x, sum_x) / max(sum_w, MIN_VAL)
        XtWz_adj = XtWz - sum_x * sum_y / max(sum_w, MIN_VAL)

        # add ridge penalty to XtWX (but not to intercept)
        XtWX_ridge = XtWX_adj + lambda_reg * I

        try:
            c, low = sp.linalg.cho_factor(XtWX_ridge, lower=True, check_finite=False)
            betas_new = sp.linalg.cho_solve((c, low), XtWz_adj, check_finite=False)
        except np.linalg.LinAlgError:
            betas_new = np.linalg.solve(XtWX_ridge, XtWz_adj)

        beta0_new = (sum_y - betas_new @ sum_x) / max(sum_w, MIN_VAL)

        # convergence check
        if np.allclose(betas_new, betas, atol=tol) and abs(beta0_new - beta0) < tol:
            break

        betas[:] = betas_new
        beta0 = beta0_new

    return beta0, betas


def poisson_irls_loop_sparse_dense_multi(mat_x_full, mat_y_full, X_multi, links=None, max_iter=100, tol=1e-3, lambda_reg=0.0):
    """
    Run Poisson IRLS (multi-covariate) over all link pairs sequentially on worker.

    Parameters
    ----------
    mat_x_full : sp.sparse.csc
        Full sparse ATAC matrix (cells x features)
    mat_y_full : sp.sparse.csc
        Full sparse RNA matrix (cells x features)
    X_multi : np.array, shape (n_cells, n_covariates)
        Additional covariates to include for every link.
    links : np.array
        Array of shape (n_links, 2), where each row is (x_idx, y_idx)

    Returns
    -------
    results : np.ndarray
        Array of shape (n_links, n_covariates + 1), with intercept in column 0
    """

    n = mat_x_full.shape[1]
    if links is None:
        links = np.stack([np.arange(n)] * 2, axis=1)

    n_links = links.shape[0]
    n_cov = X_multi.shape[1]
    results = np.zeros(n_links, dtype=np.float32)

    if mat_x_full.dtype != np.float32:
        mat_x_full = mat_x_full.astype(np.float32)
    if mat_y_full.dtype != np.float32:
        mat_y_full = mat_y_full.astype(np.float32)
    if X_multi.dtype != np.float32:
        X_multi = X_multi.astype(np.float32)

    for i, (x_idx, y_idx) in enumerate(links):

        y_i = mat_y_full[:, y_idx].toarray().ravel()
        x_i = mat_x_full[:, x_idx]

        beta0, betas = poisson_irls_sparse_dense_multi_ridge(x_i, 
            X_multi,
            y_i, 
            max_iter=max_iter, 
            tol=tol, 
            lambda_reg=lambda_reg
        )

        results[i] = betas[0]

    return results


def poisson_irls_sparse_dense_multi_ridge(X_sparse, X_dense, y, max_iter=100, tol=1e-3, lambda_reg=1.0):
    """
    Poisson IRLS with sparse main predictor and dense covariates.

    Parameters
    ----------
    X_sparse : scipy.sparse.csc or csr, shape (n_cells, n_sparse)
        Sparse main predictors (e.g., peaks)
    X_dense : np.ndarray, shape (n_cells, n_dense)
        Dense covariates (e.g., batch effects)
    y : np.ndarray, shape (n_cells,)
        Poisson counts
    lambda_reg : float
        Ridge penalty applied to coefficients (not intercept)
    """
    n_cells = y.shape[0]
    n_sparse = X_sparse.shape[1]
    n_dense = X_dense.shape[1]
    p = n_sparse + n_dense

    beta0 = 0.0
    betas = np.zeros(p, dtype=np.float32)

    MAX_EXP, MAX_VAL, MIN_VAL = 80.0, 1e10, 1e-8

    # Ridge identity, do not penalize intercept
    I = np.eye(p, dtype=np.float32)
    I[0, 0] = 0.0

    for _ in range(max_iter):

        # Linear predictor
        eta = beta0 + X_sparse @ betas[:n_sparse] + X_dense @ betas[n_sparse:]
        eta = np.clip(eta, -MAX_EXP, MAX_EXP)

        # Mean response
        mu = np.exp(eta)
        mu = np.minimum(mu, MAX_VAL)

        # Working weights and response
        W = mu
        z = eta + (y - mu) / np.maximum(mu, MIN_VAL)

        # Weighted Gram and RHS
        XtWX_sparse = (X_sparse.T @ (X_sparse.multiply(W[:, None]))).toarray()
        XtWX_dense = (X_dense.T @ (X_dense * W[:, None]))
        XtWz_sparse = (X_sparse.T @ (W * z)).ravel()
        XtWz_dense = (X_dense.T @ (W * z)).ravel()
        sum_x_sparse = (X_sparse.multiply(W[:, None])).sum(axis=0).A1
        sum_x_dense = (X_dense * W[:, None]).sum(axis=0).ravel()

        # Cross term
        cross = (X_sparse.T @ (X_dense * W[:, None]))

        # Assemble full Gram and RHS
        XtWX = np.zeros((p, p), dtype=np.float32)
        XtWz = np.zeros(p, dtype=np.float32)
        sum_x = np.zeros(p, dtype=np.float32)

        XtWX[:n_sparse, :n_sparse] = XtWX_sparse
        XtWX[n_sparse:, n_sparse:] = XtWX_dense
        XtWX[:n_sparse, n_sparse:] = cross
        XtWX[n_sparse:, :n_sparse] = cross.T

        XtWz[:n_sparse] = XtWz_sparse
        XtWz[n_sparse:] = XtWz_dense

        sum_x[:n_sparse] = sum_x_sparse
        sum_x[n_sparse:] = sum_x_dense

        # Centering for intercept
        sum_w = np.sum(W)
        sum_y = np.dot(W, z)

        XtWX_adj = XtWX - np.outer(sum_x, sum_x) / max(sum_w, MIN_VAL)
        XtWz_adj = XtWz - sum_x * sum_y / max(sum_w, MIN_VAL)

        # Ridge penalty
        XtWX_ridge = XtWX_adj + lambda_reg * I

        # Solve
        c, low = sp.linalg.cho_factor(XtWX_ridge, lower=True, check_finite=False)
        betas_new = sp.linalg.cho_solve((c, low), XtWz_adj, check_finite=False)
        beta0_new = (sum_y - np.dot(betas_new, sum_x)) / max(sum_w, MIN_VAL)

        # Convergence check
        if np.allclose(betas_new, betas, atol=tol) and abs(beta0_new - beta0) < tol:
            break

        betas, beta0 = betas_new, beta0_new

    return beta0, betas



######################### generate null #########################


def gc_content(adata, col='gene_ids', genome_file='GRCh38.p13.genome.fa.bgz'):
    
    ''' Finds GC content for peaks.
    
    Parameters
    ----------
    adata : ad.AnnData
        AnnData object of size (N,M) with atac mod containing peak range information
        and peak dataframe (in adata.uns). Also assumed to have 'gene_id' column in vars.
    col : str
        Label for peak column.
    genome_file : genome.fa.bgz file
        Reference genome in fasta file format.
    
    Returns
    ----------
    gc : np.ndarray
        Array of shape (N,) containing GC content of each peak.
        
    '''

    adata_copy = adata.copy()

    # muon get_sequences requires peaks to be index name
    # expected to be named as chrX:NNN-NNN
    assert adata_copy.var[col].apply(lambda x: bool(re.match(r'^[\w\W]+:[\d]+-[\d]+$', str(x).strip()))).all(), (
        f'Not all entries in {col} match the chrN:N-N format.'
        )
    adata_copy.var.index = adata_copy.var[col]

    # Get sequences
    atac_seqs = mu.atac.tl.get_sequences(adata_copy,None,fasta_file=genome_file)
 
    # Store each's gc content
    gc = np.empty(adata_copy.shape[1])
    i=0
    for seq in atac_seqs:
        gc[i] = gc_fraction(seq)
        i+=1
    return gc


def sub_bin(df, group_col, val_col, num_sub_bins, out_col):
    ''' Assigns nested bin labels within groups.
    ''' 

    for bin_i in df[group_col].unique():
        inds = (df[group_col] == bin_i)
        ranked = df.loc[inds, val_col].rank(method='first')
        sub_bin_labels = pd.qcut(ranked, num_sub_bins, labels=False, duplicates='drop')
        df.loc[inds, out_col] = [f"{int(bin_i)}.{int(x)}" for x in sub_bin_labels]

    return df


def get_bins(adata, num_bins=5, type='mean', col='gene_ids', layer='atac_raw', genome_file=None):
    ''' Obtains GC and MFA bins for adata peaks.
    
    Parameters
    ----------
    adata : ad.AnnData
        AnnData object
    num_bins : int or list of int
        Number of desired bins. If list, first is for mean, second for sub-binning.
    type : str
        Binning type. Options: ['mean', 'mean_var', 'mean_gc', 'chol_logsum_gc'].
    col : str
        Column name in adata.var indicating peak ID.
    layer : str
        Name of adata layer with matrix values.
    genome_file : str or None
        Path to genome file, required for GC calculations.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with peak binning information.
    '''

    if type == 'mean_gc':
        assert genome_file is not None, (
            'Must provide reference genome.'
            )
    if isinstance(num_bins, list):
        assert len(num_bins) <= 2, (
            'Maximum of 2 types of bins supported.'
            )

    bins = pd.DataFrame()
    unique = ~adata.var.duplicated(subset=col)

    bins[col] = adata.var[col][unique].values
    adata.var['index_z'] = range(len(adata.var))
    bins['ind'] = adata.var.loc[unique, 'index_z'].values

    sparse_X = adata[:, unique].layers[layer]
    bins['mean'] = sparse_X.mean(axis=0).A1
    print('Mean done.')

    mean_num_bins = num_bins[0] if isinstance(num_bins, list) else num_bins
    alt_num_bins = num_bins[1] if isinstance(num_bins, list) else None
    bins['mean_bin'] = pd.qcut(bins['mean'].rank(method='first'), mean_num_bins, labels=False, duplicates="drop")

    if type == 'mean_var':
        e2 = sparse_X.power(2).mean(axis=0).A1
        bins['var'] = e2 - bins['mean'] ** 2
        print('Var done.')
        bins['mean_var_bin'] = ''
        bins = sub_bin(bins, 'mean_bin', 'var', alt_num_bins, 'mean_var_bin')

    elif type == 'mean_gc':
        bins['gc'] = gc_content(adata[:, unique], col=col, genome_file=genome_file)
        print('GC done.')
        bins['mean_gc_bin'] = ''
        bins = sub_bin(bins, 'mean_bin', 'gc', alt_num_bins, 'mean_gc_bin')

    elif type == 'chol_logsum_gc':
        bins['sum'] = sparse_X.sum(axis=0).A1
        bins['logsum'] = np.log10(bins['sum'] + 1e-10)
        bins['gc'] = gc_content(adata[:, unique], col='peak', genome_file=genome_file)

        norm_mat = bins[['logsum', 'gc']].values.T
        chol_cov = sp.linalg.cholesky(np.cov(norm_mat))
        trans_mat = np.linalg.solve(chol_cov, norm_mat)
        bins[['chol_logsum', 'chol_gc']] = trans_mat.T

        bins['chol_logsum_bin'] = pd.qcut(bins['chol_logsum'].rank(method='first'), mean_num_bins, labels=False, duplicates='drop')
        bins['chol_logsum_gc_bin'] = ''
        bins = sub_bin(bins, 'chol_logsum_bin', 'chol_gc', alt_num_bins, 'chol_logsum_gc_bin')

    return bins


def get_value_bins(adata, num_bins=5, b=200, type='mean', col='gene_ids', layer='atac_raw', genome_file=None):

    ''' Attempts to split bins based on mean value with min size b, rather than by quantile.
    For now, the only implemented type is "mean".
    
    '''

    bins = pd.DataFrame()
    unique = ~adata.var.duplicated(subset=col)

    bins[col] = adata.var[col][unique]
    adata.var['index_z'] = range(len(adata.var))
    bins['ind'] = adata.var.index_z[unique]
    sparse_X = adata[:,unique].layers[layer]

    bins['mean'] = sparse_X.mean(axis=0).A1

    bins['mean_bin'] = pd.cut(bins['mean'], num_bins, labels=False, duplicates="drop")
    if bins.mean_bin.value_counts().min() > b:
        print('No resizing needed.')
        return bins

    max_bins = bins['mean'].nunique() // b
    multiplier = 1
    while bins.mean_bin.value_counts().min() < b:
        
        if multiplier > max_bins: # fail condition
            bins['mean_bin'] = pd.qcut(bins['mean'], num_bins, labels=False, duplicates="drop")
            return bins
            
        # take double the amount of bins
        bins['mean_bin'] = pd.cut(bins['mean'], num_bins*multiplier, labels=False, duplicates="drop")
        # leave the lower 1/4 of bins, modify the upper 3/4 back to desired num_bins
        upper_bins = num_bins - (num_bins // 2)
        bins.loc[bins.mean_bin >= upper_bins,'mean_bin'] = upper_bins + pd.qcut(bins[bins.mean_bin >= upper_bins]['mean'],
                                                                      upper_bins,
                                                                      labels=False, 
                                                                      duplicates="drop")
        multiplier += 1
        
    print('Multiplier: %d' % (multiplier))
    return bins


def create_ctrl_peaks(adata,num_bins=5,b=1000,type='mean',peak_col='gene_ids',layer='atac_raw',return_bins_df=False,genome_file=None):

    ''' Obtains GC and MFA bins for ATAC peaks.
    
    Parameters
    ----------
    adata : ad.AnnData
        AnnData object of shape (#cells,#peaks).
    num_bins : int
        Number of desired bins for groupings.
    b : int
        Number of desired random peaks per focal peak-gene pair.
    peak_col : str
        Label for column containing peak IDs.
    layer : str
        Layer in adata.layers corresponding to the matrix to base bins on.
    type : str
        Metric to base binning on. Options are ['mean','mean_var','mean_value','mean_gc', 'chol_logsum_gc'].
    
    Returns
    ----------
    ctrl_peaks : np.ndarray
        Matrix of length (#peaks,n) where n is number of random peaks generated (1000 by default).
    
    '''
    
    if type == 'mean_value':
        bins = get_value_bins(adata, num_bins=num_bins, b=b, type=type, col=peak_col, layer=layer, genome_file=genome_file)
        type = 'mean' # ensures compatibility for future types
    else:
        bins = get_bins(adata, num_bins=num_bins, type=type, col=peak_col, layer=layer, genome_file=genome_file)
    print('Get_bins done.')
    # Group indices for rand_peaks
    bins_grouped = bins[['ind',f'{type}_bin']].groupby([f'{type}_bin']).ind.apply(np.array)
    
    # Generate random peaks
    ctrl_peaks = np.empty((len(bins),b))
    # iterate through each row of bins
    for i in range(len(bins)):
        peak = bins.iloc[i]
        row_bin = bins_grouped.loc[peak[f'{type}_bin']]
        row_bin_copy = row_bin[row_bin!=peak.ind]
        ctrl_peaks[i] = np.random.choice(row_bin_copy, size=(b,), replace=False)
    print('Ctrl index array done.')
    
    # for duplicated peaks, simply copy these rows
    # since they will be compared with different genes anyway
    ind,_ = pd.factorize(adata.var[peak_col])
    ctrl_peaks = ctrl_peaks[ind,:].astype(int)

    if return_bins_df:
        return ctrl_peaks,bins
    
    return ctrl_peaks


def create_ctrl_pairs(
    adata_atac,
    adata_rna,
    atac_bins = [5, 5],
    rna_bins = [20,20],
    atac_type = 'mean',
    rna_type = 'mean_var',
    genome_file = None,
    atac_layer = 'atac_raw',
    rna_layer = 'rna_raw',
    b = 100000,
    peak_col = 'peak',
    gene_col = 'gene',
    return_bins_df = True,
    flag_combine_bin = False,
):

    ''' Obtain b control pairs for each permutation of bin features.

    Parameters
    ----------
    adata_atac : ad.AnnData
        AnnData object of shape (#cells,#peaks).
    adata_rna : ad.AnnData
        AnnData object of shape (#cells,#genes).
    atac_bins : int or list (int)
        Number of desired atac bins for groupings. If 'mean_gc' order is [# mean bins, # gc bins].
    rna_bins : int or list (int)
        Number of desired rna bins for groupings. If 'mean_var' order is [# mean bins, # var bins].
    atac_type : str
        Metric to base atac binning on. Options are ['mean','mean_gc', 'chol_logsum_gc'].
    rna_type : str
        Metric to base rna binning on. Options are ['mean','mean_var'].
    genome_file : str
        Path to file containing reference genome in fa.bgz format.
    atac_layer : str
        Layer in adata_atac.layers corresponding to the matrix to base bins on.
    rna_layer : str
        Layer in adata_rna.layers corresponding to the matrix to base bins on.
    b : int
        Number of desired random peaks per focal peak-gene pair.
    peak_col : str
        Label for column containing peak IDs in adata_atac.var.
    gene_col : str
        Label for column containing gene IDs in adata_rna.var.
    return_bins_df : bool
        If true, return atac and rna dataframes with their respective bins.
    
    Returns
    ----------
    ctrl_dic : dict
        Dictionary where keys are all permutations of rna_bins and atac_bins. Values are b random
        pairs selected from each bin.
    atac_bins_df : pd.DataFrame (optional)
        Copy of adata_atac.var containing atac_type information and bin labels.
    rna_bins_df : pd.DataFrame (optional)
        Copy of adata_rna.var containing rna_type information and bin labels.
    
    '''

    assert atac_type in ['mean', 'mean_gc', 'chol_logsum_gc'], (
        'atac_type must be mean, mean_gc, or chol_logsum_gc.'
        )
    assert rna_type in ['mean', 'mean_var'], (
        'rna_type must be mean or mean_var.'
        )
    
    atac_bins_df = get_bins(adata_atac,num_bins=atac_bins,type=atac_type,col=peak_col,layer=atac_layer,genome_file=genome_file)
    rna_bins_df = get_bins(adata_rna,num_bins=rna_bins,type=rna_type,col=gene_col,layer=rna_layer)
    print('Get_bins done.')
    if flag_combine_bin:
        rna_bins_df.loc[rna_bins_df.mean_bin == 0, 'mean_bin'] = 1
        rna_bins_df['mean_var_bin'] = rna_bins_df['mean_var_bin'].str.replace(r'^0.','1.',regex=True).values
    
    # Group indices for controls
    pairs = [(i, j) for i in atac_bins_df[f'{atac_type}_bin'].unique() for j in rna_bins_df[f'{rna_type}_bin'].unique()]
    atac_bins_grouped = atac_bins_df[['ind',f'{atac_type}_bin']].groupby([f'{atac_type}_bin']).ind.apply(np.array)
    rna_bins_grouped = rna_bins_df[['ind',f'{rna_type}_bin']].groupby([f'{rna_type}_bin']).ind.apply(np.array)

    
    # Generate random pairs
    ctrl_dic = {}
    for pair in tqdm(pairs):
        atac_inds = atac_bins_grouped.loc[pair[0]]
        rna_inds = rna_bins_grouped.loc[pair[1]]
        
        # Get Cartesian product of indices
        all_pairs = np.array(np.meshgrid(atac_inds, rna_inds)).T.reshape(-1, 2)
        n_pairs = all_pairs.shape[0]

        if n_pairs < b:
            raise ValueError(f"Not enough unique pairs: {n_pairs} available, {b} requested.")

        # Randomly sample without replacement
        selected_idx = np.random.choice(n_pairs, size=b, replace=False)
        ctrl_links = all_pairs[selected_idx]
        ctrl_dic[str(pair[0]) + '_' + str(pair[1])] = ctrl_links

    if return_bins_df:

        return ctrl_dic, atac_bins_df, rna_bins_df

    return ctrl_dic


######################### pvalue methods #########################


def initial_mcpval(ctrl_corr,corr,one_sided=True):
    
    ''' Calculates one or two-tailed Monte Carlo p-value (only for B controls).
    
    Parameters
    ----------
    ctrl_corr : np.ndarray
        Matrix of shape (gene#,n) where n is number of rand samples.
    corr : np.ndarray
        Vector of shape (gene#,), which gets reshaped to (gene#,1).
    one_sided : bool
        Indicates whether to do 1 sided or 2 sided pval.
    
    Returns
    ----------
    Vector of shape (gene#,) corresponding to the Monte Carlo p-value of each statistic.
    
    '''

    corr = corr.reshape(-1, 1)
    if not one_sided:
        ctrl_corr = np.abs(ctrl_corr)
        corr = np.abs(corr)
    indicator = np.sum(ctrl_corr >= corr.reshape(-1, 1), axis=1)
    return (1+indicator)/(1+ctrl_corr.shape[1])


def zscore_pval(ctrl_corr,corr,axis=1):
    ''' 1-sided Z-score pvalue.
    '''

    mean = np.mean(ctrl_corr,axis=axis)
    sd = np.std(ctrl_corr,axis=axis)
    z = (corr - mean)/sd
    
    p_value = 1 - sp.stats.norm.cdf(z)
    return p_value, z


def pooled_mcpval(ctrl_corr,corr,axis=1):
    ''' 1-sided MC pooled pvalue.
    '''

    # Center first
    ctrl_corr_centered,corr_centered = center_ctrls(ctrl_corr,corr,axis=axis)
    ctrl_corr_centered = np.sort(ctrl_corr_centered)
    n,b = ctrl_corr.shape
    
    # Search sort returns indices where element would be inserted
    indicator = (n*b) - np.searchsorted(ctrl_corr_centered, corr_centered, side='left')
    return (1+indicator)/(1+(n*b))


def center_ctrls(ctrl_corray,main_array,axis=0):

    ''' Centers control and focal correlation arrays according to control mean and std.
    
    Parameters
    ----------
    ctrl_corray : np.ndarray
        Array of shape (N,B) where N is number of genes and B is number
        of repetitions (typically 1000x). Contains correlation between
        focal gene and random peaks.
    main_array : np.ndarray
        Array of shape (N,) containing correlation between focal gene
        and focal peaks.
    
    Returns
    ----------
    ctrls : np.ndarray
        Array of shape (N*B,) containing centered correlations between
        focal gene and random peaks.
    main : np.ndarray
        Array of shape (N,) containing centered correlations between
        focal gene and focal peak, according to ctrl mean and std.
        
    '''
    
    
    # Takes all ctrls and centers at same time
    # then centers putative/main with same vals
    mean = np.mean(ctrl_corray,axis=axis)
    std = np.std(ctrl_corray,axis=axis)

    main = (main_array - mean) / std
    if axis == 1:
        mean = mean.reshape(-1,1)
        std = std.reshape(-1,1)
    ctrls = (ctrl_corray - mean) / std
    
    return ctrls.flatten(), main


def basic_mcpval(ctrl_corr,corr):
    ''' 1-sided MC pvalue (for single set of controls)
    '''

    ctrl_corr = np.sort(ctrl_corr)
    indicator = len(ctrl_corr) - np.searchsorted(ctrl_corr,corr,side='left')

    return (1+indicator)/(1+len(ctrl_corr))


def basic_zpval(ctrl_corr,corr):
    ''' 1-sided zscore pvalue (for single set of controls)
    '''

    mean = np.mean(ctrl_corr)
    sd = np.std(ctrl_corr)
    z = (corr - mean)/sd

    p_value = 1 - sp.stats.norm.cdf(z)
    return p_value


def mc_pval(ctrl_corr_full,corr):

    ''' Calculates MC p-value using centered control and focal correlation arrays across
    all controls (N*B).
    
    Parameters
    ----------
    ctrl_corr_full : np.ndarray
        Array of shape (N,B) where N is number of genes and B is number
        of repetitions (typically 1000x). Contains correlation/delta correlation
        between focal gene and random peaks.
    corr : np.ndarray
        Array of shape (N,) containing correlation/delta correlation between
        focal gene and focal peaks.
    
    Returns
    ----------
    full_mcpvalue : np.ndarray
        Array of shape (N,) containing MC p-value corresponding to Nth peak-gene pair
        against all centered contrl correlation/delta correlation pairs.
        
    '''

    # Center first
    ctrl_corr_full_centered,corr_centered = center_ctrls(np.abs(ctrl_corr_full),np.abs(corr))
    ctrl_corr_full_centered = np.sort(ctrl_corr_full_centered)
    n,b = ctrl_corr_full.shape
    
    # Search sort returns indices where element would be inserted
    indicator = len(ctrl_corr_full_centered) - np.searchsorted(ctrl_corr_full_centered,corr_centered)
    return (1+indicator)/(1+(n*b))


def cauchy_combination(p_values1, p_values2):

    ''' Calculates Cauchy combination test for two arrays of p-values.
    
    Parameters
    ----------
    p_values1 : np.ndarray
        Array of shape (N,) of p-values from one method.
    p_values2 : np.ndarray
        Array of shape (N,) of p-values from another method.
    
    Returns
    ----------
    combined_p_value : np.ndarray
        Array of shape (N,) of p-values combined using Cauchy
        distribution approximation.
        
    '''

    # From R code : 0.5-atan(mean(tan((0.5-Pval)*pi)))/pi
    quantiles1 = np.tan(np.pi * (0.5 - p_values1))
    quantiles2 = np.tan(np.pi * (0.5 - p_values2))

    # Combine the quantiles
    combined_quantiles = np.vstack((quantiles1, quantiles2))
    
    # Calculate the combined statistic (mean)
    combined_statistic = np.mean(combined_quantiles,axis=0)

    # Convert the combined statistic back to a p-value
    combined_p_value = 0.5-np.arctan(combined_statistic)/math.pi

    return combined_p_value


def binned_mcpval(
    cis_pairs_dic, 
    ctrl_pairs_dic,
    b=None,
):
    '''
    Compute p-values for binned evaluation data using Monte Carlo method.

    Parameters
    ----------
    cis_pairs_dic : dict
        Dictionary of numpy arrays where keys are bins and values are PR coefficients
        for cis-peak-gene pairs.
    ctrl_pairs_dic : dict
        Dictionary of numpy arrays where keys are bins and values are PR coefficients
        for control peak-gene pairs.

    Returns
    ----------
    mcpval_dic : dict
        Dictionary of numpy arrays where keys are bins and values are mc-pvalues.
    ppval_dic : dict
        Dictionary of numpy arrays where keys are bins and values are pooled-pvalues.
    '''
    bin_keys = sorted(cis_pairs_dic.keys())

    # set b, arbitrarily using the first key
    if b == None:
        b = len(ctrl_pairs_dic[bin_keys[0]])
        print(f'Setting n_ctrl to {b}.')

    mcpval_dic = {}
    centered_ctrl_ls = []
    centered_cis_ls = []

    for bin_key in bin_keys:

        coeffs = cis_pairs_dic[bin_key]
        ctrl_coeffs = ctrl_pairs_dic[bin_key].ravel()[:b]

        mcpval_dic[bin_key] = basic_mcpval(ctrl_coeffs, coeffs)

        centered_ctrls, centered_cis = center_ctrls(ctrl_coeffs,coeffs,axis=0)
        centered_ctrl_ls.append(centered_ctrls)
        centered_cis_ls.append(centered_cis)

    centered_ctrl_ls = np.concatenate(centered_ctrl_ls)
    centered_cis_ls = np.concatenate(centered_cis_ls)
    pooled_pvals = basic_mcpval(centered_ctrl_ls, centered_cis_ls)

    ppval_dic = {}
    start = 0

    for bin_key in bin_keys:
        bin_size = len(cis_pairs_dic[bin_key])
        ppval_dic[bin_key] = pooled_pvals[start:start+bin_size]
        start += bin_size

    return mcpval_dic, ppval_dic


def binned_mcpval_from_disk(
    path, 
    eval_df, 
    coeffs, 
    bin_label = 'combined_bin_5.20.20.20', 
    startswith='poiss_ctrl_', 
    pattern = r'[\d]+.[\d]+_[\d]+.[\d]+'
):
    '''
    Compute p-values for binned evaluation data using Monte Carlo method.

    Parameters
    ----------
    path : str
        Path to the directory containing control files.
    eval_df : pd.DataFrame
        Evaluation DataFrame with coefficients.
    coeffs : np.array
        Original Poisson coefficients (assumed global).
    bin_label : str 
        Column name to group bins.
    pattern : str
        Regex pattern to identify control files.

    Returns
    ----------
    tuple : (bin_mcpvals, centered_pvals)

    '''

    poiss_grp_files = [f for f in os.listdir(path) if re.search(startswith + pattern + '.npy', f)]
    
    # create dictionary for controls
    poiss_grp_dic = {
        re.findall(pattern, f)[0]: np.load(os.path.join(path, f))
        for f in tqdm(poiss_grp_files)
    }

    gt_bins = eval_df.copy()
    gt_bins['ind'] = range(len(gt_bins))
    grouped_bins = gt_bins.groupby(bin_label)['ind'].agg(list)

    n = len(gt_bins)
    bin_mcpvals = np.ones(n)
    centered_coeffs = np.ones(n)
    all_centered_ctrls = []

    for bin_i in tqdm(grouped_bins.index):
        
        indices = grouped_bins.loc[bin_i]
        bin_coeffs = coeffs[indices]
        bin_ctrls = poiss_grp_dic[bin_i].flatten()

        bin_mcpvals[indices] = basic_mcpval(bin_ctrls,bin_coeffs)

        centered_ctrl, centered_coeff = center_ctrls(bin_ctrls,bin_coeffs,axis=0)
        centered_coeffs[indices] = centered_coeff
        all_centered_ctrls.append(centered_ctrl)

    all_centered_ctrls = np.concatenate(all_centered_ctrls)
    centered_pvals = basic_mcpval(all_centered_ctrls, centered_coeffs)

    return bin_mcpvals, centered_pvals


######################### clustering #########################


def sum_by(adata: ad.AnnData, col: str) -> ad.AnnData:
    adata.strings_to_categoricals()
    assert pd.api.types.is_categorical_dtype(adata.obs[col])

    indicator = pd.get_dummies(adata.obs[col])

    return ad.AnnData(
        indicator.values.T @ adata.X, #.layers['counts'],
        var=adata.var,
        obs=pd.DataFrame(index=indicator.columns)
    )


def get_moments(adata, col, get_max=True):
    ''' Get first and second moment of Anndata matrix for each categorial variable in col.
    '''
    adata.strings_to_categoricals()
    
    indicator = pd.get_dummies(adata.obs[col])
    v_counts = indicator.sum(axis=0).values.reshape(-1,1)
    
    # mean = sum / count
    means = indicator.values.T @ adata.X / v_counts
    means = means.clip(min=1e-6)
    print('Calculated means.')

    # mean of squared X
    squared_X = adata.X.power(2) # X^2
    means_of_squares = (indicator.values.T @ squared_X) / v_counts # E[X^2]

    # V[X] = E[X^2] - (E[X])^2
    variances = means_of_squares - means ** 2
    if get_max:
        variances = np.maximum(variances,means)
    print('Calculated variances.')
    
    return means, variances



#####################################################################################
######################################## OLD ########################################
#####################################################################################


def fit_poisson(x,y,return_both=False):

    # Simple log(E[y]) ~ x equation
    exog = sm.add_constant(x)
    
    poisson_model = sm.GLM(y, exog, family=sm.families.Poisson())
    result = poisson_model.fit()

    if not return_both:
        return result.params[1]

    return result.params[0], result.params[1]


def fit_negbinom(x,y,return_both=False):

    # Simple log(E[y]) ~ x equation with y ~ NB(mu,r)
    exog = sm.add_constant(x)
    
    result = sm.NegativeBinomial(y, exog).fit(disp=0) # sm.GLM(y, exog, family=sm.families.NegativeBinomial())
    # result = negbinom_model.fit(start_params=[1,1])

    if not return_both:
        return result.params[1]

    return result.params[0], result.params[1]

