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
import sklearn as sk
import os
import re


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


def vectorized_poisson_regression(X, Y, max_iter=100, tol=1e-6):
    """ Fast poisson regression from Alistair
    
    Perform vectorized Poisson regression using IRLS.
    
    Parameters:
    - X: Predictor matrix of shape (N, P)
    - Y: Response vector of shape (N,)
    - max_iter: Maximum number of iterations
    - tol: Convergence tolerance
    
    Returns:
    - beta0: Intercept coefficients of shape (P,)
    - beta1: Slope coefficients of shape (P,)
    """
    # Start timing for the entire function
    
    N, P = X.shape
    beta0 = np.zeros(P)
    beta1 = np.zeros(P)
    if Y.ndim == 1: Y_broad = Y.reshape(-1,1)
    else: Y_broad = Y
    
    for iteration in range(max_iter):
        #eta = np.clip(beta0 + X * beta1, -10, 10)
        #mu = np.clip(np.exp(eta), 1e-8, None)
        eta = beta0 + X * beta1 # N x P
        mu = np.exp(eta) # N x P
        z = eta + (Y_broad - mu) / mu # N x P
        WX = mu * X # N x P
        WX2 = WX * X # N x P
        Sw = mu.sum(axis=0)        # (P,)
        Sx = WX.sum(axis=0)       # (P,)
        #Sx2 = WX2.sum(axis=0)     # (P,)
        Sy = (mu * z).sum(axis=0)  # (P,)
        Sxy = (WX * z).sum(axis=0)  # (P,)
        denom = WX2.sum(axis=0)  - (Sx**2) / Sw # P
        denom = np.where(denom == 0, 1e-8, denom)  # Avoid division by zero
        
        beta1_new = (Sxy - (Sx * Sy) / Sw) / denom # P
        beta0_new = (Sy - beta1_new * Sx) / Sw # P
        if np.all(np.abs(beta1_new - beta1) < tol) and np.all(np.abs(beta0_new - beta0) < tol):
            print(f"Converged after {iteration+1} iterations.")
            break
        # Update beta0 and beta1
        beta0, beta1 = beta0_new, beta1_new  # Simpler variable update

    return beta0, beta1


def vectorized_poisson_regression_final(mat_x, mat_y, max_iter=100, tol=1e-6, flag_float32=True):
    """ Fast poisson regression using IRLS, adapted from Alistair.
    Using sparsity and optionally low precision.

    
    Parameters
    ----------
    mat_x : np.ndarray
        Predictor matrix of shape (n_cell, n_pair).
    mat_y : np.ndarray
        Response vector of shape (n_cell,) or (n_cell, 1)
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
    flag_float32 : bool
        Determines whether or not dtype will be converted to float32.

    Returns
    -------
    v_beta0 : np.ndarray
        Intercept coefficients of shape (n_pair,)
    v_beta1 : np.ndarray
        Slope coefficients of shape (n_pair,)

    """

    n_cell, n_pair = mat_x.shape
    
    if flag_float32 is True:
        fct_dtype = np.float32
    else: 
        fct_dtype = float

    
    v_beta0 = np.zeros(n_pair, dtype=fct_dtype)
    v_beta1 = np.zeros(n_pair, dtype=fct_dtype)

    # change mat_y from (n_cell,) to (n_cell, n_pair) if single pair
    if mat_y.shape[1] == 1:
        # if we repeat the sparse array, it can greatly slow performance
        # (concatenated sparse vectors are not optimized for sparsity)
        if sp.sparse.issparse(mat_y) is True:
            mat_y = np.repeat(mat_y.toarray(), n_pair, axis=1)
        else:
            mat_y = np.repeat(mat_y, n_pair, axis=1)
    
    if sp.sparse.issparse(mat_x) is False:
        mat_x = sp.sparse.csc_matrix(mat_x) 
    if sp.sparse.issparse(mat_y) is False:
        mat_y = sp.sparse.csc_matrix(mat_y)

    if mat_x.dtype != fct_dtype:
        mat_x = mat_x.astype(fct_dtype)
    if mat_y.dtype != fct_dtype:
        mat_y = mat_y.astype(fct_dtype)

    if mat_y.ndim == 1: 
        mat_y = mat_y.reshape(-1,1)


    for iteration in range(max_iter):

        sum_xy = mat_x.multiply(v_beta1) + v_beta0 # (n_cell, n_pair)
        sum_w = np.exp(sum_xy) # (n_cell, n_pair)

        sum_xy = sum_xy + (mat_y - sum_w) / sum_w # (n_cell, n_pair)
        sum_x = mat_x.multiply(sum_w) # (n_cell, n_pair)
        denom = mat_x.multiply(sum_x) # (n_cell, n_pair)

        sum_y = np.multiply(sum_w,sum_xy).sum(axis=0)
        sum_xy = sum_x.multiply(sum_xy).sum(axis=0)
        sum_w = sum_w.sum(axis=0)
        sum_x = sum_x.sum(axis=0)

        denom = denom.sum(axis=0) - (np.power(sum_x,2) / sum_w)
        denom = np.maximum(denom, 1e-8)  # Avoid division by zero
        
        v_beta1_new = (sum_xy - np.multiply(sum_x,sum_y) / sum_w) / denom
        v_beta0_new = (sum_y - np.multiply(v_beta1_new,sum_x)) / sum_w

        if np.all(np.abs(v_beta1_new - v_beta1) < tol) and np.all(np.abs(v_beta0_new - v_beta0) < tol):
            print(f"Converged after {iteration+1} iterations.")
            break

        # Update beta0 and beta1
        v_beta0, v_beta1 = v_beta0_new, v_beta1_new  # Simpler variable update

    return v_beta0, v_beta1



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
        df.loc[inds, out_col] = [f"{int(x)}.{int(bin_i)}" for x in sub_bin_labels]

    return df


def get_bins(adata, num_bins=5, type='mean', col='gene_ids', layer='atac_raw', genome_file=None):
    ''' Obtains GC and MFA bins for adata peaks.
    
    Parameters
    ----------
    adata : ad.AnnData
        AnnData object
    num_bins : int or list of int
        Number of desired bins. If list, first is for sub-binning, second for mean.
    type : str
        Binning type. Options: ['mean', 'mean_var', 'mean_gc', 'cholesky'].
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

    bins = pd.DataFrame()
    unique = ~adata.var.duplicated(subset=col)

    bins[col] = adata.var[col][unique].values
    adata.var['index_z'] = range(len(adata.var))
    bins['ind'] = adata.var.loc[unique, 'index_z'].values

    sparse_X = adata[:, unique].layers[layer]
    bins['mean'] = sparse_X.mean(axis=0).A1
    print('Mean done.')

    alt_num_bins = num_bins[0] if isinstance(num_bins, list) else None
    mean_num_bins = num_bins[1] if isinstance(num_bins, list) else num_bins
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
    atac_bins = 5,
    rna_bins = [20,20],
    atac_type = 'mean',
    rna_type = 'mean_var',
    genome_file = None,
    atac_layer = 'atac_raw',
    rna_layer = 'rna_raw',
    b = 100000,
    peak_col = 'peak',
    gene_col = 'gene',
    return_bins_df = True
):

    ''' Control pairs.
    
    '''
    
    atac_bins_df = get_bins(adata_atac,num_bins=atac_bins,type=atac_type,col=peak_col,layer=atac_layer,genome_file=genome_file)
    rna_bins_df = get_bins(adata_rna,num_bins=rna_bins,type=rna_type,col=gene_col,layer=rna_layer)
    print('Get_bins done.')
    
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
        return ctrl_dic,atac_bins_df,rna_bins_df

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
        for f in poiss_grp_files
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


def add_corr(mdata):

    '''Adds pearson corr for putative links to existing mdata.

    Parameters
    ----------
    mdata : mu.MuData
        MuData object of shape (#cells,#peaks). Contains DataFrame under mdata.uns.peak_gene_pairs
        containing columns ['index_x','index_y'] that correspond to peak and gene indices in atac.X
        and rna.X respectively, as well as mdata.uns.control_peaks containing randomly generated peaks.
        If mdata.uns.control_peaks doesn't exist yet, will create it.
    
    Returns
    ----------
    mdata : mu.MuData
        Updates mdata input with mdata.uns.corr, an np.ndarray of shape (#peak-gene pairs).
    
    '''

    try:
        mdata.uns['peak_gene_pairs']
    except KeyError:
        print('Attempting to add peak-gene pairs.')
        find_peak_gene_pairs(mdata)

    # Extract peak-gene pairs and their respective atac and rna exp values
    peaks_df = mdata.uns['peak_gene_pairs']
    atac_Xs = mdata['atac'].X[:,peaks_df.index_x.values]
    rna_Xs = mdata['rna'].X[:,peaks_df.index_y.values]

    # Calculate corr
    corr = pearson_corr_sparse(atac_Xs,rna_Xs,var_filter=True)

    # Store in mdata.uns
    mdata.uns['peak_gene_corr'] = corr
    
    return mdata
    

def get_corrs(adata):

    '''Adds pearson corr for putative links to a links AnnData.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData of shape (#cells, #peak-gene pairs) containing rna and atac layers.
        See build_adata.
    
    Returns
    ----------
    adata : ad.AnnData
        Updates given AnnData with correlation between peak-gene pairs of shape
        (#cells, #peak-gene pairs with variance > 1e-6).
    
    '''

    assert type(adata) == type(ad.AnnData()), 'Must be AnnData.'

    # Calculate corr
    corr = pearson_corr_sparse(adata.layers['rna'],adata.layers['atac'],var_filter=True)

    # Remove pairs that have fail var_filter from adata obj
    adata = adata[:,corr[1]].copy()
    adata.var['corr'] = corr[0].flatten()
    
    return adata


def control_corr(adata, b=1000, update=True, ct=False):

    '''Calculates pearson corr for controls.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object of shape (#cells,#peaks). Contains DataFrame under mdata.uns.peak_gene_pairs
        containing columns ['index_x','index_y'] that correspond to peak and gene indices in atac.X
        and rna.X respectively, as well as mdata.uns.control_peaks containing randomly generated peaks.
        If mdata.uns.control_peaks does not exist, will create it.
    b : int
        Number of desired random peaks per focal peak-gene pair. Should match
        mdata.uns['control_peaks'].shape[1].
    update : bool
        If True, updates original AnnData with adata.varm['control_corr']
    ct : bool
        If True, identifies using original atac.X information, as control pairs are not limited to pairs
        highly expressed within CT.

    
    Returns
    ----------
    ctrl_corr : np.ndarray
        Array of shape (N,B).
    
    '''

    try:
        adata.varm['control_peaks']
    except KeyError:
        print('Adding control_peaks.')
        create_ctrl_peaks(adata,b=b)

    ctrl_peaks = adata.varm['control_peaks']

    if ct:
        atac_Xs = adata.uns['original_atac']
    else:
        atac_Xs = adata.layers['atac']
    rna_Xs = adata.layers['rna']
    
    ctrl_corr = np.empty([adata.shape[1], b])
    for row in tqdm(range(adata.shape[1])):
        ctrl_corr[row,:] = pearson_corr_sparse(atac_Xs[:,ctrl_peaks[row]],rna_Xs[:,[row]])

    if update:
        adata.varm['control_corr'] = ctrl_corr
    
    return ctrl_corr


def get_pvals(adata, control_metric='control_corr', metric='corr', alpha=0.05):

    ''' Adds mc_pval and mc_qval to AnnData.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object of shape (#cells,#peaks). Should contain metric under adata.var,
        and control_metric under adata.varm.
    control_metric : str
        Column name in adata.varm (pd.DataFrame of length #peaks) pertaining to 
        control metric, e.g. 'control_corr' or 'delta_control_corr'.
    metric : str
        Column name in adata.varm (pd.DataFrame of length #peaks) pertaining to 
        putative metric, e.g. 'corr' or 'delta_corr'.
    alpha : int
        Alpha threshold for BH FDR correction.
    
    Returns
    ----------
    adata : ad.AnnData
        AnnData of shape (#cells,#peaks) updated with mc_pval and mc_qval.

    '''
    
    # Adds mc_pval to AnnData
    adata.var['mc_pval'] = mc_pval(adata.varm[control_metric], adata.var[metric].values)
    
    # Adds BH FDR qvals to AnnData
    adata.var['mc_qval'] = stats.multitest.multipletests(adata.var['mc_pval'].values, alpha=alpha, method='fdr_bh')[1]
    
    return adata


def filter_lowexp(atac, rna, min_pct=0.05, min_mean=0):
    
    ''' Filter out cells with few expressing cells or lower absolute mean expression.

    Parameters
    ----------
    atac : sp.sparse_array
        Sparse array of shape (#cells,#peaks) containing raw ATAC data.
    atac : sp.sparse_array
        Sparse array of shape (#cells,#peaks) containing raw RNA data.
    min_pct : float
        Value between [0,1] of the minimum number of nonzero ATAC and RNA
        values for a given link.
    min_pct : float
        Minimum absolute mean for ATAC and RNA expression.
    
    Returns
    ----------
    lowexp_mask : np.array (dtype: bool)
        Boolean array of shape (#peaks) dictating which peaks were filtered out.

    '''

    mean_atac, mean_rna = np.abs(atac.mean(axis=0)), np.abs(rna.mean(axis=0))
    nnz_atac, nnz_rna = atac.getnnz(axis=0) / atac.shape[0], rna.getnnz(axis=0) / atac.shape[0]
    lowexp_mask = (((nnz_atac > min_pct) & (nnz_rna > min_pct)) & ((mean_atac > min_mean) & (mean_rna > min_mean))).A1
    
    return lowexp_mask



def filter_vars(adata, min_pct=0.05, min_mean=0.1):
    
    ''' Returns var filtered adata and its bool mask.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object of shape (#cells,#peaks), typically for a certain celltype.
    min_pct : float
        Value between [0,1] of the minimum number of nonzero ATAC and RNA
        values for a given link.
    min_pct : float
        Minimum absolute mean for ATAC and RNA expression.
    
    Returns
    ----------
    ct_adata : ad.AnnData
        AnnData of shape (#cells,#peaks-peaks failing filter).
    lowexp_mask : np.array (dtype: bool)
        Boolean array of shape (#peaks) dictating which peaks were filtered out.

    '''

    # Check if at least min_pct of cells are nonzero and above min_mean for ATAC and RNA values
    lowexp_mask = filter_lowexp(adata.layers['atac_raw'], adata.layers['rna_raw'])

    # Filter out lowly expressed
    ct_adata = adata[:,lowexp_mask]
    
    return ct_adata, lowexp_mask


def filter_ct(adata, ct_key):
    
    ''' Returns CT-specific adata.
    
    Parameters
    ----------
    adata : ad.AnnData
        AnnData object of shape (#cells,#peaks).
    ct_key : int or str
        The key in the adata.uns['ct_labels'] dictionary for the desired celltype.
    
    Returns
    ----------
    ct_adata : ad.AnnData
        AnnData of shape (#cells within ct,#peaks) containing only cells of a given celltype. 

    '''
    
    # Get CT label
    ct = adata.uns['ct_labels'][ct_key]

    # Filter for CT
    ct_adata = adata[adata.obs['celltype'] == ct].copy()

    # Remove general corrs if they exist
    try: 
        ct_adata.var.drop(columns=['corr','mc_pval'],inplace=True)
        del ct_adata.varm['control_corr']
    except KeyError as e: pass

    # Keep only specific label
    ct_adata.uns['ct_labels'].clear()
    ct_adata.uns['ct_labels'] = dict(ct_key=ct)
    
    return ct_adata


def build_ct_adata(adata, ct_key, min_pct=0.05, min_mean=0.1):

    ''' Returns CT-specific adata.
    
    Parameters
    ----------
    adata : ad.AnnData
        AnnData object of shape (#cells,#peaks).
    ct_key : int or str
        The key in the adata.uns['ct_labels'] dictionary for the desired celltype.
    
    Returns
    ----------
    ct_adata : ad.AnnData
        AnnData of shape (#cells within ct,#peaks) containing only cells of a given celltype,
        with lowly expressed links also filtered out. 

    '''
    
    ct_adata = filter_ct(adata,ct_key)
    ct_adata,ct_le_mask = filter_vars(ct_adata,min_pct,min_mean)

    # ct label
    ct = ct_adata.uns['ct_labels']['ct_key']
    
    # Add mask to original adata object for future reference
    adata.varm['lowexp_ct_mask'][ct] = ct_le_mask

    # we store original atac_X in uns for calculating the control corr 
    # as original control corr still includes peaks removed by lowexp masks
    ct_adata.uns['original_atac'] = adata.layers['atac'][(adata.obs['celltype'] == ct),:]

    return ct_adata


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


def get_poiss_coeff(adata,layer='raw',binarize=False,label='poiss_coeff'):

    # If binarizing, convert to bool then int
    # so that statsmodel can handle it
    if binarize:
        x = adata.layers['atac_'+layer].astype(bool).astype(int)
    else:
        x = adata.layers['atac_'+layer]
    if sp.sparse.issparse(x):
        x = x.A
    y = adata.layers['rna_'+layer]
    if sp.sparse.issparse(y):
        y = y.A

    coeffs = []
    failed = []

    # Calculate poisson coefficient for each peak gene pair
    for i in tqdm(np.arange(adata.shape[1])):
        coeff_ = fit_poisson(x[:,i],y[:, i])
        if not coeff_.any(): failed.append(i)
        else: coeffs.append(coeff_)

    # Remove pairs that have fail poisson
    idx = np.arange(adata.shape[1])
    idx = np.delete(idx,failed)
    adata = adata[:,idx].copy()
    # Save ATAC term coefficient in adata.var
    adata.var[label] = np.array(coeffs)

    return adata


def build_other_adata(adata,ct_key):
    ''' Returns adata with everything except CT in ct_key.
    '''
    ct = adata.uns['ct_labels'][ct_key]
    
    other_adata = adata[~(adata.obs['celltype'] == ct),:]
    other_adata = other_adata[:,adata.varm['lowexp_ct_mask'][ct].values].copy()
    
    other_adata.var.drop(columns=['corr','mc_pval'],inplace=True)
    del other_adata.varm['control_corr']
    
    other_adata.uns['ct_labels'].clear()
    other_adata.uns['ct_labels'] = dict(ct_key=ct)

    # for other_adata, we store original atac_X in uns for calculating the control corr 
    # as original control corr still includes peaks removed by lowexp masks
    other_adata.uns['original_atac'] = adata.layers['atac'][~(adata.obs['celltype'] == ct),:]

    return other_adata

def get_deltas(ct_adata,other_adata):
    deltas = ct_adata.var['corr'].values - other_adata.var['corr'].values
    ct_adata.var['delta_corr'] = deltas
    return deltas

def get_control_deltas(ct_adata,other_adata):
    deltas = ct_adata.varm['control_corr'] - other_adata.varm['control_corr']
    ct_adata.varm['delta_control_corr'] = deltas
    return deltas


def stratified_adata(adata,neighbors='leiden'):
    ''' Stratify ATAC and RNA information within neighborhood groupings by
    descending count.
    
    Parameters
    ----------
    adata : AnnData
        AnnData peak-gene linked object with layers ['atac_raw'] and ['rna_raw'].
    neighbors : str
        The column name in adata.obs containing neighborhood groupings.

    Returns
    -------
    adata : AnnData
        adata object with new layers ['atac_strat'] and ['rna_strat']. The vars axis
        will remain the same, while the obs axis will be sorted independently within
        neighborhood groupings in descending peak acc. and gene exp. order.
    
    '''
    
    # get the sizes of each neighborhood group
    neighborhood_sizes = adata.obs[neighbors].value_counts(sort=False)
    
    # sort adata by neighborhood grouping
    stratified_order = adata.obs[neighbors].sort_values().index
    sorted_adata = adata[stratified_order,:]
    atac = sorted_adata.layers['atac_raw'].A
    rna = sorted_adata.layers['rna_raw'].A

    stratified_atac, stratified_rna = build_strat_layers(atac, rna, neighborhood_sizes)

    adata.layers['atac_strat'] = stratified_atac
    adata.layers['rna_strat'] = stratified_rna

    return adata


def build_strat_layers(atac,rna,neighborhood_sizes):
    
    stratified_atac = []
    stratified_rna = []
    
    for i in neighborhood_sizes.values:
        
        # get first group of neighbors
        # sort in descending order
        atac_i = -np.sort(-atac[:i,:],axis=0)
        
        # remove the first group of neighbors
        atac = atac[i:,:]

        # append to list
        stratified_atac.append(atac_i)


        if rna is not None:

            rna_i = -np.sort(-rna[:i,:],axis=0)
            rna = rna[i:,:]
            stratified_rna.append(rna_i)

    # turn into arrays
    stratified_atac = np.vstack(stratified_atac)
    
    if rna is not None:
        stratified_rna = np.vstack(stratified_rna)
        return stratified_atac,stratified_rna

    return stratified_atac


def shuffled_poiss_coeff(adata,neighbors='leiden',b=200):

    ''' Shuffle peaks row-wise, then compute the poisson coefficient, b times.
    
    Parameters
    ----------
    adata : AnnData
        AnnData peak-gene linked object with layers ['atac_strat'] and ['rna_strat'].
    neighbors : str
        The column name in adata.obs containing neighborhood groupings.
    b : int
        The number of repetitions/times to shuffle.

    Returns
    -------
    coeffs : np.array
        Array of shape (#links,b), containing poisson coefficient for shuffled,
        stratified peak-gene pairs.
    
    '''

    # mantain neighborhood sizes from original adata.obs
    neighborhood_sizes = adata.obs[neighbors].value_counts(sort=False)

    coeffs = []
    
    for j in np.arange(b):
        
        atac = adata.layers['atac_strat']
        rna = adata.layers['rna_strat']
        
        # shuffle peaks only
        inds = np.arange(len(atac))
        np.random.shuffle(inds)
        atac = atac[inds]

        # arange shuffled rows into descending order within a strata
        atac = build_strat_layers(atac, None, neighborhood_sizes)
    
        coeffs_i = []
        for i in np.arange(adata.shape[1]):
            # obtain poisson coeff for all peak-gene pairs for shuffled cells 
            coeff_ = fit_poisson(atac[:,i],rna[:, i])
            coeffs_i.append(coeff_)
        coeffs_i = np.array(coeffs_i)
        coeffs.append(coeffs_i)
        
    coeffs = np.vstack(coeffs)

    return coeffs.T


