import numpy as np
import pandas as pd 
import scipy as sp
import anndata as ad
from ctar.data_loader import get_gene_coords



def null_peak_gene_pairs(rna, atac, gene_tss=None, gene_col='gene_name', peak_col='gene_ids'):

    ''' Generate null peak gene pairs in which genes and peaks are on different chromosomes.

    Parameters
    ----------
    rna : an.AnnData
        AnnData of len (#genes). Must contain rna.var DataFrame with `gene_tss` describing
        gene region and `gene_col` listing gene names. If gene_tss is None, uses ctar
        get_gene_coords to find gene body.
    atac : an.AnnData
        AnnData of len (#peaks). Must contain rna.var DataFrame with `peak_col` describing
        peak region.

    Returns
    -------
    null_pairs : pd.DataFrame
        DataFrame with gene_ids,gene_name, index_x, and index_y where index_x describes
        indices pertaining to its original AnnData atac index and index_y describes
        indices pertaining to its original AnnData rna index.
    
    '''

    # Add indices to rna and atac var
    atac.var['index_x'] = range(len(atac.var))
    rna.var['index_y'] = range(len(rna.var))

    # Add chrom information to rna and atac var
    atac.var[['chr','range']] = atac.var[peak_col].str.split(':', n=1, expand=True)
    if gene_tss is None:
        rna.var = get_gene_coords(rna.var, add_tss=False)
        gene_tss = 'interval'
    rna.var[['chr','range']] = rna.var[gene_tss].str.split(':', n=1, expand=True)

    df_list = []

    # For each unique chrom (23 + extrachromosomal)
    for chrom in atac.var.chr.unique():
        
        # Find corresponding chrom
        chrm_peaks = atac.var.loc[atac.var['chr'] == chrom][['index_x',peak_col]]
        # Find genes NOT on that chrom
        nonchrm_genes = rna.var.loc[rna.var['chr'] != chrom][['index_y',gene_col]]
        # Sample random genes
        rand_genes = nonchrm_genes.sample(n=len(chrm_peaks), replace=True)
        # Concat into one df
        rand_peak_gene_pairs = pd.concat([chrm_peaks.reset_index(drop=True),rand_genes.reset_index(drop=True)],axis=1)
        df_list += [rand_peak_gene_pairs]

    # Concat dfs for all chroms
    null_pairs = pd.concat(df_list,ignore_index=True)
    return null_pairs


def simulate_full_dataset(n_cells, n_tp_per_level, n_tn_per_level, n_bg_peaks,
                          alpha_levels, mean_expr_levels, beta, p_peak, seed=None):
    """
    Simulate different expression levels together in one dataset.

    TP genes: RNA ~ Poisson(exp(alpha + beta * assigned_peak))
    TN genes: RNA ~ Poisson(exp(alpha))   — no peak dependency

    Returns
    -------
    atac_sparse  : csc_matrix  (n_cells, n_bg_peaks)
    rna_sparse   : csc_matrix  (n_cells, n_total_genes)
    cis_links_df : DataFrame   columns: atac_idx, rna_idx, is_tp, mean_expr, alpha
    """
    rng = np.random.default_rng(seed)
    n_per_level   = n_tp_per_level + n_tn_per_level
    n_total_genes = len(alpha_levels) * n_per_level

    # Peak accessibility: Bernoulli(p_peak) for all cells and peaks
    atac_dense = rng.binomial(1, p_peak, size=(n_cells, n_bg_peaks)).astype(np.float32)
    rna_dense  = np.zeros((n_cells, n_total_genes), dtype=np.float32)

    rows        = []
    gene_offset = 0

    for alpha, mean_expr in zip(alpha_levels, mean_expr_levels):

        # TP: each gene is assigned a specific peak; RNA depends on that peak
        tp_peak_idxs = rng.choice(n_bg_peaks, size=n_tp_per_level, replace=False)
        for j in range(n_tp_per_level):
            gene_idx = gene_offset + j
            peak_idx = int(tp_peak_idxs[j])
            lam = np.exp(alpha + beta * atac_dense[:, peak_idx])
            rna_dense[:, gene_idx] = rng.poisson(lam).astype(np.float32)
            rows.append(dict(atac_idx=peak_idx, rna_idx=gene_idx,
                             is_tp=True, mean_expr=mean_expr, alpha=alpha))
        gene_offset += n_tp_per_level

        # TN: RNA is independent of any peak
        tn_peak_idxs = rng.integers(0, n_bg_peaks, size=n_tn_per_level)
        for j in range(n_tn_per_level):
            gene_idx = gene_offset + j
            peak_idx = int(tn_peak_idxs[j])
            rna_dense[:, gene_idx] = rng.poisson(np.exp(alpha), size=n_cells).astype(np.float32)
            rows.append(dict(atac_idx=peak_idx, rna_idx=gene_idx,
                             is_tp=False, mean_expr=mean_expr, alpha=alpha))
        gene_offset += n_tn_per_level

    atac_sparse  = sp.sparse.csc_matrix(atac_dense)
    rna_sparse   = sp.sparse.csc_matrix(rna_dense)
    cis_links_df = pd.DataFrame(rows).reset_index(drop=True)

    return atac_sparse, rna_sparse, cis_links_df


def build_anndata(sparse_mat, feature_ids, feature_col, layer='counts'):
    """Minimal AnnData wrapper."""
    n_cells = sparse_mat.shape[0]
    var  = pd.DataFrame({feature_col: feature_ids}, index=feature_ids)
    obs  = pd.DataFrame(index=[f'cell_{i}' for i in range(n_cells)])
    adata = ad.AnnData(X=sparse_mat, obs=obs, var=var)
    adata.layers[layer] = sparse_mat
    return adata