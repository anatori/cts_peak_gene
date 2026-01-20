from __future__ import annotations

import statsmodels.api as sm
import numpy as np
from scipy.stats import spearmanr
import pandas as pd
from tqdm import tqdm
from scipy.stats import rankdata


def residualize_atac(atac_counts, offsets, batch_labels, eps=0):
    """
    Residualizes ATAC data using a Negative Binomial (NB) Generalized Linear Model (GLM) with offsets.

    Parameters:
    ----------
    atac_counts : np.ndarray
        Array of shape (n_cells, n_peaks) representing ATAC fragment counts per peak.
    offsets : np.ndarray
        Array of shape (n_cells,) representing offsets (log fragment counts per cell).
    batch_labels : np.ndarray
        Array of shape (n_cells,) representing batch labels.
        
    Returns:
    -------
    atac_residuals : np.ndarray
        Residualized ATAC matrix of shape (n_cells, n_peaks).
    """
    
    n_cells, n_peaks = atac_counts.shape
    atac_residuals = np.zeros_like(atac_counts, dtype=float)

    # Dummy-coding batch labels
    batch_dummy = pd.get_dummies(batch_labels,drop_first=True)

    # Mean-center offset to stabilize optimization (intercept absorbs it)
    offsets = offsets - np.nanmean(offsets)

    for peak_idx in tqdm(range(n_peaks)):
        y = atac_counts[:, peak_idx]
        model = sm.GLM(y, np.column_stack([batch_dummy, np.ones_like(y)]), 
                       family=sm.families.Poisson(), offset=offsets)
        result = model.fit()
        mu = result.fittedvalues
        atac_residuals[:, peak_idx] = (y - mu) / np.sqrt(mu + eps)

    return atac_residuals


def residualize_rna(rna_counts, offsets, batch_labels, alpha=1, eps=0):
    """
    Residualizes RNA data using a Negative Binomial (NB) Generalized Linear Model (GLM) with offsets.

    Parameters:
    ----------
    rna_counts : np.ndarray
        Array of shape (n_cells, n_genes) representing RNA read counts per gene.
    offsets : np.ndarray
        Array of shape (n_cells,) representing offsets (log library size per cell).
    batch_labels : np.ndarray
        Array of shape (n_cells,) representing batch labels.
        
    Returns:
    -------
    rna_residuals : np.ndarray
        Residualized RNA matrix of shape (n_cells, n_genes).
    """

    n_cells, n_genes = rna_counts.shape
    rna_residuals = np.zeros_like(rna_counts, dtype=float)

    # Dummy-coding for batch labels
    batch_dummy = pd.get_dummies(batch_labels,drop_first=True)

    # Mean-center offset to stabilize optimization (intercept absorbs it)
    offsets = offsets - np.nanmean(offsets)

    for gene_idx in tqdm(range(n_genes)):
        y = rna_counts[:, gene_idx]
        model = sm.GLM(y, np.column_stack([batch_dummy, np.ones_like(y)]),
                       family=sm.families.NegativeBinomial(alpha=alpha), offset=offsets)
        result = model.fit()
        mu = result.fittedvalues
        var = mu + alpha * (mu ** 2)
        rna_residuals[:, gene_idx] = (y - mu) / np.sqrt(var + eps)

    return rna_residuals


def compute_spearman_correlation(atac_residuals, rna_residuals, peak_gene_pairs):
    """
    Computes Spearman correlations for ATAC and RNA residualized data over cis peak-gene pairs.

    Parameters:
    ----------
    atac_residuals : np.ndarray
        Residualized ATAC matrix of shape (n_cells, n_peaks).
    rna_residuals : np.ndarray
        Residualized RNA matrix of shape (n_cells, n_genes).
    peak_gene_pairs : pd.DataFrame
        DataFrame with columns 'peak_idx' and 'gene_idx' representing cis peak-gene pairs.
        
    Returns:
    -------
    correlations : np.array
        Array of Spearman correlation coefficients for each peak-gene pair.
    """
    correlations = []

    for _, row in peak_gene_pairs.iterrows():
        peak_idx = row['index_x']
        gene_idx = row['index_y']
        corr, _ = spearmanr(atac_residuals[:, peak_idx], rna_residuals[:, gene_idx])
        correlations.append(corr)

    return np.array(correlations)


######################### optimized spearman corr #########################


def _ranks_ordinal_fast(A: np.ndarray, dtype=np.float32) -> np.ndarray:
    """
    Fast ordinal ranks for all columns of A (n x k), ties NOT averaged.
    Returns ranks starting at 1, dtype float32/float64.
    """
    n, k = A.shape
    order = np.argsort(A, axis=0, kind="quicksort")
    R = np.empty_like(order, dtype=dtype)
    cols = np.broadcast_to(np.arange(k), (n, k))
    R[order, cols] = (np.arange(n, dtype=dtype)[:, None] + dtype(1))
    return R


def _ranks_average(A: np.ndarray, dtype=np.float32, block_size: int | None = None) -> np.ndarray:
    """
    Tie-aware ranks (average method) column-wise using SciPy. Falls back to ordinal_fast
    if SciPy is unavailable. Optional block processing to limit peak memory.
    """
    n, k = A.shape
    if not _SCIPY_AVAILABLE:
        return _ranks_ordinal_fast(A, dtype=dtype)

    if block_size is None or block_size >= k:
        cols = [_scipy_rankdata(A[:, j], method="average").astype(dtype) for j in range(k)]
        return np.column_stack(cols)

    R = np.empty((n, k), dtype=dtype)
    for start in range(0, k, block_size):
        end = min(start + block_size, k)
        block_cols = [_scipy_rankdata(A[:, j], method="average").astype(dtype) for j in range(start, end)]
        R[:, start:end] = np.column_stack(block_cols)
    return R


def build_rank_stats(A: np.ndarray, method: str = "ordinal_fast", dtype=np.float32,
                     block_size: int | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build rank matrix and per-column mean/std for A (n x k).

    Parameters
    ----------
    A : np.ndarray
        Data matrix (n_cells x n_features).
    method : {'ordinal_fast','average'}
        Rank method. 'ordinal_fast' is fastest; 'average' is tie-aware via SciPy.
    dtype : np.dtype
        Ranks dtype (float32 recommended for memory).
    block_size : int or None
        Only used for 'average' to process columns in blocks.

    Returns
    -------
    R : (n x k) rank matrix
    mean : (k,) mean of ranks per column
    std  : (k,) std of ranks per column (ddof=1)
    """
    if method == "ordinal_fast":
        R = _ranks_ordinal_fast(A, dtype=dtype)
    elif method == "average":
        R = _ranks_average(A, dtype=dtype, block_size=block_size)
    else:
        raise ValueError("method must be 'ordinal_fast' or 'average'")

    mean = R.mean(axis=0).astype(dtype)
    std = R.std(axis=0, ddof=1).astype(dtype)
    return R, mean, std


def spearman_cis_corr_precomputed_fast(
    Xr_atac: np.ndarray,
    atac_mean: np.ndarray,
    atac_std: np.ndarray,
    Yr_rna: np.ndarray,
    rna_mean: np.ndarray,
    rna_std: np.ndarray,
    cell_block_size: int = 4096,
    dtype_out: np.dtype = np.float32,
) -> np.ndarray:
    """
    Fast, memory-safe Spearman correlations for matched control peak-gene columns,
    using precomputed rank matrices and per-column rank stats.

    This function assumes:
      - Xr_atac is the ATAC rank matrix (n_cells x k) for the control peaks in a bin
      - Yr_rna  is the RNA  rank matrix (n_cells x k) for the matched control genes in the same bin
      - atac_mean/atac_std and rna_mean/rna_std are the per-column means and stds of ranks
        corresponding to the columns of Xr_atac and Yr_rna (length k)

    It computes, for each control pair j, the Spearman correlation:
        corr_j = sum_i (Xr_atac[i,j] - atac_mean[j]) * (Yr_rna[i,j] - rna_mean[j]) / ((n-1) * atac_std[j] * rna_std[j])

    Parameters
    ----------
    Xr_atac : np.ndarray
        Rank matrix for ATAC controls, shape (n_cells, k).
    atac_mean : np.ndarray
        Mean of ranks per ATAC control column, shape (k,).
    atac_std : np.ndarray
        Std of ranks per ATAC control column (ddof=1), shape (k,).
    Yr_rna : np.ndarray
        Rank matrix for RNA controls, shape (n_cells, k), aligned column-wise to Xr_atac.
    rna_mean : np.ndarray
        Mean of ranks per RNA control column, shape (k,).
    rna_std : np.ndarray
        Std of ranks per RNA control column (ddof=1), shape (k,).
    cell_block_size : int
        Number of cells to process per block to limit memory. Tune based on RAM (e.g., 2048–8192).
    dtype_out : np.dtype
        Output dtype for correlations (np.float32 recommended to save memory).

    Returns
    -------
    corr : np.ndarray
        Spearman correlations for each matched control pair, shape (k,), dtype=dtype_out.
        Entries with zero variance or invalid denominators are set to NaN.
    """
    # Basic shape checks
    n_cells, k = Xr_atac.shape
    assert Yr_rna.shape == (n_cells, k), "Yr_rna must match Xr_atac shape (n_cells, k)"
    assert atac_mean.shape[0] == k and atac_std.shape[0] == k, "ATAC mean/std must have length k"
    assert rna_mean.shape[0] == k and rna_std.shape[0] == k, "RNA mean/std must have length k"

    # Numerator accumulator per column
    num = np.zeros(k, dtype=np.float64)

    # Cell-block accumulation of centered rank cross-products
    for cstart in range(0, n_cells, cell_block_size):
        cend = min(cstart + cell_block_size, n_cells)
        # Convert to float64 for stable accumulation
        Xb = Xr_atac[cstart:cend, :].astype(np.float64, copy=False)
        Yb = Yr_rna[cstart:cend, :].astype(np.float64, copy=False)
        # Center ranks per column
        Xb_center = Xb - atac_mean[None, :]
        Yb_center = Yb - rna_mean[None, :]
        # Sum over rows for each column
        num += np.sum(Xb_center * Yb_center, axis=0)

    # Denominator per column
    den = (n_cells - 1) * atac_std.astype(np.float64) * rna_std.astype(np.float64)

    # Final correlations with validity mask
    corr = np.full(k, np.nan, dtype=dtype_out)
    valid = np.isfinite(den) & (den > 0)
    corr[valid] = (num[valid] / den[valid]).astype(dtype_out)

    return corr


def spearman_cis_corr_precomputed_chunked(
    Xr_atac: np.ndarray, atac_mean: np.ndarray, atac_std: np.ndarray,
    Yr_rna:  np.ndarray, rna_mean:  np.ndarray,  rna_std:  np.ndarray,
    pairs_df: pd.DataFrame,
    peak_col: str = "index_x",
    gene_col: str = "index_y",
    pair_batch_size: int = 10000,
    cell_block_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Memory-safe CIS Spearman correlations using precomputed ranks with chunking over pairs and cells.

    For a batch of pairs, we:
      - collect unique peaks and genes
      - accumulate cross-products M = X_ranks[:, uniq_peaks]^T @ Y_ranks[:, uniq_genes]
        in small cell blocks to avoid large allocations
      - compute corr for each pair from M and precomputed means/stds

    Parameters
    ----------
    Xr_atac, atac_mean, atac_std : ATAC ranks and stats (n_cells x n_peaks, k,)
    Yr_rna,  rna_mean,  rna_std  : RNA  ranks and stats  (n_cells x n_genes, k,)
    pairs_df : DataFrame with integer indices [peak_col, gene_col]
    pair_batch_size : number of pairs per batch
    cell_block_size : number of cells per block (tune to fit RAM); e.g., 2048–8192

    Returns
    -------
    corr_all : (n_pairs,) float32
    keep_all : (n_pairs,) bool
    """
    n_cells = Xr_atac.shape[0]
    n_pairs = pairs_df.shape[0]
    corr_all = np.full(n_pairs, np.nan, dtype=np.float32)
    keep_all = np.zeros(n_pairs, dtype=bool)

    # Iterate over pair batches
    for start in range(0, n_pairs, pair_batch_size):
        end = min(start + pair_batch_size, n_pairs)
        batch = pairs_df.iloc[start:end]

        peaks = batch[peak_col].to_numpy(dtype=int)
        genes = batch[gene_col].to_numpy(dtype=int)

        uniq_peaks = np.unique(peaks)
        uniq_genes = np.unique(genes)

        # Map original indices to positions in the unique lists
        peak_pos = {p: i for i, p in enumerate(uniq_peaks)}
        gene_pos = {g: i for i, g in enumerate(uniq_genes)}

        # Accumulate cross-products in a small matrix
        M = np.zeros((uniq_peaks.shape[0], uniq_genes.shape[0]), dtype=np.float64)

        for cstart in range(0, n_cells, cell_block_size):
            cend = min(cstart + cell_block_size, n_cells)
            X_block = Xr_atac[cstart:cend, :][:, uniq_peaks]   # (cells_block x P)
            Y_block = Yr_rna[cstart:cend, :][:, uniq_genes]    # (cells_block x G)
            # Accumulate dot-products
            M += X_block.T @ Y_block

        # Now compute corr per pair using M and rank stats
        Xstd = atac_std[peaks].astype(np.float64)
        Ystd = rna_std[genes].astype(np.float64)
        mx   = atac_mean[peaks].astype(np.float64)
        my   = rna_mean[genes].astype(np.float64)

        # Validity mask
        keep = (Xstd > 0) & (Ystd > 0) & np.isfinite(Xstd) & np.isfinite(Ystd)

        # Numerator: sum(x*y) - n * mx * my
        num = np.empty(end - start, dtype=np.float64)
        den = (n_cells - 1) * Xstd * Ystd
        for i, (p, g) in enumerate(zip(peaks, genes)):
            num[i] = M[peak_pos[p], gene_pos[g]] - n_cells * mx[i] * my[i]

        # Assign results
        corr_batch = np.full(end - start, np.nan, dtype=np.float32)
        valid = keep & np.isfinite(den) & (den != 0)
        corr_batch[valid] = (num[valid] / den[valid]).astype(np.float32)
        corr_all[start:end] = corr_batch
        keep_all[start:end] = valid

    return corr_all, keep_all


def spearman_ctrl_corr_per_gene_precomputed_blocked(
    Xr_atac: np.ndarray, atac_mean: np.ndarray, atac_std: np.ndarray,
    Yr_rna:  np.ndarray, rna_mean:  np.ndarray,  rna_std:  np.ndarray,
    pairs_df: pd.DataFrame,
    ctrl_idx: np.ndarray,
    peak_col: str = "index_x",
    gene_col: str = "index_y",
    cell_block_size: int = 4096,
) -> np.ndarray:
    """
    Control Spearman correlations per cis pair, grouped by gene, using precomputed ranks
    and cell-block accumulation for memory efficiency.

    For gene g:
      - compute centered ranks y_c = Yr[:, g] - mean_y[g]
      - gather unique control peaks across the group's pairs
      - accumulate num = sum_i Xc[i,j] * y_c[i] for all controls in cell blocks
      - corr = num / ((n-1)*std_x[j]*std_y[g])

    Returns
    -------
    ctrl_corr : (n_pairs x n_ctrl) float32 array of Spearman correlations.
    """
    n_cells = Xr_atac.shape[0]
    n_pairs = pairs_df.shape[0]
    n_ctrl = int(ctrl_idx.shape[1])
    ctrl_corr = np.full((n_pairs, n_ctrl), np.nan, dtype=np.float32)

    pairs_df = pairs_df.reset_index(drop=True)
    grouped = pairs_df.groupby(gene_col)

    for gene_idx, grp in tqdm(grouped):
        gene_idx = int(gene_idx)
        # Centered RNA ranks for this gene
        y_r = Yr_rna[:, gene_idx].astype(np.float64)
        y_c = y_r - float(rna_mean[gene_idx])
        y_std = float(rna_std[gene_idx])
        if not (np.isfinite(y_std) and y_std > 0):
            continue

        peak_inds = grp[peak_col].to_numpy(dtype=int)    # (#pairs_g,)
        ctrl_for_grp = ctrl_idx[peak_inds, :]            # (#pairs_g, n_ctrl)
        uniq_ctrl = np.unique(ctrl_for_grp.ravel())      # (n_unique_ctrl,)
        pos_map = {p: i for i, p in enumerate(uniq_ctrl)}

        Xstd = atac_std[uniq_ctrl].astype(np.float64)
        mx   = atac_mean[uniq_ctrl].astype(np.float64)

        # Accumulate num = sum_i Xc[i,j] * y_c[i] for all controls
        num = np.zeros(uniq_ctrl.shape[0], dtype=np.float64)
        for cstart in range(0, n_cells, cell_block_size):
            cend = min(cstart + cell_block_size, n_cells)
            X_block = Xr_atac[cstart:cend, :][:, uniq_ctrl].astype(np.float64)  # (cells_block x U)
            # Xc = X_block - mx[None, :]
            X_block -= mx[None, :]
            yc_block = y_c[cstart:cend]
            num += X_block.T @ yc_block

        den = (n_cells - 1) * Xstd * y_std
        ok = (Xstd > 0) & np.isfinite(den) & (den != 0)

        corr_all = np.full_like(num, np.nan, dtype=np.float32)
        corr_all[ok] = (num[ok] / den[ok]).astype(np.float32)

        # Fill outputs for each pair in this gene group
        for local_row, pair_row in enumerate(grp.index):
            ctrl_peaks = ctrl_for_grp[local_row, :]
            pos = [pos_map[p] for p in ctrl_peaks]
            ctrl_corr[pair_row, :] = corr_all[pos]

    return ctrl_corr