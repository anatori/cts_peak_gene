from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Literal, Tuple, Optional
import statsmodels.api as sm
from scipy.optimize import brentq
from scipy.stats import rankdata, spearmanr, nbinom, poisson, norm


######################### shared helpers #########################


def _ensure_log_offset(offsets: np.ndarray) -> np.ndarray:
    offsets = np.asarray(offsets, dtype=np.float64).reshape(-1)
    if np.nanmedian(offsets) > 50 and np.nanmax(offsets) > 100:
        offsets = np.log(offsets + 1e-8)
    return offsets - np.nanmean(offsets)

def _design_from_batch(batch_labels: np.ndarray) -> np.ndarray:
    X = pd.get_dummies(pd.Series(batch_labels), drop_first=True).to_numpy(dtype=np.float64)
    X = sm.add_constant(X, has_constant="add")
    return X

def _alpha_moments(y: np.ndarray, mu: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    num = np.sum((y - mu) ** 2 - mu)
    den = np.sum(mu ** 2) + 1e-12
    return float(max(num / den, 0.0))

def _alpha_chisq_match(y: np.ndarray, mu: np.ndarray, df_resid: int) -> float:
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    df_resid = max(int(df_resid), 1)
    def f(alpha: float) -> float:
        return np.sum((y - mu) ** 2 / (mu + alpha * mu ** 2 + 1e-12)) - df_resid
    lo, hi = 0.0, 1.0
    for _ in range(30):
        if f(hi) < 0.0:
            break
        hi *= 2.0
    if f(lo) * f(hi) > 0:
        return _alpha_moments(y, mu)
    return float(max(brentq(f, lo, hi), 0.0))

def _pearson_residual(y: np.ndarray, mu: np.ndarray, alpha: float, eps: float) -> np.ndarray:
    var = mu + alpha * (mu ** 2)
    return (y - mu) / np.sqrt(np.maximum(var, eps))

def _dunn_smyth_residual(y: np.ndarray, mu: np.ndarray, alpha: float, random_state: int | None = None) -> np.ndarray:
    y = y.astype(np.int64)
    mu = mu.astype(np.float64)
    if alpha <= 0:
        p_lower = poisson.cdf(y - 1, mu)
        p_upper = poisson.cdf(y, mu)
    else:
        size = 1.0 / max(alpha, 1e-12)
        p = size / (size + mu)
        p_lower = nbinom.cdf(y - 1, size, p)
        p_upper = nbinom.cdf(y, size, p)
    rng = np.random.default_rng(random_state)
    width = np.clip(p_upper - p_lower, 0.0, 1.0)
    u = p_lower + rng.uniform(0.0, 1.0, size=y.shape) * width
    u = np.clip(u, 1e-12, 1 - 1e-12)
    return norm.ppf(u).astype(np.float64)

def _post_standardize_columns(R: np.ndarray, center: bool = True, scale: bool = True) -> np.ndarray:
    """
    Per-column mean-center and/or scale to unit variance.
    Spearman ranks are invariant to this monotone transform.
    """
    Rz = R.copy()
    if center:
        Rz -= Rz.mean(axis=0, keepdims=True)
    if scale:
        sd = Rz.std(axis=0, ddof=1)
        sd[~np.isfinite(sd) | (sd == 0)] = 1.0
        Rz /= sd
    return Rz


######################### optional alpha shrink (only for low-μ) #########################


def _alpha_shrink_low_mu(
    mu_all: np.ndarray,
    alpha_hat: np.ndarray,
    mu_thresh: float = 0.1,
    tau: float = 1e4,
    clip: Tuple[float, float] = (1e-6, 30.0),
) -> np.ndarray:
    """
    Optional: shrink alpha only for low-μ genes (where α̂ is noisy).
    For μ_mean >= mu_thresh, keep α̂. For μ_mean < mu_thresh, shrink in log-space
    toward the global median log(α̂), with weight w = sum(mu^2) / (sum(mu^2) + tau).
    """
    mu_mean = mu_all.mean(axis=0)
    mu2 = (mu_all**2).sum(axis=0)
    alpha_hat = np.clip(alpha_hat, clip[0], None)

    if np.any(mu_mean > mu_thresh):
        global_med = np.median(np.log(alpha_hat[mu_mean > mu_thresh]))
    else:
        global_med = np.median(np.log(alpha_hat))

    w = mu2 / (mu2 + tau)
    log_alpha = np.log(alpha_hat)
    log_alpha_shrunk = np.where(
        mu_mean < mu_thresh,
        w * log_alpha + (1 - w) * global_med,
        log_alpha,
    )
    alpha_shrunk = np.exp(log_alpha_shrunk)
    return np.clip(alpha_shrunk, clip[0], clip[1])


######################### ATAC residualization #########################


def residualize_atac(
    atac_counts: np.ndarray,
    offsets: np.ndarray,
    batch_labels: np.ndarray,
    residual_type: Literal["pearson", "dunn_smyth"] = "dunn_smyth",
    alpha_init: float = 0.5,
    eps: float = 1e-8,
    refit_with_alpha: bool = True,
    post_standardize: bool = False,
    clip_at: Optional[float] = None,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    Y = np.asarray(atac_counts, dtype=np.float64)
    n_cells, n_peaks = Y.shape
    off = _ensure_log_offset(offsets)
    X = _design_from_batch(batch_labels)

    mu_all = np.zeros_like(Y, dtype=np.float64)
    alpha_all = np.zeros(n_peaks, dtype=np.float64)
    resid_all = np.zeros_like(Y, dtype=np.float64)
    log: list[str] = []

    for j in tqdm(range(n_peaks), desc="ATAC residuals"):
        y = Y[:, j]
        if np.all(y == 0):
            mu_all[:, j] = 0.0
            alpha_all[j] = 0.0
            resid_all[:, j] = 0.0
            log.append("all_zero_skip")
            continue

        fam = sm.families.NegativeBinomial(alpha=alpha_init)
        try:
            res = sm.GLM(y, X, family=fam, offset=off).fit()
            mu = res.fittedvalues.astype(np.float64)
            df_resid = int(getattr(res, "df_resid", len(y) - X.shape[1]))
            note = "nb_init"
            if not res.converged:
                res = sm.GLM(y, X, family=fam, offset=off).fit_regularized(alpha=1e-6, L1_wt=0.0)
                mu = res.fittedvalues.astype(np.float64)
                df_resid = int(getattr(res, "df_resid", len(y) - X.shape[1]))
                note = "nb_regularized"
        except Exception:
            res = sm.GLM(y, X, family=sm.families.Poisson(), offset=off).fit()
            mu = res.fittedvalues.astype(np.float64)
            df_resid = int(getattr(res, "df_resid", len(y) - X.shape[1]))
            note = "poisson_fallback"

        a_hat = _alpha_chisq_match(y, mu, df_resid)
        if refit_with_alpha and a_hat > 0:
            try:
                res2 = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=max(a_hat, 1e-8)), offset=off).fit()
                mu = res2.fittedvalues.astype(np.float64)
                note += "->nb_refit"
            except Exception:
                note += "->refit_fail"

        mu_all[:, j] = mu
        alpha_all[j] = a_hat

        r = _pearson_residual(y, mu, a_hat, eps) if residual_type == "pearson" \
            else _dunn_smyth_residual(y, mu, a_hat, random_state)
        if clip_at is not None:
            r = np.clip(r, -clip_at, clip_at)
        resid_all[:, j] = r
        log.append(note)

    if post_standardize:
        resid_all = _post_standardize_columns(resid_all, center=True, scale=True)

    return mu_all, alpha_all, resid_all, log


######################### RNA residualization (no shrink by default) #########################


def residualize_rna(
    rna_counts: np.ndarray,
    offsets: np.ndarray,
    batch_labels: np.ndarray,
    residual_type: Literal["pearson", "dunn_smyth"] = "pearson",
    alpha_init: float = 0.5,
    eps: float = 1e-8,
    refit_with_alpha: bool = True,
    shrink_alpha_low_mu: bool = False,   # keep available, but OFF by default to preserve var≈1 calibration
    mu_thresh: float = 0.1,
    tau: float = 1e4,
    clip_at: float = 10.0,
    post_standardize: bool = True,       # standardize for cleaner diagnostics (Spearman ranks invariant)
    random_state: Optional[int] = 12345,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    """
    RNA residualization WITHOUT any pre-filtering (matches ATAC behavior):
      - Per-gene NB fit, χ² α̂, optional refit
      - Optional shrink α only for low-μ genes (default: off)
      - Residuals: Pearson (default) or Dunn–Smyth; clipping
      - Optional post-standardization to mean 0, var 1 (Spearman-safe)

    Returns
    -------
    mu_all : (n_cells x n_genes) fitted means
    alpha_hat : (n_genes,) per-gene dispersion (pre-shrink)
    resid_all : (n_cells x n_genes) residuals
    log : list of per-gene fit notes
    alpha_used : (n_genes,) α actually used for residuals (possibly shrunk)
    """
    Y = np.asarray(rna_counts)
    if not np.issubdtype(Y.dtype, np.integer):
        raise ValueError("rna_counts must be integer UMI counts.")
    Y = Y.astype(np.int64)

    n_cells, n_genes = Y.shape
    off = _ensure_log_offset(offsets)
    X = _design_from_batch(batch_labels)

    mu_all = np.zeros((n_cells, n_genes), dtype=np.float64)
    alpha_hat = np.zeros(n_genes, dtype=np.float64)
    resid_all = np.zeros((n_cells, n_genes), dtype=np.float64)
    log: list[str] = []

    # Fit per gene, estimate alpha
    for g in tqdm(range(n_genes), desc="RNA fit"):
        y = Y[:, g].astype(np.float64)
        if np.all(y == 0):
            mu_all[:, g] = 0.0
            alpha_hat[g] = 0.0
            resid_all[:, g] = 0.0
            log.append("all_zero_skip")
            continue

        fam = sm.families.NegativeBinomial(alpha=alpha_init)
        try:
            res = sm.GLM(y, X, family=fam, offset=off).fit()
            mu = res.fittedvalues.astype(np.float64)
            df_resid = int(getattr(res, "df_resid", len(y) - X.shape[1]))
            note = "nb_init"
            if not res.converged:
                res = sm.GLM(y, X, family=fam, offset=off).fit_regularized(alpha=1e-6, L1_wt=0.0)
                mu = res.fittedvalues.astype(np.float64)
                df_resid = int(getattr(res, "df_resid", len(y) - X.shape[1]))
                note = "nb_regularized"
        except Exception:
            res = sm.GLM(y, X, family=sm.families.Poisson(), offset=off).fit()
            mu = res.fittedvalues.astype(np.float64)
            df_resid = int(getattr(res, "df_resid", len(y) - X.shape[1]))
            note = "poisson_fallback"

        a_hat = _alpha_chisq_match(y, mu, df_resid)

        if refit_with_alpha and a_hat > 0:
            try:
                res2 = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=max(a_hat, 1e-8)), offset=off).fit()
                mu = res2.fittedvalues.astype(np.float64)
                note += "->nb_refit"
            except Exception:
                note += "->refit_fail"

        mu_all[:, g] = mu
        alpha_hat[g] = a_hat
        log.append(note)

    # Use α̂ as-is (best calibration for Pearson). Optionally shrink only low-μ genes.
    alpha_use = _alpha_shrink_low_mu(mu_all, alpha_hat, mu_thresh=mu_thresh, tau=tau) if shrink_alpha_low_mu else alpha_hat.copy()

    rng = np.random.default_rng(random_state)
    for g in tqdm(range(n_genes), desc="RNA residuals"):
        y = Y[:, g]
        mu = mu_all[:, g]
        a = float(alpha_use[g])

        r = _pearson_residual(y, mu, a, eps) if residual_type == "pearson" \
            else _dunn_smyth_residual(y, mu, a, random_state=rng.integers(0, 2**31 - 1))
        if clip_at is not None:
            r = np.clip(r, -clip_at, clip_at)
        resid_all[:, g] = r

    if post_standardize:
        # Spearman-invariant cosmetic step
        resid_all = (resid_all - resid_all.mean(axis=0, keepdims=True))
        sd = resid_all.std(axis=0, ddof=1)
        sd[~np.isfinite(sd) | (sd == 0)] = 1.0
        resid_all = resid_all / sd

    return mu_all, alpha_hat, resid_all, log, alpha_use


######################### standard spearman corr #########################


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


def spearman_ctrl_corr_by_bin_precomputed_fast(
    ctrl_pairs_by_bin: Dict[Union[int, str], Union[pd.DataFrame, np.ndarray, List[Tuple[int, int]]]],
    Xr_atac: np.ndarray,
    atac_mean: np.ndarray,
    atac_std: np.ndarray,
    Yr_rna: np.ndarray,
    rna_mean: np.ndarray,
    rna_std: np.ndarray,
    cell_block_size: int = 4096,
    dtype_out: np.dtype = np.float32,
    peak_col: str = "index_x",
    gene_col: str = "index_y",
    show_progress: bool = True,
) -> Dict[Union[int, str], np.ndarray]:
    """
    Option A (minimal deviation): For each bin, slice matched columns and call your
    spearman_cis_corr_precomputed_fast.

    Parameters
    ----------
    ctrl_pairs_by_bin : dict[bin] -> pairs (DataFrame, ndarray (n,2), or list of tuples)
    Xr_atac, atac_mean, atac_std : precomputed ATAC rank matrix and stats for all peaks
    Yr_rna,  rna_mean,  rna_std  : precomputed RNA  rank matrix and stats for all genes
    cell_block_size : cells per block for accumulation
    dtype_out : output dtype
    peak_col, gene_col : column names if values are DataFrames
    show_progress : show tqdm over bins

    Returns
    -------
    corr_by_bin : dict[bin] -> (n_pairs_bin,) float32 Spearman correlations (NaN for invalid)
    """
    bins = list(ctrl_pairs_by_bin.keys())
    iterator = tqdm(bins, desc="Spearman ctrl per bin (fast)", disable=not show_progress)
    results: Dict[Union[int, str], np.ndarray] = {}

    for b in iterator:
        pairs = ctrl_pairs_by_bin[b]
        peaks, genes = pairs[:,0], pairs[:,1]
        if peaks.size == 0:
            results[b] = np.array([], dtype=dtype_out)
            continue

        # Slice matched columns (aligned pair-wise)
        X_sel = Xr_atac[:, peaks]
        Y_sel = Yr_rna[:, genes]
        mx = atac_mean[peaks]
        sx = atac_std[peaks]
        my = rna_mean[genes]
        sy = rna_std[genes]

        corr = spearman_cis_corr_precomputed_fast(
            X_sel, mx, sx, Y_sel, my, sy, cell_block_size=cell_block_size, dtype_out=dtype_out
        )
        results[b] = corr

    return results


def spearman_ctrl_corr_by_bin_precomputed_chunked(
    ctrl_pairs_by_bin: Dict[Union[int, str], Union[pd.DataFrame, np.ndarray, List[Tuple[int, int]]]],
    Xr_atac: np.ndarray,
    atac_mean: np.ndarray,
    atac_std: np.ndarray,
    Yr_rna: np.ndarray,
    rna_mean: np.ndarray,
    rna_std: np.ndarray,
    cell_block_size: int = 4096,
    dtype_out: np.dtype = np.float32,
    peak_col: str = "index_x",
    gene_col: str = "index_y",
    show_progress: bool = True,
) -> Dict[Union[int, str], np.ndarray]:
    """
    Option B (chunked cross-product): For each bin, gather unique peaks and genes,
    accumulate M = sum_i X[i, uniq_peaks]^T @ Y[i, uniq_genes] in cell blocks, then
    compute each pair's numerator as M[p,g] - n*mx[p]*my[g]. Avoids materializing
    full (n_cells x n_pairs) slices if memory is tight.

    Returns
    -------
    corr_by_bin : dict[bin] -> (n_pairs_bin,) float32 Spearman correlations (NaN for invalid)
    """
    n_cells = Xr_atac.shape[0]
    bins = list(ctrl_pairs_by_bin.keys())
    iterator = tqdm(bins, desc="Spearman ctrl per bin (chunked)", disable=not show_progress)
    results: Dict[Union[int, str], np.ndarray] = {}

    for b in iterator:
        pairs = ctrl_pairs_by_bin[b]
        peaks, genes = pairs[:,0], pairs[:,1]
        n_pairs = peaks.size
        if n_pairs == 0:
            results[b] = np.array([], dtype=dtype_out)
            continue

        uniq_peaks = np.unique(peaks)
        uniq_genes = np.unique(genes)
        peak_pos = {p: i for i, p in enumerate(uniq_peaks)}
        gene_pos = {g: i for i, g in enumerate(uniq_genes)}

        # Accumulate dot-products across cell blocks
        M = np.zeros((uniq_peaks.shape[0], uniq_genes.shape[0]), dtype=np.float64)
        for cstart in range(0, n_cells, cell_block_size):
            cend = min(cstart + cell_block_size, n_cells)
            Xb = Xr_atac[cstart:cend, :][:, uniq_peaks].astype(np.float64, copy=False)
            Yb = Yr_rna[cstart:cend, :][:, uniq_genes].astype(np.float64, copy=False)
            M += Xb.T @ Yb

        # Per-pair stats
        mx = atac_mean[peaks].astype(np.float64)
        sx = atac_std[peaks].astype(np.float64)
        my = rna_mean[genes].astype(np.float64)
        sy = rna_std[genes].astype(np.float64)

        den = (n_cells - 1) * sx * sy
        num = np.empty(n_pairs, dtype=np.float64)
        for i, (p, g) in enumerate(zip(peaks, genes)):
            num[i] = M[peak_pos[p], gene_pos[g]] - n_cells * mx[i] * my[i]

        corr = np.full(n_pairs, np.nan, dtype=dtype_out)
        valid = np.isfinite(den) & (den > 0)
        corr[valid] = (num[valid] / den[valid]).astype(dtype_out)

        results[b] = corr

    return results


def spearman_ctrl_corr_by_bin(
    ctrl_pairs_by_bin: Dict[Union[int, str], Union[pd.DataFrame, np.ndarray, List[Tuple[int, int]]]],
    Xr_atac: np.ndarray,
    atac_mean: np.ndarray,
    atac_std: np.ndarray,
    Yr_rna: np.ndarray,
    rna_mean: np.ndarray,
    rna_std: np.ndarray,
    method: str = "fast",  # {'fast','chunked'}
    cell_block_size: int = 4096,
    dtype_out: np.dtype = np.float32,
    peak_col: str = "index_x",
    gene_col: str = "index_y",
    show_progress: bool = True,
) -> Dict[Union[int, str], np.ndarray]:
    """
    Compute Spearman correlations per bin using precomputed ranks and stats.

    method='fast'    -> slice matched columns per bin and call fast routine (closest to your approach)
    method='chunked' -> accumulate cross-products per bin (more memory-safe at very large n_cells)
    """
    if method == "fast":
        return spearman_ctrl_corr_by_bin_precomputed_fast(
            ctrl_pairs_by_bin, Xr_atac, atac_mean, atac_std, Yr_rna, rna_mean, rna_std,
            cell_block_size=cell_block_size, dtype_out=dtype_out,
            peak_col=peak_col, gene_col=gene_col, show_progress=show_progress,
        )
    elif method == "chunked":
        return spearman_ctrl_corr_by_bin_precomputed_chunked(
            ctrl_pairs_by_bin, Xr_atac, atac_mean, atac_std, Yr_rna, rna_mean, rna_std,
            cell_block_size=cell_block_size, dtype_out=dtype_out,
            peak_col=peak_col, gene_col=gene_col, show_progress=show_progress,
        )
    else:
        raise ValueError("method must be 'fast' or 'chunked'")