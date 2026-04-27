import numpy as np
import scipy as sp
import math
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess


# Module 1: Per-feature parameter estimation


def estimate_s(x):
    """Per-cell library sizes from a raw count matrix (N, M) or vector (N,)."""
    if sp.sparse.issparse(x):
        return np.asarray(x.sum(axis=1)).ravel()
    return np.asarray(x).sum(axis=1)


def estimate_poisson(x, s):
    """
    Moment-based Poisson parameter estimates for a single feature.

    Parameters
    ----------
    x : (N,) raw counts
    s : (N,) per-cell library sizes

    Returns
    -------
    mu, sigma2_latent, total_var
    """
    if sp.sparse.issparse(x):
        x = np.asarray(x.todense()).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    r = x / s
    mu = r.mean()
    total_var = r.var()
    noise_var = mu * np.mean(1.0 / s)
    sigma2_latent = max(total_var - noise_var, 1e-8)
    return mu, sigma2_latent, total_var


def estimate_zip(x, s):
    """
    Moment-based Zero-Inflated Poisson parameter estimates for a single feature.

    Parameters
    ----------
    x : (N,) raw counts
    s : (N,) per-cell library sizes

    Returns
    -------
    mu, sigma2_latent, total_var, pi, lam
    """
    if sp.sparse.issparse(x):
        x = np.asarray(x.todense()).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    r = x / s
    mu = r.mean()
    total_var = r.var()

    p_zero_obs = np.mean(x == 0)
    # second-order saddlepoint approximation: E[exp(-lam*s)] ≈ exp(-lam*E[s] + 0.5*lam²*Var(s))
    p_zero_poi = np.exp(-mu * np.mean(s) + 0.5 * mu ** 2 * np.var(s))
    pi = max(p_zero_obs - p_zero_poi, 0) / (1 - p_zero_poi + 1e-8)
    lam = mu / (1 - pi + 1e-8)

    noise_var = (1 - pi) * lam * np.mean(1.0 / s) + pi * (1 - pi) * lam ** 2
    sigma2_latent = max(total_var - noise_var, 1e-8)
    return mu, sigma2_latent, total_var, pi, lam


def estimate_nb(x, s):
    """
    Moment-based Negative Binomial parameter estimates for a single feature.

    Parameters
    ----------
    x : (N,) raw counts
    s : (N,) per-cell library sizes

    Returns
    -------
    mu, sigma2_latent, total_var, phi
    """
    if sp.sparse.issparse(x):
        x = np.asarray(x.todense()).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    r = x / s
    mu = r.mean()
    total_var = r.var()

    excess = max(total_var - mu * np.mean(1.0 / s), 1e-6)
    phi = mu ** 2 / excess
    # E[Var(x/s)] = mu*E[1/s] + mu²/phi  (correct NB noise decomposition)
    noise_var = mu * np.mean(1.0 / s) + mu ** 2 / phi
    sigma2_latent = max(total_var - noise_var, 1e-8)
    return mu, sigma2_latent, total_var, phi


def estimate_nb_deseq2(x, s, mu_all, phi_raw_all, phi_trend_fn=None):
    """
    NB parameter estimates with DESeq2-style phi shrinkage toward a lowess trend.

    phi_trend_fn : callable(mu_arr) -> phi_trend_arr, prefit on (mu_all, phi_raw_all).
                   If None, fits the lowess trend inline (slow; avoid in production).

    Returns
    -------
    mu, sigma2_latent, total_var, phi_shrunk
    """
    if sp.sparse.issparse(x):
        x = np.asarray(x.todense()).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    r = x / s
    mu = r.mean()
    total_var = r.var()

    excess = max(total_var - mu * np.mean(1.0 / s), 1e-6)
    phi_raw = mu ** 2 / excess

    w = np.sum(x > 0) / len(x)  # fraction nonzero as shrinkage weight toward phi_raw

    if phi_trend_fn is not None:
        phi_trend = float(phi_trend_fn(np.array([mu]))[0])
    else:
        log_mu_all  = np.log(np.maximum(mu_all, 1e-8))
        log_phi_all = np.log(np.maximum(phi_raw_all, 1e-8))
        smoothed = sm_lowess(log_phi_all, log_mu_all, frac=0.3, return_sorted=True)
        phi_trend = float(np.exp(np.interp(
            np.log(max(mu, 1e-8)), smoothed[:, 0], smoothed[:, 1]
        )))

    phi_shrunk = w * phi_raw + (1 - w) * phi_trend
    noise_var = mu * np.mean(1.0 / s) + mu ** 2 / phi_shrunk
    sigma2_latent = max(total_var - noise_var, 1e-8)
    return mu, sigma2_latent, total_var, phi_shrunk


# Module 2: Covariance estimator


def estimate_covariance(x_peak, s_atac, x_gene, s_rna, mu_p, mu_g):
    """
    Model-free covariance between one peak and one gene on depth-normalised counts.

    Parameters
    ----------
    x_peak : (N,) raw ATAC counts
    s_atac : (N,) ATAC per-cell library sizes
    x_gene : (N,) raw RNA counts
    s_rna  : (N,) RNA per-cell library sizes
    mu_p, mu_g : precomputed feature means of x/s

    Returns
    -------
    sigma_pg : float
    """
    return float(np.mean((x_peak / s_atac) * (x_gene / s_rna)) - mu_p * mu_g)


# Module 3: Correlation statistics


def r_latent(sigma_pg, sigma2_p, sigma2_g):
    """
    Latent correlation. Can exceed [-1, 1] for unstable features; do NOT clip.
    """
    denom = math.sqrt(sigma2_p * sigma2_g)
    if denom < 1e-8:
        return 0.0
    return sigma_pg / denom


def r_shrinkage(sigma_pg, total_var_p, total_var_g):
    """
    Shrinkage correlation that telescopes to Pearson on depth-normalised counts.
    Always in [-1, 1] by Cauchy-Schwarz; no clipping needed.
    """
    denom = math.sqrt(total_var_p * total_var_g)
    if denom < 1e-8:
        return 0.0
    return sigma_pg / denom


# Vectorised genome-wide precomputation


def _sparse_mean_and_var(counts, inv_s):
    """
    Compute mean(x/s) and var(x/s) for all features simultaneously.

    Parameters
    ----------
    counts : (N, M) CSC sparse matrix
    inv_s  : (N,) = 1 / per-cell library sizes

    Returns
    -------
    mu : (M,), total_var : (M,)
    """
    r = counts.multiply(inv_s[:, None])
    mu = np.asarray(r.mean(axis=0)).ravel()
    e_r2 = np.asarray(r.power(2).mean(axis=0)).ravel()
    total_var = np.maximum(e_r2 - mu ** 2, 0.0)
    return mu, total_var


def precompute_atac_poisson(counts_atac, s_atac):
    """Vectorised Poisson precomputation for all ATAC peaks.

    Returns dict with arrays of shape (n_peaks,):
        mu, sigma2, total_var, sigma2_raw
    """
    counts_atac = counts_atac.tocsc()
    inv_s = 1.0 / s_atac
    mu, total_var = _sparse_mean_and_var(counts_atac, inv_s)
    noise_var = mu * inv_s.mean()
    sigma2_raw = total_var - noise_var
    return dict(mu=mu, sigma2=np.maximum(sigma2_raw, 1e-8),
                total_var=total_var, sigma2_raw=sigma2_raw)


def precompute_atac_zip(counts_atac, s_atac):
    """Vectorised ZIP precomputation for all ATAC peaks.

    Returns dict with arrays of shape (n_peaks,):
        mu, sigma2, total_var, sigma2_raw, pi, lam
    """
    counts_atac = counts_atac.tocsc()
    n_cells = counts_atac.shape[0]
    inv_s = 1.0 / s_atac
    mu, total_var = _sparse_mean_and_var(counts_atac, inv_s)

    nnz = np.diff(counts_atac.indptr)                 # (P,) nnz per peak
    mean_s    = s_atac.mean()                         # global scalar E[s]
    var_s     = s_atac.var()                          # global scalar Var[s]
    mean_inv_s = inv_s.mean()                         # global scalar E[1/s]

    p_zero_obs = 1.0 - nnz / n_cells
    # second-order: E[exp(-lam*s)] ≈ exp(-lam*E[s] + 0.5*lam²*Var(s))
    p_zero_poi = np.exp(-mu * mean_s + 0.5 * mu ** 2 * var_s)
    pi  = np.maximum(p_zero_obs - p_zero_poi, 0) / (1 - p_zero_poi + 1e-8)
    lam = mu / (1 - pi + 1e-8)

    noise_var  = (1 - pi) * lam * mean_inv_s + pi * (1 - pi) * lam ** 2
    sigma2_raw = total_var - noise_var
    return dict(mu=mu, sigma2=np.maximum(sigma2_raw, 1e-8),
                total_var=total_var, sigma2_raw=sigma2_raw, pi=pi, lam=lam)


def precompute_rna_poisson(counts_rna, s_rna):
    """Vectorised Poisson precomputation for all RNA genes.

    Returns dict with arrays of shape (n_genes,):
        mu, sigma2, total_var, sigma2_raw
    """
    counts_rna = counts_rna.tocsc()
    inv_s = 1.0 / s_rna
    mu, total_var = _sparse_mean_and_var(counts_rna, inv_s)

    noise_var  = mu * inv_s.mean()     # global E[1/s], same for all genes
    sigma2_raw = total_var - noise_var
    return dict(mu=mu, sigma2=np.maximum(sigma2_raw, 1e-8),
                total_var=total_var, sigma2_raw=sigma2_raw)


def precompute_rna_nb(counts_rna, s_rna):
    """Vectorised NB (analytical phi) precomputation for all RNA genes.

    Returns dict with arrays of shape (n_genes,):
        mu, sigma2, total_var, sigma2_raw, phi, phi_raw
    """
    counts_rna = counts_rna.tocsc()
    inv_s = 1.0 / s_rna
    mu, total_var = _sparse_mean_and_var(counts_rna, inv_s)

    excess    = np.maximum(total_var - mu * inv_s.mean(), 1e-6)
    phi_raw   = mu ** 2 / excess
    noise_var = mu * inv_s.mean() + mu ** 2 / phi_raw   # E[Var(x/s)] = mu*E[1/s] + mu²/phi
    sigma2_raw = total_var - noise_var
    return dict(mu=mu, sigma2=np.maximum(sigma2_raw, 1e-8),
                total_var=total_var, sigma2_raw=sigma2_raw,
                phi=phi_raw, phi_raw=phi_raw)


def build_phi_trend_fn(mu_all, phi_raw_all, frac=0.3):
    """
    Fit lowess of log(phi_raw) ~ log(mu) across all genes.

    Returns a vectorised callable phi_trend_fn(mu_arr) -> phi_trend_arr.
    """
    log_mu  = np.log(np.maximum(mu_all, 1e-8))
    log_phi = np.log(np.maximum(phi_raw_all, 1e-8))
    ok = np.isfinite(log_mu) & np.isfinite(log_phi)
    smoothed = sm_lowess(log_phi[ok], log_mu[ok], frac=frac, return_sorted=True)
    grid_x, grid_y = smoothed[:, 0], smoothed[:, 1]

    def phi_trend_fn(mu_arr):
        lm = np.log(np.maximum(np.asarray(mu_arr, dtype=np.float64), 1e-8))
        return np.exp(np.interp(lm, grid_x, grid_y))

    return phi_trend_fn


def precompute_rna_nb_deseq2(counts_rna, s_rna, nb_params, phi_trend_fn):
    """Apply DESeq2-style phi shrinkage to precomputed NB params.

    Parameters
    ----------
    nb_params     : output of precompute_rna_nb
    phi_trend_fn  : output of build_phi_trend_fn

    Returns dict with arrays of shape (n_genes,):
        mu, sigma2, total_var, sigma2_raw, phi (shrunk), phi_raw
    """
    counts_rna = counts_rna.tocsc()
    n_cells    = counts_rna.shape[0]
    nnz        = np.diff(counts_rna.indptr)
    w          = nnz / n_cells               # fraction nonzero → weight toward phi_raw

    phi_raw   = nb_params['phi_raw']
    mu_all    = nb_params['mu']
    phi_trend = phi_trend_fn(mu_all)
    phi_shrunk = w * phi_raw + (1 - w) * phi_trend

    inv_s      = 1.0 / s_rna
    noise_var  = mu_all * inv_s.mean() + mu_all ** 2 / phi_shrunk
    sigma2_raw = nb_params['total_var'] - noise_var
    return dict(mu=mu_all, sigma2=np.maximum(sigma2_raw, 1e-8),
                total_var=nb_params['total_var'], sigma2_raw=sigma2_raw,
                phi=phi_shrunk, phi_raw=phi_raw)


# Batch correlation computation


def compute_latent_corr_batch(counts_atac, counts_rna, s_atac, s_rna,
                               atac_idx, rna_idx,
                               atac_params, rna_params, stat_name):
    """
    Compute latent correlations for a batch of (atac_idx, rna_idx) pairs.

    Control peaks and genes use their own precomputed parameters from
    atac_params and rna_params (keyed by integer feature index).

    Parameters
    ----------
    counts_atac, counts_rna : CSC sparse (N, P) and (N, G) raw count matrices
    s_atac, s_rna           : (N,) library sizes
    atac_idx, rna_idx       : integer arrays of length B (feature indices)
    atac_params, rna_params : dicts with 'mu', 'sigma2', 'total_var' arrays
    stat_name               : 'r_latent' or 'r_shrinkage'

    Returns
    -------
    corr : np.ndarray of shape (B,), float32
    """
    atac_idx = np.asarray(atac_idx, dtype=int)
    rna_idx  = np.asarray(rna_idx,  dtype=int)

    X_atac = counts_atac[:, atac_idx]
    X_rna  = counts_rna[:,  rna_idx]
    if sp.sparse.issparse(X_atac):
        X_atac = X_atac.toarray()
    if sp.sparse.issparse(X_rna):
        X_rna = X_rna.toarray()

    R_atac = X_atac.astype(np.float64) / s_atac[:, None]
    R_rna  = X_rna.astype(np.float64)  / s_rna[:,  None]

    cross   = (R_atac * R_rna).mean(axis=0)
    mu_p    = atac_params['mu'][atac_idx]
    mu_g    = rna_params['mu'][rna_idx]
    sigma_pg = cross - mu_p * mu_g

    if stat_name == 'r_latent':
        denom = np.sqrt(atac_params['sigma2'][atac_idx] * rna_params['sigma2'][rna_idx])
    else:  # r_shrinkage
        denom = np.sqrt(atac_params['total_var'][atac_idx] * rna_params['total_var'][rna_idx])

    corr = np.where(denom < 1e-8, 0.0, sigma_pg / denom)
    return corr.astype(np.float32)


# Diagnostics


def collect_diagnostics(atac_params, rna_params, run_cfg):
    """
    Collect per-run diagnostics.

    Returns
    -------
    dict of scalar floats / percentile values.
    """
    diag = {}
    diag['frac_negative_sigma2_atac'] = float(np.mean(atac_params['sigma2_raw'] < 0))
    diag['frac_negative_sigma2_rna']  = float(np.mean(rna_params['sigma2_raw'] < 0))

    tv_p = np.maximum(atac_params['total_var'], 1e-30)
    tv_g = np.maximum(rna_params['total_var'],  1e-30)
    diag['frac_low_rel_atac'] = float(np.mean(atac_params['sigma2'] / tv_p < 0.05))
    diag['frac_low_rel_rna']  = float(np.mean(rna_params['sigma2']  / tv_g < 0.05))

    if run_cfg['atac'] == 'zip':
        for pct, val in zip([5, 25, 50, 75, 95],
                            np.percentile(atac_params['pi'], [5, 25, 50, 75, 95])):
            diag[f'pi_p{pct}'] = float(val)

    if run_cfg['rna'] == 'nb':
        for pct, val in zip([5, 25, 50, 75, 95],
                            np.percentile(rna_params['phi'], [5, 25, 50, 75, 95])):
            diag[f'phi_p{pct}'] = float(val)

    return diag
