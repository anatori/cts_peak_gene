import numpy as np
import pandas as pd
import scipy as sp
import anndata as ad
import scanpy as sc


def mean_by_group(adata, group_key="celltype", layer=None, use_raw=False, groups=None):
    # pick matrix
    if use_raw:
        X = adata.raw.X
        var_names = adata.raw.var_names
        obs = adata.obs
    else:
        obs = adata.obs
        var_names = adata.var_names
        if layer is None:
            X = adata.X
        else:
            X = adata.layers[layer]

    if groups is None:
        groups = obs[group_key].astype("category")
    else:
        groups = pd.Series(np.asarray(groups), index=obs.index).astype("category")

    codes = groups.cat.codes.to_numpy()
    n_groups = len(groups.cat.categories)

    # one-hot (cells x groups)
    G = sp.sparse.csr_matrix(
        (np.ones(X.shape[0]), (np.arange(X.shape[0]), codes)),
        shape=(X.shape[0], n_groups),
    )

    # sums: (groups x genes)
    sums = (G.T @ X) if sp.sparse.issparse(X) else (G.T @ sp.sparse.csr_matrix(X))

    counts = np.asarray(G.sum(axis=0)).ravel()  # cells per group
    means = sums.multiply(1.0 / counts[:, None]) if sp.sparse.issparse(sums) else (sums / counts[:, None])

    # to DataFrame (dense)
    means = means.toarray() if sp.sparse.issparse(means) else np.asarray(means)
    means_df = pd.DataFrame(means, index=groups.cat.categories, columns=var_names).T
    means_df = means_df.reset_index(names='feature')
    return means_df


def entropy_specificity(x, min_total=0.0, eps=1e-6):
    """
    Entropy-based specificity:

        p_i = x_i / sum_j x_j
        H(p) = - sum_i p_i * log(p_i)
        specificity = 1 - H(p) / log(K)

    Parameters
    ----------
    x
        1D signal array of shape (k,) or 2D of shape (n, k) for one feature.
    eps
        Additive smoothing constant. Acts as filter with eps proportional to x_i (constant depends on normalization).
    Returns
    -------
    Specificity in [0,1], or np.nan if total signal is too low.
    Returns float if 1D and (n,) array if 2D.
    """
    signal = np.asarray(x, dtype=float)
    if signal.ndim not in (1, 2):
        raise ValueError("x must be 1D (one feature) or 2D (n_features, k).")

    is_single_feature = signal.ndim == 1
    if is_single_feature:
        signal = signal[None, :]

    feature_totals = np.nansum(signal, axis=1)
    n_celltypes = signal.shape[1]

    smoothed = np.clip(signal, 0, None) + eps
    with np.errstate(invalid='ignore', divide='ignore'):
        proportions = smoothed / smoothed.sum(axis=1, keepdims=True)
        log_proportions = np.zeros_like(proportions)
        np.log(proportions, where=proportions > 0, out=log_proportions)
        entropy = -(proportions * log_proportions).sum(axis=1)
        specificity = 1.0 - entropy / np.log(n_celltypes)

    specificity[feature_totals < min_total] = np.nan

    return float(specificity[0]) if is_single_feature else specificity


def dominant_subtype(x):
    x = np.asarray(x, dtype=float)
    if np.all(np.isnan(x)):
        return np.nan
    return int(np.nanargmax(x))


def compute_feature_specificity(mean_df, feature_col='feature', subtype_cols=None, min_total=0.0, eps=1e-6):
    """
    mean_df: dataframe with one row per feature and subtype mean columns
    returns dataframe with specificity and dominant subtype and mean columns per subtype
    """
    if subtype_cols is None:
        subtype_cols = [c for c in mean_df.columns if c != feature_col]

    out = mean_df[[feature_col]].copy()

    mean_values = mean_df[subtype_cols].to_numpy(dtype=float)

    out['specificity'] = entropy_specificity(mean_values, min_total=min_total, eps=eps)
    out['dominant_subtype_idx'] = [
        dominant_subtype(feature_means) for feature_means in mean_values
    ]
    out['dominant_subtype'] = [
        subtype_cols[i] if pd.notna(i) else np.nan
        for i in out['dominant_subtype_idx']
    ]

    return out


def _mean_table(adata, labels, layer, target_sum, alpha):
    """
    Smoothed cell-type mean table on the linear scale:

        normalized_c = counts_c / sum(counts_c) * target_sum    (normalize_total, no log1p)
        mean         = mean of normalized_c within each cell type, over cells with a non-NaN label
        pseudocount  = alpha * (target_sum / median_depth) / n_labeled_cells
        table        = mean + pseudocount

    The pseudocount is one count at median depth spread over all cells, in normalized
    units. Median depth is taken over all cells on raw counts, before dropping NaN labels.
    Normalization and cell-type means are one sparse product. Membership rows carry each
    labeled cell's target_sum / depth scale.
    """
    counts = adata.X if layer is None else adata.layers[layer]
    counts = sp.sparse.csr_matrix(counts, dtype=np.float64)
    depth = np.asarray(counts.sum(axis=1)).ravel()
    scale = np.divide(target_sum, depth, out=np.zeros_like(depth), where=depth > 0)

    labels = np.asarray(labels)
    labeled_cells = np.flatnonzero(pd.notna(labels))
    celltypes = pd.Categorical(labels[labeled_cells])
    n_celltypes = len(celltypes.categories)
    scaled_membership = sp.sparse.csr_matrix(
        (scale[labeled_cells], (celltypes.codes, labeled_cells)), shape=(n_celltypes, counts.shape[0]))
    group_sizes = np.bincount(celltypes.codes, minlength=n_celltypes).astype(np.float64)
    means = np.asarray((scaled_membership @ counts).todense()) / group_sizes[:, None]   # (n_celltypes, n_features)

    pseudocount = alpha * (target_sum / np.median(depth)) / len(labeled_cells)
    return pd.DataFrame(np.clip(means.T, 0, None) + pseudocount, index=adata.var_names, columns=celltypes.categories)


def _pseudobulk_table(adata, labels, layer, alpha):
    """
    Smoothed library-size-scaled pseudobulk table:

        table[feature, t] = sum_{c in t} counts[c, feature] / N_t + alpha / mean_t(N_t)
        N_t               = total raw counts of cell type t in this modality, before the pseudocount

    Raw counts are summed within each cell type (ratio of sums), divided by that cell
    type's total counts, then smoothed with alpha counts at the average cell-type depth,
    the same in every cell type. No CPM scaling; a global constant cancels in the entropy.
    Cells with NaN labels are dropped before N_t is computed.
    """
    counts = adata.X if layer is None else adata.layers[layer]
    if not sp.sparse.issparse(counts):
        counts = sp.sparse.csr_matrix(np.asarray(counts))

    labels = np.asarray(labels)
    labeled_cells = np.flatnonzero(pd.notna(labels))
    celltypes = pd.Categorical(labels[labeled_cells])
    n_celltypes = len(celltypes.categories)

    membership = sp.sparse.csr_matrix(
        (np.ones(len(labeled_cells)), (celltypes.codes, labeled_cells)),
        shape=(n_celltypes, counts.shape[0]),
    )
    summed_counts = np.asarray((membership @ counts).todense(), dtype=np.float64)   # (n_celltypes, n_features)

    celltype_totals = summed_counts.sum(axis=1)
    proportions = summed_counts / celltype_totals[:, None] + alpha / celltype_totals.mean()
    return pd.DataFrame(proportions.T, index=adata.var_names, columns=celltypes.categories)


def celltype_tables(adata_rna, adata_atac, celltype_col='celltype', method='pseudobulk',
                    layer='counts', groups=None, target_sum=1e4, alpha=1.0):
    """
    Smoothed feature x cell-type tables for RNA and ATAC, for joint_entropy_specificity().

    Parameters
    ----------
    adata_rna, adata_atac
        Cell x feature AnnData holding raw counts, same cells in the same order.
    celltype_col
        adata_rna.obs column with cell-type labels; NaN labels are dropped.
    method
        'pseudobulk' (default): summed raw counts / cell-type total + alpha / mean cell-type total.
        'mean': normalize_total(target_sum), per-cell-type mean + depth-calibrated pseudocount
        (the original construction; kept to reproduce earlier results).
    layer
        Layer with raw counts. None uses .X.
    groups
        Optional array-like of labels replacing adata_rna.obs[celltype_col], used for both modalities.
    target_sum
        'mean' only: normalization target.
    alpha
        Pseudocount weight for either method.

    Returns
    -------
    rna_table, atac_table
        DataFrames indexed by var_names with one column per cell type (same columns, same order).
    """
    if method not in ('pseudobulk', 'mean'):
        raise ValueError("method must be 'pseudobulk' or 'mean'.")
    if not np.array_equal(adata_rna.obs_names, adata_atac.obs_names):
        raise ValueError("RNA and ATAC cells must be in the same order.")
    if groups is None:
        if celltype_col not in adata_rna.obs.columns:
            raise ValueError(f"'{celltype_col}' not found in adata_rna.obs.")
        groups = adata_rna.obs[celltype_col].to_numpy()

    tables = []
    for adata in (adata_rna, adata_atac):
        if method == 'pseudobulk':
            tables.append(_pseudobulk_table(adata, groups, layer, alpha))
        else:
            tables.append(_mean_table(adata, groups, layer, target_sum, alpha))
    rna_table, atac_table = tables
    if not rna_table.columns.equals(atac_table.columns):
        raise ValueError("RNA and ATAC cell-type columns differ.")
    return rna_table, atac_table


def joint_entropy_specificity(rna_table, atac_table, gene_idx, peak_idx, chunk_size=200000,
                              rna_log=None, atac_log=None):
    """
    Link specificity from the elementwise product of smoothed cell-type tables.
        joint_i = rna_i(gene) * atac_i(peak)
        entropy_specificity(joint)

    Gathers rows by index in chunks instead of taking pre-gathered (n_links, K) arrays,
    so control sets fit in memory. The entropy is evaluated as log(S) - sum(j log j) / S 
    with logs taken once per feature for efficiency.

    Parameters
    ----------
    rna_table, atac_table
        (n_genes, K) and (n_peaks, K) smoothed, positive tables with the same K columns in the
        same order, e.g. from celltype_tables(). DataFrames or arrays.
    gene_idx, peak_idx
        (n_links,) row positions into rna_table and atac_table for each link.
    chunk_size
        Number of links scored per block.
    rna_log, atac_log
        Optional precomputed log of the tables (zero where the table is 0), so callers scoring
        many link batches against the same tables take the logs once.

    Returns
    -------
    (n_links,) array of specificity in [0,1].
    """
    rna_values = np.asarray(rna_table, dtype=np.float64)
    atac_values = np.asarray(atac_table, dtype=np.float64)
    if rna_log is None:
        rna_log = np.log(rna_values, where=rna_values > 0, out=np.zeros_like(rna_values))
    if atac_log is None:
        atac_log = np.log(atac_values, where=atac_values > 0, out=np.zeros_like(atac_values))
    gene_idx = np.asarray(gene_idx, dtype=np.intp)
    peak_idx = np.asarray(peak_idx, dtype=np.intp)
    if gene_idx.shape != peak_idx.shape or gene_idx.ndim != 1:
        raise ValueError("gene_idx and peak_idx must be 1D arrays of the same length.")
    if rna_values.shape[1] != atac_values.shape[1]:
        raise ValueError("rna_table and atac_table must have the same cell-type columns.")

    log_n_celltypes = np.log(rna_values.shape[1])
    specificity = np.empty(len(gene_idx), dtype=np.float64)
    for start in range(0, len(gene_idx), chunk_size):
        stop = start + chunk_size
        gene_chunk = gene_idx[start:stop]
        peak_chunk = peak_idx[start:stop]

        joint = rna_values[gene_chunk] # joint_i = rna_i * atac_i
        joint *= atac_values[peak_chunk]
        log_joint = rna_log[gene_chunk] # log joint_i = log rna_i + log atac_i
        log_joint += atac_log[peak_chunk]

        joint_total = joint.sum(axis=1)
        joint_log_sum = np.einsum('ij,ij->i', joint, log_joint)
        with np.errstate(invalid='ignore', divide='ignore'):
            # entropy H = -sum p log p = log(total) - sum(joint * log joint) / total
            entropy = np.log(joint_total) - joint_log_sum / joint_total
            chunk_specificity = 1.0 - entropy / log_n_celltypes
        chunk_specificity[joint_total == 0] = 0.0
        specificity[start:stop] = chunk_specificity
    return specificity


def marginal_specificity(rna_table, atac_table):
    """
    Entropy specificity of each feature's own cell-type profile, from the smoothed tables of
    celltype_tables() (same pseudocount as the joint score).

    Parameters
    ----------
    rna_table, atac_table
        (n_genes, K) and (n_peaks, K) smoothed, positive tables. DataFrames or arrays.

    Returns
    -------
    gene_specificity, peak_specificity
        Arrays of length n_genes and n_peaks in table row order, in [0,1]. Index them with the
        same gene / peak positions used for the links.
    """
    gene_specificity = entropy_specificity(np.asarray(rna_table, dtype=np.float64), eps=0.0)
    peak_specificity = entropy_specificity(np.asarray(atac_table, dtype=np.float64), eps=0.0)
    return gene_specificity, peak_specificity


def dominant_celltypes(rna_table, atac_table, gene_idx, peak_idx, chunk_size=200000):
    """
    Dominant cell type of each link's gene, peak and joint profile, from the smoothed tables of celltype_tables().

        gene_dominant = argmax_t rna_table[gene, t]
        peak_dominant = argmax_t atac_table[peak, t]
        link_dominant = argmax_t rna_table[gene, t] * atac_table[peak, t]

    A profile that is constant across cell types (for example a feature with no counts, which holds only the
    pseudocount) has no dominant cell type and gets None, as does a link with a negative index (missing feature).

    Parameters
    ----------
    rna_table, atac_table
        DataFrames from celltype_tables(), features x cell types with the same columns.
    gene_idx, peak_idx
        (n_links,) row positions into rna_table and atac_table; negative marks a missing link.

    Returns
    -------
    gene_dominant, peak_dominant, link_dominant
        Object arrays of cell type names, one entry per link (None where undefined).
    concordant
        Float array: 1.0 if gene_dominant == peak_dominant, 0.0 if they differ, NaN if either is undefined.
    """
    celltypes = np.asarray(rna_table.columns, dtype=object)
    rna_values = np.asarray(rna_table, dtype=np.float64)
    atac_values = np.asarray(atac_table, dtype=np.float64)
    gene_idx = np.asarray(gene_idx, dtype=np.intp)
    peak_idx = np.asarray(peak_idx, dtype=np.intp)

    # argmax per row; -1 where the row is constant (no dominant cell type)
    gene_code_by_feature = np.where(rna_values.max(axis=1) > rna_values.min(axis=1), rna_values.argmax(axis=1), -1)
    peak_code_by_feature = np.where(atac_values.max(axis=1) > atac_values.min(axis=1), atac_values.argmax(axis=1), -1)
    gene_code = np.where(gene_idx >= 0, gene_code_by_feature[np.maximum(gene_idx, 0)], -1)
    peak_code = np.where(peak_idx >= 0, peak_code_by_feature[np.maximum(peak_idx, 0)], -1)

    link_code = np.full(len(gene_idx), -1)
    scored_links = np.flatnonzero((gene_idx >= 0) & (peak_idx >= 0))
    for start in range(0, len(scored_links), chunk_size):
        links = scored_links[start:start + chunk_size]
        joint = rna_values[gene_idx[links]] * atac_values[peak_idx[links]]
        link_code[links] = np.where(joint.max(axis=1) > joint.min(axis=1), joint.argmax(axis=1), -1)

    concordant = np.where((gene_code >= 0) & (peak_code >= 0), (gene_code == peak_code).astype(float), np.nan)
    return (np.where(gene_code >= 0, celltypes[np.maximum(gene_code, 0)], None),
            np.where(peak_code >= 0, celltypes[np.maximum(peak_code, 0)], None),
            np.where(link_code >= 0, celltypes[np.maximum(link_code, 0)], None), concordant)


def coexp_context(adata_rna, adata_atac, celltype_col='celltype', layer='counts', groups=None,
                  target_sum=None, alpha=1.0, log_rna=None, log_atac=None):
    """
    Preprocessing for coexp_specificity(), called once for efficiency.

    Pseudobulk of co-expression ('coexp_log1p') for a link (gene g, peak p) and cell type t is

        P_t = mean_{c in t} [ log1p(normalized rna_c(g)) * log1p(normalized atac_c(p)) ] + pseudocount
        normalized x_c = target * x_c / depth_c (normalize_total; default target = median depth)
        pseudocount = eps_rna * eps_atac, eps = alpha * log1p(target / median_depth) / n_labeled_cells

    P_t is the cell-type average of a per-cell product (multiply, then average), unlike the joint score,
    which averages then multiplies. eps is one raw count at median depth on the log1p scale. By default,
    target = median depth, and pseudocount eps is alpha * log(2) / n_labeled. Note that median depth is 
    over all cells with non-zero depth, before dropping NaN labels (as in scanpy).

    Parameters
    ----------
    adata_rna, adata_atac
        Cell x feature AnnData holding raw counts, same cells in the same order.
    celltype_col
        adata_rna.obs column with cell-type labels; NaN labels are dropped.
    layer
        Layer with raw counts. None uses .X.
    groups
        Optional array-like of labels replacing adata_rna.obs[celltype_col].
    target_sum
        normalize_total target. None (default) is median depth per modality, i.e. sc.pp.normalize_total's default.
        Otherwise, values are passed into sc.pp.normalize_total.
    alpha
        Pseudocount weight.
    log_rna, log_atac
        Optional precomputed sc.pp.normalize_total + sc.pp.log1p matrices over all cells (cells x features),
        CTAR's default matrices for correlation can be passed in here!

    Returns
    -------
    dict with
        log_rna, log_atac   (n_labeled, n_genes) / (n_labeled, n_peaks) csc log1p matrices; columns keep
                            var order, so link indices into adata.var index them directly
        membership          (K, n_labeled) csr one-hot cell-type membership
        group_sizes         (K,) cells per cell type
        pseudocount         float
        celltypes           (K,) cell-type names, in the same order as celltype_tables()
    """
    if not np.array_equal(adata_rna.obs_names, adata_atac.obs_names):
        raise ValueError("RNA and ATAC cells must be in the same order.")
    if groups is None:
        if celltype_col not in adata_rna.obs.columns:
            raise ValueError(f"'{celltype_col}' not found in adata_rna.obs.")
        groups = adata_rna.obs[celltype_col].to_numpy()
    labels = np.asarray(groups)
    labeled_cells = np.flatnonzero(pd.notna(labels))
    celltypes = pd.Categorical(labels[labeled_cells])
    n_labeled = len(labeled_cells)

    eps, log_matrices = [], []
    for adata, precomputed in ((adata_rna, log_rna), (adata_atac, log_atac)):
        counts = adata.X if layer is None else adata.layers[layer]
        depth = np.asarray(counts.sum(axis=1)).ravel()
        median_depth = np.median(depth[depth > 0]) # scanpy's default normalize_total target
        target = median_depth if target_sum is None else target_sum
        eps.append(alpha * np.log1p(target / median_depth) / n_labeled)
        if precomputed is None:
            # counts-only adata so scanpy does not copy other layers
            normalized = ad.AnnData(X=sp.sparse.csr_matrix(counts, dtype=np.float32))
            sc.pp.normalize_total(normalized, target_sum=target_sum)
            sc.pp.log1p(normalized)
            precomputed = normalized.X
        log_matrices.append(sp.sparse.csc_matrix(sp.sparse.csr_matrix(precomputed)[labeled_cells], dtype=np.float32))

    n_celltypes = len(celltypes.categories)
    membership = sp.sparse.csr_matrix(
        (np.ones(n_labeled), (celltypes.codes, np.arange(n_labeled))), shape=(n_celltypes, n_labeled))
    return {
        'log_rna': log_matrices[0],
        'log_atac': log_matrices[1],
        'membership': membership,
        'group_sizes': np.bincount(celltypes.codes, minlength=n_celltypes).astype(np.float64),
        'pseudocount': float(eps[0] * eps[1]),
        'celltypes': np.asarray(celltypes.categories, dtype=object),
    }


def coexp_specificity(context, gene_idx, peak_idx, chunk_size=100000):
    """
    Pseudobulk-of-co-expression ('coexp_log1p') specificity and dominant cell type for links.

        P_t  = group_sums_t / n_t + pseudocount
        spec = entropy_specificity(P, eps=0)

    Parameters
    ----------
    context
        dict from coexp_context().
    gene_idx, peak_idx
        (n_links,) column positions into context['log_rna'] / context['log_atac'].

    Returns
    -------
    specificity
        (n_links,) float64 in [0, 1].
    dominant_code
        (n_links,) int, argmax_t P_t; -1 where P is constant across cell types (no co-expression anywhere).
    """
    # cell-type sums of the per-cell product
    # cost scales with the non-zeros of the gene and peak columns
    gene_idx = np.asarray(gene_idx, dtype=np.intp)
    peak_idx = np.asarray(peak_idx, dtype=np.intp)
    order = np.argsort(gene_idx, kind='stable')
    sums = np.empty((len(gene_idx), context['membership'].shape[0]), dtype=np.float64)
    for start in range(0, len(order), chunk_size):
        rows = order[start:start + chunk_size]
        products = context['log_rna'][:, gene_idx[rows]].multiply(context['log_atac'][:, peak_idx[rows]]).tocsc()
        sums[rows] = np.asarray((context['membership'] @ products).todense()).T
    profile = sums / context['group_sizes'][None, :] + context['pseudocount']
    dominant_code = np.where(profile.max(axis=1) > profile.min(axis=1), profile.argmax(axis=1), -1)
    return entropy_specificity(profile, eps=0.0), dominant_code


def groupwise_z(W, codes, n_groups=None, n_cells=None, group_sizes=None, var_w=None):
    """
    One-vs-rest z per-cell score for Wilcoxon test.

        S_i      = sum_{c in i} w_c
        E[S_i]   = n_i * mean(w)
        Var[S_i] = n_i * (N - n_i) / (N - 1) * var(w)
        z_i      = ( S_i - E[S_i] ) / sqrt( Var[S_i] )


    Parameters
    ----------
    W
        (n_rows, m) sparse or dense score matrix. Columns are features or links.
    codes
        Integer group code for each row of W.
    n_cells, group_sizes
        Population N and n_i. Pass these when W holds only a subset of cells,
        the omitted ones being implicitly zero; defaults to W's own rows.
    var_w
        Override for var(w), per column or scalar. Pass (N^2-1)/12 to reproduce
        the uncorrected rank-sum variance.

    Returns
    -------
    (K, m) array of z.
    """
    codes = np.asarray(codes)
    K = int(n_groups) if n_groups is not None else int(codes.max()) + 1
    N = int(n_cells) if n_cells is not None else W.shape[0]

    if group_sizes is None:
        n_i = np.bincount(codes, minlength=K).astype(float)
    else:
        n_i = np.asarray(group_sizes, dtype=float)

    G = sp.sparse.csr_matrix(
        (np.ones(W.shape[0]), (codes, np.arange(W.shape[0]))), shape=(K, W.shape[0])
    )

    if sp.sparse.issparse(W):
        S = np.asarray((G @ W).todense())
        sum_w = np.asarray(W.sum(axis=0)).ravel()
        sumsq_w = np.asarray(W.multiply(W).sum(axis=0)).ravel()
    else:
        W = np.asarray(W, dtype=float)
        S = G @ W
        sum_w = W.sum(axis=0)
        sumsq_w = (W ** 2).sum(axis=0)

    mean_w = sum_w / N
    if var_w is None:
        var_w = np.maximum(sumsq_w / N - mean_w ** 2, 0.0)
    else:
        var_w = np.broadcast_to(np.asarray(var_w, dtype=float), mean_w.shape)

    var_factor = n_i * (N - n_i) / (N - 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        Z = (S - n_i[:, None] * mean_w[None, :]) / np.sqrt(
            var_factor[:, None] * var_w[None, :]
        )
    Z[:, var_w <= 0] = np.nan
    Z[var_factor <= 0, :] = 0.0
    return Z


def wilcoxon_z_by_group(adata, group_key="celltype", layer=None, use_raw=False, groups=None,
                        tie_correct=True):
    """
    One-vs-rest Wilcoxon rank-sum z-score per feature per cell type. Sparse-aware version of
    sc.tl.rank_genes_groups(method='wilcoxon') 'scores'.
 
        R_i = sum of ranks of the feature's values among cells of type i
        z_i = ( R_i - n_i(N+1)/2 ) / sqrt( n_i * (N - n_i) * (N+1) / 12 )
 
    Parameters
    ----------
    groups
        Optional array-like of group labels replacing adata.obs[group_key].
    tie_correct
        Standardize by the empirical rank variance, which accounts for the tied
        zero block.

    Returns
    -------
    Dataframe with one row per feature and one column per cell type, in the
    same shape and column order as mean_by_group().
    """
    if use_raw:
        X = adata.raw.X; var_names = adata.raw.var_names; obs = adata.obs
    else:
        obs = adata.obs; var_names = adata.var_names
        X = adata.X if layer is None else adata.layers[layer]
 
    if groups is None:
        groups = obs[group_key].astype("category")
    else:
        groups = pd.Series(np.asarray(groups), index=obs.index).astype("category")
    codes = groups.cat.codes.to_numpy()
    cats = groups.cat.categories
    K = len(cats)
 
    Xs = sp.sparse.csc_matrix(X) if sp.sparse.issparse(X) else sp.sparse.csc_matrix(np.asarray(X))
    n_cells, n_feat = Xs.shape
 
    # Rank nonzeros only using lexsort
    indptr, data = Xs.indptr, Xs.data
    nnz = len(data)
    col = np.repeat(np.arange(n_feat), np.diff(indptr))
    n_zero = (n_cells - np.diff(indptr)).astype(float)
 
    if nnz:
        order = np.lexsort((data, col))
        sc_, sd_ = col[order], data[order]
        rank_nz = (np.arange(nnz) - indptr[sc_]) + 1.0 # 1-based within column
        # average tied ranks consistent with scanpy
        newgrp = np.empty(nnz, dtype=bool); newgrp[0] = True
        newgrp[1:] = (sc_[1:] != sc_[:-1]) | (sd_[1:] != sd_[:-1])
        starts = np.flatnonzero(newgrp)
        sums = np.add.reduceat(rank_nz, starts)
        cnts = np.diff(np.append(starts, nnz))
        rank_nz = np.repeat(sums / cnts, cnts) + n_zero[sc_] # shift past the zero block
        Rnz = np.empty(nnz); Rnz[order] = rank_nz
    else:
        Rnz = np.empty(0)
 
    # center each column's nonzero ranks on its zero-block rank
    Tnz = Rnz - (n_zero[col] + 1.0) / 2.0 if nnz else Rnz
    Tmat = sp.sparse.csc_matrix((Tnz, Xs.indices, Xs.indptr), shape=Xs.shape)

    Z = groupwise_z(
        Tmat, codes, n_groups=K,
        var_w=None if tie_correct else (n_cells ** 2 - 1) / 12.0,
    )

    out = pd.DataFrame(Z.T, index=var_names, columns=cats)
    return out.reset_index(names='feature')


def link_score_matrix(rna_adata, atac_adata, gene_names, peak_names,
                      rna_layer=None, atac_layer=None):
    """
    Per-cell link scores w_c = rna_c(gene) * atac_c(peak), as a sparse
    (n_cells, n_kept) matrix.

    Build this once and reuse it with groupwise_z() whenever the same links are
    scored under many group assignments for permutation null.

    Returns
    -------
    W
        (n_cells, n_kept) csc matrix of per-cell link scores.
    cols
        Positions in gene_names/peak_names that W's columns correspond to.
    """
    if rna_adata.n_obs != atac_adata.n_obs:
        raise ValueError("rna_adata and atac_adata must have the same cells.")
    if not np.array_equal(np.asarray(rna_adata.obs_names), np.asarray(atac_adata.obs_names)):
        raise ValueError("rna_adata and atac_adata cells must be in the same order.")

    gene_names = np.asarray(gene_names)
    peak_names = np.asarray(peak_names)
    if gene_names.shape != peak_names.shape:
        raise ValueError("gene_names and peak_names must have the same length.")

    A = rna_adata.X if rna_layer is None else rna_adata.layers[rna_layer]
    B = atac_adata.X if atac_layer is None else atac_adata.layers[atac_layer]
    A = A.tocsc() if sp.sparse.issparse(A) else sp.sparse.csc_matrix(A)
    B = B.tocsc() if sp.sparse.issparse(B) else sp.sparse.csc_matrix(B)
    N = A.shape[0]

    gene_idx = pd.Index(rna_adata.var_names).get_indexer(gene_names)
    peak_idx = pd.Index(atac_adata.var_names).get_indexer(peak_names)
    valid = np.flatnonzero((gene_idx >= 0) & (peak_idx >= 0))
    if len(valid) == 0:
        return sp.sparse.csc_matrix((N, 0)), np.array([], dtype=int)

    order = valid[np.argsort(gene_idx[valid], kind='stable')]
    bounds = np.flatnonzero(np.r_[True, np.diff(gene_idx[order]) != 0])

    a_dense = np.zeros(N)
    cols, rows = [], []
    for blk in np.split(order, bounds[1:]):
        g = gene_idx[blk[0]]
        sl = slice(A.indptr[g], A.indptr[g + 1])
        nz, vals = A.indices[sl], A.data[sl]
        if len(nz) == 0:
            continue                                  # gene never detected

        # scale this gene's peak columns by its per-cell values
        P = B[:, peak_idx[blk]].tocsc().astype(np.float64)
        a_dense[nz] = vals
        P.data *= a_dense[P.indices]
        a_dense[nz] = 0.0
        P.eliminate_zeros()

        cols.append(P)
        rows.append(blk)

    if not cols:
        return sp.sparse.csc_matrix((N, 0)), np.array([], dtype=int)

    W = sp.sparse.hstack(cols, format='csc') if len(cols) > 1 else cols[0]
    return W, np.concatenate(rows)


def joint_cell_z(
    rna_adata,
    atac_adata,
    gene_names,
    peak_names,
    group_key="celltype",
    groups=None,
    rna_layer=None,
    atac_layer=None,
    gene_col='gene',
    peak_col='peak',
    chunk_size=20000,
    min_nonzero=0,
):
    """
    One-vs-rest z for a per-cell link score: the product is formed per cell and
    then tested, instead of combining two per-celltype summaries.

        w_c      = rna_c(gene) * atac_c(peak)
        S_i      = sum_{c in i} w_c
        E[S_i]   = n_i * mean(w)
        Var[S_i] = n_i * (N - n_i) / (N - 1) * var(w)
        z_i      = ( S_i - E[S_i] ) / sqrt( Var[S_i] )

    Parameters
    ----------
    rna_adata, atac_adata
        Cell x feature AnnData for the two modalities, same cells in the same
        order.
    gene_names, peak_names
        Aligned link endpoints, one entry per link.
    groups
        Optional array-like of group labels replacing rna_adata.obs[group_key].
    chunk_size
        Number of links scored per block.
    min_nonzero
        Links with fewer than this many cells where w != 0 return NaN. Under
        sparsity the statistic degenerates toward a co-detection test.

    Returns
    -------
    Link x cell-type table with one row per input link, in the same shape and
    column order as wilcoxon_z_by_group() but keyed by gene_col and peak_col.
    """
    gene_names = np.asarray(gene_names)
    peak_names = np.asarray(peak_names)

    if groups is None:
        groups = rna_adata.obs[group_key].astype("category")
    else:
        groups = pd.Series(np.asarray(groups), index=rna_adata.obs_names).astype("category")
    codes = groups.cat.codes.to_numpy()
    cats = groups.cat.categories
    K = len(cats)

    N = rna_adata.n_obs
    n_i = np.bincount(codes, minlength=K).astype(float)

    Z = np.full((len(gene_names), K), np.nan)
    for start in range(0, len(gene_names), chunk_size):
        block = slice(start, start + chunk_size)
        W, cols = link_score_matrix(rna_adata, atac_adata,
                                    gene_names[block], peak_names[block],
                                    rna_layer=rna_layer, atac_layer=atac_layer)
        if W.shape[1] == 0:
            continue
        z = groupwise_z(W, codes, n_groups=K, n_cells=N, group_sizes=n_i)
        if min_nonzero:
            z[:, np.diff(W.indptr) < min_nonzero] = np.nan
        Z[start + cols] = z.T

    out = pd.DataFrame(Z, columns=cats)
    out.insert(0, peak_col, peak_names)
    out.insert(0, gene_col, gene_names)
    return out


def attach_link_specificity(
    link_df,
    gene_spec_df,
    peak_spec_df,
    rna_means=None,
    atac_means=None,
    link_z=None,
    subtypes_dict=None,
    gene_col='gene',
    peak_col='peak',
    gene_spec_col='specificity',
    peak_spec_col='specificity',
    out_col='link_specificity',
    weight_concordance=False,
    eps_rna=1e-6,
    eps_atac=1e-6,
):
    """
    Attach link-level cell-type specificity statistics.

    Parameters
    ----------
    link_df
        One row per peak-gene link.

    gene_spec_df, peak_spec_df
        Feature-level specificity tables returned by
        compute_feature_specificity(). These should contain:
            feature, specificity, dominant_subtype

    rna_means, atac_means
        Feature x cell-type mean tables returned by mean_by_group().
        Required for joint entropy specificity.

    link_z
        Link x cell-type z table from joint_cell_z(), keyed by gene_col and
        peak_col. Reduced here to max_i z_i and its cell type.

    subtypes_dict
        Optional mapping from subtype -> broader cell type.

    Returns
    -------
    out
        link_df with gene/peak specificity, dominant subtype,
        joint entropy specificity, geometric specificity,
        minimum specificity, and concordance.
    """
    gene_features = gene_spec_df.rename(columns={
        'feature': gene_col,
        gene_spec_col: 'gene_specificity',
        'dominant_subtype': 'gene_dominant_subtype'
    })[[gene_col, 'gene_specificity', 'gene_dominant_subtype']]

    peak_features = peak_spec_df.rename(columns={
        'feature': peak_col,
        peak_spec_col: 'peak_specificity',
        'dominant_subtype': 'peak_dominant_subtype'
    })[[peak_col, 'peak_specificity', 'peak_dominant_subtype']]

    if subtypes_dict is not None:
        gene_features['gene_dominant_celltype'] = gene_features['gene_dominant_subtype'].map(subtypes_dict)
        peak_features['peak_dominant_celltype'] = peak_features['peak_dominant_subtype'].map(subtypes_dict)

    out = link_df.merge(gene_features, on=gene_col, how='left').merge(peak_features, on=peak_col, how='left')

    out[f'{out_col}_joint'] = np.nan
    if rna_means is not None and atac_means is not None:
        subtype_cols = [
            c for c in rna_means.columns
            if c in atac_means.columns
            and c not in {'feature', gene_col, peak_col}
        ]
        if not subtype_cols:
            raise ValueError(
                "No shared cell-type columns found in rna_means and atac_means."
            )

        gene_row = pd.Index(rna_means['feature']).get_indexer(out[gene_col])
        peak_row = pd.Index(atac_means['feature']).get_indexer(out[peak_col])
        has_both_features = (gene_row >= 0) & (peak_row >= 0)

        if np.any(has_both_features):
            rna_means_values = rna_means[subtype_cols].to_numpy(dtype=np.float64, copy=False)
            atac_means_values = atac_means[subtype_cols].to_numpy(dtype=np.float64, copy=False)
            out.loc[has_both_features, f'{out_col}_joint'] = joint_entropy_specificity(
                np.clip(rna_means_values, 0, None) + eps_rna,
                np.clip(atac_means_values, 0, None) + eps_atac,
                gene_row[has_both_features], peak_row[has_both_features],
            )

    out[f'{out_col}_min'] = np.minimum(out['gene_specificity'].to_numpy(), out['peak_specificity'].to_numpy())
    out[f'{out_col}_geom'] = np.sqrt(out['gene_specificity'].to_numpy() * out['peak_specificity'].to_numpy())

    dom_nan_mask = out['gene_dominant_subtype'].notna() & out['peak_dominant_subtype'].notna()
    out.loc[dom_nan_mask, 'dominant_subtype_concordant'] = (out.loc[dom_nan_mask, 'gene_dominant_subtype'] == out.loc[dom_nan_mask, 'peak_dominant_subtype']).astype(float)

    if subtypes_dict is not None:
        mapped_celltypes = (
            out['gene_dominant_celltype'].notna() &
            out['peak_dominant_celltype'].notna()
        )
        out['dominant_celltype_concordant'] = np.nan
        out.loc[mapped_celltypes, 'dominant_celltype_concordant'] = (
            out.loc[mapped_celltypes, 'gene_dominant_celltype'] ==
            out.loc[mapped_celltypes, 'peak_dominant_celltype']
        ).astype(float)

    out[f'{out_col}_cellz'] = np.nan
    out[f'{out_col}_cellz_subtype'] = np.nan
    if link_z is not None:
        z_cols = [c for c in link_z.columns if c not in {gene_col, peak_col, 'feature'}]
        if not z_cols:
            raise ValueError("No cell-type columns found in link_z.")

        link_z = link_z.drop_duplicates([gene_col, peak_col])
        z_link_keys = pd.MultiIndex.from_frame(link_z[[gene_col, peak_col]])
        z_row = z_link_keys.get_indexer(pd.MultiIndex.from_frame(out[[gene_col, peak_col]]))
        has_z = z_row >= 0

        if np.any(has_z):
            z_values = link_z[z_cols].to_numpy(dtype=np.float64, copy=False)[z_row[has_z]]
            is_scored = ~np.isnan(z_values).all(axis=1)
            with np.errstate(invalid='ignore'):
                out.loc[has_z, f'{out_col}_cellz'] = np.where(
                    is_scored,
                    np.nanmax(np.where(is_scored[:, None], z_values, 0.0), axis=1),
                    np.nan,
                )
            best_celltype_idx = np.full(len(z_values), np.nan)
            best_celltype_idx[is_scored] = np.nanargmax(z_values[is_scored], axis=1)
            out.loc[has_z, f'{out_col}_cellz_subtype'] = [
                z_cols[int(i)] if np.isfinite(i) else np.nan for i in best_celltype_idx
            ]

    if weight_concordance:
        if subtypes_dict is not None:
            print(
                "Weighting link specificity by dominant_celltype_concordant; "
                "rows with unmapped celltypes remain NaN-weighted.",
                flush=True,
            )
            out[f'{out_col}_geom'] = out[f'{out_col}_geom'] * out['dominant_celltype_concordant']
            out[f'{out_col}_joint'] = out[f'{out_col}_joint'] * out['dominant_celltype_concordant']
        else:
            print(
                "weight_concordance requested without subtypes_dict; "
                "falling back to dominant_subtype_concordant.",
                flush=True,
            )
            out[f'{out_col}_geom'] = out[f'{out_col}_geom'] * out['dominant_subtype_concordant']
            out[f'{out_col}_joint'] = out[f'{out_col}_joint'] * out['dominant_subtype_concordant']

    return out


def permutation_replicate_stats(
    seed, rna_adata, atac_adata, gene_names, peak_names, group_key="celltype",
    stats=("link_specificity_joint", "link_specificity_cellz",
           "gene_specificity", "peak_specificity"),
    eps_rna=1e-6, eps_atac=1e-6, W=None, cols=None, cats=None,
):
    """
    One label-permutation replicate's link-level specificity statistics.
    Shuffles rna_adata.obs[group_key] under `seed` and recomputes the whole
    per-celltype-mean -> feature-specificity -> link-specificity pipeline
    under that shuffle. For 'link_specificity_cellz', pass the per-cell score
    matrix W (from link_score_matrix, independent of any label assignment) 
    and re-standardized via groupwise_z.

    Parameters
    ----------
    rna_adata, atac_adata
        AnnData where NaN counts have been removed, same cell order.
    gene_names, peak_names
        Aligned links to score (i.e. cis-links).
    stats
        Must be columns attach_link_specificity can produce.
    seed
        This replicate's child seed.
    W, cols, cats
        link_score_matrix's output + rna_adata.obs[group_key]'s category
        list, built once and reused read-only across every replicate. Pass 
        None (default) if 'link_specificity_cellz' isn't in stats.

    Returns
    -------
    {stat: (n_links,) ndarray}
    """
    labels = rna_adata.obs[group_key].to_numpy()
    rng = np.random.default_rng(seed)
    perm_labels = rng.permutation(labels)

    ct_rna_means = mean_by_group(rna_adata, group_key, groups=perm_labels)
    ct_atac_means = mean_by_group(atac_adata, group_key, groups=perm_labels)
    rna_spec = compute_feature_specificity(ct_rna_means)
    atac_spec = compute_feature_specificity(ct_atac_means)

    out = attach_link_specificity(
        pd.DataFrame({"gene": gene_names, "peak": peak_names}),
        rna_spec, atac_spec, rna_means=ct_rna_means, atac_means=ct_atac_means,
        eps_rna=eps_rna, eps_atac=eps_atac,
    )
    row = {s: out[s].to_numpy() for s in stats if s != "link_specificity_cellz"}
    if "link_specificity_cellz" in stats:
        perm_codes = pd.Categorical(perm_labels, categories=cats).codes
        z = groupwise_z(W, perm_codes, n_groups=len(cats), n_cells=rna_adata.n_obs)
        cellz = np.full(len(gene_names), np.nan)
        cellz[cols] = np.fmax.reduce(z, axis=0)
        row["link_specificity_cellz"] = cellz
    return row


def attach_ground_truth_specificity(
    link_df,
    truth_spec_df,
    link_key_cols=('tenk10k_id', 'gt_gene'),
    truth_key_cols=('tenk10k_id', 'gt_gene'),
    truth_spec_col='specificity',
    out_col='specificity',
    duplicate_strategy='error',
):
    """
    Attach a ground-truth specificity metric to evaluation rows by key columns.

    Parameters
    ----------
    link_df
        Evaluation dataframe to annotate.
    truth_spec_df
        Specificity table keyed by ground-truth identifiers.
    link_key_cols
        Column names in link_df used for the join.
    truth_key_cols
        Column names in truth_spec_df used for the join.
    truth_spec_col
        Specificity column in truth_spec_df.
    out_col
        Output specificity column name in the returned dataframe.
    duplicate_strategy
        How to handle duplicated keys in truth_spec_df:
        'error', 'first', or 'mean'.
    """
    link_key_cols = list(link_key_cols)
    truth_key_cols = list(truth_key_cols)

    if len(link_key_cols) != len(truth_key_cols):
        raise ValueError("link_key_cols and truth_key_cols must have the same length.")

    missing_link = [c for c in link_key_cols if c not in link_df.columns]
    if missing_link:
        raise ValueError(f"Missing link_df key columns: {missing_link}")

    required_truth_cols = truth_key_cols + [truth_spec_col]
    missing_truth = [c for c in required_truth_cols if c not in truth_spec_df.columns]
    if missing_truth:
        raise ValueError(f"Missing truth_spec_df columns: {missing_truth}")

    truth = truth_spec_df[required_truth_cols].copy()

    dup_mask = truth.duplicated(truth_key_cols, keep=False)
    if dup_mask.any():
        if duplicate_strategy == 'error':
            dup_rows = truth.loc[dup_mask, truth_key_cols].drop_duplicates()
            raise ValueError(
                "Duplicate keys found in truth_spec_df. "
                f"Example duplicated keys: {dup_rows.head().to_dict(orient='records')}"
            )
        if duplicate_strategy == 'first':
            truth = truth.drop_duplicates(truth_key_cols, keep='first')
        elif duplicate_strategy == 'mean':
            truth = (
                truth.groupby(truth_key_cols, as_index=False)[truth_spec_col]
                .mean()
            )
        else:
            raise ValueError(
                "duplicate_strategy must be one of {'error', 'first', 'mean'}."
            )

    rename_map = {
        truth_key_col: link_key_col
        for link_key_col, truth_key_col in zip(link_key_cols, truth_key_cols)
        if link_key_col != truth_key_col
    }
    truth = truth.rename(columns=rename_map)

    out = link_df.merge(truth, on=link_key_cols, how='left')
    if truth_spec_col != out_col:
        out = out.rename(columns={truth_spec_col: out_col})
    return out


def pip_entropy_specificity(pip_matrix, eps=1e-12):
    """
    pip_matrix: (n, k) matrix where n is #links, k is #subtypes

    Returns normalized entropy specificity in [0, 1] when defined.
    Rows with zero total signal return np.nan.
    """
    pip_matrix = np.asarray(pip_matrix, dtype=float)

    if pip_matrix.ndim != 2:
        raise ValueError("pip_matrix must be a 2D array.")

    if np.any(pip_matrix < 0):
        raise ValueError("pip_matrix must be nonnegative.")

    k = pip_matrix.shape[1]
    if k <= 1:
        return np.full(pip_matrix.shape[0], np.nan, dtype=float)

    row_sums = pip_matrix.sum(axis=1)
    out = np.full(pip_matrix.shape[0], np.nan, dtype=float)

    valid = row_sums > eps
    if not np.any(valid):
        return out

    q = pip_matrix[valid] / row_sums[valid, None]

    # Use the standard convention 0 * log(0) = 0.
    log_q = np.zeros_like(q)
    positive = q > 0
    log_q[positive] = np.log(q[positive])

    h = -(q * log_q).sum(axis=1)
    h_norm = h / np.log(k)
    out[valid] = 1.0 - h_norm

    return out
