import numpy as np
import pandas as pd
import scipy as sp


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
    x = np.asarray(x, dtype=float)
    if x.ndim not in (1, 2):
        raise ValueError("x must be 1D (one feature) or 2D (n_features, k).")

    flag_scalar = x.ndim == 1
    if flag_scalar:
        x = x[None, :]
 
    row_totals = np.nansum(x, axis=1)
    k = x.shape[1]
 
    xc = np.clip(x, 0, None) + eps
    with np.errstate(invalid='ignore', divide='ignore'):
        p = xc / xc.sum(axis=1, keepdims=True)
        log_p = np.zeros_like(p)
        np.log(p, where=p > 0, out=log_p)
        h = -(p * log_p).sum(axis=1)
        spec = 1.0 - h / np.log(k)
 
    spec[row_totals < min_total] = np.nan
 
    return float(spec[0]) if flag_scalar else spec


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

    vals = mean_df[subtype_cols].to_numpy(dtype=float)

    out['specificity'] = entropy_specificity(vals, min_total=min_total, eps=eps)
    out['dominant_subtype_idx'] = [
        dominant_subtype(row) for row in vals
    ]
    out['dominant_subtype'] = [
        subtype_cols[i] if pd.notna(i) else np.nan
        for i in out['dominant_subtype_idx']
    ]

    return out


def joint_entropy_specificity(rna_x, atac_x, min_total=0.0, eps_rna=1e-6, eps_atac=1e-6, eps_overall=0.0):
    """
    Compute link specificity from the elementwise product.
        joint_i = rna_i * atac_i
        entropy_specificity(joint)

    Parameters
    ----------
    rna_x
       1D RNA array of shape (k,) or 2D of shape (n, k) for one gene.
    atac_x
       1D ATAC array of shape (k,) or 2D of shape (n, k) for one peak.
    eps_rna, eps_atac
        Smoothing value for each modality. Independent so neither dominates.
    eps_overall
        Instead, you can try adding smoothing to joint value.
    Returns
    -------
    Returns float if 1D and (n,) array if 2D.
    """
    rna_x = np.asarray(rna_x, dtype=float)
    atac_x = np.asarray(atac_x, dtype=float)

    if rna_x.shape != atac_x.shape:
        raise ValueError(
            "rna_x and atac_x must have the same shape to compute joint entropy specificity."
        )
    if rna_x.ndim not in (1, 2):
        raise ValueError("rna_x and atac_x must be 1D (one link) or 2D (n_links, k).")

    joint_x = (np.clip(rna_x, 0, None) + eps_rna) * (np.clip(atac_x, 0, None) + eps_atac)
    spec = entropy_specificity(joint_x, min_total=min_total, eps=eps_overall)

    if joint_x.ndim == 1:
        return 0.0 if joint_x.sum() == 0 else spec
    else:
        spec = np.array(spec, copy=True)
        spec[joint_x.sum(axis=1) == 0] = 0.0
        return spec


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
    min_total=0.0,
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
    g = gene_spec_df.rename(columns={
        'feature': gene_col,
        gene_spec_col: 'gene_specificity',
        'dominant_subtype': 'gene_dominant_subtype'
    })[[gene_col, 'gene_specificity', 'gene_dominant_subtype']]

    p = peak_spec_df.rename(columns={
        'feature': peak_col,
        peak_spec_col: 'peak_specificity',
        'dominant_subtype': 'peak_dominant_subtype'
    })[[peak_col, 'peak_specificity', 'peak_dominant_subtype']]

    if subtypes_dict is not None:
        g['gene_dominant_celltype'] = g['gene_dominant_subtype'].map(subtypes_dict)
        p['peak_dominant_celltype'] = p['peak_dominant_subtype'].map(subtypes_dict)

    out = link_df.merge(g, on=gene_col, how='left').merge(p, on=peak_col, how='left')

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

        rna_idx = pd.Index(rna_means['feature']).get_indexer(out[gene_col])
        atac_idx = pd.Index(atac_means['feature']).get_indexer(out[peak_col])
        valid = (rna_idx >= 0) & (atac_idx >= 0)

        if np.any(valid):

            rx = rna_means[subtype_cols].to_numpy(dtype=np.float64, copy=False)[rna_idx[valid]]
            ax = atac_means[subtype_cols].to_numpy(dtype=np.float64, copy=False)[atac_idx[valid]]
            out.loc[valid, f'{out_col}_joint'] = joint_entropy_specificity(rx, ax, min_total=min_total, eps_rna=eps_rna, eps_atac=eps_atac)

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
        z_keys = pd.MultiIndex.from_frame(link_z[[gene_col, peak_col]])
        zidx = z_keys.get_indexer(pd.MultiIndex.from_frame(out[[gene_col, peak_col]]))
        zvalid = zidx >= 0

        if np.any(zvalid):
            Z = link_z[z_cols].to_numpy(dtype=np.float64, copy=False)[zidx[zvalid]]
            scored = ~np.isnan(Z).all(axis=1)
            with np.errstate(invalid='ignore'):
                out.loc[zvalid, f'{out_col}_cellz'] = np.where(
                    scored, np.nanmax(np.where(scored[:, None], Z, 0.0), axis=1), np.nan
                )
            idx = np.full(len(Z), np.nan)
            idx[scored] = np.nanargmax(Z[scored], axis=1)
            out.loc[zvalid, f'{out_col}_cellz_subtype'] = [
                z_cols[int(i)] if np.isfinite(i) else np.nan for i in idx
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
