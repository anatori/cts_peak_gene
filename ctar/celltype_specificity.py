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


def joint_entropy_specificity(rna_x, atac_x, min_total=0.0, eps=1e-6):
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

    joint_x = np.clip(rna_x, 0, None) * np.clip(atac_x, 0, None)
    spec = entropy_specificity(joint_x, min_total=min_total, eps=eps)

    if joint_x.ndim == 1:
        return 0.0 if joint_x.sum() == 0 else spec
    else:
        spec = np.array(spec, copy=True)
        spec[joint_x.sum(axis=1) == 0] = 0.0
        return spec


def wilcoxon_z_by_group(adata, group_key="celltype", layer=None, use_raw=False, groups=None):
    """
    One-vs-rest Wilcoxon rank-sum z-score per feature per cell type. Same as
    sc.tl.rank_genes_groups(method='wilcoxon', tie_correct=False) 'scores' but sparse-aware.
 
        R_i = sum of ranks of the feature's values among cells of type i
        z_i = ( R_i - n_i(N+1)/2 ) / sqrt( n_i * (N - n_i) * (N+1) / 12 )
 
    Parameters
    ----------
    groups
        Optional array-like of group labels replacing adata.obs[group_key].
 
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
 
    Rmat = sp.sparse.csc_matrix((Rnz, Xs.indices, Xs.indptr), shape=Xs.shape)
    NZmat = sp.sparse.csc_matrix((np.ones(nnz), Xs.indices, Xs.indptr), shape=Xs.shape)
 
    G = sp.sparse.csr_matrix(
        (np.ones(n_cells), (np.arange(n_cells), codes)), shape=(n_cells, K)
    )
    Rsum_nz = np.asarray((G.T @ Rmat).todense())
    cnt_nz = np.asarray((G.T @ NZmat).todense())
    n1 = np.asarray(G.sum(axis=0)).ravel()[:, None]
    Rsum = Rsum_nz + (n1 - cnt_nz) * (n_zero[None, :] + 1.0) / 2.0
 
    n2 = n_cells - n1
    with np.errstate(invalid='ignore', divide='ignore'):
        Z = (Rsum - n1 * (n_cells + 1) / 2.0) / np.sqrt(n1 * n2 * (n_cells + 1) / 12.0)
    Z = np.where((n1 == 0) | (n2 == 0), 0.0, Z).T
 
    out = pd.DataFrame(Z, index=var_names, columns=cats)
    return out.reset_index(names='feature')
 
 
def marker_concordance(rna_z, atac_z):
    """
    Link specificity from standard marker statistics:
        T = max_i min( z_rna_i , z_atac_i )
 
    Parameters
    ----------
    rna_z
       1D RNA z array of shape (k,) or 2D of shape (n, k).
    atac_z
       1D ATAC z array of shape (k,) or 2D of shape (n, k).
    Returns
    -------
    Returns float if 1D and (n,) array if 2D.
    """
    rna_z = np.asarray(rna_z, dtype=float)
    atac_z = np.asarray(atac_z, dtype=float)
    if rna_z.shape != atac_z.shape:
        raise ValueError("rna_z and atac_z must have the same shape.")
    if rna_z.ndim not in (1, 2):
        raise ValueError("rna_z and atac_z must be 1D (one link) or 2D (n_links, k).")
 
    scalar = rna_z.ndim == 1
    if scalar:
        rna_z = rna_z[None, :]; atac_z = atac_z[None, :]
 
    T = np.minimum(rna_z, atac_z).max(axis=1)
    return float(T[0]) if scalar else T


def attach_link_specificity(
    link_df,
    gene_spec_df,
    peak_spec_df,
    rna_means=None,
    atac_means=None,
    rna_z=None,
    atac_z=None,
    subtypes_dict=None,
    gene_col='gene',
    peak_col='peak',
    gene_spec_col='specificity',
    peak_spec_col='specificity',
    out_col='link_specificity',
    weight_concordance=False,
    min_total=0.0,
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

    rna_z, atac_z
        Feature x cell-type Wilcoxon rank-sum z tables from
        wilcoxon_z_by_group().

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
    out[f'{out_col}_overlap'] = np.nan
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
            out.loc[valid, f'{out_col}_joint'] = joint_entropy_specificity(rx, ax, min_total=min_total)

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

    out[f'{out_col}_marker'] = np.nan
    if rna_z is not None and atac_z is not None:
        z_cols = [c for c in rna_z.columns if c in atac_z.columns and c not in {'feature', gene_col, peak_col}]
        if not z_cols:
            raise ValueError("No shared celltype columns found in rna_z and atac_z.")
 
        rna_zidx = pd.Index(rna_z['feature']).get_indexer(out[gene_col])
        atac_zidx = pd.Index(atac_z['feature']).get_indexer(out[peak_col])
        zvalid = (rna_zidx >= 0) & (atac_zidx >= 0)
        if np.any(zvalid):
            rz = rna_z[z_cols].to_numpy(dtype=np.float64, copy=False)[rna_zidx[zvalid]]
            az = atac_z[z_cols].to_numpy(dtype=np.float64, copy=False)[atac_zidx[zvalid]]
            out.loc[zvalid, f'{out_col}_marker'] = marker_concordance(rz, az)

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
