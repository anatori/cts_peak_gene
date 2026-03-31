import numpy as np
import pandas as pd
import scipy.sparse as sp


def mean_by_group(adata, group_key="celltype", layer=None, use_raw=False):
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

    groups = obs[group_key].astype("category")
    codes = groups.cat.codes.to_numpy()
    n_groups = len(groups.cat.categories)
    n_genes = X.shape[1]

    # one-hot (cells x groups)
    G = sp.csr_matrix(
        (np.ones(X.shape[0]), (np.arange(X.shape[0]), codes)),
        shape=(X.shape[0], n_groups),
    )

    # sums: (groups x genes)
    sums = (G.T @ X) if sp.issparse(X) else (G.T @ sp.csr_matrix(X))

    counts = np.asarray(G.sum(axis=0)).ravel()  # cells per group
    means = sums.multiply(1.0 / counts[:, None]) if sp.issparse(sums) else (sums / counts[:, None])

    # to DataFrame (dense)
    means = means.toarray() if sp.issparse(means) else np.asarray(means)
    means_df = pd.DataFrame(means, index=groups.cat.categories, columns=var_names).T
    means_df = means_df.reset_index(names='feature')
    return means_df


def entropy_specificity(x, eps=1e-6, min_total=0.0):
    """
    x: 1D array-like of subtype-level means for one feature
    returns specificity in [0,1], or np.nan if total signal is too low
    """
    x = np.asarray(x, dtype=float)

    if np.nansum(x) < min_total:
        return np.nan

    x = np.clip(x, 0, None)
    x = x + eps
    p = x / x.sum()

    k = len(p)
    h = -(p * np.log(p)).sum()
    h_norm = h / np.log(k)

    return 1.0 - h_norm


def dominant_subtype(x):
    x = np.asarray(x, dtype=float)
    if np.all(np.isnan(x)):
        return np.nan
    return int(np.nanargmax(x))


def compute_feature_specificity(mean_df, feature_col='feature', subtype_cols=None,
                                eps=1e-6, min_total=0.0):
    """
    mean_df: dataframe with one row per feature and subtype mean columns
    returns dataframe with specificity and dominant subtype
    """
    if subtype_cols is None:
        subtype_cols = [c for c in mean_df.columns if c != feature_col]

    out = mean_df[[feature_col]].copy()

    vals = mean_df[subtype_cols].to_numpy(dtype=float)

    out['specificity'] = [
        entropy_specificity(row, eps=eps, min_total=min_total)
        for row in vals
    ]
    out['dominant_subtype_idx'] = [
        dominant_subtype(row) for row in vals
    ]
    out['dominant_subtype'] = [
        subtype_cols[i] if pd.notna(i) else np.nan
        for i in out['dominant_subtype_idx']
    ]

    return out


def attach_link_specificity(link_df,
                            gene_spec_df, peak_spec_df,
                            gene_col='gene', peak_col='peak',
                            gene_spec_col='specificity',
                            peak_spec_col='specificity'):
    """
    link_df: one row per peak-gene link
    gene_spec_df: feature specificity table for genes
    peak_spec_df: feature specificity table for peaks
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

    out = link_df.merge(g, on=gene_col, how='left').merge(p, on=peak_col, how='left')

    out['link_specificity_min'] = np.minimum(
        out['gene_specificity'], out['peak_specificity']
    )
    out['link_specificity_geom'] = np.sqrt(
        out['gene_specificity'] * out['peak_specificity']
    )

    out['dominant_subtype_concordant'] = (
        out['gene_dominant_subtype'] == out['peak_dominant_subtype']
    ).astype(float)

    return out