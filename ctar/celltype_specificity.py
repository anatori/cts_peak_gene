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


def joint_entropy_specificity(rna_x, atac_x, eps=1e-6, min_total=0.0):
    """
    Compute link specificity from the entropy of the subtype-wise joint signal.

    Parameters
    ----------
    rna_x
        1D array-like RNA subtype vector for one gene.
    atac_x
        1D array-like ATAC subtype vector for one peak.

    Notes
    -----
    This uses the aligned subtype-wise product of the two vectors as the
    link-level signal and then computes entropy specificity on that joint
    subtype profile. Although this is sometimes described informally as a
    "dot product", the entropy is computed on the per-subtype product vector,
    not on a collapsed scalar.
    """
    rna_x = np.asarray(rna_x, dtype=float)
    atac_x = np.asarray(atac_x, dtype=float)

    if rna_x.shape != atac_x.shape:
        raise ValueError(
            "rna_x and atac_x must have the same shape to compute joint entropy specificity."
        )

    joint_x = np.clip(rna_x, 0, None) * np.clip(atac_x, 0, None)
    return entropy_specificity(joint_x, eps=eps, min_total=min_total)


def attach_link_specificity(link_df,
                            gene_spec_df, peak_spec_df,
                            subtypes_dict=None,
                            gene_col='gene', peak_col='peak',
                            gene_spec_col='specificity',
                            peak_spec_col='specificity',
                            out_col='link_specificity',
                            weight_concordance=False):
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

    if subtypes_dict is not None:
        g['gene_dominant_celltype'] = g['gene_dominant_subtype'].map(subtypes_dict)
        p['peak_dominant_celltype'] = p['peak_dominant_subtype'].map(subtypes_dict)

    out = link_df.merge(g, on=gene_col, how='left').merge(p, on=peak_col, how='left')

    subtype_cols = [
        c for c in gene_spec_df.columns
        if c in peak_spec_df.columns and c not in {gene_col, peak_col, 'feature'}
    ]
    subtype_cols = [
        c for c in subtype_cols
        if pd.api.types.is_numeric_dtype(gene_spec_df[c])
        and pd.api.types.is_numeric_dtype(peak_spec_df[c])
        and c not in {
            gene_spec_col,
            peak_spec_col,
            'dominant_subtype_idx',
        }
    ]

    out[f'{out_col}_joint'] = np.nan

    if subtype_cols:
        g_joint = gene_spec_df[[gene_col, *subtype_cols]].copy() if gene_col in gene_spec_df.columns else (
            gene_spec_df.rename(columns={'feature': gene_col})[[gene_col, *subtype_cols]].copy()
        )
        p_joint = peak_spec_df[[peak_col, *subtype_cols]].copy() if peak_col in peak_spec_df.columns else (
            peak_spec_df.rename(columns={'feature': peak_col})[[peak_col, *subtype_cols]].copy()
        )

        g_joint = g_joint.rename(columns={c: f'{c}_rna' for c in subtype_cols})
        p_joint = p_joint.rename(columns={c: f'{c}_atac' for c in subtype_cols})

        out = out.merge(g_joint, on=gene_col, how='left').merge(p_joint, on=peak_col, how='left')

        out[f'{out_col}_joint'] = [
            joint_entropy_specificity(
                row[[f'{c}_rna' for c in subtype_cols]].to_numpy(dtype=float),
                row[[f'{c}_atac' for c in subtype_cols]].to_numpy(dtype=float),
            )
            for _, row in out.iterrows()
        ]

    out[f'{out_col}_min'] = np.minimum(
        out['gene_specificity'], out['peak_specificity']
    )
    out[f'{out_col}_geom'] = np.sqrt(
        out['gene_specificity'] * out['peak_specificity']
    )

    out['dominant_subtype_concordant'] = (
        out['gene_dominant_subtype'] == out['peak_dominant_subtype']
    ).astype(float)

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
