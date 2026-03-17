import numpy as np
import pandas as pd
import muon as mu
import scipy.sparse as sp
import statsmodels.formula.api as smf



def _get_matrix(adata, layer="counts"):
    if layer is not None and layer in adata.layers:
        return adata.layers[layer]
    return adata.X


def _row_sums(X):
    if sp.issparse(X):
        return np.asarray(X.sum(axis=1)).ravel()
    return np.asarray(X.sum(axis=1)).ravel()


def _nonzero_per_row(X):
    if sp.issparse(X):
        return np.asarray(X.getnnz(axis=1)).ravel()
    return np.count_nonzero(X, axis=1)


def _col_means(X):
    if sp.issparse(X):
        return np.asarray(X.mean(axis=0)).ravel()
    return np.asarray(X.mean(axis=0)).ravel()


def _fraction_top_signal(col_means, top_frac=0.1):
    col_means = np.asarray(col_means, dtype=float)
    if col_means.size == 0:
        return np.nan
    total = col_means.sum()
    if total <= 0:
        return np.nan
    k = max(1, int(np.ceil(col_means.size * top_frac)))
    top_vals = np.partition(col_means, -k)[-k:]
    return float(top_vals.sum() / total)


def _summarize_matrix(X, n_features, prefix, top_frac=0.1):
    depth = _row_sums(X)
    detected = _nonzero_per_row(X)
    detect_rate = detected / n_features
    col_means = _col_means(X)

    return {
        f"{prefix}_mean_depth": float(np.mean(depth)),
        f"{prefix}_median_depth": float(np.median(depth)),
        f"{prefix}_mean_detected": float(np.mean(detected)),
        f"{prefix}_median_detected": float(np.median(detected)),
        f"{prefix}_detect_rate": float(np.mean(detect_rate)),
        f"{prefix}_sparsity": float(1.0 - np.mean(detect_rate)),
        f"{prefix}_top{int(top_frac * 100)}pct_signal_frac": _fraction_top_signal(
            col_means, top_frac=top_frac
        ),
    }


def _load_celltype_map(ct_file):
    ct_df = pd.read_csv(ct_file)

    if "pgboost_ct" not in ct_df.columns:
        raise ValueError(f"{ct_file} must contain a 'pgboost_ct' column.")

    other_cols = [c for c in ct_df.columns if c != "pgboost_ct"]

    if len(other_cols) == 0:
        # assume index already stores cell ids
        return ct_df["pgboost_ct"].to_dict()

    # assume first non-pgboost_ct column contains cell ids
    cell_col = other_cols[0]
    return dict(zip(ct_df[cell_col], ct_df["pgboost_ct"]))


def summarize_dataset_and_celltypes(mu_files, ct_files=None, layer="counts", top_frac=0.1):
    """
    Build one tidy QC dataframe with:
      - dataset-wide rows labeled celltype='all'
      - within-celltype rows labeled with actual celltype

    Returns
    -------
    qc_df : pd.DataFrame
        One row per dataset or dataset-celltype summary.
    """
    if ct_files is None:
        ct_files = {}

    rows = []

    for name, file in mu_files.items():
        mdata = mu.read(file)
        rna = mdata.mod["rna"]
        atac = mdata.mod["atac"]

        # attach celltype annotations if available
        if name in ct_files:
            ct_map = _load_celltype_map(ct_files[name])
            rna.obs["pgboost_ct"] = rna.obs_names.map(ct_map)
            atac.obs["pgboost_ct"] = atac.obs_names.map(ct_map)

        X_rna = _get_matrix(rna, layer=layer)
        X_atac = _get_matrix(atac, layer=layer)

        # dataset-wide summary
        ds_row = {
            "multiome_dataset": name,
            "celltype": "all",
            "n_cells": int(rna.n_obs),
            "log_n_cells": float(np.log1p(rna.n_obs)),
            "rna_n_features": int(rna.n_vars),
            "atac_n_features": int(atac.n_vars),
        }
        ds_row.update(_summarize_matrix(X_rna, rna.n_vars, prefix="rna", top_frac=top_frac))
        ds_row.update(_summarize_matrix(X_atac, atac.n_vars, prefix="atac", top_frac=top_frac))
        rows.append(ds_row)

        # within-celltype summaries
        if "pgboost_ct" not in rna.obs.columns:
            continue

        ct_series = rna.obs["pgboost_ct"]
        valid_cts = ct_series.dropna().unique()

        for ct in valid_cts:
            rna_mask = (rna.obs["pgboost_ct"] == ct).to_numpy()
            atac_mask = (atac.obs["pgboost_ct"] == ct).to_numpy()

            n_ct_rna = int(rna_mask.sum())
            n_ct_atac = int(atac_mask.sum())

            if n_ct_rna == 0 or n_ct_atac == 0:
                continue

            # slice matrices directly; no .copy()
            X_rna_ct = X_rna[rna_mask]
            X_atac_ct = X_atac[atac_mask]

            ct_row = {
                "multiome_dataset": name,
                "celltype": ct,
                "n_cells": n_ct_rna,
                "log_n_cells": float(np.log1p(n_ct_rna)),
                "rna_n_features": int(rna.n_vars),
                "atac_n_features": int(atac.n_vars),
            }
            ct_row.update(_summarize_matrix(X_rna_ct, rna.n_vars, prefix="rna", top_frac=top_frac))
            ct_row.update(_summarize_matrix(X_atac_ct, atac.n_vars, prefix="atac", top_frac=top_frac))
            rows.append(ct_row)

    qc_df = pd.DataFrame(rows).sort_values(["multiome_dataset", "celltype"]).reset_index(drop=True)
    return qc_df


def bootstrap_regression(df, model_string, dataset_col="dataset", weight_col=None, n_boot=2000, seed=1):
    ''' Bootstrap per dataset.
    '''
    rng = np.random.default_rng(seed)
    datasets = df[dataset_col].unique()
    boot_coefs = []
    for i in range(n_boot):
        sampled = rng.choice(datasets, size=len(datasets), replace=True)
        df_boot = pd.concat([
            df[df[dataset_col] == d] for d in sampled
        ])
        if weight_col is None:
            model = smf.ols(
                model_string,
                df_boot,
            ).fit()
        else:
            model = smf.wls(
                model_string,
                df_boot,
                weights=df_boot[weight_col],
            ).fit()
        boot_coefs.append(model.params)
    boot_coefs = pd.DataFrame(boot_coefs)
    return boot_coefs
    

def bootstrap_pvalue(samples):
    return 2 * min(
        (samples <= 0).mean(),
        (samples >= 0).mean()
    )