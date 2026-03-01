import numpy as np
import pandas as pd 
import scipy as sp
import warnings
from ctar.data_loader import get_gene_coords



def null_peak_gene_pairs(rna, atac):

    ''' Generate null peak gene pairs in which genes and peaks are on different chromosomes.

    Parameters
    ----------
    rna : an.AnnData
        AnnData of len (#genes). Must contain rna.var DataFrame with 'intervals' describing
        gene region and 'gene_name' listing gene names.
    atac : an.AnnData
        AnnData of len (#peaks). Must contain rna.var DataFrame with 'gene_ids' describing
        peak region and 'gene_ids' listing gene names.

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
    atac.var[['chr','range']] = atac.var['gene_ids'].str.split(':', n=1, expand=True)
    rna.var[['chr','range']] = rna.var['interval'].str.split(':', n=1, expand=True)

    df_list = []

    # For each unique chrom (23 + extrachromosomal)
    for chrom in atac.var.chr.unique():
        
        # Find corresponding chrom
        chrm_peaks = atac.var.loc[atac.var['chr'] == chrom][['index_x','gene_ids']]
        # Find genes NOT on that chrom
        nonchrm_genes = rna.var.loc[rna.var['chr'] != chrom][['index_y','gene_name']]
        # Sample random genes
        rand_genes = nonchrm_genes.sample(n=len(chrm_peaks))
        # Concat into one df
        rand_peak_gene_pairs = pd.concat([chrm_peaks.reset_index(drop=True),rand_genes.reset_index(drop=True)],axis=1)
        df_list += [rand_peak_gene_pairs]

    # Concat dfs for all chroms
    null_pairs = pd.concat(df_list,ignore_index=True)
    return null_pairs


def topk_or_by_regime(
    df,
    score_col,            # p-values or scores for ONE method
    regime_mask,          # boolean mask for the regime
    truth_col="score",
    K=100,
    smaller_is_better=True,
    continuity=0.5,
    alpha=0.05
):
    d = df[regime_mask].copy()
    d[truth_col] = d[truth_col].astype(bool)

    if len(d) == 0:
        return None

    # rank within regime
    if smaller_is_better:
        d["rank"] = d[score_col].rank(method="average")
    else:
        d["rank"] = (-d[score_col]).rank(method="average")

    d["topK"] = d["rank"] <= min(K, len(d))  # avoid empty "not topK" when n<K

    # contingency table counts
    a = int(((d["topK"]) & (d[truth_col])).sum())
    b = int(((d["topK"]) & (~d[truth_col])).sum())
    c = int(((~d["topK"]) & (d[truth_col])).sum())
    d_ = int(((~d["topK"]) & (~d[truth_col])).sum())

    table = np.array([[a, b],
                      [c, d_]])

    # Fisher test (exact; may return inf/0 when zeros)
    try:
        or_val, p = sp.stats.fisher_exact(table)
    except Exception:
        or_val, p = np.nan, np.nan

    # continuity-corrected OR (stable for display + CI)
    ac, bc, cc, dc = (a + continuity, b + continuity, c + continuity, d_ + continuity)
    or_cc = (ac * dc) / (bc * cc)

    # Wald CI on log(OR_cc)
    z = sp.stats.norm.ppf(1 - alpha / 2)
    se_logOR_cc = np.sqrt(1/ac + 1/bc + 1/cc + 1/dc)
    logOR_cc = np.log(or_cc)
    ci_low = np.exp(logOR_cc - z * se_logOR_cc)
    ci_high = np.exp(logOR_cc + z * se_logOR_cc)

    # log2 scale versions (handy for heatmaps if you prefer)
    log2OR = np.log2(or_cc)
    log2_ci_low = np.log2(ci_low)
    log2_ci_high = np.log2(ci_high)

    return {
        "a": a, "b": b, "c": c, "d": d_,
        "OR": or_val,
        "OR_cc": or_cc,
        "OR_ci_low": ci_low,
        "OR_ci_high": ci_high,
        "logOR_cc": logOR_cc,
        "se_logOR_cc": se_logOR_cc,
        "log2OR": log2OR,
        "log2OR_ci_low": log2_ci_low,
        "log2OR_ci_high": log2_ci_high,
        "p": p,
        "n": int(a + b + c + d_)
    }


def global_topk_or_by_regime(
    df,
    score_col,
    regimes, # list of (regime_name, mask) or dict {name: mask}
    truth_col="score",
    K=100,
    smaller_is_better=True,
    continuity=0.5,
    alpha=0.05,
):
    """
    Global Top-K is computed ONCE over all rows.
    Then for each regime mask, compute 2x2 OR of True vs False for TopK vs not-TopK
    restricted to that regime.

    regimes: list of tuples [("regime1", mask1), ...] OR dict {"regime1": mask1, ...}
    """
    d = df.copy()
    d[truth_col] = d[truth_col].astype(bool)

    # global rank + global topK
    if smaller_is_better:
        d["_rank"] = d[score_col].rank(method="average")
    else:
        d["_rank"] = (-d[score_col]).rank(method="average")

    d["_topK_global"] = d["_rank"] <= K

    # normalize regimes input
    if isinstance(regimes, dict):
        regimes_iter = list(regimes.items())
    else:
        regimes_iter = list(regimes)

    rows = []
    z = sp.stats.norm.ppf(1 - alpha/2)

    for regime_name, mask in regimes_iter:
        sub = d.loc[mask].copy()
        if len(sub) == 0:
            continue

        a = int(((sub["_topK_global"]) & (sub[truth_col])).sum())
        b = int(((sub["_topK_global"]) & (~sub[truth_col])).sum())
        c = int(((~sub["_topK_global"]) & (sub[truth_col])).sum())
        d_ = int(((~sub["_topK_global"]) & (~sub[truth_col])).sum())

        table = np.array([[a, b],
                          [c, d_]])

        # Fisher exact
        try:
            or_val, p = sp.stats.fisher_exact(table)
        except Exception:
            or_val, p = np.nan, np.nan

        # continuity-corrected OR + Wald CI on log(OR_cc)
        ac, bc, cc, dc = a+continuity, b+continuity, c+continuity, d_+continuity
        or_cc = (ac * dc) / (bc * cc)
        se = np.sqrt(1/ac + 1/bc + 1/cc + 1/dc)
        ci_low = np.exp(np.log(or_cc) - z*se)
        ci_high = np.exp(np.log(or_cc) + z*se)

        rows.append({
            "regime": regime_name,
            "a": a, "b": b, "c": c, "d": d_,
            "OR": or_val,
            "OR_cc": or_cc,
            "OR_ci_low": ci_low,
            "OR_ci_high": ci_high,
            "log2OR": np.log2(or_cc),
            "p": p,
            "n_regime": int(a+b+c+d_),
            "K_global": int(K),
        })

    return pd.DataFrame(rows)


def global_fdr_or_by_regime(
    df,
    fdr_col,
    regimes, # list of (regime_name, mask) or dict {name: mask}
    truth_col="score",
    fdr_level=0.1,
    continuity=0.5,
    alpha=0.05,
):
    """
    Global Top-K is computed ONCE over all rows.
    Then for each regime mask, compute 2x2 OR of True vs False for TopK vs not-TopK
    restricted to that regime.

    regimes: list of tuples [("regime1", mask1), ...] OR dict {"regime1": mask1, ...}
    """
    d = df.copy()
    d[truth_col] = d[truth_col].astype(bool)

    d["_fdr_global"] = d[fdr_col] <= fdr_level

    # normalize regimes input
    if isinstance(regimes, dict):
        regimes_iter = list(regimes.items())
    else:
        regimes_iter = list(regimes)

    rows = []
    z = sp.stats.norm.ppf(1 - alpha/2)

    for regime_name, mask in regimes_iter:
        sub = d.loc[mask].copy()
        if len(sub) == 0:
            continue

        a = int(((sub["_fdr_global"]) & (sub[truth_col])).sum())
        b = int(((sub["_fdr_global"]) & (~sub[truth_col])).sum())
        c = int(((~sub["_fdr_global"]) & (sub[truth_col])).sum())
        d_ = int(((~sub["_fdr_global"]) & (~sub[truth_col])).sum())

        table = np.array([[a, b],
                          [c, d_]])

        # Fisher exact
        try:
            or_val, p = sp.stats.fisher_exact(table)
        except Exception:
            or_val, p = np.nan, np.nan

        # continuity-corrected OR + Wald CI on log(OR_cc)
        ac, bc, cc, dc = a+continuity, b+continuity, c+continuity, d_+continuity
        or_cc = (ac * dc) / (bc * cc)
        se = np.sqrt(1/ac + 1/bc + 1/cc + 1/dc)
        ci_low = np.exp(np.log(or_cc) - z*se)
        ci_high = np.exp(np.log(or_cc) + z*se)

        rows.append({
            "regime": regime_name,
            "a": a, "b": b, "c": c, "d": d_,
            "OR": or_val,
            "OR_cc": or_cc,
            "OR_ci_low": ci_low,
            "OR_ci_high": ci_high,
            "log2OR": np.log2(or_cc),
            "p": p,
            "n_regime": int(a+b+c+d_)
        })

    return pd.DataFrame(rows)
