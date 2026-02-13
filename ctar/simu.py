import numpy as np
import pandas as pd 
import scipy as sp
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from sklearn import metrics
import os
import warnings
from typing import Optional, Tuple, Dict, Any
import pybedtools
import seaborn as sns
from ctar.data_loader import get_gene_coords
from statsmodels.stats.contingency_tables import Table2x2



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


def odds_ratio(y_arr, label_arr, return_table=False, smoothed=False, epsilon=1e-6, flag_zero=False):

    # (tp * tn) / (fp * fn)
    
    tp = np.sum((label_arr == 1) & y_arr)
    fp = np.sum((label_arr == 0) & y_arr)
    fn = np.sum((label_arr == 1) & ~y_arr)
    tn = np.sum((label_arr == 0) & ~y_arr)
    table = [[tp, fp], [fn, tn]]

    stat, pval = sp.stats.fisher_exact(table)
    
    if smoothed:
        # haldane-anscombe correction
        tp_s = tp + epsilon
        fp_s = fp + epsilon
        fn_s = fn + epsilon
        tn_s = tn + epsilon
        stat = (tp_s * tn_s) / (fp_s * fn_s)

    if flag_zero:
        if 0 in np.array(table):
            stat = np.nan

    if return_table:
        return table, stat, pval

    return stat, pval


def enrichment(scores, pvals, top_n, treat_missing='exclude', smoothed=False, epsilon=1e-6):

    ''' ( sum(scores of top nlinks) / top nlinks ) / ( sum(scores of all links) / all links )

    Parameters
    ----------
    scores : np.array
        Array of scores from ground truth data.
    pvals : np.array
        Array of p-values from method.
    top_n : int
        Number of top smallest p-values to consider true.
    treat_missing :  str
        'exclude' (default) -> compute numerator/denominator only on observed scores;
        'zero' -> treat missing scores as 0.

    Returns
    -------
    enrichment : float
    
    '''
    
    if top_n <= 0:
        return np.nan

    sorted_idx = np.argsort(pvals)
    take_n = min(int(top_n), len(pvals))
    top_idx = sorted_idx[:take_n]

    if treat_missing == 'zero':
        scores_filled = np.where(np.isfinite(scores), scores, 0.0)
        # numerator averaged over exactly top_n slots (missing => 0)
        numerator = np.nanmean(scores_filled[top_idx]) if take_n > 0 else np.nan
        denominator = np.nanmean(scores_filled) if scores_filled.size > 0 else np.nan
        if not np.isfinite(denominator) or denominator == 0:
            return np.nan
        return numerator / denominator

    # else 'exclude' behaviour: compute using observed (finite) scores only
    observed_mask = np.isfinite(scores)
    obs_idx = np.nonzero(observed_mask)[0]
    if obs_idx.size == 0:
        return np.nan

    # restrict top indices to observed rows
    top_obs_idx = [i for i in top_idx if observed_mask[i]]
    if len(top_obs_idx) == 0:
        # no observed rows among top_n -> no signal in top
        return 0.0

    mean_top = float(np.nanmean(scores[top_obs_idx]))
    mean_all = float(np.nanmean(scores[obs_idx]))
    if not np.isfinite(mean_all) or mean_all == 0:
        return np.nan
    return mean_top / mean_all


def contingency(link_list_sig, link_list_all, link_list_true):
    """Score links based on the gold/silver standard set.
    
    Parameters
    ----------
    link_list_sig : list
        List of peak-gene pairs w/ q<threshold. List of tuples: [(peak1,gene1), (peak2, gene2), ...]
    link_list_all : list
        List of all peak-gene pairs considered. List of tuples: [(peak1,gene1), (peak2, gene2), ...]
    link_list_true : list
        List of true peak-gene pairs based on a ground truth. List of tuples: [(peak1,gene1), (peak2, gene2), ...]

    Returns
    -------
    """

    set_sig = set(link_list_sig)
    set_all = set(link_list_all)
    set_true = set(link_list_true)

    # Enrichment w/o p-value
    a = len(set_sig & set_true) # Sig. & true
    b = len(set_sig) # Sig.
    c = len(set_all & set_true) # All & true
    d = len(set_all) # All

    enrich = (a / b) / (c / d)

    # Odd ratio w/ everything
    pvalue, oddsratio, or_ub, or_lb = test_overlap(
        link_list_sig,
        [x for x in link_list_true if x in set_all],
        link_list_all,
    )
    return enrich, pvalue, oddsratio, or_ub, or_lb


def odds_ratio_with_ci(y_bool, label_arr, ci=0.95, smoothed=True, correction=0.5, fn_cap=None):
    """
    y_bool: boolean predictions (e.g., pval <= alpha)
    label_arr: 0/1 ground truth labels
    Returns: (or_stat, fisher_p, ci_low, ci_high, table)
    """
    y_bool = np.asarray(y_bool, dtype=bool)
    label_arr = np.asarray(label_arr)

    tp = np.sum((label_arr == 1) & y_bool)
    fp = np.sum((label_arr == 0) & y_bool)
    fn = np.sum((label_arr == 1) & ~y_bool)
    tn = np.sum((label_arr == 0) & ~y_bool)

    table = np.array([[tp, fp],
                      [fn, tn]], dtype=float)

    # Fisher exact on raw integer counts
    fisher_or, fisher_p = sp.stats.fisher_exact([[tp, fp], [fn, tn]])

    if fn_cap is not None and fn < fn_cap:
        return np.nan, fisher_p, np.nan, np.nan, table

    # CI: Table2x2 can fail / be inf with zeros; use Haldane-Anscombe if smoothed
    table_for_ci = table.copy()
    if smoothed:
        table_for_ci = table_for_ci + correction

    try:
        t22 = Table2x2(table_for_ci,shift_zeros=smoothed)
        or_stat = t22.oddsratio
        alpha = 1.0 - ci
        ci_low, ci_high = t22.oddsratio_confint(alpha=alpha)
    except Exception:
        # Fallback: keep fisher OR but no CI
        or_stat = fisher_or
        ci_low, ci_high = np.nan, np.nan

    return or_stat, fisher_p, ci_low, ci_high, table


def auprc_enrichment(y_true, y_scores):
    """Enrichment with based on label.
    
    Parameters
    ----------
    y_true: np.array, bool
        Array of true labels.
    y_scores : np.array
        Array of scores.

    Returns
    -------
    auprc, precision, recall, enrichment
    """

    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    n_links = len(y_true)
    n_relevant = int(np.sum(y_true))
    if n_links == 0 or n_relevant == 0:
        return np.nan, None, None, None
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_scores, pos_label=1)
    enrichment = precision * (n_links / n_relevant)
    try:
        auprc = metrics.auc(recall, enrichment)
    except Exception:
        auprc = np.nan
    return auprc, precision, recall, enrichment


def stratified_bootstrap_auprc(y_true, y_scores, n_boot=1000, seed=0):
    ''' Bootstrap AUPRC that resamples positives & negatives separately.
    '''
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true==1)[0]
    neg_idx = np.where(y_true==0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    boot_vals = []
    for _ in range(n_boot):
        pos_sample = rng.choice(pos_idx, size=n_pos, replace=True) if n_pos>0 else np.array([],dtype=int)
        neg_sample = rng.choice(neg_idx, size=n_neg, replace=True) if n_neg>0 else np.array([],dtype=int)
        idx = np.concatenate([pos_sample, neg_sample])
        yb = y_true[idx]; sb = y_scores[idx]
        if yb.sum()==0 or yb.sum()==len(yb):
            continue
        val, _, _ = auprc_enrichment(yb, sb)
        if np.isfinite(val):
            boot_vals.append(val)
    return np.array(boot_vals)


def pgb_enrichment_recall(df, col, min_recall, max_recall, ascending=True, 
                          gold_col='label', full_info=False, extrapolate=False):
    '''Adapted from Dorans et al. NG 2025.
    
    Supports both binary labels (0/1) and continuous scores (0.0-1.0) in gold_col.
    For continuous scores: a value of 0.5 contributes 0.5 to cumulative gold. 
    
    Changes from original:
    - Added total_gold and total_length calculation before filtering (handles NaN)
    - Modified "linked" calculation to check . notna() in addition to > 0
    - Modified "gold_cum" to only count gold items that are linked
    - Fixed recall calculation to use total_gold instead of tmp["gold_cum"].max()
    - Fixed enrich_denom to use total_gold/total_length instead of max/len of filtered
    - Modified unscored identification to check .notna() & (col > 0)
    - Fixed extrapolation:  new_enrichment uses (i+1) instead of i
    '''
    tmp = df[[col, gold_col]].copy()
    total_gold = tmp[gold_col].sum()
    total_length = len(tmp)
    
    tmp = tmp.sort_values(by=col, ascending=ascending).reset_index(drop=True)
    
    # check .notna() in addition to > 0
    tmp["linked"] = 1 * (tmp[col].notna() & (tmp[col] > 0))
    tmp["linked_cum"] = tmp["linked"].cumsum()
    tmp["gold_cum"] = (tmp[gold_col] * tmp["linked"]).cumsum()
    tmp["recall"] = tmp["gold_cum"] / total_gold 
    enrich_denom = total_gold / total_length
    tmp["enrichment"] = (tmp["gold_cum"].divide(tmp["linked_cum"])) / enrich_denom
    
    # identify unscored as NaN or <= 0 using .notna() check
    unscored = tmp[~(tmp[col].notna() & (tmp[col] > 0))].reset_index(drop=True)
    tmp = tmp[tmp[col].notna() & (tmp[col] > 0)][["recall", "enrichment"]]
    
    if extrapolate and len(unscored) > 0 and unscored[gold_col].sum() > 0:
        last_point = tmp.iloc[-1]
        last_recall, last_enrichment = last_point["recall"], last_point["enrichment"]
        
        num_new_points = int(unscored[gold_col].sum())
        recall_increment = (1 - last_recall) / num_new_points
        enrichment_increment = (last_enrichment - 1) / num_new_points
        new_recall = [last_recall + (recall_increment * (i + 1)) for i in range(num_new_points)]
        new_enrichment = [last_enrichment - (enrichment_increment * (i + 1)) for i in range(num_new_points)]
        
        extrapolated_er = pd.DataFrame({"recall": new_recall, "enrichment": new_enrichment})
        tmp = pd.concat([tmp, extrapolated_er], ignore_index=True)
    
    tmp = tmp[(tmp["recall"] <= max_recall) & (tmp["recall"] >= min_recall)]
    
    if full_info:
        return tmp.drop_duplicates(subset=["recall"], keep="first")
    else:
        return tmp[["recall", "enrichment"]].drop_duplicates(subset=["recall"], keep="first")
    

def pgb_auerc(df, col, min_recall = 0.0, max_recall = 1.0, ascending = True, gold_col='label', extrapolate = False, weighted = False):
    ''' Adapted from Dorans et al. NG 2025.
    '''
    er = pgb_enrichment_recall(df, col, min_recall, max_recall, ascending = ascending, extrapolate = extrapolate, gold_col=gold_col)
    if weighted == True:
        try:
            return np.average(er["enrichment"], weights = 1 - er["recall"])
        except:
            return np.nan
    else:
        return np.average(er["enrichment"])


def boostrap_pgb_auerc(df, col, ci, n_bs = 1000, method='basic', **auerc_kwargs):
    estimate = pgb_auerc(df, col, **auerc_kwargs)
    boots = np.empty(n_bs)
    N = len(df)
    for i in range(n_bs):
        idx = np.random.randint(0, N, size=N)
        sample = df.iloc[idx].reset_index(drop=True)
        boots[i] = pgb_auerc(sample, col, **auerc_kwargs)
    alpha = 1.0 - ci
    if method == "percentile":
        lower = np.nanquantile(boots, alpha / 2.0)
        upper = np.nanquantile(boots, 1.0 - alpha / 2.0)
    elif method == "basic":
        q_low = np.nanquantile(boots, alpha / 2.0)
        q_high = np.nanquantile(boots, 1.0 - alpha / 2.0)
        lower = 2.0 * estimate - q_high
        upper = 2.0 * estimate - q_low
    out = {"estimate": estimate, "ci_lower": lower, "ci_upper": upper, "bootstraps": boots}
    return out


def dedup_df(
    work_df,
    key_cols, # list/tuple of columns to group by (e.g. ["peak_id", "gene_id"])
    gold_col="label",
    score_cols=None,
    handle_dup="any",
    tie="zero",
):
    assert key_cols is not None and len(key_cols) > 0, "Provide key_cols for grouping."

    def any_positive(x):
        return float((x.astype(float) > 0).any())

    def consensus_majority(x):
        x = x.astype(float)
        pos = (x > 0).sum()
        neg = len(x) - pos
        if pos > neg:
            return 1.0
        elif pos < neg:
            return 0.0
        else:
            return {"zero": 0.0, "half": 0.5, "prefer_positive": 1.0, "prefer_negative": 0.0}[tie]

    def proportion_positive(x):
        x = x.astype(float)
        return float((x > 0).mean())

    if handle_dup == "any":
        label_agg = any_positive
    elif handle_dup == "consensus":
        label_agg = consensus_majority
    elif handle_dup == "proportion":
        label_agg = proportion_positive
    else:
        raise ValueError("handle_dup must be one of: 'any', 'consensus', 'proportion'")

    agg_dic = {gold_col: label_agg}
    if score_cols:
        for sc in score_cols:
            agg_dic[sc] = "min"

    return work_df.groupby(list(key_cols), as_index=False).agg(agg_dic)


def compute_bootstrap_table(
    all_df,
    methods,
    gold_col="label",
    reference_method="CTAR",
    n_bootstrap=1000,
    ci=0.95,
    fillna=True,
    random_state=None,
    handle_dup=None,
    dup_key_cols=None,
    tie='zero',
    or_alpha=0.05,
    or_smoothed=True,
    or_correction=0.5,
    or_fn_cap=None,
    or_multiple_testing=True,
    **auerc_kwargs
):
    """
    Paired bootstrap across methods: one resample index per replicate, compute AUERC (and OR) for all methods,
    then form paired differences vs reference on the same replicate.

    Returns DataFrame with columns:
      method,
      estimate, ci_lower, ci_upper,
      diff_estimate, diff_ci_lower, diff_ci_upper, p_value,
      odds_ratio, or_ci_lower, or_ci_upper, fisher_p,
      n_bootstrap
    """
    rng = np.random.default_rng(random_state)

    work_df = all_df.copy()
    for method in methods:
        work_df[method] = work_df[method].clip(1e-100)
        if fillna:
            work_df[method] = work_df[method].fillna(1.0)

    if handle_dup:
        print("Deduplicating..")
        work_df = dedup_df(
            work_df,
            key_cols=dup_key_cols,
            handle_dup=handle_dup,
            tie=tie,
            gold_col=gold_col,
            score_cols=methods,
        )

    if reference_method not in methods:
        raise ValueError(f"reference_method {reference_method} not in provided methods.")

    alpha = 1.0 - ci
    N = len(work_df)

    # Point estimates on full data (AUERC)
    auerc_est = {m: pgb_auerc(work_df, m, gold_col=gold_col, **auerc_kwargs) for m in methods}

    # Point estimates on full data (OR)
    or_est = {}
    or_fisher_p = {}
    or_ci = {}
    for m in methods:
        p = work_df[m].to_numpy()
        if or_multiple_testing:
            pval_corrected = np.empty(p.shape)
            pval_corrected.fill(np.nan)
            mask = np.isfinite(p)
            pval_corrected[mask] = multipletests(p[mask],method='fdr_bh')[1]
        else:
            pval_corrected = p
        y_bool = (pval_corrected <= or_alpha)
        print(m, y_bool.sum())
        or_stat, fisher_p, ci_low, ci_high, _ = odds_ratio_with_ci(
            y_bool=y_bool,
            label_arr=work_df[gold_col].to_numpy(),
            ci=ci,
            smoothed=or_smoothed,
            correction=or_correction,
            fn_cap=or_fn_cap,
        )
        or_est[m] = or_stat
        or_fisher_p[m] = fisher_p
        or_ci[m] = (ci_low, ci_high)

    # Bootstrap storage
    boot_auerc = {m: np.empty(n_bootstrap, dtype=float) for m in methods}
    boot_or = {m: np.empty(n_bootstrap, dtype=float) for m in methods}

    for b in range(n_bootstrap):
        idx = rng.integers(0, N, size=N)
        sample = work_df.iloc[idx].reset_index(drop=True)

        labels = sample[gold_col].to_numpy()
        for m in methods:
            # AUERC
            boot_auerc[m][b] = pgb_auerc(sample, m, gold_col=gold_col, **auerc_kwargs)
            # OR
            yb = (sample[m].to_numpy() <= or_alpha)
            or_stat_b, _, _, _, _ = odds_ratio_with_ci(
                y_bool=yb,
                label_arr=labels,
                ci=ci,
                smoothed=or_smoothed,
                correction=or_correction,
                fn_cap=or_fn_cap,
            )
            boot_or[m][b] = or_stat_b

    ref_auerc_boot = boot_auerc[reference_method]
    ref_auerc_est = auerc_est[reference_method]

    rows = []
    for m in tqdm(methods):
        # AUERC
        mboot = boot_auerc[m]
        pair_mask = np.isfinite(mboot) & np.isfinite(ref_auerc_boot)
        mboot_p = mboot[pair_mask]
        refboot_p = ref_auerc_boot[pair_mask]

        row = {
            "method": m,
            "estimate": auerc_est[m],
            "ci_lower": np.nanquantile(mboot[np.isfinite(mboot)], alpha / 2.0) if np.isfinite(mboot).any() else np.nan,
            "ci_upper": np.nanquantile(mboot[np.isfinite(mboot)], 1.0 - alpha / 2.0) if np.isfinite(mboot).any() else np.nan,
            "diff_estimate": ref_auerc_est - auerc_est[m],
            "diff_ci_lower": np.nan,
            "diff_ci_upper": np.nan,
            "p_value": np.nan,
            # OR
            "odds_ratio": or_est[m],
            "or_ci_lower": or_ci[m][0],
            "or_ci_upper": or_ci[m][1],
            "fisher_p": or_fisher_p[m],

            "n_bootstrap": n_bootstrap,
        }

        if mboot_p.size > 0 and np.isfinite(auerc_est[m]) and np.isfinite(ref_auerc_est):
            diffs = refboot_p - mboot_p
            row["diff_ci_lower"] = np.nanquantile(diffs, alpha / 2.0)
            row["diff_ci_upper"] = np.nanquantile(diffs, 1.0 - alpha / 2.0)

            # two-sided sign test style p-value
            n = diffs.size
            n_lt = np.sum(diffs < 0)
            n_gt = np.sum(diffs > 0)
            n_eq = n - n_lt - n_gt
            p_lower = (n_lt + 0.5 * n_eq) / n
            p_upper = (n_gt + 0.5 * n_eq) / n
            pval = 2.0 * min(p_lower, p_upper)
            row["p_value"] = min(1.0, max(0.0, pval))

        # OR boostrap CI
        ob = boot_or[m]
        ob = ob[np.isfinite(ob)]
        row["or_ci_lower"] = np.nanquantile(ob, alpha / 2.0) if ob.size else np.nan
        row["or_ci_upper"] = np.nanquantile(ob, 1.0 - alpha / 2.0) if ob.size else np.nan

        rows.append(row)

    return pd.DataFrame(rows)

    
def test_overlap(list1, list2, list_background):
    """
    Test overlap of two gene sets using Fisher's exact test
    """

    set1 = set(list1)
    set2 = set(list2)
    set_background = set(list_background)

    n1 = len(set1)
    n2 = len(set2)
    n_overlap = len(set1 & set2)
    n_other = len(set_background - set1 - set2)

    oddsratio, pvalue = sp.stats.fisher_exact(
        [[n_other, n1 - n_overlap], [n2 - n_overlap, n_overlap]]
    )

    if (
        (n_overlap == 0)
        | (n_other == 0)
        | ((n2 - n_overlap) == 0)
        | ((n1 - n_overlap) == 0)
    ):
        return pvalue, oddsratio, 0, 0
    else:
        se_log_or = np.sqrt(
            1 / (n1 - n_overlap) + 1 / (n2 - n_overlap) + 1 / n_overlap + 1 / n_other
        )
        or_ub = np.exp(np.log(oddsratio) + 1.96 * se_log_or)
        or_lb = np.exp(np.log(oddsratio) - 1.96 * se_log_or)
        return pvalue, oddsratio, or_ub, or_lb


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
