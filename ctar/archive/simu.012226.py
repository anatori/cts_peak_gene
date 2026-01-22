import numpy as np
import pandas as pd 
import scipy as sp
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from sklearn import metrics
import os
import warnings
from typing import Optional, Tuple, Dict, Any
import pybedtools
import seaborn as sns
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


def odds_ratio(y_arr, label_arr, return_table=False, smoothed=False, epsilon=1e-6, fn_cap=None):

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

    if fn_cap and (fn < fn_cap):
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
    auprc, recall, enrichment
    """

    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    n_links = len(y_true)
    n_relevant = int(np.sum(y_true))
    if n_links == 0 or n_relevant == 0:
        return np.nan, None, None
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_scores, pos_label=1)
    enrichment = precision * (n_links / n_relevant)
    try:
        auprc = metrics.auc(recall, enrichment)
    except Exception:
        auprc = np.nan
    return auprc, recall, enrichment


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
    For continuous scores:  a value of 0.5 contributes 0.5 to cumulative gold. 
    
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
    
    # Store totals from original df BEFORE filtering (to handle NaN correctly)
    # Works with continuous gold_col values (sums the total "gold mass")
    total_gold = tmp[gold_col].sum()
    total_length = len(tmp)
    
    tmp = tmp.sort_values(by=col, ascending=ascending).reset_index(drop=True)
    
    # Check .notna() in addition to > 0
    tmp["linked"] = 1 * (tmp[col].notna() & (tmp[col] > 0))
    tmp["linked_cum"] = tmp["linked"].cumsum()
    
    # Only count gold in linked items (multiply by "linked" mask)
    # For continuous scores:  this accumulates the weighted "gold mass"
    tmp["gold_cum"] = (tmp[gold_col] * tmp["linked"]).cumsum()
    
    # Use total_gold instead of tmp["gold_cum"].max()
    # Recall = fraction of total "gold mass" recovered
    tmp["recall"] = tmp["gold_cum"] / total_gold
    
    # Use total_gold / total_length instead of tmp["gold_cum"].max() / len(tmp)
    # Enrichment denom = average "gold density" across all items
    enrich_denom = total_gold / total_length
    tmp["enrichment"] = (tmp["gold_cum"].divide(tmp["linked_cum"])) / enrich_denom
    
    # Identify unscored as NaN or <= 0 using .notna() check
    unscored = tmp[~(tmp[col].notna() & (tmp[col] > 0))].reset_index(drop=True)
    tmp = tmp[tmp[col].notna() & (tmp[col] > 0)][["recall", "enrichment"]]
    
    if extrapolate and len(unscored) > 0 and unscored[gold_col].sum() > 0:
        last_point = tmp.iloc[-1]
        last_recall, last_enrichment = last_point["recall"], last_point["enrichment"]
        
        # Remaining "gold mass" in unscored items
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
    effective_score=False,
    effective_alpha=1.0, 
    handle_dup=None,
    dup_key_cols=None,
    tie='zero',
    **auerc_kwargs
):
    """
    Paired bootstrap across methods: one resample index per replicate, compute AUERC for all methods,
    then form differences vs reference on the same replicate.

    Returns DataFrame with columns:
      method, estimate, ci_lower, ci_upper,
      diff_estimate, diff_ci_lower, diff_ci_upper, p_value, n_bootstrap
    """
    rng = np.random.default_rng(random_state)

    raw_df = all_df.copy()

    # Preprocess once
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

    # Point estimates on full data
    estimates = {m: pgb_auerc(work_df, m, gold_col=gold_col, **auerc_kwargs) for m in methods}
    # Storage for bootstrap statistics per method
    boot_stats = {m: np.empty(n_bootstrap, dtype=float) for m in methods}

    N = len(work_df)
    for b in range(n_bootstrap):
        idx = rng.integers(0, N, size=N)  # same indices for all methods
        sample = work_df.iloc[idx].reset_index(drop=True)
        for m in methods:
            boot_stats[m][b] = pgb_auerc(sample, m, gold_col=gold_col, **auerc_kwargs)
    alpha = 1.0 - ci

    # Reference bootstrap and estimate
    if reference_method not in methods:
        raise ValueError(f"reference_method {reference_method} not in provided methods.")
    ref_boot = boot_stats[reference_method]
    ref_est = estimates[reference_method]

    rows = []
    for m in tqdm(methods):
        boots = boot_stats[m]
        boots = boots[np.isfinite(boots)]
        ref_boot_clean = ref_boot[np.isfinite(ref_boot)]
        nmin = min(len(boots), len(ref_boot_clean))

        # Base row (existing AUERC)
        row = {
            "method": m,
            "estimate": estimates[m],
            "ci_lower": np.nanquantile(boots, alpha / 2.0) if boots.size else np.nan,
            "ci_upper": np.nanquantile(boots, 1.0 - alpha / 2.0) if boots.size else np.nan,
            "diff_estimate": ref_est - estimates[m],
            "diff_ci_lower": np.nan,
            "diff_ci_upper": np.nan,
            "p_value": np.nan,
            "n_bootstrap": n_bootstrap,
        }

        if nmin > 0 and np.isfinite(estimates[m]) and np.isfinite(ref_est):
            diffs = ref_boot_clean[:nmin] - boots[:nmin]
            row["diff_ci_lower"] = np.nanquantile(diffs, alpha / 2.0)
            row["diff_ci_upper"] = np.nanquantile(diffs, 1.0 - alpha / 2.0)

            n = diffs.size
            n_lt = np.sum(diffs < 0)
            n_gt = np.sum(diffs > 0)
            n_eq = n - n_lt - n_gt
            p_lower = (n_lt + 0.5 * n_eq) / n
            p_upper = (n_gt + 0.5 * n_eq) / n
            pval = 2.0 * min(p_lower, p_upper)
            row["p_value"] = min(1.0, max(0.0, pval))

        rows.append(row)

    return pd.DataFrame(rows)


def scent_enrichment_setup(annotations_df,
    genomes_file,
    causal_variants_bed,
    ):
    ''' Setup dataframes for scent_enrichment.
    Parameters
    ----------
    genomes_file : BED file path containing common SNPs in hg38. Must be tab delimited.
    causal_variants_bed : pybedtools object containing causal variants (PIP < 0.2 in SCENT paper).

    Returns
    -------
    causal_var_in_annot, common_var_in_annot, annotations_df, genomes_df, windows_df

    '''

    # create annotations bed
    annotations_df[['peak','gene']] = annotations_df.reset_index()['index'].str.split(';',expand=True).values
    annotations_df[['chr','start','end']] = annotations_df.peak.str.split(':|-',expand=True).values
    annotations_df[['start','end']] = annotations_df[['start','end']].astype(int)
    annotations_df = annotations_df.reset_index(names='idx')
    annotations_df.index = annotations_df['idx']
    annotations_bed = pybedtools.BedTool.from_dataframe(annotations_df[['chr','start','end','idx']])
    
    # load common snps
    genomes_df = pd.read_csv(genomes_file,header=None, sep='\t')
    genomes_df.columns = ['CHROM','START','END','rsid']
    common_variants_bed = pybedtools.BedTool.from_dataframe(genomes_df)
    
    # precompute common variants in annotations
    common_var_in_annot = common_variants_bed.intersect(annotations_bed, wa=True, wb=True)
    common_var_in_annot = common_var_in_annot.to_dataframe(names=['CHROM', 'POS', 'POS2', 'RSID', 'CHROM_ANN', 'START_ANN', 'END_ANN', 'GENE_ANN'])
    common_var_in_annot[['PEAK_ANN','GENE_ANN']] = common_var_in_annot['GENE_ANN'].str.split(';',expand=True).values
    
    windows_df = get_gene_coords(annotations_df,gene_col='gene',gene_id_type='ensembl_gene_id')
    windows_df.dropna(subset=['chr', 'start', 'end'],inplace=True)
    windows_df['CIS_START'] = windows_df['start'].astype(int) - 500000
    windows_df['CIS_START'] = windows_df['CIS_START'].clip(lower=0)  # Ensure CIS_START is not negative
    windows_df['CIS_END'] = windows_df['end'].astype(int) + 500000
    windowsdf_bed = pybedtools.BedTool.from_dataframe(
        windows_df[['chr', 'CIS_START', 'CIS_END', 'gene']]
    )
    
    # precompute causal variants in annotations
    causal_var_in_annot = causal_variants_bed.intersect(annotations_bed, wa=True, wb=True)
    causal_var_in_annot = causal_var_in_annot.to_dataframe(names=['CHROM', 'POS', 'POS2', 'PROB', 'CHROM_ANN', 'START_ANN', 'END_ANN', 'GENE_ANN'])
    causal_var_in_annot[['eQTL','eQTL_GENE','eQTL_PROB','eQTL_TISSUE']] = causal_var_in_annot['PROB'].str.split(';',expand=True).values
    causal_var_in_annot['eQTL_PROB'] = causal_var_in_annot['eQTL_PROB'].astype(float)
    causal_var_in_annot.index = causal_var_in_annot['GENE_ANN']
    causal_var_in_annot.index.name = ''
    causal_var_in_annot[['PEAK_ANN','GENE_ANN']] = causal_var_in_annot['GENE_ANN'].str.split(';',expand=True).values
    
    return causal_var_in_annot, common_var_in_annot, annotations_df, genomes_df, windows_df


def scent_enrichment(causal_var_in_annot,
                    common_var_in_annot,
                    annotations_df,
                    ref_df,
                    genomes_df,
                    windows_df,
                    method,
                    threshold,
                    flag_debug=False,
    ):
    """
    Vectorized replacement for the per-gene loop.

    Parameters
    ----------
    causal_var_in_annot: DataFrame with columns ['eQTL_GENE','GENE_ANN','eQTL_PROB', method, ...]
    common_var_in_annot: DataFrame with column 'GENE_ANN' (1000G variant in annotation peaks)
    ref_df: DataFrame with column 'GENE' and 'Probability'
    genomes_df: DataFrame with columns ['CHROM','START','END'] for common variants (hg38)
    windows_df: DataFrame with columns ['gene', 'chr', 'CIS_START', 'CIS_END']
                  (chr should include 'chr' prefix to match genomes_df['CHROM'])
    method: column name in causal_var_in_annot to threshold (smaller is better if you use '< threshold' as in your loop)
    threshold: threshold applied as causal_var_in_annot[method] < threshold

    Returns
    -------
    overall_enrichment: scalar (mean over included genes)
    per_gene_enrichment: pandas Series indexed by gene name for included genes (same ordering as original logic)
    """
    # gene universe (match original: intersection of annotations and reference genes)
    genes_eQTL = ref_df['GENE'].astype(str).unique()
    genes_ANN  = annotations_df['gene'].astype(str).unique()
    genes = np.intersect1d(genes_eQTL, genes_ANN) # sorted unique

    # precompute pip totals by eQTL_GENE (pip_causal_var_genei)
    pip_total = ref_df.groupby('GENE')['Probability'].sum()
    # reindex to full gene list (missing -> 0)
    pip_total_arr = pip_total.reindex(genes).fillna(0.0).to_numpy(dtype=float)

    if isinstance(threshold, int) and threshold >= 1:
        # treat smaller values as better (like p-values). Select rows with smallest `method` values.
        scores = causal_var_in_annot[method].to_numpy()
        # put NaNs at the end by replacing with +inf so they are not selected
        scores_clean = np.where(np.isfinite(scores), scores, np.inf)
        # argsort and take top-N indices
        order = np.argsort(scores_clean)
        top_n = threshold
        if top_n >= len(scores_clean):
            selected_idx = order  # all
        else:
            selected_idx = order[:top_n]
        mask_method = np.zeros(len(causal_var_in_annot), dtype=bool)
        mask_method[selected_idx] = True
    else:
        # original numeric threshold behavior: select rows with method value < threshold
        mask_method = causal_var_in_annot[method] < threshold

    # precompute pip in annotated peaks where the method condition holds AND GENE matches (eQTL_GENE == GENE_ANN)
    mask_same = causal_var_in_annot['eQTL_GENE'].astype(str) == causal_var_in_annot['GENE_ANN'].astype(str)
    df_linked = causal_var_in_annot.loc[mask_method & mask_same]
    pip_in_annot = df_linked.groupby('eQTL_GENE')['eQTL_PROB'].sum()
    pip_in_annot_arr = pip_in_annot.reindex(genes).fillna(0.0).to_numpy(dtype=float)

    # precompute num_common_var_in_annot_genei from common_var_in_annot grouped by GENE_ANN
    num_common_annot = common_var_in_annot.groupby('GENE_ANN').size()
    num_common_annot_arr = num_common_annot.reindex(genes).fillna(0).to_numpy(dtype=int)

    # compute num_common_var_in_cis_genei efficiently per-chromosome using numpy.searchsorted
    genomes = genomes_df.copy()
    # assume genomes['START'] is int; if not, convert
    genomes['START'] = genomes['START'].astype(int)
    # group genomes by CHROM and build sorted arrays of STARTs
    genomes_grouped = {chrom: arr['START'].to_numpy(dtype=int) for chrom, arr in genomes.groupby('CHROM')}
    for chrom in genomes_grouped:
        genomes_grouped[chrom] = np.sort(genomes_grouped[chrom])

    # arrange windows by chrom and compute counts vectorized per-chrom
    # create arrays aligned with genes
    gene_to_index = {g: i for i, g in enumerate(genes)}
    num_common_cis_arr = np.zeros(len(genes), dtype=int)
    # filter windows_df to only genes present in our gene list
    w = windows_df[windows_df['gene'].astype(str).isin(genes)].copy()
    # standardize 'chr' column name
    if 'chr' in w.columns:
        chrom_col = 'chr'
    elif 'CHROM' in w.columns:
        chrom_col = 'CHROM'
    else:
        raise KeyError("windows_df must contain a 'chr' or 'CHROM' column")

    # ensure same 'chr' formatting: convert to string and keep 'chr' prefix if present in genome
    w[chrom_col] = w[chrom_col].astype(str)

    # group by chromosome and do vectorized searchsorted per chrom
    for chrom, grp in w.groupby(chrom_col):
        starts = genomes_grouped.get(chrom)
        if starts is None or starts.size == 0:
            # zero counts for all genes on this chrom
            for idx, row in grp.iterrows():
                gi = gene_to_index[str(row['gene'])]
                num_common_cis_arr[gi] = 0
            continue
        # vector of window starts/ends for this chrom
        cis_starts = grp['CIS_START'].to_numpy(dtype=int)
        cis_ends   = grp['CIS_END'].to_numpy(dtype=int)
        # searchsorted supports vectorized inputs
        left_idx  = np.searchsorted(starts, cis_starts, side='left')
        right_idx = np.searchsorted(starts, cis_ends, side='right')
        counts = (right_idx - left_idx).astype(int)
        # assign to array based on genes in this group
        for gene_name, cnt in zip(grp['gene'].astype(str).tolist(), counts.tolist()):
            gi = gene_to_index[gene_name]
            num_common_cis_arr[gi] = int(cnt)

    # vectorized logic implementing the same control flow as the loop:
    # - if pip_total == 0 -> enrichment = 0 (include)
    # - elif num_common_annot == 0 -> enrichment = 0 (include)
    # - elif num_common_cis == 0 -> skip (do not include)
    # - else compute enrichment = (pip_in_annot / num_common_annot) / (pip_total / num_common_cis)

    pip_total = pip_total_arr
    pip_annot = pip_in_annot_arr
    n_common_annot = num_common_annot_arr
    n_common_cis = num_common_cis_arr

    G = len(genes)
    enrichment_vals = np.empty(G, dtype=float)
    included_mask = np.zeros(G, dtype=bool)

    # case A: pip_total == 0 -> 0 and include
    mask_pip0 = (pip_total == 0.0)
    enrichment_vals[mask_pip0] = 0.0
    included_mask[mask_pip0] = True

    # case B: pip_total > 0 -> consider further
    mask_remaining = ~mask_pip0

    # case B1: n_common_annot == 0 => enrichment 0 and include
    mask_annot0 = mask_remaining & (n_common_annot == 0)
    enrichment_vals[mask_annot0] = 0.0
    included_mask[mask_annot0] = True

    # case C: have both pip_total>0 and n_common_annot>0 => potential compute
    mask_to_compute = mask_remaining & (n_common_annot > 0)

    # from these, those with n_common_cis == 0 are skipped
    mask_cis0 = mask_to_compute & (n_common_cis == 0)
    # do nothing for mask_cis0 (excluded)

    # final compute mask: pip_total>0 & n_common_annot>0 & n_common_cis>0
    mask_compute_final = mask_to_compute & (n_common_cis > 0)

    if np.any(mask_compute_final):
        numer = pip_annot[mask_compute_final] / n_common_annot[mask_compute_final].astype(float)
        denom = pip_total[mask_compute_final] / n_common_cis[mask_compute_final].astype(float)
        if flag_debug:
            print('pip_annot',pip_annot[mask_compute_final])
            print('n_common_annot',n_common_annot[mask_compute_final].astype(float))
            print('pip_total',pip_total[mask_compute_final])
            print('n_common_cis',n_common_cis[mask_compute_final].astype(float))
            print('')
            print('numer',numer)
            print('denom',denom)
        # guard against zero denom (shouldn't happen because n_common_cis >0 and pip_total>0)
        with np.errstate(divide='ignore', invalid='ignore'):
            enr = numer / denom
        # where denom==0 or nan set enr to np.nan (but the original code wouldn't hit denom==0)
        enr = np.where(np.isfinite(enr), enr, np.nan)
        enrichment_vals[mask_compute_final] = enr
        included_mask[mask_compute_final] = True

    # now compute the final list exactly like original: include entries for genes with included_mask True
    included_enrichments = enrichment_vals[included_mask]

    # overall enrichment: mean of included values
    overall_enrichment = float(np.nanmean(included_enrichments)) if included_enrichments.size > 0 else float('nan')

    # build a pd.Series of per-gene enrichment for included genes (to inspect)
    included_genes = genes[included_mask]
    per_gene_series = pd.Series(included_enrichments, index=included_genes)

    return overall_enrichment, per_gene_series

    
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


def adaptive_nlinks(label_arr, N_points=20, k_min_tp=5, n_lower=None, n_upper=None):
    """
    Adaptively determine top N links such that we have at least k positives at our first evaluation
    point.

    Interpretation:
    - A value is counted as a positive only if (value == True).
    - NaNs and False are treated as non-positives for the numerator.
    - The denominator N is the full length of label_arr (so NaNs are counted in N).
    - Returns a sorted array of nlink sizes (integers). Returns empty array if there are no positives.
    """
    arr = np.asarray(label_arr, dtype=object)  # preserve NaNs and booleans uniformly
    N = arr.size
    if N == 0:
        return np.array([])

    # Count strict True values as positives. Everything else (False / None / np.nan / other) -> non-positive.
    # Use equality to True which is robust for object/boolean arrays.
    mask_true = (arr == True)
    positives = int(np.sum(mask_true))

    # p = positives / total N (NaNs contribute to denominator)
    if positives == 0:
        return np.array([])

    p = positives / float(N)

    # compute n_min / n_max same as before but using p and total N
    n_min = max(1, int(np.ceil(k_min_tp / p)))  # ensure we expect at least k positives
    n_max = N - 1

    if n_lower is not None:
        if n_lower == 'pos': n_lower = positives
        n_min = min(n_min, n_lower)
    if n_upper is not None:
        n_max = min(n_max, n_upper)
    if n_min >= n_max:
        return np.array([min(n_min, n_max)])

    # hybrid: near n_min linear small steps, then logspace to n_max
    small_linear = np.arange(n_min, min(n_min + 10, n_max + 1))
    start = max(n_min, 10)
    log_spaced = np.unique(np.round(np.logspace(np.log10(start), np.log10(n_max), N_points)).astype(int))
    nlinks = np.unique(np.concatenate([small_linear, log_spaced]))
    nlinks = nlinks[(nlinks > 0) & (nlinks <= n_max)]

    return np.sort(nlinks)


def analyze_odds_ratio_bootstrap(eval_df, nlinks_ls=None, methods=None, n_bs_samples=1000):
    """
    Perform odds ratio analysis with bootstrap confidence intervals
    for multiple p-value methods in a DataFrame.

    Parameters
    ----------
    eval_df: pd.DataFrame
      DataFrame containing 'label' column and p-value columns.
    nlinks_ls: list of int, optional
      List of numbers of links to consider. Default: [500, 1000, 1500, 2000, 2500]
    n_bs_samples: int, optional
      Number of bootstrap samples to use. Default: 1000

    Returns
    ----------
    odds_df, pval_df, lower_ci_df, upper_ci_df, std_ci_df, bs_dic: pd.DataFrame
      DataFrames containing the computed statistics indexed by `nlinks_ls`
      and columns as methods.

    TODO:
    - flag_adaptive_nlinks option
    """

    if nlinks_ls is None:
        nlinks_ls = [500, 1000, 1500, 2000, 2500]

    if methods is None:
        methods = [s for s in eval_df.columns if 'pval' in s]

    limit_dic = {
        method: (eval_df[method] == eval_df[method].min()).sum()
        for method in methods if 'mc' in method
    }

    bs_dic = {n: {m: [] for m in methods} for n in nlinks_ls}

    results_shape = (len(nlinks_ls), len(methods))
    odds_arr = np.full(results_shape, np.nan)
    pval_arr = np.full(results_shape, np.nan)
    lower_ci_arr = np.full(results_shape, np.nan)
    upper_ci_arr = np.full(results_shape, np.nan)
    std_ci_arr = np.full(results_shape, np.nan)

    bootstrap_idx = np.random.randint(0, len(eval_df), size=(n_bs_samples, len(eval_df)))
    label_arr = eval_df['label'].values

    for mi, method in enumerate(tqdm(methods)):
        pval_arr_col = eval_df[method].values
        sorted_idx = np.argsort(pval_arr_col)
        notna_mask = pd.notna(label_arr) & np.isfinite(pval_arr_col)

        for ni, nlinks in enumerate(nlinks_ls):
            y_score = np.zeros(len(eval_df), dtype=bool)
            y_score[sorted_idx[:nlinks]] = True

            stat, pval = odds_ratio(y_score[notna_mask], label_arr[notna_mask])
            odds_arr[ni, mi] = stat
            pval_arr[ni, mi] = pval

            if method in limit_dic and nlinks < limit_dic[method]:
                odds_arr[ni, mi] = np.nan
                pval_arr[ni, mi] = np.nan
                continue

            bs_stats = np.zeros(n_bs_samples)
            for i in range(n_bs_samples):
                bs_idx = bootstrap_idx[i]
                bs_y_score = y_score[bs_idx]
                bs_label = label_arr[bs_idx]

                stat, _ = odds_ratio(bs_y_score, bs_label)
                bs_stats[i] = stat

            bs_dic[nlinks][method] = bs_stats
            lower_ci_arr[ni, mi], upper_ci_arr[ni, mi] = np.percentile(bs_stats, [2.5, 97.5])
            std_ci_arr[ni, mi] = np.std(bs_stats)

    odds_df = pd.DataFrame(odds_arr, index=nlinks_ls, columns=methods)
    pval_df = pd.DataFrame(pval_arr, index=nlinks_ls, columns=methods)
    lower_ci_df = pd.DataFrame(lower_ci_arr, index=nlinks_ls, columns=methods)
    upper_ci_df = pd.DataFrame(upper_ci_arr, index=nlinks_ls, columns=methods)
    std_ci_df = pd.DataFrame(std_ci_arr, index=nlinks_ls, columns=methods)

    for df in [odds_df, pval_df, lower_ci_df, upper_ci_df, std_ci_df]:
        df.index.name = 'nlinks'

    return odds_df, pval_df, lower_ci_df, upper_ci_df, std_ci_df, bs_dic


def split_by_chromosome(df,col='peak',sep=':'):
    ''' Return categorical variables of chromosomes.
    '''
    return df[col].str.split(sep).str[0].astype('category').values


def analyze_odds_ratio_jacknife(eval_df, categorical_arr, label='label', methods=None, nlinks_ls=None, flag_print=False, **odds_ratio_kwargs):
    """
    Perform odds ratio analysis with jacknife confidence intervals by
    chromosome blocks for multiple p-value methods in a DataFrame.

    From Nick Patterson notes on 
    F.M.T.A. Busing, E. Meijer, and R. van der Leeden. 
    Delete-m jackknife for unequal m. Statistics and Computing, 9:3–8, 1999.

    Parameters
    ----------
    eval_df: pd.DataFrame
      DataFrame containing label and p-value columns.
    categorical_arr : np.array
      Array of values corresponding to category to split on for jacknifing (e.g. loci blocks)
    label : str
      Column name containing ground truth labels.
    nlinks_ls: list of int, optional
      List of numbers of links to consider. Default: [500, 1000, 1500, 2000, 2500]

    Returns
    ----------
    odds_df, pval_df, lower_ci_df, upper_ci_df, std_ci_df, jn_dic: pd.DataFrame
      DataFrames containing the computed statistics indexed by `nlinks_arr`
      and columns as methods.

    TODO:
    - flag_adaptive_nlinks option
    """

    if nlinks_ls is None:
        nlinks_ls = [500, 1000, 1500, 2000, 2500]
    nlinks_arr = np.sort(np.array(nlinks_ls))

    if methods is None:
        methods = [s for s in eval_df.columns if 'pval' in s]

    limit_dic = {
        method: (eval_df[method] == eval_df[method].min()).sum()
        for method in methods if 'mc' in method
    }

    jn_dic = {n: {m: [] for m in methods} for n in nlinks_arr}

    results_shape = (len(nlinks_arr), len(methods))
    odds_arr = np.full(results_shape, np.nan)
    pval_arr = np.full(results_shape, np.nan)
    lower_ci_arr = np.full(results_shape, np.nan)
    upper_ci_arr = np.full(results_shape, np.nan)
    std_ci_arr = np.full(results_shape, np.nan)

    label_arr = eval_df[label].values

    for mi, method in enumerate(tqdm(methods)):
        pval_arr_col = eval_df[method].values
        sorted_idx = np.argsort(pval_arr_col)

        for ni, nlinks in enumerate(nlinks_arr):
            y_score = np.zeros(len(eval_df), dtype=bool)
            y_score[sorted_idx[:nlinks]] = True

            tbl, odds_arr[ni, mi], pval_arr[ni, mi] = odds_ratio(y_score, label_arr, 
                return_table=True,
                **odds_ratio_kwargs
                )
            if flag_print:
                print(tbl)

            if method in limit_dic and nlinks < limit_dic[method]:
                odds_arr[ni, mi] = np.nan
                pval_arr[ni, mi] = np.nan
                continue

            blocks_arr = np.unique(categorical_arr)
            n_blocks = len(blocks_arr)
            ratio_arr = np.zeros(n_blocks)
            jn_stats = np.zeros(n_blocks)

            for bi,block in enumerate(blocks_arr):
                ind = np.argwhere(categorical_arr != block).flatten()
                jn_y_score = y_score[ind]
                jn_label = label_arr[ind]

                jn_stats[bi], _ = odds_ratio(jn_y_score, jn_label)
                ratio_arr[bi] = len(categorical_arr) / sum(categorical_arr == block)
            
            tau = ratio_arr * odds_arr[ni,mi] - (ratio_arr - 1) * jn_stats
            jn_estimates = np.sum(odds_arr[ni,mi] - jn_stats) + np.sum((1/ratio_arr) * jn_stats)

            jn_dic[nlinks][method] = jn_stats
            lower_ci_arr[ni, mi], upper_ci_arr[ni, mi] = np.percentile(jn_stats, [2.5, 97.5])
            std_ci_arr[ni, mi] = (1/n_blocks) * np.sum((tau - jn_estimates)**2 / (ratio_arr - 1))
            std_ci_arr[ni, mi] = np.sqrt(std_ci_arr[ni, mi])

    odds_df = pd.DataFrame(odds_arr, index=nlinks_arr, columns=methods)
    pval_df = pd.DataFrame(pval_arr, index=nlinks_arr, columns=methods)
    lower_ci_df = pd.DataFrame(lower_ci_arr, index=nlinks_arr, columns=methods)
    upper_ci_df = pd.DataFrame(upper_ci_arr, index=nlinks_arr, columns=methods)
    std_ci_df = pd.DataFrame(std_ci_arr, index=nlinks_arr, columns=methods)

    for df in [odds_df, pval_df, lower_ci_df, upper_ci_df, std_ci_df]:
        df.index.name = 'nlinks'

    return odds_df, pval_df, lower_ci_df, upper_ci_df, std_ci_df, jn_dic


def analyze_enrichment_bootstrap(eval_df, score_col='gt_pip', nlinks_ls=None, methods=None, n_bs_samples=1000):
    """
    Perform enrichment analysis with bootstrap confidence intervals
    for multiple p-value methods in a DataFrame.

    Parameters
    ----------
    eval_df: pd.DataFrame
      DataFrame containing 'label' column and p-value columns.
    score_col : str, optional
      Column name for column containing evaluation scores.
    nlinks_ls: list of int, optional
      List of numbers of links to consider. Default: [500, 1000, 1500, 2000, 2500]
    n_bs_samples: int, optional
      Number of bootstrap samples to use. Default: 1000

    Returns
    ----------
    odds_df, lower_ci_df, upper_ci_df, std_ci_df, bs_dic: pd.DataFrame
      DataFrames containing the computed statistics indexed by `nlinks_ls`
      and columns as methods.

    TODO:
    - flag_adaptive_nlinks option
    """

    if nlinks_ls is None:
        nlinks_ls = [500, 1000, 1500, 2000, 2500]

    if methods is None:
        methods = [s for s in eval_df.columns if 'pval' in s]

    limit_dic = {
        method: (eval_df[method] == eval_df[method].min()).sum()
        for method in methods if 'mc' in method
    }

    bs_dic = {n: {m: [] for m in methods} for n in nlinks_ls}

    results_shape = (len(nlinks_ls), len(methods))
    enrich_arr = np.full(results_shape, np.nan)
    lower_ci_arr = np.full(results_shape, np.nan)
    upper_ci_arr = np.full(results_shape, np.nan)
    std_ci_arr = np.full(results_shape, np.nan)

    bootstrap_idx = np.random.randint(0, len(eval_df), size=(n_bs_samples, len(eval_df)))
    score_arr = eval_df[score_col].values

    for mi, method in enumerate(tqdm(methods)):
        pval_arr_col = eval_df[method].values

        for ni, nlinks in enumerate(nlinks_ls):

            stat = enrichment(score_arr, pval_arr_col, nlinks)
            enrich_arr[ni, mi] = stat

            if method in limit_dic and nlinks < limit_dic[method]:
                enrich_arr[ni, mi] = np.nan
                continue

            bs_stats = np.zeros(n_bs_samples)
            for i in range(n_bs_samples):
                bs_idx = bootstrap_idx[i]
                stat = enrichment(score_arr[bs_idx], pval_arr_col[bs_idx], nlinks)
                bs_stats[i] = stat

            bs_dic[nlinks][method] = bs_stats
            lower_ci_arr[ni, mi], upper_ci_arr[ni, mi] = np.percentile(bs_stats, [2.5, 97.5])
            std_ci_arr[ni, mi] = np.std(bs_stats)

    enrich_df = pd.DataFrame(enrich_arr, index=nlinks_ls, columns=methods)
    lower_ci_df = pd.DataFrame(lower_ci_arr, index=nlinks_ls, columns=methods)
    upper_ci_df = pd.DataFrame(upper_ci_arr, index=nlinks_ls, columns=methods)
    std_ci_df = pd.DataFrame(std_ci_arr, index=nlinks_ls, columns=methods)

    for df in [enrich_df, lower_ci_df, upper_ci_df, std_ci_df]:
        df.index.name = 'nlinks'

    return enrich_df, lower_ci_df, upper_ci_df, std_ci_df, bs_dic


def analyze_enrichment_jacknife(eval_df, categorical_arr, score_col='gt_pip', methods=None, nlinks_ls=None, n_cap=None, treat_missing='exclude'):
    """
    Perform enrichment analysis with jacknife confidence intervals by
    chromosome blocks for multiple p-value methods in a DataFrame.

    Parameters
    ----------
    eval_df: pd.DataFrame
      DataFrame containing 'label' column and p-value columns.
    categorical_arr : np.array
      Array of values corresponding to category to split on for jacknifing (e.g. loci blocks)
    score_col : str, optional
      Column name for column containing evaluation scores.
    nlinks_ls: list of int, optional
      List of numbers of links to consider. Default: [500, 1000, 1500, 2000, 2500]

    Returns
    ----------
    odds_df, lower_ci_df, upper_ci_df, std_ci_df, jn_dic: pd.DataFrame
      DataFrames containing the computed statistics indexed by `nlinks_arr`
      and columns as methods.

    TODO:
    - flag_adaptive_nlinks option
    """

    if nlinks_ls is None:
        nlinks_ls = [500, 1000, 1500, 2000, 2500]
    nlinks_arr = np.sort(np.array(nlinks_ls)).astype(int)

    if methods is None:
        methods = [s for s in eval_df.columns if 'pval' in s]

    # limit dic preserved for 'mc' methods as in previous code
    limit_dic = {
        method: (eval_df[method] == eval_df[method].min()).sum()
        for method in methods if 'mc' in method
    }

    jn_dic = {n: {m: [] for m in methods} for n in nlinks_arr}
    results_shape = (len(nlinks_arr), len(methods))
    enrich_arr = np.full(results_shape, np.nan)
    lower_ci_arr = np.full(results_shape, np.nan)
    upper_ci_arr = np.full(results_shape, np.nan)
    std_ci_arr = np.full(results_shape, np.nan)

    score_arr_full = eval_df[score_col].values
    observed_mask_full = np.isfinite(score_arr_full)
    total_observed = int(np.sum(observed_mask_full))

    # optionally cap nlinks to observed rows to avoid instability
    if n_cap is None and total_observed > 0:
        # keep nlinks strictly less than total_observed (so leave at least one observed positive out)
        n_cap_eff = total_observed - 1
    else:
        n_cap_eff = n_cap

    if n_cap_eff is not None:
        nlinks_arr = nlinks_arr[nlinks_arr <= int(n_cap_eff)]

    blocks_arr = np.unique(categorical_arr)
    per_block_observed = {block: int(np.sum((categorical_arr == block) & observed_mask_full)) for block in blocks_arr}

    for mi, method in enumerate(tqdm(methods)):
        pval_arr_col = eval_df[method].values
        sorted_idx = np.argsort(pval_arr_col)  # NaNs to end

        for ni, nlinks in enumerate(nlinks_arr):
            # build top-n mask on full data
            take_n = min(nlinks, len(eval_df))
            y_score = np.zeros(len(eval_df), dtype=bool)
            y_score[sorted_idx[:take_n]] = True

            # compute enrichment on observed only (or treat_missing='zero')
            if treat_missing == 'zero':
                enrich_arr[ni, mi] = enrichment(score_arr_full, pval_arr_col, nlinks, treat_missing='zero')
            else:
                obs_idx = np.nonzero(observed_mask_full)[0]
                if obs_idx.size == 0:
                    enrich_arr[ni, mi] = np.nan
                else:
                    top_obs = np.in1d(np.argsort(pval_arr_col)[:take_n], obs_idx)
                    # reuse function logic on observed rows: compute means on observed subset
                    top_idx = np.argsort(pval_arr_col)[:take_n]
                    top_obs_idx = [i for i in top_idx if observed_mask_full[i]]
                    if len(top_obs_idx) == 0:
                        enrich_arr[ni, mi] = 0.0
                    else:
                        mean_top = float(np.nanmean(score_arr_full[top_obs_idx]))
                        mean_all = float(np.nanmean(score_arr_full[obs_idx]))
                        enrich_arr[ni, mi] = (mean_top / mean_all) if (mean_all not in [0, np.nan]) else np.nan

            if method in limit_dic and nlinks < limit_dic[method]:
                enrich_arr[ni, mi] = np.nan
                continue

            # jackknife
            n_blocks = len(blocks_arr)
            ratio_arr = np.full(n_blocks, np.nan)
            jn_stats = np.full(n_blocks, np.nan)

            for bi, block in enumerate(blocks_arr):
                m_i = per_block_observed.get(block, 0)
                if m_i == 0:
                    ratio_arr[bi] = np.nan
                    jn_stats[bi] = np.nan
                    continue

                # keep observed rows not in this block
                keep_mask = (categorical_arr != block) & observed_mask_full
                ind = np.nonzero(keep_mask)[0]
                if ind.size == 0:
                    ratio_arr[bi] = np.nan
                    jn_stats[bi] = np.nan
                    continue

                # top-n mask on full but restricted to observed & excluding block
                take_n_local = min(nlinks, len(eval_df))
                top_full_idx = np.argsort(pval_arr_col)[:take_n_local]
                top_ind_mask = np.in1d(top_full_idx, ind)
                top_ind_idx = top_full_idx[top_ind_mask]

                if treat_missing == 'zero':
                    # consider missing as zeros; but ind already excludes non-observed
                    jn_stats[bi] = enrichment(score_arr_full[ind], pval_arr_col[ind], len(top_ind_idx), treat_missing='zero')
                else:
                    if len(top_ind_idx) == 0:
                        jn_stats[bi] = 0.0
                    else:
                        mean_top_jn = float(np.nanmean(score_arr_full[top_ind_idx]))
                        mean_all_jn = float(np.nanmean(score_arr_full[ind]))
                        jn_stats[bi] = (mean_top_jn / mean_all_jn) if (mean_all_jn not in [0, np.nan]) else np.nan

                ratio_arr[bi] = total_observed / float(m_i)

            jn_dic[nlinks][method] = jn_stats.copy()

            valid_mask = np.isfinite(ratio_arr) & np.isfinite(jn_stats)
            if valid_mask.sum() == 0:
                lower_ci_arr[ni, mi] = np.nan
                upper_ci_arr[ni, mi] = np.nan
                std_ci_arr[ni, mi] = np.nan
                continue

            r_valid = ratio_arr[valid_mask]
            jn_valid = jn_stats[valid_mask]
            try:
                lower_ci_arr[ni, mi], upper_ci_arr[ni, mi] = np.percentile(jn_valid, [2.5, 97.5])
            except Exception:
                lower_ci_arr[ni, mi], upper_ci_arr[ni, mi] = np.nan, np.nan

            tau = r_valid * enrich_arr[ni, mi] - (r_valid - 1) * jn_valid
            jn_estimates = np.sum(enrich_arr[ni, mi] - jn_valid) + np.sum((1.0 / r_valid) * jn_valid)
            try:
                std_val = (1.0 / valid_mask.sum()) * np.sum((tau - jn_estimates) ** 2 / (r_valid - 1.0))
                std_ci_arr[ni, mi] = np.sqrt(std_val) if std_val >= 0 else np.nan
            except Exception:
                std_ci_arr[ni, mi] = np.nan

    enrich_df = pd.DataFrame(enrich_arr, index=nlinks_arr, columns=methods)
    lower_ci_df = pd.DataFrame(lower_ci_arr, index=nlinks_arr, columns=methods)
    upper_ci_df = pd.DataFrame(upper_ci_arr, index=nlinks_arr, columns=methods)
    std_ci_df = pd.DataFrame(std_ci_arr, index=nlinks_arr, columns=methods)

    for df in [enrich_df, lower_ci_df, upper_ci_df, std_ci_df]:
        df.index.name = 'nlinks'

    return enrich_df, lower_ci_df, upper_ci_df, std_ci_df, jn_dic


def get_top_n_mask(pvals, n):
    """Return a boolean mask for the top n smallest p-values (on full array)."""
    pvals = np.asarray(pvals)
    take_n = min(int(n), len(pvals))
    idx = np.argsort(pvals)[:take_n]
    mask = np.zeros(len(pvals), dtype=bool)
    mask[idx] = True
    return mask


def analyze_auc_midpoint_jackknife(eval_df, categorical_arr, score_col='gt_pip', methods=None, nlinks_ls=None, stat_method='enrichment', label_col=None, n_cap=None):
    """
    Compute enrichment AUC via midpoint rule using jackknife confidence intervals.
    
    Parameters
    ----------
    eval_df : pd.DataFrame
        DataFrame with p-value columns and a score column.
    categorical_arr : np.array
        Jackknife blocking array (e.g., chromosomes).
    score_col : str
        Name of the score column to use for enrichment ranking.
    nlinks_ls : list of int
        List of N link cutoffs (e.g. [500, 1000, ...]).
    stat_method : str
        'enrichment' or 'odds_ratio'
    
    Returns
    ----------
    auc_df, lower_ci_df, upper_ci_df, std_ci_df, jn_dic : pd.DataFrame, ...
        AUCs and jackknife CIs, all indexed by method.
    """

    if nlinks_ls is None:
        nlinks_ls = [500, 1000, 1500, 2000, 2500]
    nlinks_arr = np.sort(np.array(nlinks_ls)).astype(int)

    if methods is None:
        methods = [s for s in eval_df.columns if 'pval' in s]

    score_arr_full = eval_df[score_col].values
    observed_mask_full = np.isfinite(score_arr_full)
    total_observed = int(np.sum(observed_mask_full))

    # optionally cap nlinks to observed rows to avoid FN->0 instability
    if n_cap is None and total_observed > 0:
        n_cap_eff = total_observed - 1
    else:
        n_cap_eff = n_cap

    if n_cap_eff is not None:
        nlinks_arr = nlinks_arr[nlinks_arr <= int(n_cap_eff)]

    blocks_arr = np.unique(categorical_arr)
    per_block_observed = {block: int(np.sum((categorical_arr == block) & observed_mask_full)) for block in blocks_arr}
    n_blocks = len(blocks_arr)

    auc_vals = []
    lower_ci_vals = []
    upper_ci_vals = []
    std_ci_vals = []
    jn_dic = {0: {}}

    for method in tqdm(methods):
        pval_arr = eval_df[method].values
        sorted_idx_full = np.argsort(pval_arr)

        # ensure at least two nlinks to get midpoints
        nlinks_arr_eff = nlinks_arr[nlinks_arr < len(eval_df)]
        if nlinks_arr_eff.size < 2:
            auc_vals.append(np.nan)
            lower_ci_vals.append(np.nan)
            upper_ci_vals.append(np.nan)
            std_ci_vals.append(np.nan)
            jn_dic[0][method] = np.full(n_blocks, np.nan)
            continue

        midpoints = np.round((nlinks_arr_eff[:-1] + nlinks_arr_eff[1:]) / 2).astype(int)
        deltas = np.diff(nlinks_arr_eff)

        # compute stat at midpoints (stat computed on observed subset)
        stat_at_mid = []
        for n in midpoints:
            top_mask = np.zeros(len(eval_df), dtype=bool)
            take_n = min(n, len(eval_df))
            top_mask[sorted_idx_full[:take_n]] = True

            obs_idx = np.nonzero(observed_mask_full)[0]
            top_obs = top_mask[obs_idx]
            score_obs = score_arr_full[obs_idx]

            if stat_method == 'enrichment':
                if top_obs.sum() == 0 or score_obs.size == 0:
                    stat_at_mid.append(0.0)
                else:
                    mean_top = float(np.nanmean(score_obs[top_obs]))
                    mean_all = float(np.nanmean(score_obs))
                    stat_at_mid.append((mean_top / mean_all) if (mean_all not in [0, np.nan]) else np.nan)

            elif stat_method == 'odds_ratio':
                if label_col is None:
                    raise ValueError("When stat_method=='odds_ratio' you must pass label_col with binary ground truth.")
                label_arr_full = eval_df[label_col].values
                label_obs_full = label_arr_full[obs_idx]
                if np.sum(label_obs_full == True) == 0:
                    stat_at_mid.append(np.nan)
                else:
                    y_obs = top_obs
                    or_val, _ = odds_ratio(y_obs, label_obs_full)
                    stat_at_mid.append(or_val)
            else:
                raise ValueError("stat_method must be 'enrichment' or 'odds_ratio'")

        stat_at_mid = np.array(stat_at_mid, dtype=float)
        stat_at_mid[np.isinf(stat_at_mid)] = 0.0
        auc = np.sum(stat_at_mid * deltas)
        auc_vals.append(auc)

        # jackknife
        jn_estimates = np.zeros(n_blocks)
        ratio_arr = np.zeros(n_blocks)

        for bi, block in enumerate(blocks_arr):
            m_i = per_block_observed.get(block, 0)
            if m_i == 0:
                jn_estimates[bi] = np.nan
                ratio_arr[bi] = np.nan
                continue

            # remove block: keep observed rows not in this block
            keep_mask = (categorical_arr != block) & observed_mask_full
            ind = np.nonzero(keep_mask)[0]

            score_jn = score_arr_full[ind]
            pval_jn = pval_arr[ind]

            if len(score_jn) == 0 or len(pval_jn) == 0:
                jn_estimates[bi] = np.nan
                ratio_arr[bi] = np.nan
                continue

            sorted_idx_jn = np.argsort(pval_jn)
            stat_jn = []
            for n in midpoints:
                take_n_jn = min(n, len(pval_jn))
                top_mask_jn = np.zeros(len(pval_jn), dtype=bool)
                top_mask_jn[sorted_idx_jn[:take_n_jn]] = True

                if stat_method == 'enrichment':
                    if top_mask_jn.sum() == 0:
                        stat_jn.append(0.0)
                    else:
                        mean_top_jn = float(np.nanmean(score_jn[top_mask_jn]))
                        mean_all_jn = float(np.nanmean(score_jn))
                        stat_jn.append((mean_top_jn / mean_all_jn) if (mean_all_jn not in [0, np.nan]) else np.nan)
                else:  # odds_ratio
                    label_jn = eval_df[label_col].values[ind]
                    if np.sum(label_jn == True) == 0:
                        stat_jn.append(np.nan)
                    else:
                        or_jn, _ = odds_ratio(top_mask_jn, label_jn)
                        stat_jn.append(or_jn)

            stat_jn = np.array(stat_jn, dtype=float)
            stat_jn[np.isinf(stat_jn)] = 0.0
            auc_jn = np.sum(stat_jn * deltas[:len(stat_jn)])
            jn_estimates[bi] = auc_jn
            ratio_arr[bi] = total_observed / float(m_i)

        jn_dic[0][method] = jn_estimates.copy()

        valid = np.isfinite(jn_estimates) & np.isfinite(ratio_arr)
        if valid.sum() == 0:
            lower_ci_vals.append(np.nan)
            upper_ci_vals.append(np.nan)
            std_ci_vals.append(np.nan)
            continue

        jn_valid = jn_estimates[valid]
        r_valid = ratio_arr[valid]
        try:
            lower, upper = np.percentile(jn_valid, [2.5, 97.5])
        except Exception:
            lower, upper = np.nan, np.nan

        mean_jn = np.sum(auc - jn_valid) + np.sum((1.0 / r_valid) * jn_valid)
        tau = r_valid * auc - (r_valid - 1) * jn_valid
        try:
            std = np.sqrt((1.0 / valid.sum()) * np.sum((tau - mean_jn)**2 / (r_valid - 1.0)))
        except Exception:
            std = np.nan

        lower_ci_vals.append(lower)
        upper_ci_vals.append(upper)
        std_ci_vals.append(std)

    index = [0]
    auc_df = pd.DataFrame([auc_vals], columns=methods, index=index)
    lower_ci_df = pd.DataFrame([lower_ci_vals], columns=methods, index=index)
    upper_ci_df = pd.DataFrame([upper_ci_vals], columns=methods, index=index)
    std_ci_df = pd.DataFrame([std_ci_vals], columns=methods, index=index)

    for df in [auc_df, lower_ci_df, upper_ci_df, std_ci_df]:
        df.index.name = 'nlinks'

    return auc_df, lower_ci_df, upper_ci_df, std_ci_df, jn_dic


def compute_pairwise_delta_or_with_fdr(selected_methods, bs_dic, odds_df, nlinks_ls):
    """
    Compare methods pairwise using bootstrap distributions of odds ratios.

    Parameters
    ----------
    selected_methods: list of str
      Methods to include in the pairwise comparisons.
    bs_dic: dict
      Dictionary of bootstrap samples: bs_dic[nlinks][method] -> np.array
    odds_df: pd.DataFrame
      DataFrame with odds ratios from the main analysis.
    nlinks_ls: list of int
      List of `nlinks` values used in the main analysis.

    Returns
    ----------
    corrected_pval_lookup: dict
      {(focal_method, nlink, comparison_method): (raw_pval, corrected_pval)}
    all_delta_or_data: dict
      Focal method -> array of delta ORs (focal - comparator)
    all_yerr: dict
      Focal method -> array of 1.96 * std deviation of bootstrap differences
    """

    all_pvals = []
    meta_info = []

    all_delta_or_data = {}
    all_yerr = {}

    for method in selected_methods:
        focal_method = method
        comparison_methods = [m for m in selected_methods if m != method]

        # same dataframes as original, initialized with NaN
        delta_std_ci_df = pd.DataFrame(index=nlinks_ls, columns=comparison_methods, dtype=float)
        delta_mean_ci_df = pd.DataFrame(index=nlinks_ls, columns=comparison_methods, dtype=float)

        for key in nlinks_ls:
            # if bootstrap dict doesn't have this key, leave NaNs
            if key not in bs_dic:
                continue

            for m in comparison_methods:
                # require both methods present in bootstrap dict for this key
                if focal_method not in bs_dic[key] or m not in bs_dic[key]:
                    continue

                # get bootstrap arrays
                arr_f = np.asarray(bs_dic[key][focal_method], dtype=float)
                arr_m = np.asarray(bs_dic[key][m], dtype=float)

                # if shapes mismatch, try to handle by broadcasting or skip
                if arr_f.size != arr_m.size:
                    # try to broadcast/truncate to the min length if both are >0
                    min_len = min(arr_f.size, arr_m.size)
                    if min_len == 0:
                        # can't compute difference
                        continue
                    arr_f = arr_f[:min_len]
                    arr_m = arr_m[:min_len]

                delta_or = arr_f - arr_m
                # compute mean/std robustly (ignore NaNs)
                delta_mean = np.nanmean(delta_or)
                delta_std = np.nanstd(delta_or, ddof=0)

                delta_mean_ci_df.loc[key, m] = delta_mean
                delta_std_ci_df.loc[key, m] = delta_std

        # convert to numpy arrays
        # std_vals and mean_vals shaped (n_comparisons x n_nlinks) to match your original code
        std_vals = delta_std_ci_df.values.T.astype(float)   # shape (len(comparison_methods), len(nlinks_ls))
        mean_vals = delta_mean_ci_df.values.T.astype(float)

        # compute z and p-values safely: only where std is finite and > 0
        z = np.full_like(mean_vals, np.nan, dtype=float)
        pvals = np.full_like(mean_vals, np.nan, dtype=float)
        valid_mask = np.isfinite(mean_vals) & np.isfinite(std_vals) & (std_vals > 0)
        if np.any(valid_mask):
            z[valid_mask] = mean_vals[valid_mask] / std_vals[valid_mask]
            # two-sided p-values
            pvals[valid_mask] = 2 * (1 - sp.stats.norm.cdf(np.abs(z[valid_mask])))

        # Collect p-values and meta info in the same iteration order as your original code
        for i, m in enumerate(comparison_methods):
            for j, key in enumerate(nlinks_ls):
                all_pvals.append(pvals[i, j])
                meta_info.append((focal_method, key, m))

        # Build delta_or array from odds_df just like your original code, but mask invalid entries
        try:
            delta_or = (
                odds_df.loc[nlinks_ls, focal_method].values.reshape(-1, 1) -
                odds_df.loc[nlinks_ls, comparison_methods].values
            )  # shape (len(nlinks_ls), len(comparison_methods))
        except Exception:
            # if odds_df lookup fails, build NaN array of correct shape
            delta_or = np.full((len(nlinks_ls), len(comparison_methods)), np.nan, dtype=float)

        # mask delta_or values where we don't have valid bootstrap-based mean (i.e., delta_mean_ci_df is NaN)
        mask_missing = delta_mean_ci_df.isna().values  # shape (len(nlinks_ls), len(comparison_methods))
        # delta_or is shape (nlinks, ncomparisons) whereas mask_missing is (nlinks, ncomparisons) - same orientation
        delta_or = delta_or.astype(float)
        delta_or[mask_missing] = np.nan

        all_delta_or_data[focal_method] = delta_or

        # all_yerr uses std_vals transposed in your original code path; keep same orientation
        # std_vals currently shape (n_comp, n_nlinks) — multiply by 1.96 and keep that
        all_yerr[focal_method] = std_vals * 1.96

    # Global FDR correction across all valid p-values
    all_pvals_arr = np.array(all_pvals, dtype=float)
    pvals_corrected_all = np.full_like(all_pvals_arr, np.nan, dtype=float)
    valid_mask = ~np.isnan(all_pvals_arr)

    if valid_mask.sum() > 0:
        rej, pvals_corrected, _, _ = multipletests(all_pvals_arr[valid_mask], alpha=0.05, method='fdr_bh')
        pvals_corrected_all[valid_mask] = pvals_corrected
    else:
        # no valid tests -> pvals_corrected_all already NaN
        pass

    # Build lookup dictionary mapping (focal, key, comp) -> (raw_pval, corrected_pval)
    corrected_pval_lookup = {}
    for (meta, raw, corr) in zip(meta_info, all_pvals_arr.tolist(), pvals_corrected_all.tolist()):
        corrected_pval_lookup[meta] = (raw, corr)

    return corrected_pval_lookup, all_delta_or_data, all_yerr


def analyze_odds_ratio_jacknife_bins(eval_df, binning_col, nlinks_ls=None):
    """
    Perform odds ratio analysis with jackknife confidence intervals for
    multiple p-value methods within predefined bins in a DataFrame.

    Parameters
    ----------
    eval_df: pd.DataFrame
      DataFrame containing 'label' column, p-value columns, and the `binning_col`.
    binning_col: str
      Column name in `eval_df` used for defining bins (e.g., 'rna_mean_var_bin_5.5').
    nlinks_ls: list of int, optional
      List of numbers of links to consider. Default: [500, 1000, 1500, 2000, 2500]

    Returns
    ----------
    odds_df_all_bins, pval_df_all_bins, lower_ci_df_all_bins, upper_ci_df_all_bins, std_ci_df_all_bins, jn_dic_all_bins: tuple of pd.DataFrame and dict
      DataFrames containing the computed statistics indexed by `nlinks_ls` and columns as methods (multi-indexed by bin).
      The `jn_dic_all_bins` stores the jackknife samples for each nlink, bin, and method.
    """
    if nlinks_ls is None:
        nlinks_ls = [500, 1000, 1500, 2000, 2500]

    methods = [s for s in eval_df.columns if 'pval' in s]
    bins = eval_df[binning_col].sort_values().unique()

    # Initialize dictionaries to store results for all bins
    odds_results = {}
    pval_results = {}
    lower_ci_results = {}
    upper_ci_results = {}
    std_ci_results = {}
    jn_dic_all_bins = {n: {b: {m: [] for m in methods} for b in bins} for n in nlinks_ls}

    for mi, method in enumerate(methods):
        method_odds_df = pd.DataFrame(index=nlinks_ls, columns=bins)
        method_pval_df = pd.DataFrame(index=nlinks_ls, columns=bins)
        method_lower_ci_df = pd.DataFrame(index=nlinks_ls, columns=bins)
        method_upper_ci_df = pd.DataFrame(index=nlinks_ls, columns=bins)
        method_std_ci_df = pd.DataFrame(index=nlinks_ls, columns=bins)

        for bi, bin_val in enumerate(tqdm(bins, desc=f"Processing method {method}")):
            eval_subset_df = eval_df.loc[eval_df[binning_col] == bin_val].copy()
            
            if eval_subset_df.empty:
                continue

            pval_arr_col = eval_subset_df[method].values
            label_arr = eval_subset_df['label'].values
            
            # Use chromosome as categorical for jackknife within each bin
            # Assuming 'peak' column exists in eval_df and can be used to derive chromosome.
            categorical_arr_subset = split_by_chromosome(eval_subset_df, col='peak')

            sorted_idx = np.argsort(pval_arr_col)
            
            # Limit dictionary for this subset
            limit_dic_subset = {
                method: (eval_subset_df[method] == eval_subset_df[method].min()).sum()
                if 'mc' in method else 0
            }

            for ni, nlinks in enumerate(nlinks_ls):
                y_score = np.zeros(len(eval_subset_df), dtype=bool)
                y_score[sorted_idx[:nlinks]] = True

                stat, pval = odds_ratio(y_score, label_arr, smoothed=True)
                method_odds_df.loc[nlinks, bin_val] = stat
                method_pval_df.loc[nlinks, bin_val] = pval

                if method in limit_dic_subset and nlinks < limit_dic_subset[method]:
                    method_odds_df.loc[nlinks, bin_val] = np.nan
                    method_pval_df.loc[nlinks, bin_val] = np.nan
                    method_lower_ci_df.loc[nlinks, bin_val] = np.nan
                    method_upper_ci_df.loc[nlinks, bin_val] = np.nan
                    method_std_ci_df.loc[nlinks, bin_val] = np.nan
                    continue

                blocks_arr = np.unique(categorical_arr_subset)
                n_blocks = len(blocks_arr)
                if n_blocks == 0: # Handle cases where a bin might have no data with chromosome info
                    continue

                jn_stats = np.zeros(n_blocks)

                for block_idx, block in enumerate(blocks_arr):
                    # Exclude the current block (chromosome) for jackknife sample
                    ind = np.argwhere(categorical_arr_subset != block).flatten()
                    
                    if len(ind) == 0: # Ensure there's data left after exclusion
                        jn_stats[block_idx] = np.nan # Or handle as appropriate
                        continue

                    jn_y_score = y_score[ind]
                    jn_label = label_arr[ind]

                    jn_stats[block_idx], _ = odds_ratio(jn_y_score, jn_label, smoothed=True)
                
                # Filter out NaN jackknife samples before calculating statistics
                jn_stats_valid = jn_stats[~np.isnan(jn_stats)]
                if len(jn_stats_valid) > 1: # Need at least 2 samples for std dev
                    std_err_jackknife = np.sqrt(((len(jn_stats_valid) - 1) / len(jn_stats_valid)) * np.sum((jn_stats_valid - np.mean(jn_stats_valid))**2))
                    method_lower_ci_df.loc[nlinks, bin_val] = stat - 1.96 * std_err_jackknife
                    method_upper_ci_df.loc[nlinks, bin_val] = stat + 1.96 * std_err_jackknife
                    method_std_ci_df.loc[nlinks, bin_val] = std_err_jackknife
                else:
                    method_lower_ci_df.loc[nlinks, bin_val] = np.nan
                    method_upper_ci_df.loc[nlinks, bin_val] = np.nan
                    method_std_ci_df.loc[nlinks, bin_val] = np.nan

                jn_dic_all_bins[nlinks][bin_val][method] = jn_stats

        odds_results[method] = method_odds_df
        pval_results[method] = method_pval_df
        lower_ci_results[method] = method_lower_ci_df
        upper_ci_results[method] = method_upper_ci_df
        std_ci_results[method] = method_std_ci_df
    
    # Combine results into multi-indexed DataFrames
    odds_df_all_bins = pd.concat(odds_results, axis=1, keys=methods)
    pval_df_all_bins = pd.concat(pval_results, axis=1, keys=methods)
    lower_ci_df_all_bins = pd.concat(lower_ci_results, axis=1, keys=methods)
    upper_ci_df_all_bins = pd.concat(upper_ci_results, axis=1, keys=methods)
    std_ci_df_all_bins = pd.concat(std_ci_results, axis=1, keys=methods)

    for df in [odds_df_all_bins, pval_df_all_bins, lower_ci_df_all_bins, upper_ci_df_all_bins, std_ci_df_all_bins]:
        df.index.name = 'nlinks'
        df.columns.names = ['method', binning_col]

    return odds_df_all_bins, pval_df_all_bins, lower_ci_df_all_bins, upper_ci_df_all_bins, std_ci_df_all_bins, jn_dic_all_bins


def compute_pairwise_delta_or_with_fdr_bins(
    focal_method, comparison_method, nlinks_ls, bins, jn_dic_all_bins, odds_df_all_bins
):
    """
    Compare two methods (focal and comparison) across different bins and nlinks values
    using jackknife distributions of odds ratios, and apply FDR correction.

    Parameters
    ----------
    focal_method : str
        The name of the focal method.
    comparison_method : str
        The name of the comparison method.
    nlinks_ls : list of int
        List of nlinks values used in the analysis.
    bins : list
        List of unique bin values.
    jn_dic_all_bins : dict
        Dictionary containing jackknife samples for all nlinks, bins, and methods.
        Format: jn_dic_all_bins[nlink][bin_val][method] -> np.array of jackknife samples.
    odds_df_all_bins : pd.DataFrame
        DataFrame with overall odds ratios for all methods and bins, multi-indexed.

    Returns
    -------
    corrected_pval_lookup : dict
        A dictionary mapping (bin, nlink) to a tuple of (raw_pval, corrected_pval).
    all_delta_or_data : dict
        A dictionary where keys are bin values and values are lists of delta ORs
        (focal - comparison) for each nlink.
    all_yerr : dict
        A dictionary where keys are bin values and values are arrays of 1.96 * std deviation
        of jackknife differences for each nlink.
    """
    all_pvals = []
    meta_info = []

    all_delta_or_data = {}
    all_yerr = {}

    for bi, bin_ in enumerate(tqdm(bins, desc="Computing delta ORs per bin")):
        delta_mean_per_nlink = []
        delta_std_per_nlink = []

        for nlinks in nlinks_ls:
            jn_focal = jn_dic_all_bins[nlinks][bin_][focal_method]
            jn_comparison = jn_dic_all_bins[nlinks][bin_][comparison_method]

            # Ensure both arrays are valid and have data
            jn_focal_valid = jn_focal[~np.isnan(jn_focal)]
            jn_comparison_valid = jn_comparison[~np.isnan(jn_comparison)]
            
            # For jackknife, the comparison is usually between the overall estimate and leave-one-out estimates
            # For comparing two methods' jackknife distributions, we can approximate the standard error of the difference
            # by assuming independence for large samples, or directly compute the differences of samples if they are paired.
            # Assuming jackknife samples for each method within a bin are derived independently,
            # the variance of the difference is the sum of variances.
            # However, a simpler approach for a z-test based on jackknife standard errors is often used:
            
            if len(jn_focal_valid) > 1 and len(jn_comparison_valid) > 1:
                # Calculate the difference of the *original* odds ratios for the mean
                current_focal_or = odds_df_all_bins.loc[nlinks, (focal_method, bin_)]
                current_comparison_or = odds_df_all_bins.loc[nlinks, (comparison_method, bin_)]
                
                delta_or_mean = current_focal_or - current_comparison_or

                # Calculate the standard error of the odds ratios from jackknife for each method
                # This uses the formula: SE(theta_hat) = sqrt(((n-1)/n) * sum((theta_i - theta_bar)^2))
                # where theta_i are the jackknife estimates
                std_err_focal = np.sqrt(((len(jn_focal_valid) - 1) / len(jn_focal_valid)) * np.sum((jn_focal_valid - np.mean(jn_focal_valid))**2))
                std_err_comparison = np.sqrt(((len(jn_comparison_valid) - 1) / len(jn_comparison_valid)) * np.sum((jn_comparison_valid - np.mean(jn_comparison_valid))**2))
                
                # Standard error of the difference (assuming independence of methods)
                std_err_delta = np.sqrt(std_err_focal**2 + std_err_comparison**2)

                delta_mean_per_nlink.append(delta_or_mean)
                delta_std_per_nlink.append(std_err_delta)

                # Compute uncorrected p-value using z-test
                z_score = delta_or_mean / (std_err_delta + 1e-10)
                pval = 2 * (1 - sp.stats.norm.cdf(np.abs(z_score)))
                all_pvals.append(pval)
                meta_info.append((bin_, nlinks))
            else:
                delta_mean_per_nlink.append(np.nan)
                delta_std_per_nlink.append(np.nan)
                all_pvals.append(np.nan)
                meta_info.append((bin_, nlinks)) # Still add meta info for NaN p-values

        all_delta_or_data[bin_] = np.array(delta_mean_per_nlink)
        all_yerr[bin_] = np.array(delta_std_per_nlink) * 1.96 # For 95% CI

    # Global FDR correction
    all_pvals_arr = np.array(all_pvals)
    non_nan_pval_inds = np.argwhere(~np.isnan(all_pvals_arr)).flatten()
    
    pvals_corrected_arr = np.full(all_pvals_arr.shape, np.nan)
    if len(non_nan_pval_inds) > 0:
        rej, pvals_corrected, _, _ = multipletests(all_pvals_arr[non_nan_pval_inds], alpha=0.05, method='fdr_bh')
        pvals_corrected_arr[non_nan_pval_inds] = pvals_corrected

    # Build lookup for corrected and raw p-values
    corrected_pval_lookup = {
        (bin_val, nlink): (raw, corr)
        for (bin_val, nlink), raw, corr in zip(meta_info, all_pvals_arr, pvals_corrected_arr)
    }

    return corrected_pval_lookup, all_delta_or_data, all_yerr


def analyze_delta_enrichment_auc_jackknife(eval_df, categorical_arr, bin_arr,
                                             score_col='gt_pip', methods=None,
                                             nlinks_ls=None):
    """
    Compute enrichment AUC via midpoint rule using jackknife confidence intervals,
    and also compute per-bin contributions (leave-one-bin-out) with jackknife SEs.
    
    Parameters
    ----------
    eval_df : pd.DataFrame
        DataFrame with p-value columns and a score column.
    categorical_arr : np.array
        Jackknife blocking array (e.g., chromosomes).
    bin_arr : np.array
        Bin labels (same length as eval_df), e.g. '1.1.1.1'.
    score_col : str
        Name of the score column to use for enrichment ranking.
    nlinks_ls : list of int
        List of N link cutoffs (e.g. [500, 1000, ...]).
    
    Returns
    ----------
    auc_df, lower_ci_df, upper_ci_df, std_ci_df, jn_dic, bin_contribs_df
        - auc_df etc. are method-level AUCs and jackknife CIs
        - bin_contribs_df: DataFrame with columns [bin, method, contribution, se]
    """
    if nlinks_ls is None:
        nlinks_ls = [500, 1000, 1500, 2000, 2500]
    nlinks_arr = np.sort(np.array(nlinks_ls))

    if methods is None:
        methods = [s for s in eval_df.columns if 'pval' in s]

    score_arr = eval_df[score_col].values
    blocks_arr = np.unique(categorical_arr)
    n_blocks = len(blocks_arr)

    # Midpoint and delta arrays
    midpoints = (nlinks_arr[:-1] + nlinks_arr[1:]) / 2
    deltas = np.diff(nlinks_arr)

    auc_vals = []
    lower_ci_vals = []
    upper_ci_vals = []
    std_ci_vals = []
    jn_dic = {0:{}}

    # To collect bin-level results
    bin_contribs_mean = {method:{} for method in methods}
    bin_contribs_jn   = {method:{b:[] for b in np.unique(bin_arr)} for method in methods}

    # Helper
    def compute_auc(scores, pvals):
        nan_mask = ~np.isnan(scores) & ~np.isnan(pvals)
        if nan_mask.sum() == 0:
            return np.nan
        enrich_at_mid = np.array([
            ctar.simu.enrichment(scores[nan_mask], pvals[nan_mask], int(n)) for n in midpoints
        ])
        return np.sum(enrich_at_mid * deltas)

    for method in tqdm(methods):
        pval_arr = eval_df[method].values

        auc = compute_auc(score_arr, pval_arr)
        auc_vals.append(auc)

        for b in np.unique(bin_arr):
            mask = (bin_arr != b)
            auc_sub = compute_auc(score_arr[mask], pval_arr[mask])
            if np.isnan(auc_sub):
                contrib = np.nan
            else:
                contrib = auc - auc_sub
            bin_contribs_mean[method][b] = contrib

        jn_estimates = np.zeros(n_blocks)
        ratio_arr = np.zeros(n_blocks)

        for bi, block in enumerate(blocks_arr):
            mask_block = categorical_arr != block
            score_sub = score_arr[mask_block]
            pval_sub = pval_arr[mask_block]
            ratio_arr[bi] = len(categorical_arr) / np.sum(categorical_arr == block)

            auc_block = compute_auc(score_sub, pval_sub)
            if np.isnan(auc_block):
                jn_estimates[bi] = np.nan
                ratio_arr[bi] = np.nan
                continue
            jn_estimates[bi] = auc_block

            for b in np.unique(bin_arr):
                mask_bin = (bin_arr != b) & mask_block
                auc_block_bin = compute_auc(score_arr[mask_bin], pval_arr[mask_bin])
                if np.isnan(auc_block_bin):
                    continue
                contrib_block = auc_block - auc_block_bin
                bin_contribs_jn[method][b].append(contrib_block)

        jn_dic[0][method] = jn_estimates

        # Remove NaNs before CI/stat calculation
        valid = ~np.isnan(jn_estimates) & ~np.isnan(ratio_arr)
        jn_estimates = jn_estimates[valid]
        ratio_arr = ratio_arr[valid]

        if len(jn_estimates) == 0:
            lower_ci_vals.append(np.nan)
            upper_ci_vals.append(np.nan)
            std_ci_vals.append(np.nan)
            continue

        # CI and Std
        lower, upper = np.percentile(jn_estimates, [2.5, 97.5])
        mean_jn = np.sum(auc - jn_estimates) + np.sum((1/ratio_arr) * jn_estimates)
        tau = ratio_arr * auc - (ratio_arr - 1) * jn_estimates
        std = np.sqrt((1 / n_blocks) * np.sum((tau - mean_jn)**2 / (ratio_arr - 1)))

        lower_ci_vals.append(lower)
        upper_ci_vals.append(upper)
        std_ci_vals.append(std)

    index = [0]
    auc_df = pd.DataFrame([auc_vals], columns=methods, index=index)
    lower_ci_df = pd.DataFrame([lower_ci_vals], columns=methods, index=index)
    upper_ci_df = pd.DataFrame([upper_ci_vals], columns=methods, index=index)
    std_ci_df = pd.DataFrame([std_ci_vals], columns=methods, index=index)

    for df in [auc_df, lower_ci_df, upper_ci_df, std_ci_df]:
        df.index.name = 'nlinks'

    records = []
    for method in methods:
        for b, contrib_mean in bin_contribs_mean[method].items():
            contrib_reps = np.array(bin_contribs_jn[method][b])
            if len(contrib_reps) > 1:
                se = np.std(contrib_reps, ddof=1) * np.sqrt(len(contrib_reps)-1)
            else:
                se = np.nan
            records.append({
                'bin': b,
                'method': method,
                'contribution': contrib_mean,
                'se': se
            })
    bin_contribs_df = pd.DataFrame(records)

    return auc_df, lower_ci_df, upper_ci_df, std_ci_df, jn_dic, bin_contribs_df


class ZeroInflatedPoisson:
    ''' Custom distribution for ZIP QQ plot.
    '''

    def __init__(self, pi, lam):
        self.pi = pi
        self.lam = lam
        self.poisson = sp.stats.poisson(mu=lam)

    def ppf(self, q):
        ''' Percent point function (inverse of CDF).  Crucial for QQ plots.
        '''
        # Handle edge cases for q (quantiles/probabilities)
        q = np.asarray(q)
        q = np.clip(q, 0, 1) # Make sure q is between 0 and 1.

        # Pre-calculate the Poisson quantiles (for efficiency)
        poisson_quantiles = self.poisson.ppf(q) # Use scipy.stats ppf

        # Calculate the ZIP quantiles
        zip_quantiles = np.where(q <= self.pi, 0, poisson_quantiles)
        return zip_quantiles

    def rvs(self, size=1):
        ''' Random variate generation.
        '''
        poisson_rvs = self.poisson.rvs(size=size) # Use scipy.stats rvs
        zeros = np.random.binomial(1, self.pi, size=size)
        return np.where(zeros == 1, 0, poisson_rvs)
    
    @property
    def name(self):
        return "ZeroInflatedPoisson"


class ZeroInflatedNegativeBinomial:
    ''' Custom distribution for ZINB QQ plot.
    '''

    def __init__(self, pi, mu, alpha):
        self.pi = pi
        self.mu = mu
        self.alpha = alpha
        self.nbinom = sp.stats.nbinom(n=mu,p=alpha)

    def ppf(self, q):
        ''' Percent point function (inverse of CDF).  Crucial for QQ plots.
        '''
        # Handle edge cases for q (quantiles/probabilities)
        q = np.asarray(q)
        q = np.clip(q, 0, 1) # Make sure q is between 0 and 1.

        # Pre-calculate the Poisson quantiles (for efficiency)
        nb_quantiles = self.nbinom.ppf(q) # Use scipy.stats ppf

        # Calculate the ZIP quantiles
        zinb_quantiles = np.where(q <= self.pi, 0, nb_quantiles)
        return zinb_quantiles

    def rvs(self, size=1):
        ''' Random variate generation.
        '''
        nbinom_rvs = self.nbinom.rvs(size=size) # Use scipy.stats rvs
        zeros = np.random.binomial(1, self.pi, size=size)
        return np.where(zeros == 1, 0, nbinom_rvs)
    
    @property
    def name(self):
        return "ZeroInflatedNegativeBinomial"


def analyze_poisson(Y, X, display=True):
    
    # regression
    X_with_intercept = sm.add_constant(X)
    model = sm.GLM(Y, X_with_intercept, family=sm.families.Poisson())
    results = model.fit()

    if display:
        # results
        print(results.summary())

        plt.plot(Y, results.fittedvalues, 'o', alpha=0.3)
        plt.plot(Y, Y, ':', label='Y = X',c='grey')
        plt.ylabel("fitted value")
        plt.xlabel("observed value")
        plt.legend()
        plt.show()

        f, axes = plt.subplots(1, 2, figsize=(17, 6))
        axes[0].plot(Y, results.resid_response, 'o')
        axes[0].set_ylabel("Residuals")
        axes[0].set_xlabel("$Y$")
        axes[1].plot(Y, results.resid_pearson, 'o')
        axes[1].axhline(y=-1, linestyle=':', color='black', label='$\\pm 1$')
        axes[1].axhline(y=+1, linestyle=':', color='black')
        axes[1].set_ylabel("Standardized residuals")
        axes[1].set_xlabel("$Y$")
        plt.legend()
        plt.show()

    return results


def analyze_nb(Y, X, method='nm',optim_kwds_prelim=dict(method='nm', disp=1), display=True):
    
    # nb regression
    X_with_intercept = sm.add_constant(X)
    model = NegativeBinomial(Y, X_with_intercept)
    results = model.fit(method=method,optim_kwds_prelim=optim_kwds_prelim)

    if display:
        # results
        print(results.summary())

        plt.plot(Y, results.fittedvalues, 'o', alpha=0.3)
        plt.plot(Y, Y, ':', label='Y = X',c='grey')
        plt.ylabel("fitted value")
        plt.xlabel("observed value")
        plt.legend()
        plt.show()

        f, axes = plt.subplots(1, 2, figsize=(17, 6))
        axes[0].plot(Y, results.resid_response, 'o')
        axes[0].set_ylabel("Residuals")
        axes[0].set_xlabel("$Y$")
        axes[1].plot(Y, results.resid_pearson, 'o')
        axes[1].axhline(y=-1, linestyle=':', color='black', label='$\\pm 1$')
        axes[1].axhline(y=+1, linestyle=':', color='black')
        axes[1].set_ylabel("Standardized residuals")
        axes[1].set_xlabel("$Y$")
        plt.legend()
        plt.show()

    return results

