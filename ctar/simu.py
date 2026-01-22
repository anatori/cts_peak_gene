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

