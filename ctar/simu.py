import numpy as np
import pandas as pd 
import scipy as sp
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm


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


def odds_ratio(y_arr, label_arr, return_table=False, smoothed=False, epsilon=1e-6):

    # (tp * tn) / (fp * fn)
    
    tp = np.sum((label_arr == 1) & y_arr)
    fp = np.sum((label_arr == 0) & y_arr)
    fn = np.sum((label_arr == 1) & ~y_arr)
    tn = np.sum((label_arr == 0) & ~y_arr)
    table = [[tp, fp], [fn, tn]]

    stat, pval = sp.stats.fisher_exact(table)
    
    if ( np.isnan(stat)) and smoothed: # np.isinf(stat) or

        tp_s = tp + epsilon
        fp_s = fp + epsilon
        fn_s = fn + epsilon
        tn_s = tn + epsilon
        stat = (tp_s * tn_s) / (fp_s * fn_s)

    if return_table:
        return table, stat, pval

    return stat, pval


def enrichment(scores, pvals, top_n, smoothed=False, epsilon=1e-6):

    ''' ( sum(scores of top nlinks) / top nlinks ) / ( sum(scores of all links) / all links )

    Parameters
    ----------
    scores : np.array
        Array of scores from ground truth data.
    pvals : np.array
        Array of p-values from method.
    top_n : int
        Number of top smallest p-values to consider true.

    Returns
    -------
    enrichment : float
    
    '''
    
    numerator = np.argsort(pvals)
    numerator = np.sum(scores[numerator][:top_n]) / top_n
    denominator = np.sum(scores) / len(scores)

    return numerator / denominator


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

        for ni, nlinks in enumerate(nlinks_ls):
            y_score = np.zeros(len(eval_df), dtype=bool)
            y_score[sorted_idx[:nlinks]] = True

            stat, pval = odds_ratio(y_score, label_arr)
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


def analyze_odds_ratio_jacknife(eval_df, categorical_arr, methods=None, nlinks_ls=None):
    """
    Perform odds ratio analysis with jacknife confidence intervals by
    chromosome blocks for multiple p-value methods in a DataFrame.

    From Nick Patterson notes on 
    F.M.T.A. Busing, E. Meijer, and R. van der Leeden. 
    Delete-m jackknife for unequal m. Statistics and Computing, 9:3–8, 1999.

    Parameters
    ----------
    eval_df: pd.DataFrame
      DataFrame containing 'label' column and p-value columns.
    categorical_arr : np.array
      Array of values corresponding to category to split on for jacknifing (e.g. loci blocks)
    nlinks_ls: list of int, optional
      List of numbers of links to consider. Default: [500, 1000, 1500, 2000, 2500]

    Returns
    ----------
    odds_df, pval_df, lower_ci_df, upper_ci_df, std_ci_df, jn_dic: pd.DataFrame
      DataFrames containing the computed statistics indexed by `nlinks_arr`
      and columns as methods.
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

    label_arr = eval_df['label'].values

    for mi, method in enumerate(tqdm(methods)):
        pval_arr_col = eval_df[method].values
        sorted_idx = np.argsort(pval_arr_col)

        for ni, nlinks in enumerate(nlinks_arr):
            y_score = np.zeros(len(eval_df), dtype=bool)
            y_score[sorted_idx[:nlinks]] = True

            odds_arr[ni, mi], pval_arr[ni, mi] = odds_ratio(y_score, label_arr)

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


def analyze_enrichment_jacknife(eval_df, categorical_arr, score_col='gt_pip', methods=None, nlinks_ls=None):
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
    enrich_arr = np.full(results_shape, np.nan)
    lower_ci_arr = np.full(results_shape, np.nan)
    upper_ci_arr = np.full(results_shape, np.nan)
    std_ci_arr = np.full(results_shape, np.nan)

    score_arr = eval_df[score_col].values

    for mi, method in enumerate(tqdm(methods)):
        pval_arr_col = eval_df[method].values

        for ni, nlinks in enumerate(nlinks_arr):

            enrich_arr[ni, mi] = enrichment(score_arr, pval_arr_col, nlinks)

            if method in limit_dic and nlinks < limit_dic[method]:
                enrich_arr[ni, mi] = np.nan
                continue

            blocks_arr = np.unique(categorical_arr)
            n_blocks = len(blocks_arr)
            ratio_arr = np.zeros(n_blocks)
            jn_stats = np.zeros(n_blocks)

            for bi,block in enumerate(blocks_arr):
                ind = np.argwhere(categorical_arr != block).flatten()
                jn_stats[bi] = enrichment(score_arr[ind], pval_arr_col[ind], nlinks)
                ratio_arr[bi] = len(categorical_arr) / sum(categorical_arr == block)
            
            tau = ratio_arr * enrich_arr[ni,mi] - (ratio_arr - 1) * jn_stats
            jn_estimates = np.sum(enrich_arr[ni,mi] - jn_stats) + np.sum((1/ratio_arr) * jn_stats)

            jn_dic[nlinks][method] = jn_stats
            lower_ci_arr[ni, mi], upper_ci_arr[ni, mi] = np.percentile(jn_stats, [2.5, 97.5])
            std_ci_arr[ni, mi] = (1/n_blocks) * np.sum((tau - jn_estimates)**2 / (ratio_arr - 1))
            std_ci_arr[ni, mi] = np.sqrt(std_ci_arr[ni, mi])

    enrich_df = pd.DataFrame(enrich_arr, index=nlinks_arr, columns=methods)
    lower_ci_df = pd.DataFrame(lower_ci_arr, index=nlinks_arr, columns=methods)
    upper_ci_df = pd.DataFrame(upper_ci_arr, index=nlinks_arr, columns=methods)
    std_ci_df = pd.DataFrame(std_ci_arr, index=nlinks_arr, columns=methods)

    for df in [enrich_df, lower_ci_df, upper_ci_df, std_ci_df]:
        df.index.name = 'nlinks'

    return enrich_df, lower_ci_df, upper_ci_df, std_ci_df, jn_dic


def get_top_n_mask(pvals, n):
    """Return a boolean mask for the top n smallest p-values."""
    idx = np.argsort(pvals)[:n]
    mask = np.zeros(len(pvals), dtype=bool)
    mask[idx] = True
    return mask


def analyze_auc_midpoint_jackknife(eval_df, categorical_arr, score_col='gt_pip', methods=None, nlinks_ls=None, stat_method='enrichment'):
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
    nlinks_arr = np.sort(np.array(nlinks_ls))

    if methods is None:
        methods = [s for s in eval_df.columns if 'pval' in s]

    score_arr = eval_df[score_col].values
    blocks_arr = np.unique(categorical_arr)
    n_blocks = len(blocks_arr)

    auc_vals = []
    lower_ci_vals = []
    upper_ci_vals = []
    std_ci_vals = []
    jn_dic = {0:{}}

    for method in tqdm(methods):
        pval_arr = eval_df[method].values

        nan_mask = np.isfinite(score_arr) & np.isfinite(pval_arr)
        score = score_arr[nan_mask]
        pval = pval_arr[nan_mask]
        cat = np.asarray(categorical_arr)[nan_mask]
        n_rows = len(pval)
        blocks = np.unique(cat)

        nlinks_arr_eff = nlinks_arr[nlinks_arr < n_rows]
        midpoints = np.round((nlinks_arr_eff[:-1] + nlinks_arr_eff[1:]) / 2).astype(int)
        deltas = np.diff(nlinks_arr_eff)

        if stat_method == 'enrichment':
            stat_at_mid = np.array([enrichment(score, pval, int(n)) for n in midpoints])
        elif stat_method == 'odds_ratio':
            label_arr = score
            stat_at_mid = np.array([
                odds_ratio(get_top_n_mask(pval, n), label_arr)[0] for n in midpoints
            ])
        else:
            raise ValueError("stat_method must be either 'enrichment' or 'odds_ratio'")

        stat_at_mid[np.isinf(stat_at_mid)] = 0.0
        auc = np.sum(stat_at_mid * deltas)
        auc_vals.append(auc)

        # Jackknife
        jn_estimates = np.zeros(n_blocks)
        ratio_arr = np.zeros(n_blocks)
        enrich_jn = np.zeros((n_blocks, len(midpoints)))

        for bi, block in enumerate(blocks_arr):
            mask_block = cat != block
            ratio_arr[bi] = len(cat) / np.sum(cat == block)
            score_sub = score[mask_block]
            pval_sub = pval[mask_block]
            label_sub = label_arr[mask_block] if stat_method == 'odds_ratio' else None

            if len(score_sub) == 0 or len(pval_sub) == 0:
                print(f'Excluding {method}, {block} from JN computation...')
                enrich_jn[bi] = np.nan
                jn_estimates[bi] = np.nan
                ratio_arr[bi] = np.nan
                continue

            if stat_method == 'enrichment':
                stat_jn = np.array([
                    enrichment(score_sub, pval_sub, int(n)) for n in midpoints
                ])
            elif stat_method == 'odds_ratio':
                stat_jn = np.array([
                    odds_ratio(get_top_n_mask(pval_sub, n), label_sub)[0] for n in midpoints
                ])

            stat_jn[np.isinf(stat_jn)] = 0.0
            auc_jn = np.sum(stat_jn * deltas)
            jn_estimates[bi] = auc_jn

        jn_dic[0][method] = jn_estimates

        # Remove NaNs before CI/stat calculation
        valid = np.isfinite(jn_estimates) & np.isfinite(ratio_arr)
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

