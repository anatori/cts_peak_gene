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


def analyze_odds_ratio_bootstrap(eval_df, nlinks_ls=None, n_bs_samples=1000):
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
    odds_df, pval_df, lower_ci_df, upper_ci_df, std_ci_df: pd.DataFrame
      DataFrames containing the computed statistics indexed by `nlinks_ls`
      and columns as methods.
    """

    if nlinks_ls is None:
        nlinks_ls = [500, 1000, 1500, 2000, 2500]

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

    return odds_df, pval_df, lower_ci_df, upper_ci_df, std_ci_df


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

        delta_std_ci_df = pd.DataFrame(index=nlinks_ls, columns=comparison_methods)
        delta_mean_ci_df = pd.DataFrame(index=nlinks_ls, columns=comparison_methods)

        for key in nlinks_ls:
            for m in comparison_methods:
                delta_or = bs_dic[key][focal_method] - bs_dic[key][m]
                delta_std_ci_df.loc[key, m] = np.std(delta_or)
                delta_mean_ci_df.loc[key, m] = np.mean(delta_or)

        std_vals = delta_std_ci_df.values.T.astype(float)
        mean_vals = delta_mean_ci_df.values.T.astype(float)

        # Compute uncorrected p-values using z-test
        z = mean_vals / std_vals
        pvals = 2 * (1 - sp.stats.norm.cdf(np.abs(z)))

        # Store raw p-values and metadata
        for i, m in enumerate(comparison_methods):
            for j, key in enumerate(nlinks_ls):
                all_pvals.append(pvals[i, j])
                meta_info.append((focal_method, key, m))

        # Store for visualization or further processing
        delta_or = (
            odds_df.loc[nlinks_ls, focal_method].values.reshape(-1, 1) -
            odds_df.loc[nlinks_ls, comparison_methods].values
        )
        all_delta_or_data[focal_method] = delta_or
        all_yerr[focal_method] = std_vals * 1.96

    # Global FDR correction
    all_pvals_arr = np.array(all_pvals)
    pvals_corrected_all = np.array(all_pvals)
    rej, pvals_corrected, _, _ = multipletests(
        all_pvals_arr[~np.isnan(all_pvals_arr)],
        alpha=0.05,
        method='fdr_bh'
    )
    pvals_corrected_all[~np.isnan(all_pvals_arr)] = pvals_corrected

    # Lookup dictionary for corrected p-values
    corrected_pval_lookup = {
        (focal, key, comp): (raw, corr)
        for (focal, key, comp), raw, corr in zip(meta_info, all_pvals, pvals_corrected_all)
    }

    return corrected_pval_lookup, all_delta_or_data, all_yerr


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

