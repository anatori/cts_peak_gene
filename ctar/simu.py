import numpy as np
import pandas as pd 
import scipy as sp
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial


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


def odds_ratio(y_arr, label_arr):

    tp = np.sum((label_arr == 1) & y_arr)
    fp = np.sum((label_arr == 0) & y_arr)
    fn = np.sum((label_arr == 1) & ~y_arr)
    tn = np.sum((label_arr == 0) & ~y_arr)
    table = [[tp, fp], [fn, tn]]

    stat, pval = sp.stats.fisher_exact(table)

    return stat, pval


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

