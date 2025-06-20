import pytest
import numpy as np
import scipy as sp
from pytest import approx
import ctar


### mc pval tests ###
def test_initial_mcpval_one_sided_positive_corr():
    ctrl_corr = np.array([[1, 2, 3, 4, 5]])  # 1 gene, 5 controls
    corr = np.array([3.5])                   # 1 gene focal corr
    # ctrl_corr >= 3.5  -> [F,F,F,T,T] -> sum = 2
    # pval = (1+2)/(1+5) = 3/6 = 0.5
    pval = ctar.method.initial_mcpval(ctrl_corr, corr, one_sided=True)
    assert pval[0] == approx(0.5)


def test_initial_mcpval_two_sided_positive_corr():
    ctrl_corr = np.array([[1, 2, 3, 4, 5]])
    corr = np.array([3.5])
    # abs(ctrl_corr) >= abs(3.5) -> sum = 2
    # pval = (1+2)/(1+5) = 3/6 = 0.5
    pval = ctar.method.initial_mcpval(ctrl_corr, corr, one_sided=False)
    assert pval[0] == approx(0.5)


def test_initial_mcpval_one_sided_negative_corr():
    ctrl_corr = np.array([[-1, -2, -3, -4, -5]])
    corr = np.array([-3.5])
    # [-1,-2,-3,-4,-5] >= -3.5 -> [T,T,T,F,F] -> sum = 3
    # pval = (1+3)/(1+5) = 4/6
    pval = ctar.method.initial_mcpval(ctrl_corr, corr, one_sided=True)
    assert pval[0] == approx(4/6)


def test_initial_mcpval_two_sided_negative_corr():
    ctrl_corr = np.array([[-1, -2, -3, -4, -5]])
    corr = np.array([-3.5])
    # abs vals: [1,2,3,4,5] vs 3.5 -> sum = 2
    # pval = (1+2)/(1+5) = 0.5
    pval = ctar.method.initial_mcpval(ctrl_corr, corr, one_sided=False)
    assert pval[0] == approx(0.5)


def test_initial_mcpval_multiple_genes():
    ctrl_corr = np.array([[1,2,3], [10,11,12]])
    corr = np.array([1.5, 12.5])
    # Gene 1: sum([T,T,F]) = 2. pval = (1+2)/(1+3) = 0.75
    # Gene 2: sum([F,F,F]) = 0. pval = (1+0)/(1+3) = 0.25
    expected_pvals = np.array([0.75, 0.25])
    pvals = ctar.method.initial_mcpval(ctrl_corr, corr, one_sided=True)
    assert np.allclose(pvals, expected_pvals)


### zscore pval tests ###
def test_zscore_pval_zero_z():
    ctrl_corr = np.array([[1,2,3,4,5]]) # mean=3, std=sqrt(2)
    corr = np.array([3.0]) # z = (3-3)/std = 0
    p_val, z = ctar.method.zscore_pval(ctrl_corr, corr, axis=1)
    assert z[0] == approx(0.0)
    assert p_val[0] == approx(0.5) # 1 - norm.cdf(0) = 0.5


def test_zscore_pval_positive_z():
    ctrl_corr = np.array([[1,2,3,4,5]]) # mean=3, std=sqrt(2)
    corr = np.array([5.0]) # z = (5-3)/sqrt(2) = sqrt(2)
    p_val, z = ctar.method.zscore_pval(ctrl_corr, corr, axis=1)
    assert z[0] == approx(np.sqrt(2))
    assert p_val[0] == approx(1 - sp.stats.norm.cdf(np.sqrt(2)))


def test_zscore_pval_zero_std_corr_greater_than_mean():
    ctrl_corr = np.array([[3,3,3,3,3]]) # mean=3, std=0
    corr = np.array([4.0])
    p_val, z = ctar.method.zscore_pval(ctrl_corr, corr, axis=1)
    assert z[0] == np.inf # (4-3)/0 = inf
    assert p_val[0] == approx(0.0) # 1 - cdf(inf) = 0
    

def test_zscore_pval_zero_std_corr_less_than_mean():
    ctrl_corr = np.array([[3,3,3,3,3]]) # mean=3, std=0
    corr = np.array([2.0])
    p_val, z = ctar.method.zscore_pval(ctrl_corr, corr, axis=1)
    assert z[0] == -np.inf # (2-3)/0 = -inf
    assert p_val[0] == approx(1.0) # 1 - cdf(-inf) = 1