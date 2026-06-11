import numpy as np
import pandas as pd
import pytest

from ctar.method import binned_empirical_fdr

def test_binned_empirical_fdr_basic():
    """All ctrl at 0.5, pairs split cleanly above/below."""
    # 4 pairs in one ctrl bin, null entirely at 0.5
    df = pd.DataFrame({
        'ctrl_bin': list('AAAA'),
        'corr':     [0.9, 0.7, 0.3, 0.1],
    })
    ctrl_dic = {'A': np.full(10, 0.5)}

    # At t=0.9, t=0.7: null_counts=0  -> expected_fp=0, fdr=0
    # At t=0.3, t=0.1: null_counts=10 -> expected_fp = 4*(10/10) = 4
    #   fdr(t=0.3) = 4/3 -> clipped to 1.0
    #   fdr(t=0.1) = 4/4 = 1.0
    result = binned_empirical_fdr(
        df, ctrl_dic, stat_col='corr', ctrl_bin_col='ctrl_bin', pi0=1.0
    )
    expected = np.array([0.0, 0.0, 1.0, 1.0])
    np.testing.assert_allclose(result['q_value'].values, expected)


def test_binned_empirical_fdr_fdr_bin_weighted_sum():
    """fdr_bin_col crosses two ctrl bins; expected_fp is weighted sum."""
    df = pd.DataFrame({
        'ctrl_bin': list('AABB'),
        'fdr_bin':  list('DDDD'),
        'corr':     [0.9, 0.7, 0.5, 0.3],
    })
    # ctrl A: 4 nulls at 0.4 (below 0.5, so irrelevant above t=0.4)
    # ctrl B: 4 nulls at 0.6 (relevant at t<=0.6)
    ctrl_dic = {
        'A': np.full(4, 0.4),
        'B': np.full(4, 0.6),
    }

    # n_disc_k = {'A': 2, 'B': 2}
    # expected_fp = 2*null_A/4 + 2*null_B/4
    # t=0.9: null_A=0, null_B=0 -> efp=0, fdr=0/1=0
    # t=0.7: null_A=0, null_B=0 -> efp=0, fdr=0/2=0
    # t=0.5: null_A=0, null_B=4 -> efp=2, fdr=2/3=0.67
    # t=0.3: null_A=4, null_B=4 -> efp=4, fdr=4/4=1.0

    result = binned_empirical_fdr(
        df, ctrl_dic, stat_col='corr',
        ctrl_bin_col='ctrl_bin', fdr_bin_col='fdr_bin', pi0=1.0
    )
    expected = np.array([0.0, 0.0, 2/3, 1.0])
    np.testing.assert_allclose(result['q_value'].values, expected, rtol=1e-6)
