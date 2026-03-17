import pytest
import numpy as np
import pandas as pd
import anndata as ad
import scipy as sp
import ctar


def load_toy_data():
    ''' Hard-coded toy data.
    '''
    peak_ids = [f'peak{i}' for i in range(1, 6)]
    data_matrix = np.array([
        [1, 2, 10, 20, 100],
        [2, 3, 12, 22, 101],
        [3, 4, 14, 24, 102]
    ], dtype=float)
    var_df = pd.DataFrame({'gene_ids': peak_ids}, index=peak_ids)
    adata = ad.AnnData(X=data_matrix.copy(), var=var_df.copy())
    adata.layers['atac_raw'] = sp.sparse.csr_matrix(data_matrix.copy())
    # Expected means: (1+2+3)/3=2, (2+3+4)/3=3, (10+12+14)/3=12, (20+22+24)/3=22, (100+101+102)/3=101
    expected_means = np.array([2., 3., 12., 22., 101.])
    return adata, peak_ids, expected_means, data_matrix


def test_sub_bin():
    ''' Test sub-binning function.
    '''
    df = pd.DataFrame({
        'main_bin': [0, 0, 0, 1, 1, 1],
        'value_to_bin': [10, 30, 20, 5, 15, 10],
        'id': ['a', 'b', 'c', 'd', 'e', 'f']
    })
    df_binned = ctar.method.sub_bin(df.copy(), 'main_bin', 'value_to_bin', 2, 'sub_binned_col')
    expected_sub_bins = pd.Series(['0.0', '0.1', '0.0', '1.0', '1.1', '1.0'], name='sub_binned_col')
    pd.testing.assert_series_equal(df_binned['sub_binned_col'], expected_sub_bins, check_dtype=False)


def test_get_bins_mean_var():
    ''' Test function for calculating mean, var, and mean_var bins.
    '''
    adata, _, expected_means, data_matrix = load_toy_data()
    expected_vars = np.var(data_matrix, axis=0, ddof=0)

    bins_df = ctar.method.get_bins(adata, num_bins=[2,2], type='mean_var', col='gene_ids', layer='atac_raw')
    np.testing.assert_array_almost_equal(bins_df['mean'].values, expected_means)
    np.testing.assert_array_almost_equal(bins_df['var'].values, expected_vars)

    expected_mean_var_bins = ['0.0', '0.0', '0.1', '1.1', '1.0']
    actual_mean_var_bins = list(bins_df['mean_var_bin'].values)

    err_msg = (
        "expected_mean_var_bins=%s\nactual_mean_var_bins=%s"
        % (expected_mean_var_bins, actual_mean_var_bins)
    )
    assert (actual_mean_var_bins == expected_mean_var_bins), err_msg