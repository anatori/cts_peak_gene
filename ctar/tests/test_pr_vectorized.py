import pytest
import numpy as np
import scipy as sp
import ctar


def load_toy_data():
    ''' Generate toy data from poisson distribution.
    '''
    np.random.seed(123)

    mat_x = np.random.poisson(size=(100,20))
    mat_y = np.random.poisson(size=(100,20))
    return mat_x, mat_y


def test_vectorized_poisson_regression():
    ''' Test vectorized (sparse-loop) PR.
    '''

    mat_x, mat_y = load_toy_data()
    links = np.column_stack([np.arange(mat_x.shape[1]), np.arange(mat_y.shape[1])])
    result = ctar.method.poisson_irls_loop_sparse(
        sp.sparse.csc_matrix(mat_x.astype(np.float64)),
        sp.sparse.csc_matrix(mat_y.astype(np.float64)),
        links=links,
        tol=1e-3,
        flag_float32=False,
    )
    v_beta0, v_beta1 = result[:, 0], result[:, 1]
    
    v_beta0_true = np.zeros((mat_x.shape[1],))
    v_beta1_true = np.zeros((mat_x.shape[1],))
    for k in range(mat_x.shape[1]):
        v_beta0_true[k], v_beta1_true[k] = ctar.method.fit_poisson(mat_x[:,k], mat_y[:,k], return_both=True)

    # v_beta0
    err_msg = (
        "rel_avg_abs_beta0_mean_dif=%0.2e"
        % (np.mean(np.abs(v_beta0_true - v_beta0)) / np.mean(np.abs(v_beta0_true)))
    )
    assert np.allclose(v_beta0, v_beta0_true, atol=1e-3, equal_nan=True), err_msg

    # v_beta1
    err_msg = (
        "rel_avg_abs_beta1_mean_dif=%0.2e"
        % (np.mean(np.abs(v_beta1_true - v_beta1)) / np.mean(np.abs(v_beta1_true)))
    )
    assert np.allclose(v_beta1, v_beta1_true, atol=1e-3, equal_nan=True), err_msg