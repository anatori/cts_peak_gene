import numpy as np
import scipy as sp
import statsmodels.api as sm

import ctar


SEED = 123
N_CELLS = 400
N_LINKS = 8
BETA0 = 0.35
BETA1_VALUES = np.array([-0.30, -0.20, -0.10, 0.00, 0.10, 0.20, 0.30, 0.40])


def simulate_poisson_pairs(
    n_cells=N_CELLS,
    beta0=BETA0,
    beta1_values=BETA1_VALUES,
    seed=SEED,
):
    """Simulate aligned peak-gene pairs under a Poisson GLM."""
    rng = np.random.default_rng(seed)

    mat_x = rng.poisson(lam=1.2, size=(n_cells, len(beta1_values))).astype(np.float64)
    mat_y = np.zeros_like(mat_x)

    for j, beta1 in enumerate(beta1_values):
        eta = beta0 + beta1 * mat_x[:, j]
        mu = np.exp(np.clip(eta, -20.0, 20.0))
        mat_y[:, j] = rng.poisson(lam=mu)

    return mat_x, mat_y


def fit_poisson_reference(x, y):
    """Reference Poisson GLM fit from statsmodels."""
    exog = sm.add_constant(x)
    result = sm.GLM(y, exog, family=sm.families.Poisson()).fit()
    return result.params[0], result.params[1], result.bse[1]


def test_poisson_beta_matches_statsmodels_dense():
    mat_x, mat_y = simulate_poisson_pairs()

    beta_hat = []
    beta_ref = []

    for j in range(mat_x.shape[1]):
        _, beta1 = ctar.method.poisson_irls_single(
            mat_x[:, j],
            mat_y[:, j],
            tol=1e-8,
            max_iter=200,
            flag_float32=False,
        )
        beta_hat.append(beta1)

        _, beta1_ref, _ = fit_poisson_reference(mat_x[:, j], mat_y[:, j])
        beta_ref.append(beta1_ref)

    beta_hat = np.asarray(beta_hat)
    beta_ref = np.asarray(beta_ref)

    assert np.allclose(beta_hat, beta_ref, atol=1e-5, rtol=1e-5), (
        "IRLS poissonb does not match statsmodels.\n"
        f"max_abs_err={np.max(np.abs(beta_hat - beta_ref)):.3e}"
    )


def test_poisson_se_matches_statsmodels_dense():
    mat_x, mat_y = simulate_poisson_pairs()

    se_hat = []
    se_ref = []

    for j in range(mat_x.shape[1]):
        se_beta1, _ = ctar.method.poisson_irls_single(
            mat_x[:, j],
            mat_y[:, j],
            tol=1e-8,
            max_iter=200,
            flag_float32=False,
            flag_se=True,
        )
        se_hat.append(se_beta1)

        _, _, se_beta1_ref = fit_poisson_reference(mat_x[:, j], mat_y[:, j])
        se_ref.append(se_beta1_ref)

    se_hat = np.asarray(se_hat)
    se_ref = np.asarray(se_ref)

    assert np.allclose(se_hat, se_ref, atol=1e-5, rtol=1e-5), (
        "IRLS se(poissonb) does not match statsmodels.\n"
        f"max_abs_err={np.max(np.abs(se_hat - se_ref)):.3e}"
    )


def test_poisson_sparse_path_matches_dense_reference():
    mat_x, mat_y = simulate_poisson_pairs()
    links = np.column_stack([np.arange(mat_x.shape[1]), np.arange(mat_y.shape[1])])

    sparse_result = ctar.method.poisson_irls_loop_sparse(
        sp.sparse.csc_matrix(mat_x),
        sp.sparse.csc_matrix(mat_y),
        links=links,
        tol=1e-8,
        max_iter=200,
        flag_float32=False,
        flag_se=True,
    )

    dense_reference = np.vstack(
        [
            ctar.method.poisson_irls_single(
                mat_x[:, j],
                mat_y[:, j],
                tol=1e-8,
                max_iter=200,
                flag_float32=False,
                flag_se=True,
            )
            for j in range(mat_x.shape[1])
        ]
    )

    assert np.allclose(sparse_result, dense_reference, atol=1e-8, rtol=1e-8), (
        "Sparse IRLS path disagrees with dense IRLS reference.\n"
        f"max_abs_err={np.max(np.abs(sparse_result - dense_reference)):.3e}"
    )
