import numpy as np
import scipy as sp

from numba import njit, prange


def poisson_irls_loop_multi_numba(mat_x_full, mat_y_full, X_multi, links, **kwargs):
    ''' Wrapper to send dense matrices to numba.
    '''
    x_cols = mat_x_full[:, links[:,0]].toarray().astype(np.float32)
    y_cols = mat_y_full[:, links[:,1]].toarray().astype(np.float32)
    return poisson_irls_loop_multi_numba_core(x_cols, y_cols, X_multi.astype(np.float32), **kwargs)


@njit(parallel=True, fastmath=True)
def poisson_irls_loop_multi_numba_core(x_cols, y_cols, X_multi, max_iter=100, tol=1e-3,
                                       ridge=False, lambda_reg=1.0):
    ''' Use prange to solve in parallel on thread, hopefully maximizing CPU use.
    Also store copy of covariates for each regression.
    '''

    n_cells, n_links = x_cols.shape
    n_cov = X_multi.shape[1]

    results = np.zeros(n_links, dtype=np.float32)

    for i in prange(n_links):
       
        # thread-local buffer for design matrix
        X_i = np.empty((n_cells, n_cov + 1), dtype=np.float32)
        X_i[:, 0] = x_cols[:, i]
        X_i[:, 1:] = X_multi

        y_i = y_cols[:, i]

        _, betas = poisson_irls_multi_numba(
            X_i, y_i, max_iter, tol,
            lambda_reg if ridge else 0.0
        )
        results[i] = betas[0]

    return results


@njit(fastmath=True)
def solve_float32_cholesky(A, b):
    """Solve A x = b for SPD float32 matrix A using Cholesky decomposition."""
    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.float32)

    # Cholesky factorization
    for i in range(n):
        for j in range(i + 1):
            s = np.float32(0.0)
            for k in range(j):
                s += L[i, k] * L[j, k]
            if i == j:
                val = A[i, i] - s
                if val <= 0.0:
                    val = np.float32(1e-6)
                L[i, j] = np.sqrt(val)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]

    # Solve L y = b
    y = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s = np.float32(0.0)
        for k in range(i):
            s += L[i, k] * y[k]
        y[i] = (b[i] - s) / L[i, i]

    # Solve L.T x = y
    x = np.zeros(n, dtype=np.float32)
    for i in range(n - 1, -1, -1):
        s = np.float32(0.0)
        for k in range(i + 1, n):
            s += L[k, i] * x[k]
        x[i] = (y[i] - s) / L[i, i]

    return x


@njit(fastmath=True)
def poisson_irls_multi_numba(X, y, max_iter=100, tol=1e-3, lambda_reg=0.0):
    """Stepwise float32 Poisson IRLS with optional ridge."""
    n, p = X.shape
    beta0 = np.float32(0.0)
    betas = np.zeros(p, dtype=np.float32)

    MAX_EXP = np.float32(80.0)
    MAX_VAL = np.float32(1e10)
    MIN_VAL = np.float32(1e-8)

    eta = np.zeros(n, dtype=np.float32)
    mu = np.zeros(n, dtype=np.float32)
    W = np.zeros(n, dtype=np.float32)
    z = np.zeros(n, dtype=np.float32)
    WX = np.zeros((n, p), dtype=np.float32)
    XtWX = np.zeros((p, p), dtype=np.float32)
    XtWz = np.zeros(p, dtype=np.float32)
    sum_x = np.zeros(p, dtype=np.float32)

    for _ in range(max_iter):
        # compute eta = beta0 + X @ betas elementwise
        for i in range(n):
            s = beta0
            for j in range(p):
                s += X[i, j] * betas[j]
            # clip eta
            if s > MAX_EXP:
                s = MAX_EXP
            elif s < -MAX_EXP:
                s = -MAX_EXP
            eta[i] = s

        # compute mu = exp(eta), clamp
        for i in range(n):
            val = np.exp(eta[i])
            if val > MAX_VAL:
                val = MAX_VAL
            elif val < MIN_VAL:
                val = MIN_VAL
            mu[i] = val
            W[i] = val
            z[i] = eta[i] + (y[i] - val) / val

        # WX = X * W[:, None]
        for i in range(n):
            wi = W[i]
            for j in range(p):
                WX[i, j] = X[i, j] * wi

        # XtWX = WX.T @ X
        for j in range(p):
            for k in range(p):
                s = np.float32(0.0)
                for i in range(n):
                    s += WX[i, j] * X[i, k]
                XtWX[j, k] = s

        # XtWz = WX.T @ z
        for j in range(p):
            s = np.float32(0.0)
            for i in range(n):
                s += WX[i, j] * z[i]
            XtWz[j] = s

        # compute sums for centering
        sum_w = np.float32(0.0)
        sum_y = np.float32(0.0)
        for i in range(n):
            sum_w += W[i]
            sum_y += W[i] * z[i]

        for j in range(p):
            s = np.float32(0.0)
            for i in range(n):
                s += WX[i, j]
            sum_x[j] = s

        # centering adjustment
        for j in range(p):
            for k in range(p):
                XtWX[j, k] -= (sum_x[j] * sum_x[k]) / max(sum_w, MIN_VAL)
        for j in range(p):
            XtWz[j] -= (sum_x[j] * sum_y) / max(sum_w, MIN_VAL)

        # add ridge penalty
        if lambda_reg > 0.0:
            for j in range(p):
                XtWX[j, j] += lambda_reg

        # numerical jitter
        for j in range(p):
            XtWX[j, j] += MIN_VAL

        # solve for betas
        betas_new = solve_float32_cholesky(XtWX, XtWz)
        beta0_new = (sum_y - np.float32(np.dot(betas_new, sum_x))) / max(sum_w, MIN_VAL)

        # convergence
        maxdiff = np.float32(0.0)
        for j in range(p):
            d = abs(betas_new[j] - betas[j])
            if d > maxdiff:
                maxdiff = d
        if maxdiff < tol and abs(beta0_new - beta0) < tol:
            return beta0_new, betas_new

        # update
        for j in range(p):
            betas[j] = betas_new[j]
        beta0 = beta0_new

    return beta0, betas
