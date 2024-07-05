import pandas as pd
import numpy as np
import scipy as sp

def gearys_c(adata, vals):
    """
    Compute Geary's C statistics for an AnnData.

    Adopted from https://github.com/ivirshup/scanpy/blob/metrics/scanpy/metrics/_gearys_c.py

    :math:`C=\\frac{(N - 1)\\sum_{i,j} w_{i,j} (x_i - x_j)^2}{2W \\sum_i (x_i - \\bar{x})^2}`

    Parameters
    ----------
    adata : AnnData object
        adata.obsp["Connectivities] should contain the connectivity graph,
        with shape (n_obs, n_obs).
    vals : array-like
        Values to calculate Geary's C for. If one dimensional, should have
        shape (n_obs,).

    Returns
    -------
    C : float
        Geary's C statistics.
    """
    graph = adata.obsp["connectivities"]
    assert graph.shape[0] == graph.shape[1]
    graph_data = graph.data.astype(np.float_, copy=False)
    assert graph.shape[0] == vals.shape[0]
    assert np.ndim(vals) == 1

    W = graph_data.sum()
    N = len(graph.indptr) - 1
    vals_bar = vals.mean()
    vals = vals.astype(np.float_)

    # numerators
    total = 0.0
    for i in range(N):
        s = slice(graph.indptr[i], graph.indptr[i + 1])
        # indices of corresponding neighbors
        i_indices = graph.indices[s]
        # corresponding connecting weights
        i_data = graph_data[s]
        total += np.sum(i_data * ((vals[i] - vals[i_indices]) ** 2))

    numer = (N - 1) * total
    denom = 2 * W * ((vals - vals_bar) ** 2).sum()
    C = numer / denom

    return C


def gearys_c_pw(adata, vals1, vals2):
    """
    Compute Geary's C statistics for an AnnData.

    Adopted from https://github.com/ivirshup/scanpy/blob/metrics/scanpy/metrics/_gearys_c.py

    :math:`C=\\frac{(N - 1)\\sum_{i,j} w_{i,j} (x_i - x_j)^2}{2W \\sum_i (x_i - \\bar{x})^2}`

    Parameters
    ----------
    adata : AnnData object
        adata.obsp["Connectivities] should contain the connectivity graph,
        with shape (n_obs, n_obs).
    vals : array-like
        Values to calculate Geary's C for. If one dimensional, should have
        shape (n_obs,).

    Returns
    -------
    C : float
        Geary's C statistics.
    """
    graph = adata.obsp["connectivities"]
    assert graph.shape[0] == graph.shape[1]
    graph_data = graph.data.astype(np.float_, copy=False)
    assert graph.shape[0] == vals1.shape[0]
    assert np.ndim(vals1) == 1

    W = graph_data.sum()
    N = len(graph.indptr) - 1
    vals_bar1 = vals1.mean()
    vals1 = vals1.astype(np.float_)
    vals_bar2 = vals2.mean()
    vals2 = vals2.astype(np.float_)

    # numerators
    total = 0.0
    for i in range(N):
        s = slice(graph.indptr[i], graph.indptr[i + 1])
        # indices of corresponding neighbors
        i_indices = graph.indices[s]
        # corresponding connecting weights
        i_data = graph_data[s]
        total += np.sum(i_data * ((vals1[i] - vals2[i_indices]) ** 2))

    numer = (N - 1) * total
    denom = 2 * W * np.sqrt(((vals1 - vals_bar1) ** 2).sum() * ((vals2 - vals_bar2) ** 2).sum())
#     print(numer, denom)
    C = numer / denom

    return C