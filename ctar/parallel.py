import numpy as np
import scipy as sp
import os
import dask
import gc
import math

from dask.distributed import Client, LocalCluster, wait
from multiprocessing import cpu_count
from typing import Union, Optional, List

from ctar.method import poisson_irls_loop


def process_bins(bin_links, atac_sparse, rna_sparse, bin_name=None, out_path=None, save_files=False, max_iter=100, tol=1e-3):
    """
    Worker task to process one bin of links, run poisson IRLS, and save output if save_files.
    """
    result = poisson_irls_loop(
        atac_sparse, rna_sparse, bin_links,
        max_iter=max_iter, tol=tol,
    )
    if save_files:
        np.save(os.path.join(out_path, f"poissonb_{bin_name}"), result[:, 1])

    return result[:, 1]


def preprocess_batch(bin_links, atac_sparse, rna_sparse):
    """
    Extract submatrices and reindex bin_links to local indices.
    Returns: (bin_links_reindexed, atac_sub, rna_sub)
    """
    needed_cols_x = np.unique(bin_links[:, 0])
    needed_cols_y = np.unique(bin_links[:, 1])

    atac_sub = atac_sparse[:, needed_cols_x]
    rna_sub = rna_sparse[:, needed_cols_y]

    # Map old indices to new submatrix-local indices
    x_map_array = np.full(needed_cols_x.max() + 1, -1, dtype=int)
    x_map_array[needed_cols_x] = np.arange(len(needed_cols_x))
    y_map_array = np.full(needed_cols_y.max() + 1, -1, dtype=int)
    y_map_array[needed_cols_y] = np.arange(len(needed_cols_y))

    bin_links_reindexed = np.column_stack((
        x_map_array[bin_links[:, 0]],
        y_map_array[bin_links[:, 1]],
    ))

    return bin_links_reindexed, atac_sub, rna_sub


def batch_links_array(ctrl_links_arr, batch_size):
    """
    Splits ctrl_links_arr (n_links, 2) into a list of arrays of size up to batch_size each.
    """
    n_links = ctrl_links_arr.shape[0]
    batches = [
        ctrl_links_arr[i:i+batch_size]
        for i in range(0, n_links, batch_size)
    ]

    return batches


def multiprocess_poisson_irls_batched(
    ctrl_links: Union[np.ndarray, List[np.ndarray]],
    atac_sparse,
    rna_sparse,
    save_files: bool = False,
    ctrl_labels_ls: Optional[List[str]] = None,
    out_path: Optional[str] = None,
    batch_size: int = 200,
    max_iter: int = 100,
    tol: float = 1e-3,
    n_workers: Optional[int] = None,
    memory_limit: Union[str, int] = "auto",
    local_directory: str = "./dask_temp",
    client: Optional[Client] = None,
):
    """
    Runs Poisson IRLS regression in parallel across batches using Dask.

    Parameters:
    ----------
    ctrl_links : np.ndarray or list of np.ndarray
        Either a single array of shape (n_links, 2) or a list of such arrays for each bin.
    atac_sparse : scipy.sparse matrix
        ATAC-seq sparse matrix (cells x peaks).
    rna_sparse : scipy.sparse matrix
        RNA-seq sparse matrix (cells x genes).
    save_files : bool, optional
        Whether to save output files for each bin.
    ctrl_labels_ls : list of str, optional
        Unique labels/names for each bin (required if save_files is True).
    out_path : str, optional
        Directory to save files (required if save_files is True).
    batch_size : int
        Number of bins to process per batch.
    max_iter : int
        Maximum iterations for IRLS.
    tol : float
        Tolerance for IRLS convergence.
    n_workers : int, optional
        Number of Dask workers. Defaults to (CPU count - 1).
    memory_limit : str or int
        Per-worker memory limit.
    local_directory : str
        Temporary directory for Dask workers.
    client : dask.distributed.Client
        If provided, uses this Dask client instead of creating a LocalCluster.

    Returns:
    -------
    list or None
        Returns a list of results if save_files is False. Otherwise, returns None.
    """

    # Validate save conditions
    if save_files:
        assert ctrl_labels_ls is not None and out_path is not None, (
            'Must provide ctrl_labels_ls and out_path when save_files=True.')

    # Create output directory if needed
    if out_path is not None and not os.path.exists(out_path):
        os.makedirs(out_path)

    # Dynamically get n_workers if not provided
    if n_workers is None:
        n_workers = max(multiprocessing.cpu_count() - 1, 1)

    # Chunk links if necessary
    if isinstance(ctrl_links, np.ndarray):
        assert ctrl_links.shape[1] == 2, (
            'Must be shape (n_links, 2), with col 0: atac idx, col 1: rna idx.')
        ctrl_links = batch_links_array(ctrl_links, batch_size)

    n_bins = len(ctrl_links)

    # Ensure all bins have labels
    if ctrl_labels_ls:
        assert len(ctrl_labels_ls) == n_bins, (
            'Length of ctrl_labels_ls must match number of bins in ctrl_links.')

    # Configure cluster if none provided
    if not client:
        is_internal_client = True
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,
            memory_limit=memory_limit,
            local_directory=local_directory,
        )
        client = Client(cluster)
    else:
        is_internal_client = False

    print(f"# Dask dashboard available at: {client.dashboard_link}", flush=True)
    print(f"# Dask scheduler address: {client.scheduler.address}", flush=True)

    total_results = []

    try:

        for i in range(0, n_bins, batch_size):

            print(f"# Processing batch {i // batch_size + 1} / {math.ceil(n_bins / batch_size)}", flush=True)

            batch_links = ctrl_links[i:i+batch_size]
            batch_labels = ctrl_labels_ls[i:i + batch_size] if ctrl_labels_ls else [None] * len(batch_links)

            futures = []

            for j, bin_links in enumerate(batch_links):

                # Map old col indices to new ones
                bin_links_reindexed, atac_sub, rna_sub = preprocess_batch(bin_links, atac_sparse, rna_sparse)

                # Scatter data to workers
                atac_sub_future = client.scatter(atac_sub, broadcast=False)
                rna_sub_future = client.scatter(rna_sub, broadcast=False)
                bin_links_future = client.scatter(bin_links_reindexed, broadcast=False)

                fut = client.submit(
                    process_bins,
                    bin_links_future,
                    atac_sub_future,
                    rna_sub_future,
                    out_path=out_path,
                    bin_name=batch_labels[j],
                    max_iter=max_iter,
                    tol=tol,
                    save_files=save_files,
                )
                futures.append(fut)

            if save_files:
                wait(futures)

            else:
                batch_results = client.gather(futures)
                total_results.extend(batch_results)

            client.cancel(futures)
            client.run(gc.collect)
            gc.collect()

    finally:
        if is_internal_client:
            client.close()
            cluster.close()

    return None if save_files else total_results

