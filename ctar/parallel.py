import numpy as np
import scipy as sp
import os
import dask
import gc
import math

from dask import delayed, compute
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster, wait, as_completed
from multiprocessing import cpu_count
from typing import Union, Optional, List, Dict
from itertools import islice

from ctar.method import poisson_irls_loop



def process_sub_batch(subatch_links, atac_sparse, rna_sparse, bin_name=None, out_path=None, save_files=False, max_iter=100, tol=1e-3):
    """
    Worker task to process one sub-batch of links, run poisson IRLS, and save output if save_files.
    """
    result = poisson_irls_loop(
        atac_sparse, rna_sparse, subatch_links,
        max_iter=max_iter, tol=tol,
    )
    if save_files: 
        np.save(os.path.join(out_path, f"poissonb_{bin_name}"), result[:, 1])

    return result[:, 1]


def preprocess_batch(subatch_links, atac_sparse, rna_sparse):
    """
    Extract submatrices and reindex subatch_links to local indices.
    Returns: (bin_links_reindexed, atac_sub, rna_sub)
    """
    needed_cols_x = np.unique(subatch_links[:, 0])
    needed_cols_y = np.unique(subatch_links[:, 1])

    atac_sub = atac_sparse[:, needed_cols_x].copy()
    rna_sub = rna_sparse[:, needed_cols_y].copy()

    # Map old indices to new submatrix-local indices
    x_map_array = np.full(needed_cols_x.max() + 1, -1, dtype=int)
    x_map_array[needed_cols_x] = np.arange(len(needed_cols_x))
    y_map_array = np.full(needed_cols_y.max() + 1, -1, dtype=int)
    y_map_array[needed_cols_y] = np.arange(len(needed_cols_y))

    links_reindexed = np.column_stack((
        x_map_array[subatch_links[:, 0]],
        y_map_array[subatch_links[:, 1]],
    ))

    return links_reindexed, atac_sub, rna_sub


def batch_links_array(link_arr, sub_batch_size):
    """
    Splits link_arr (n_links, 2) into a list of arrays of size up to sub_batch_size each.
    """
    n_links = link_arr.shape[0]
    batches = [
        link_arr[i:i+sub_batch_size]
        for i in range(0, n_links, sub_batch_size)
    ]

    return batches


def multiprocess_poisson_irls_client(
    links: Union[np.ndarray, List[np.ndarray]],
    atac_sparse,
    rna_sparse,
    save_files: bool = False,
    labels_ls: Optional[List[str]] = None,
    out_path: Optional[str] = None,
    sub_batch_size: Optional[int] = 1000,
    batch_size: int = 200,
    max_iter: int = 100,
    tol: float = 1e-3,
    n_workers: Optional[int] = None,
    client: Optional[Client] = None,
    **cluster_kwargs,
):
    """
    Runs Poisson IRLS regression in parallel across batches using Dask.

    Parameters:
    ----------
    links : np.ndarray or list of np.ndarray
        Either a single array of shape (n_links, 2) or a list of such arrays for each bin.
    atac_sparse : scipy.sparse matrix
        ATAC-seq sparse matrix (cells x peaks).
    rna_sparse : scipy.sparse matrix
        RNA-seq sparse matrix (cells x genes).
    save_files : bool, optional
        Whether to save output files for each sub-batch.
    labels_ls : list of str, optional
        Unique labels/names for each sub-batch (required if save_files is True).
    out_path : str, optional
        Directory to save files (required if save_files is True).
    sub_batch_size : int
        Number of links to process per sub-batch.
    batch_size : int
        Number of sub-batches to process per batch.
    max_iter : int
        Maximum iterations for IRLS.
    tol : float
        Tolerance for IRLS convergence.
    client : dask.distributed.Client
        If provided, uses this Dask client instead of creating a LocalCluster.
    cluster_kwargs : dict, optional
        Additional keyword arguments passed to dask.distributed.LocalCluster.

    Returns:
    -------
    np.array or None
        Returns a np.array of results if save_files is False. Otherwise, returns None.
    """

    # Validate save conditions
    if save_files:
        assert labels_ls is not None and out_path is not None, (
            'Must provide labels_ls and out_path when save_files=True.')

    # Create output directory if needed
    if out_path is not None and not os.path.exists(out_path):
        os.makedirs(out_path)

    # Dynamically get n_workers if not provided
    if n_workers is None:
        n_workers = max(cpu_count() - 1, 1)

    # Chunk links if necessary
    if isinstance(links, np.ndarray):
        assert links.shape[1] == 2, (
            'Must be shape (n_links, 2), with col 0: atac idx, col 1: rna idx.')
        links = batch_links_array(links, sub_batch_size)

    n_batch = len(links)

    # Ensure all bins have labels
    if labels_ls:
        assert len(labels_ls) == n_batch, (
            'Length of labels_ls must match number of bins in links.')

    # Configure cluster if none provided
    if not client:
        is_internal_client = True
        cluster = LocalCluster(
            n_workers=n_workers,
            **cluster_kwargs
        )
        client = Client(cluster)
    else:
        is_internal_client = False

    print(f"# Dask dashboard available at: {client.dashboard_link}", flush=True)
    print(f"# Dask scheduler address: {client.scheduler.address}", flush=True)

    total_results = []

    try:

        for i in range(0, n_batch, batch_size):

            print(f"# Processing batch {i // batch_size + 1} / {math.ceil(n_batch / batch_size)}", flush=True)

            batch_links = links[i:i+batch_size]
            batch_labels = labels_ls[i:i + batch_size] if labels_ls else [None] * len(batch_links)

            futures = []

            for j, subatch_links in enumerate(batch_links):

                # Map old col indices to new ones
                links_reindexed, atac_sub, rna_sub = preprocess_batch(subatch_links, atac_sparse, rna_sparse)

                # Scatter data to workers
                atac_sub_future = client.scatter(atac_sub, broadcast=False)
                rna_sub_future = client.scatter(rna_sub, broadcast=False)
                links_future = client.scatter(links_reindexed, broadcast=False)

                fut = client.submit(
                    process_sub_batch,
                    links_future,
                    atac_sub_future,
                    rna_sub_future,
                    out_path=out_path,
                    bin_name=batch_labels[j],
                    max_iter=max_iter,
                    tol=tol,
                    save_files=save_files,
                )
                futures.append(fut)

                # Clean up immediately
                del atac_sub, rna_sub, links_reindexed, atac_sub_future, rna_sub_future, links_future
                gc.collect()

            if save_files:
                wait(futures)

            else:
                results = client.gather(futures)
                total_results.extend(results)

            # After batch, delete futures and gather state
            client.cancel(futures)
            client.run(gc.collect)
            gc.collect()

    finally:

        if is_internal_client:
            client.shutdown()
            client.close()
            cluster.close()

    return None if save_files else np.concatenate(total_results)


def process_sub_batch_tuple(subatch_links, atac_sparse, rna_sparse, bin_name=None, out_path=None, save_files=False, max_iter=100, tol=1e-3):
    """
    Worker task to process one sub-batch of links, run poisson IRLS, and save output if save_files.
    """
    result = poisson_irls_loop(
        atac_sparse, rna_sparse, subatch_links,
        max_iter=max_iter, tol=tol,
    )
    if save_files: 
        np.save(os.path.join(out_path, f"poissonb_{bin_name}"), result[:, 1])

    return (bin_name, result[:, 1])


def multiprocess_poisson_irls(
    links_dict: Dict[str, np.ndarray],
    atac_sparse,
    rna_sparse,
    save_files: bool = False,
    out_path: Optional[str] = None,
    max_iter: int = 100,
    tol: float = 1e-3,
    n_workers: Optional[int] = None,
    **compute_kwargs,
):
    """
    Runs Poisson IRLS regression in parallel across bins using Dask.

    Parameters:
    ----------
    links_dict : dict
        Dictionary mapping bin names to link arrays (n_links, 2).
    atac_sparse : scipy.sparse matrix
        ATAC-seq sparse matrix (cells x peaks).
    rna_sparse : scipy.sparse matrix
        RNA-seq sparse matrix (cells x genes).
    save_files : bool, optional
        Whether to save output files.
    out_path : str, optional
        Directory to save files (required if save_files is True).
    max_iter : int
        Maximum iterations for IRLS.
    tol : float
        Tolerance for IRLS convergence.
    n_workers : int, optional
        Number of workers.
    compute_kwargs : dict, optional
        Additional keyword arguments passed to dask.distributed.LocalCluster.

    Returns:
    -------
    dict[str, np.ndarray] or None
        Returns a dict of results if save_files is False. Otherwise, returns None.
    """
    if save_files:
        assert out_path is not None, (
            "Must provide out_path when save_files=True.")

    if out_path is not None and not os.path.exists(out_path):
        os.makedirs(out_path)

    if n_workers is None:
        n_workers = max(cpu_count() - 1, 1)

    n_batches = len(links_dict)
    results = []

    for bin_key, link_array in links_dict.items():

        # Reindex + extract local submatrix
        links_reindexed, atac_sub, rna_sub = preprocess_batch(link_array, atac_sparse, rna_sparse)

        task = delayed(process_sub_batch_tuple)(
            links_reindexed,
            atac_sub,
            rna_sub,
            bin_name=bin_key,
            out_path=out_path,
            save_files=save_files,
            max_iter=max_iter,
            tol=tol,
        )
        results.append(task)

        del atac_sub, rna_sub, links_reindexed
        gc.collect()

    with ProgressBar():
        results = compute(*results, num_workers=n_workers, **compute_kwargs)

    if save_files:
        return None

    results_dict = {bin_key: result for bin_key, result in results}
    return results_dict


def chunked_iterable(iterable, size):
    """Yield successive chunks from iterable list of length 'size'."""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk


def multiprocess_poisson_irls_chunked(
    links_dict: Dict[str, np.ndarray],
    atac_sparse,
    rna_sparse,
    save_files: bool = False,
    out_path: Optional[str] = None,
    batch_size: int = 50,  # number of dict entries to process in parallel
    max_iter: int = 100,
    tol: float = 1e-3,
    n_workers: Optional[int] = None,
    **compute_kwargs,
):

    """
    Runs Poisson IRLS regression across batch sizes using Dask.

    Parameters:
    ----------
    links_dict : dict
        Dictionary mapping bin names to link arrays (n_links, 2).
    atac_sparse : scipy.sparse matrix
        ATAC-seq sparse matrix (cells x peaks).
    rna_sparse : scipy.sparse matrix
        RNA-seq sparse matrix (cells x genes).
    save_files : bool, optional
        Whether to save output files.
    out_path : str, optional
        Directory to save files (required if save_files is True).
    max_iter : int
        Maximum iterations for IRLS.
    tol : float
        Tolerance for IRLS convergence.
    n_workers : int, optional
        Number of workers.
    compute_kwargs : dict, optional
        Additional keyword arguments passed to dask.compute.

    Returns:
    -------
    dict[str, np.ndarray] or None
        Returns a dict of results if save_files is False. Otherwise, returns None.
    """

    if save_files:
        assert out_path is not None, (
            "Must provide out_path when save_files=True.")

    if out_path is not None and not os.path.exists(out_path):
        os.makedirs(out_path)

    if n_workers is None:
        n_workers = max(cpu_count() - 1, 1)

    results_dict = {} if not save_files else None

    with ProgressBar():

        for chunk in chunked_iterable(links_dict.items(), batch_size): # chunk_size
            tasks = []
            keys_for_chunk = []

            for bin_key, link_array in chunk:
                links_reindexed, atac_sub, rna_sub = preprocess_batch(
                    link_array, atac_sparse, rna_sparse
                )

                task = delayed(process_sub_batch)(
                    links_reindexed,
                    atac_sub,
                    rna_sub,
                    bin_name=bin_key,
                    out_path=out_path,
                    save_files=save_files,
                    max_iter=max_iter,
                    tol=tol,
                )
                tasks.append(task)
                keys_for_chunk.append(bin_key)

                del atac_sub, rna_sub, links_reindexed # might not be necessary
                gc.collect() # might not be necessary

            results = compute(*tasks, num_workers=n_workers, **compute_kwargs)

            if not save_files:
                for key, res in zip(keys_for_chunk, results):
                    results_dict[key] = res

            del tasks, results
            gc.collect()

    return results_dict
