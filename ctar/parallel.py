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

from ctar.method import poisson_irls_loop, poisson_irls_loop_multi, poisson_irls_loop_sparse, pearson_corr_sparse, pearson_corr_within_donor_sparse
from ctar.celltype_specificity import joint_entropy_specificity, permutation_replicate_stats, link_score_matrix, coexp_specificity



def process_sub_batch(subatch_links, atac_sparse, rna_sparse, 
    bin_name=None, 
    out_path=None, 
    save_files=False, 
    max_iter=100, 
    tol=1e-3, 
    ridge=False, 
    flag_se = False, 
    flag_ll = False,
    **irls_kwargs
):
    """
    Worker task to process one sub-batch of links, run poisson IRLS, and save output if save_files.
    Note: now using sparse-ware IRLS. NOT compatible with ridge.
    """
    result = poisson_irls_loop_sparse(
        atac_sparse, rna_sparse, subatch_links,
        max_iter=max_iter, tol=tol, ridge=ridge, flag_se=flag_se, flag_ll=flag_ll,
        **irls_kwargs,
    )
    if save_files: 
        np.save(os.path.join(out_path, f"poissonb_{bin_name}"), result[:, 1])

    if flag_se or flag_ll:
        # TODO documentation
        # dictionary will be nx2 arrays where col1
        # flag_se -- col1 : se_beta, col2: beta
        # flag_ll -- col1 : loglikelihood, col2: beta
        return result
    
    # beta only
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

    links_reindexed = np.column_stack((
        np.searchsorted(needed_cols_x, subatch_links[:, 0]),
        np.searchsorted(needed_cols_y, subatch_links[:, 1]),
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


def chunk_links_dict(link_arr, dic_batch_size=512):
    """
    Splits link_arr (n_links, 2) into a dict {i: chunk} of arrays of size up to dic_batch_size each.
    """
    return {
        i: link_arr[i * dic_batch_size: (i + 1) * dic_batch_size]
        for i in range((len(link_arr) + dic_batch_size - 1) // dic_batch_size)
    }


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

                del atac_sub, rna_sub, links_reindexed, atac_sub_future, rna_sub_future, links_future

            if save_files:
                wait(futures)

            else:
                results = client.gather(futures)
                total_results.extend(results)

            client.cancel(futures)
            gc.collect()

    finally:

        if is_internal_client:
            client.shutdown()
            client.close()
            cluster.close()

    return None if save_files else np.concatenate(total_results)


def batched_iterable(iterable, size):
    """Yield successive batches from iterable list of length 'size'."""
    it = iter(iterable)
    while True:
        batch = list(islice(it, size))
        if not batch:
            break
        yield batch


def multiprocess_poisson_irls(
    links_dict: Dict[str, np.ndarray],
    atac_sparse,
    rna_sparse,
    save_files: bool = False,
    out_path: Optional[str] = None,
    batch_size: int = 50,  # number of dict entries to process in parallel
    max_iter: int = 100,
    tol: float = 1e-3,
    n_workers: Optional[int] = None,
    ridge: bool = False,
    flag_float32: bool = True, 
    flag_se: bool = False,
    flag_ll: bool = False,
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
    n_total = math.ceil(len(links_dict) / batch_size)

    with ProgressBar():

        for batch_idx, batch in enumerate(batched_iterable(links_dict.items(), batch_size), start=1):
            
            print(f"# Processing batch {batch_idx} / {n_total}", flush=True)
            tasks = []
            keys_for_batch = []

            for bin_key, link_array in batch:
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
                    ridge=ridge,
                    flag_float32=flag_float32, 
                    flag_se=flag_se,
                    flag_ll=flag_ll,
                )
                tasks.append(task)
                keys_for_batch.append(bin_key)

                del atac_sub, rna_sub, links_reindexed

            results = compute(*tasks, num_workers=n_workers, **compute_kwargs)

            if not save_files:
                for key, res in zip(keys_for_batch, results):
                    results_dict[key] = res

            del tasks, results
            gc.collect()

    return results_dict


def _pearson_corr_worker(subatch_links, atac_sparse, rna_sparse):
    """Worker: compute Pearson correlation for a batch of link pairs."""
    atac = atac_sparse[:, subatch_links[:, 0]]
    rna  = rna_sparse[:,  subatch_links[:, 1]]
    return pearson_corr_sparse(rna, atac).flatten()


def _pearson_corr_within_donor_worker(subatch_links, atac_sparse, rna_sparse, donor_labels):
    """Worker: compute within-donor Pearson correlation for a batch of link pairs."""
    atac = atac_sparse[:, subatch_links[:, 0]]
    rna  = rna_sparse[:,  subatch_links[:, 1]]
    return pearson_corr_within_donor_sparse(atac, rna, donor_labels).flatten()


def multiprocess_pearson_corr(
    links_dict: Dict[str, np.ndarray],
    atac_sparse,
    rna_sparse,
    batch_size: int = 50,
    n_workers: Optional[int] = None,
    donor_labels=None,
    **compute_kwargs,
):
    """Parallel Pearson correlation across link batches using Dask.

    Parameters
    ----------
    links_dict : dict
        Dictionary mapping batch/bin keys to link arrays of shape (n_links, 2).
        ATAC index in column 0, RNA index in column 1.
    atac_sparse : scipy.sparse matrix
        ATAC-seq matrix (cells x peaks).
    rna_sparse : scipy.sparse matrix
        RNA-seq matrix (cells x genes).
    batch_size : int
        Number of dict entries to process per Dask batch.
    n_workers : int, optional
        Number of parallel workers.
    donor_labels : array-like, shape (n_cells,), optional
        If provided, computes within-donor Pearson correlation (donor fixed effects)
        without materializing a dense matrix.

    Returns
    -------
    dict[key, np.ndarray]
        Mapping from link batch key to 1-D correlation array of shape (n_links,).
    """
    if n_workers is None:
        n_workers = max(cpu_count() - 1, 1)

    results_dict = {}
    n_total = math.ceil(len(links_dict) / batch_size)

    with ProgressBar():
        for batch_idx, batch in enumerate(batched_iterable(links_dict.items(), batch_size), start=1):
            print(f"# Pearson batch {batch_idx} / {n_total}", flush=True)
            tasks, keys = [], []

            for bin_key, link_array in batch:
                links_re, atac_sub, rna_sub = preprocess_batch(link_array, atac_sparse, rna_sparse)
                if donor_labels is not None:
                    tasks.append(delayed(_pearson_corr_within_donor_worker)(links_re, atac_sub, rna_sub, donor_labels))
                else:
                    tasks.append(delayed(_pearson_corr_worker)(links_re, atac_sub, rna_sub))
                keys.append(bin_key)
                del atac_sub, rna_sub, links_re

            results = compute(*tasks, num_workers=n_workers, **compute_kwargs)
            for k, r in zip(keys, results):
                results_dict[k] = r
            del tasks, results
            gc.collect()

    return results_dict


def process_sub_batch_multi(subatch_links, atac_sparse, rna_sparse, covar_mat, bin_name=None, out_path=None, save_files=False, **irls_kwargs):
    """
    Worker task to process one sub-batch of links, run poisson IRLS, and save output if save_files.
    """
    result = poisson_irls_loop_multi(
        atac_sparse, rna_sparse, covar_mat, subatch_links,
        **irls_kwargs
    )
    if save_files:
        np.save(os.path.join(out_path, f"poissonb_{bin_name}"), result)
    return result


def multiprocess_poisson_irls_multivar(
    links_dict: Dict[str, np.ndarray],
    atac_sparse,
    rna_sparse,
    covar_mat: np.ndarray,
    save_files: bool = False,
    out_path: Optional[str] = None,
    batch_size: int = 50,
    max_iter: int = 100,
    tol: float = 1e-3,
    n_workers: Optional[int] = None,
    ridge: bool = False,
    lambda_reg: float = 1.0,
    flag_se: bool = False,
    **compute_kwargs,
):
    """
    Multiprocess Poisson IRLS regression for multi-covariates.

    Parameters
    ----------
    links_dict : dict
        Dictionary mapping bin names to link arrays (n_links, 2)
        ATAC [x] should be in position [:,0]
        RNA [y] should be in position [:,1]
    atac_sparse : scipy.sparse matrix
        ATAC-seq sparse matrix (cells x peaks)
    rna_sparse : scipy.sparse matrix
        RNA-seq sparse matrix (cells x genes)
    covar_mat : np.array, shape (n_cells, n_covariates)
        Covariates shared for all links
    ridge : bool
        Whether to include Ridge penalty
    lambda_reg : float
        Ridge regularization strength
    save_files : bool
        Whether to save each bin result
    out_path : str
        Directory to save files
    batch_size : int
        Number of bins per Dask batch
    max_iter, tol : IRLS parameters
    n_workers : int
        Number of workers
    compute_kwargs : dict
        Additional arguments passed to dask.compute

    Returns
    -------
    dict[str, np.ndarray] or None
        Mapping bin -> IRLS results (n_links x n_covariates+1)

    TODO:
    - implement return se + beta
    """

    if save_files:
        assert out_path is not None, "Must provide out_path when save_files=True."
        os.makedirs(out_path, exist_ok=True)

    if n_workers is None:
        n_workers = max(cpu_count() - 1, 1)

    results_dict = {} if not save_files else None
    n_total = math.ceil(len(links_dict) / batch_size)

    with ProgressBar():

        for batch_idx, batch in enumerate(batched_iterable(links_dict.items(), batch_size), start=1):

            print(f"# Processing batch {batch_idx} / {n_total}", flush=True)
            tasks = []
            keys_for_batch = []

            for bin_key, link_array in batch:
                links_reindexed, atac_sub, rna_sub = preprocess_batch(link_array, atac_sparse, rna_sparse)

                task = delayed(process_sub_batch_multi)(
                    links_reindexed,
                    atac_sub,
                    rna_sub,
                    covar_mat,
                    bin_name=bin_key,
                    out_path=out_path,
                    save_files=save_files,
                    max_iter=max_iter,
                    tol=tol,
                    ridge=ridge,
                    lambda_reg=lambda_reg,
                    flag_se=flag_se,
                )
                tasks.append(task)
                keys_for_batch.append(bin_key)

                del atac_sub, rna_sub, links_reindexed

            results = compute(*tasks, num_workers=n_workers, **compute_kwargs)

            if not save_files:
                for key, res in zip(keys_for_batch, results):
                    results_dict[key] = res

            del tasks, results
            gc.collect()

    return results_dict


def multiprocess_joint_entropy_specificity(
    links,
    rna_table,
    atac_table,
    chunk_size: int = 200000,
    batch_size: int = 50,
    n_workers: Optional[int] = None,
    array_chunk_size: int = 1000000,
    **compute_kwargs,
):
    """Parallel joint entropy specificity across link batches using Dask.

    Needs no cell submatrices, only the two small cell-type tables, so
    `scheduler="threads"` shares them read-only across workers! Logs of the tables are
    taken once here, not once per batch.

    Parameters
    ----------
    links : dict or np.ndarray
        Either a dictionary mapping batch/bin keys to link arrays of shape (n_links, 2),
        or a single (n_links, 2) array. ATAC index in column 0, RNA index in column 1.
    rna_table, atac_table : pd.DataFrame or np.ndarray
        (n_genes, K) and (n_peaks, K) smoothed cell-type tables from
        ctar.celltype_specificity.celltype_tables().
    chunk_size : int
        Links scored per block inside each batch (memory bound).
    batch_size : int
        Number of dict entries to process per Dask batch.
    n_workers : int, optional
        Number of parallel workers.
    array_chunk_size : int
        Array input only: links per dict entry after chunking.

    Returns
    -------
    dict[key, np.ndarray] for dict input, mapping each key to a 1-D specificity array of
    shape (n_links,). A single 1-D array, in input order, for array input.
    """
    if n_workers is None:
        n_workers = max(cpu_count() - 1, 1)
    compute_kwargs.setdefault("scheduler", "threads")

    return_array = isinstance(links, np.ndarray)
    links_dict = chunk_links_dict(links, dic_batch_size=array_chunk_size) if return_array else links

    # table logs once here, not once per batch
    rna_values = np.asarray(rna_table, dtype=np.float64)
    atac_values = np.asarray(atac_table, dtype=np.float64)
    rna_log = np.log(rna_values, where=rna_values > 0, out=np.zeros_like(rna_values))
    atac_log = np.log(atac_values, where=atac_values > 0, out=np.zeros_like(atac_values))

    specificity_by_key = {}
    n_total = math.ceil(len(links_dict) / batch_size)

    with ProgressBar():
        for batch_idx, batch in enumerate(batched_iterable(links_dict.items(), batch_size), start=1):
            print(f"# Specificity batch {batch_idx} / {n_total}", flush=True)
            batch_keys = [bin_key for bin_key, _ in batch]
            tasks = [
                delayed(joint_entropy_specificity)(
                    rna_values, atac_values, link_array[:, 1], link_array[:, 0], chunk_size=chunk_size,
                    rna_log=rna_log, atac_log=atac_log,
                )
                for _, link_array in batch
            ]
            batch_results = compute(*tasks, num_workers=n_workers, **compute_kwargs)
            for bin_key, result in zip(batch_keys, batch_results):
                specificity_by_key[bin_key] = result
            del tasks, batch_results
            gc.collect()

    if return_array:
        if not specificity_by_key:
            return np.empty(0, dtype=np.float64)
        return np.concatenate([specificity_by_key[key] for key in sorted(specificity_by_key)])
    return specificity_by_key


def multiprocess_coexp_specificity(
    links,
    context,
    chunk_size: int = 100000,
    batch_size: int = 50,
    n_workers: Optional[int] = None,
    array_chunk_size: int = 1000000,
    **compute_kwargs,
):
    """Parallel pseudobulk-of-co-expression specificity across link batches using Dask.

    Same input and output layout as multiprocess_joint_entropy_specificity, plus the dominant cell type.
    Unlike the joint score, this needs the cell-level log1p matrices (per-cell products), built once in
    ctar.celltype_specificity.coexp_context() and shared read-only: the default `scheduler="threads"`
    avoids copying them to every worker (the sparse slicing / products run in compiled scipy code).

    Parameters
    ----------
    links : dict or np.ndarray
        Either a dictionary mapping batch/bin keys to link arrays of shape (n_links, 2),
        or a single (n_links, 2) array. ATAC index in column 0, RNA index in column 1.
    context : dict
        From ctar.celltype_specificity.coexp_context().
    chunk_size : int
        Links per sparse product inside each task (memory bound).
    batch_size : int
        Number of dict entries per Dask batch.
    n_workers : int, optional
        Number of parallel workers.
    array_chunk_size : int
        Array input only: links per dict entry after chunking.

    Returns
    -------
    specificity, dominant_code
        For dict input, two dicts mapping each key to (n_links,) arrays (float64 specificity, int
        dominant cell-type code; -1 = none). For array input, two 1-D arrays in input order.
        Codes index context['celltypes'].
    """
    if n_workers is None:
        n_workers = max(cpu_count() - 1, 1)
    compute_kwargs.setdefault("scheduler", "threads")

    return_array = isinstance(links, np.ndarray)
    links_dict = chunk_links_dict(links, dic_batch_size=array_chunk_size) if return_array else links

    specificity_by_key, dominant_by_key = {}, {}
    n_total = math.ceil(len(links_dict) / batch_size)

    with ProgressBar():
        for batch_idx, batch in enumerate(batched_iterable(links_dict.items(), batch_size), start=1):
            print(f"# Coexp specificity batch {batch_idx} / {n_total}", flush=True)
            batch_keys = [bin_key for bin_key, _ in batch]
            tasks = [
                delayed(coexp_specificity)(context, link_array[:, 1], link_array[:, 0], chunk_size=chunk_size)
                for _, link_array in batch
            ]
            batch_results = compute(*tasks, num_workers=n_workers, **compute_kwargs)
            for bin_key, (specificity, dominant) in zip(batch_keys, batch_results):
                specificity_by_key[bin_key] = specificity
                dominant_by_key[bin_key] = dominant
            del tasks, batch_results
            gc.collect()

    if return_array:
        if not specificity_by_key:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.intp)
        keys = sorted(specificity_by_key)
        return (np.concatenate([specificity_by_key[k] for k in keys]),
                np.concatenate([dominant_by_key[k] for k in keys]))
    return specificity_by_key, dominant_by_key


def multiprocess_link_specificity(
    links,
    spec_stat,
    rna_table,
    atac_table,
    coexp_context=None,
    batch_size: int = 50,
    n_workers: Optional[int] = None,
):
    """Link specificity for an (n_links, 2) link array or a dict of them, by statistic.

    Parameters
    ----------
    links : dict or np.ndarray
        As in multiprocess_joint_entropy_specificity / multiprocess_coexp_specificity.
    spec_stat : str
        'coexp_log1p' (pseudobulk of co-expression; needs coexp_context) or 'joint'.
    rna_table, atac_table
        Cell-type tables from ctar.celltype_specificity.celltype_tables() (used by 'joint').
    coexp_context : dict, optional
        From ctar.celltype_specificity.coexp_context() (used by 'coexp_log1p').
    batch_size, n_workers
        Passed through.

    Returns
    -------
    specificity, dominant_code
        In the input's layout. dominant_code indexes coexp_context['celltypes'] (-1 = none) for 'coexp_log1p'
        and is None for 'joint' (its dominant cell types come from celltype_specificity.dominant_celltypes()).
    """
    if spec_stat == "joint":
        specificity = multiprocess_joint_entropy_specificity(
            links, rna_table, atac_table, batch_size=batch_size, n_workers=n_workers)
        return specificity, None
    return multiprocess_coexp_specificity(links, coexp_context, batch_size=batch_size, n_workers=n_workers)


def multiprocess_permutation_null(
    rna_adata,
    atac_adata,
    gene_names,
    peak_names,
    group_key="celltype",
    n_perm=100,
    stats=("link_specificity_joint", "link_specificity_cellz",
           "gene_specificity", "peak_specificity"),
    eps_rna=1e-6,
    eps_atac=1e-6,
    seed=None,
    batch_size=200,
    n_workers=None,
    **compute_kwargs,
):
    """
    Multiprocess ctar.celltype_specificity.permutation_replicate_stats across
    n_perm replicates. Builds link_score_matrix's W/cols and the group_key category 
    list once (shared read-only across every replicate, not rebuilt per task) 
    if 'link_specificity_cellz' is in stats. `scheduler="threads"` is suggested.

    Parameters
    ----------
    rna_adata, atac_adata, gene_names, peak_names, group_key, stats,
    eps_rna, eps_atac
        Passed through to permutation_replicate_stats for every replicate.
    n_perm
        Number of permutation replicates.
    seed
        Parent seed; each replicate draws an independent child stream
        (np.random.SeedSequence.spawn) so results reproduce regardless of the
        order parallel workers finish in.
    batch_size
        Number of replicates per Dask batch.
    n_workers
        Number of parallel workers.

    Returns
    -------
    {stat: (n_perm, n_links) ndarray}.
    """
    if n_workers is None:
        n_workers = max(cpu_count() - 1, 1)
    compute_kwargs.setdefault("scheduler", "threads")

    W = cols = cats = None
    if "link_specificity_cellz" in stats:
        W, cols = link_score_matrix(rna_adata, atac_adata, gene_names, peak_names)
        labels = rna_adata.obs[group_key]
        cats = labels.cat.categories if hasattr(labels, "cat") else labels.astype("category").cat.categories

    child_seeds = np.random.SeedSequence(seed).spawn(n_perm)
    rows = [None] * n_perm
    n_total = math.ceil(n_perm / batch_size)

    with ProgressBar():
        for batch_idx, batch_idxs in enumerate(batched_iterable(range(n_perm), batch_size), start=1):
            print(f"# Permutation batch {batch_idx} / {n_total}", flush=True)
            tasks = [
                delayed(permutation_replicate_stats)(
                    child_seeds[i], rna_adata, atac_adata, gene_names, peak_names,
                    group_key=group_key, stats=stats, eps_rna=eps_rna, eps_atac=eps_atac,
                    W=W, cols=cols, cats=cats,
                )
                for i in batch_idxs
            ]
            batch_results = compute(*tasks, num_workers=n_workers, **compute_kwargs)
            for i, r in zip(batch_idxs, batch_results):
                rows[i] = r
            del tasks, batch_results
            gc.collect()

    return {s: np.stack([r[s] for r in rows], axis=0) for s in stats}
