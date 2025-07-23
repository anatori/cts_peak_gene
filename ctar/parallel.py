import numpy as np
import scipy as sp
import os
import re
import dask

from dask import delayed, compute
from dask.distributed import Client, LocalCluster
from multiprocessing import shared_memory, cpu_count
from concurrent.futures import ProcessPoolExecutor

from ctar.method import vectorized_poisson_regression_safe_converged



@delayed
def run_sparse_batch(atac_sparse, rna_sparse, atac_idx, rna_idx, tol=1e-3):
    """
    Run vectorized IRLS Poisson regression on a batch of column pairs.
    atac_sparse, rna_sparse: full CSC matrices (per worker)
    atac_idx, rna_idx: np.arrays (batch slice indices)
    Returns flattened beta1 array of length len(atac_idx).
    """
    X = atac_sparse[:, atac_idx]
    Y = rna_sparse[:, rna_idx]

    _, beta1, _ = vectorized_poisson_regression_safe_converged(X, Y, tol=tol)
    return beta1.flatten()


def run_parallel_dask(
    atac_sparse,
    rna_sparse,
    ctrl_links,
    final_corr_path,
    batch_size=5000,
    tol=1e-3,
    n_workers=None,
    threads_per_worker=1
):
    """
    Parallel Poisson regression using Dask.
    - atac_sparse, rna_sparse: CSC sparse matrices
    - ctrl_links: np.ndarray of shape (n_pairs, 2)
    - Saves batch files to final_corr_path.
    """
    # Setup cluster
    n_workers = n_workers or os.cpu_count()
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)
    client = Client(cluster)

    tasks = []
    n_pairs = ctrl_links.shape[0]

    for start in range(0, n_pairs, batch_size):
        end = min(start + batch_size, n_pairs)
        batch = ctrl_links[start:end]
        atac_idx = batch[:, 0]
        rna_idx = batch[:, 1]

        task = run_sparse_batch(atac_sparse, rna_sparse, atac_idx, rna_idx, tol)
        tasks.append((start, end, task))

    print(f"Launching {len(tasks)} Dask batch tasks for ~{n_pairs} models...")

    results = [t[2] for t in tasks]
    computed = compute(*results)

    for (start, end, _), beta_flat in zip(tasks, computed):
        fname = os.path.join(final_corr_path, f'poissonb_{start}_{end}.npy')
        np.save(fname, beta_flat)

    client.close()
    cluster.close()
    return True


def create_shared_sparse(mat):
    """
    Create shared memory blocks for scipy.sparse.csc_matrix 'mat'.
    Returns shared_memory objects, shape, and dtypes.
    """
    assert sp.sparse.isspmatrix_csc(mat), "Only CSC sparse format supported"

    shm_data = shared_memory.SharedMemory(create=True, size=mat.data.nbytes)
    shm_indices = shared_memory.SharedMemory(create=True, size=mat.indices.nbytes)
    shm_indptr = shared_memory.SharedMemory(create=True, size=mat.indptr.nbytes)

    np.ndarray(mat.data.shape, dtype=mat.data.dtype, buffer=shm_data.buf)[:] = mat.data
    np.ndarray(mat.indices.shape, dtype=mat.indices.dtype, buffer=shm_indices.buf)[:] = mat.indices
    np.ndarray(mat.indptr.shape, dtype=mat.indptr.dtype, buffer=shm_indptr.buf)[:] = mat.indptr

    return (shm_data, shm_indices, shm_indptr), mat.shape, mat.data.dtype, mat.indices.dtype, mat.indptr.dtype


def attach_shared_sparse(shm_names, shape, dtype_data, dtype_indices, dtype_indptr):
    """
    Attach to shared memory blocks and reconstruct sparse matrix.
    shm_names: tuple/list of 3 shared_memory names
    """
    shm_data = shared_memory.SharedMemory(name=shm_names[0])
    shm_indices = shared_memory.SharedMemory(name=shm_names[1])
    shm_indptr = shared_memory.SharedMemory(name=shm_names[2])

    data = np.ndarray((shm_data.size // np.dtype(dtype_data).itemsize,), dtype=dtype_data, buffer=shm_data.buf)
    indices = np.ndarray((shm_indices.size // np.dtype(dtype_indices).itemsize,), dtype=dtype_indices, buffer=shm_indices.buf)
    indptr = np.ndarray((shm_indptr.size // np.dtype(dtype_indptr).itemsize,), dtype=dtype_indptr, buffer=shm_indptr.buf)

    mat = sp.sparse.csc_matrix((data, indices, indptr), shape=shape)
    return mat, (shm_data, shm_indices, shm_indptr)


def process_ctrl_file_shared_files(args):
    """
    Worker: attach to shared sparse matrices, slice columns, run Poisson regression.
    """
    (ctrl_file_idx, ctrl_files, atac_shm_names, rna_shm_names,
     atac_shape, rna_shape, atac_dtypes, rna_dtypes,
     ctrl_path, BIN_CONFIG, n_ctrl, final_corr_path) = args

    try:
        # Attach to shared sparse matrices
        atac_mat, atac_shms = attach_shared_sparse(atac_shm_names, atac_shape, *atac_dtypes)
        rna_mat, rna_shms = attach_shared_sparse(rna_shm_names, rna_shape, *rna_dtypes)

        ctrl_file = ctrl_files[ctrl_file_idx]
        ctrl_links = np.load(os.path.join(ctrl_path, f'ctrl_links_{BIN_CONFIG}', ctrl_file))[:n_ctrl]

        # Slice sparse matrices with ctrl_links indices
        atac_data = atac_mat[:, ctrl_links[:, 0]]
        rna_data = rna_mat[:, ctrl_links[:, 1]]

        # Run your Poisson regression function (expects sparse matrices)
        _, ctrl_poissonb, _ = vectorized_poisson_regression_safe_converged(atac_data, rna_data, tol=1e-3)

        # Save output
        output_path = os.path.join(final_corr_path, f'poissonb_{os.path.basename(ctrl_file)}')
        np.save(output_path, ctrl_poissonb.flatten())

        # Cleanup
        for shm in atac_shms + rna_shms:
            shm.close()

        return f"# Completed {ctrl_file}"

    except Exception as e:
        return f"Error processing {ctrl_file}: {str(e)}"


def process_ctrl_links_shared_memory(args):
    """
    Worker for in-memory ctrl_links instead of file-based ones.
    """
    (ctrl_link_idx, ctrl_links_list,
     atac_shm_names, rna_shm_names,
     atac_shape, rna_shape,
     atac_dtypes, rna_dtypes,
     final_corr_path) = args

    try:
        atac_mat, atac_shms = attach_shared_sparse(atac_shm_names, atac_shape, *atac_dtypes)
        rna_mat, rna_shms = attach_shared_sparse(rna_shm_names, rna_shape, *rna_dtypes)

        ctrl_links = ctrl_links_list[ctrl_link_idx]

        atac_data = atac_mat[:, ctrl_links[:, 0]]
        rna_data = rna_mat[:, ctrl_links[:, 1]]

        _, ctrl_poissonb, _ = vectorized_poisson_regression_safe_converged(atac_data, rna_data, tol=1e-3)

        output_path = os.path.join(final_corr_path, f'poissonb_ctrl_{ctrl_link_idx}.npy')
        np.save(output_path, ctrl_poissonb.flatten())

        for shm in atac_shms + rna_shms:
            shm.close()

        return f"# Completed poissonb_ctrl_{ctrl_link_idx}.npy"

    except Exception as e:
        return f"Error in poissonb_ctrl_{ctrl_link_idx}: {str(e)}"


def parallel_poisson_shared_sparse(
    ctrl_path, BIN_CONFIG, start, end, n_ctrl,
    adata_atac, adata_rna, final_corr_path,
    n_jobs=-1, mode='file', links_arr=None
):
    """
    Parallel Poisson regression using shared sparse memory to avoid dense conversion and copying.
    Note that slicing still creates copies, but avoiding copying anndata reduces memory usage. 

    Parameters
    ----------
    ctrl_path : str
        Path to control files or data.
    BIN_CONFIG : str
        Binning configuration name.
    start : int
        Start index of ctrl files or in-memory links.
    end : int
        End index (non-inclusive).
    n_ctrl : int
        Number of control features.
    adata_atac : AnnData
        ATAC matrix (counts in .layers).
    adata_rna : AnnData
        RNA matrix (counts in .layers).
    final_corr_path : str
        Output path to save poisson results.
    n_jobs : int
        Number of workers.
    mode : str, default='file'
        'file' for disk-based ctrl_links; 'memory' for in-memory ctrl_links.
    links_arr : np.ndarray, optional
        Used only if mode='memory'. Should be shape (n_pairs, 2), where links are rows.
    """

    # Extract sparse matrices in CSC format
    atac_sparse = adata_atac.layers['counts']
    rna_sparse = adata_rna.layers['counts']
    if not sp.sparse.isspmatrix_csc(atac_sparse):
        atac_sparse = atac_sparse.tocsc()
    if not sp.sparse.isspmatrix_csc(rna_sparse):
        rna_sparse = rna_sparse.tocsc()

    # CPU fallback
    if n_jobs is None or n_jobs <= 0:
        n_jobs = os.cpu_count() or 1

    # Shared memory for ATAC and RNA matrices
    atac_shms, atac_shape, *atac_dtypes = create_shared_sparse(atac_sparse)
    rna_shms, rna_shape, *rna_dtypes = create_shared_sparse(rna_sparse)

    results = []

    try:
        if mode == 'file':

            ctrl_files_all = os.listdir(os.path.join(ctrl_path, f'ctrl_links_{BIN_CONFIG}'))
            ctrl_files = ctrl_files_all[start:end]

            args_list = [
                (
                    i, ctrl_files,
                    (atac_shms[0].name, atac_shms[1].name, atac_shms[2].name),
                    (rna_shms[0].name, rna_shms[1].name, rna_shms[2].name),
                    atac_shape, rna_shape,
                    (atac_dtypes[0], atac_dtypes[1], atac_dtypes[2]),
                    (rna_dtypes[0], rna_dtypes[1], rna_dtypes[2]),
                    ctrl_path, BIN_CONFIG, n_ctrl, final_corr_path
                )
                for i in range(len(ctrl_files))
            ]

            print(f"# Processing {len(ctrl_files)} control files with shared sparse memory...")

            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                results = list(executor.map(process_ctrl_file_shared_files, args_list))

        elif mode == 'memory':

            if links_arr is None:
                raise ValueError("links_arr must be provided in 'memory' mode.")

            ctrl_peaks = np.load(os.path.join(ctrl_path, f'ctrl_peaks_{BIN_CONFIG}.npy'))
            adata_atac.varm['ctrl_peaks'] = ctrl_peaks[:, :n_ctrl]
            adata_rna.var['ind'] = range(len(adata_rna.var))

            links_arr = links_arr[:, start:end]
            ctrl_peaks = adata_atac[:, links_arr[0]].varm['ctrl_peaks']
            ctrl_genes = adata_rna[:, links_arr[1]].var['ind'].values

            # Build ctrl_links in-memory
            ctrl_links_list = [
                np.vstack((ctrl_peaks[i], np.full(n_ctrl, ctrl_genes[i]))).T
                for i in range(end - start)
            ]

            args_list = [
                (
                    i, ctrl_links_list,
                    (atac_shms[0].name, atac_shms[1].name, atac_shms[2].name),
                    (rna_shms[0].name, rna_shms[1].name, rna_shms[2].name),
                    atac_shape, rna_shape,
                    (atac_dtypes[0], atac_dtypes[1], atac_dtypes[2]),
                    (rna_dtypes[0], rna_dtypes[1], rna_dtypes[2]),
                    final_corr_path
                )
                for i in range(len(ctrl_links_list))
            ]

            print(f"# Processing {len(ctrl_links_list)} in-memory control links with shared sparse memory...")

            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                results = list(executor.map(process_ctrl_links_shared_memory, args_list))

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        print(f"# Successful: {len([r for r in results if 'Completed' in r])} / {len(results)}")

    finally:
        # Cleanup
        for shm in atac_shms + rna_shms:
            shm.close()
            shm.unlink()

    return results