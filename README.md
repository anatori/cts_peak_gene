# Data Exploration

The `data_exploration_cts_links.ipynb` file contains our peak-gene pair linking method and some example usage. It is split into two parts: **Methods** and **Examples**.

## Methods

This section contains useful annotated functions for our method.

## Examples

This section provides a few examples of how I used  functions found in Methods.

The following files are needed to run this section:

1. `pbmc10k.h5mu` file containing ATAC and RNA MuData objects.
2. `n_other.npy` files for n={0,11} corresponding to correlation across all peak-gene pairs across all samples EXCLUDING focal cell type *n*.
3. `n_ctrl_ct_1000x.npy` files for n={0,11} corresponding to correlation across all peak-gene pairs across samples of a focal cell type *n*.
4. `n_ctrl_other_1000x.npy` files for n={0,11} corresponding to correlation of control peaks to focal gene across all esamples EXCLUDING focal cell type *n*.

All of which can be found in my folder, `home/asprieto/10x_testing`.