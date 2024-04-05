### Installation
Install `ctar` in developer mode: go to the repo and use `pip install -e .`
- Functions are in `./ctar`. Do `import ctar` and call functions via `ctar.{module_name}.{function_name}`
- Make sure to add the following to make sure the changes in the package are updated in notebooks
- Experiments go to `/experiments`
- Upload code. Don't upload datasets.

The following magic lines in Jupyter Notebook will automatically update the package in the notebook after you make changes
```
%load_ext autoreload
%autoreload 2
```

# Data Exploration Notebook

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

# Methods Demo Notebook

The `methods_demo.ipynb` file contains examples of how to use some of the methods functions to carry out a mini `ctar` analysis! The following files are required for the notebook: `pbmc10k_csc.h5mu`, which is just the same as the original `pbmc10k.h5mu` file but the matrices are converted into `sp.csc_matrix` to save time.
