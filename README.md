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
