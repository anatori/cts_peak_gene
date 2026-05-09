import numpy as np
import anndata as ad
import muon as mu
import sys
import ctar
import os
import pandas as pd
import pybedtools
import pickle

multiome_file = ''
ctrl_dic_file = ''

bmmc_mu = mu.read(multiome_file)

with open(ctrl_dic_file, 'rb') as file:
    ctrl_links_dic = pickle.load(file)
rna_sparse = adata_rna.layers['counts']
atac_sparse = adata_atac.layers['counts']

for bin_i in bins:
    