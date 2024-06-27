# Importing libraries
import numpy as np
import anndata as ad
import sys
import ctar


# edited 12.06.24
# indicates current node
batch = int(sys.argv[1])
# read your h5ad object
h5ad_file = '/projects/zhanglab/users/ana/multiome/processed/neatseq/neat.h5ad'
adata = ad.read_h5ad(h5ad_file)
# adjust batch size. for neat dataset, batch sizes of 1000 takes ~40mins each.
batch_size = 1273
start,end = batch_size * batch, batch_size * (batch+1)
# adjust to handle last set
if batch == 50:
    end = adata.shape[1]

atac = adata.layers['atac_raw'].A
ctrl_coeff = []
for i in np.arange(start,end):
    ctrl_coeff_i = []
    rna = adata[:,i].layers['rna_raw'].A
    ind = adata[:,i].varm['control_peaks'][0].astype(int)
    for j in np.arange(200): # generate coefficient for all controls in b
        ctrl_coeff_i.append(ctar.method.fit_poisson(atac[:,[ind[j]]],
                                                    rna,return_none=False))
    ctrl_coeff.append(ctrl_coeff_i)

np.save('poiss_ctrl_'+str(start)+'_'+str(end)+'.npy',np.array(ctrl_coeff))

