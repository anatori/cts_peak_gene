import numpy as np
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import os
import anndata as ad
import scanpy as sc
import muon as mu
import pickle

import ctar


pbmc_mu = mu.read('/projects/zhanglab/users/ana/multiome/processed/pbmc_sorted_10k/celltypes/pbmc_filt_ctmatched.h5mu')

info_file = '/projects/zhanglab/users/ana/multiome/results/ctar/final_eval/pbmc/pbmc_filtered_5.5.5.5.1000'
with open(f'{info_file}/cis_links_dic.pkl','rb') as f:
    cis_links_dic = pickle.load(f)
with open(f'{info_file}/cis_idx_dic.pkl','rb') as f:
    cis_idx_dic = pickle.load(f)
with open(f'{info_file}/ctrl_links_dic.pkl','rb') as f:
    ctrl_links_dic = pickle.load(f)
cis_links_df = pd.read_csv(f'{info_file}/cis_links_df.csv',index_col=0)

celltypes = pbmc_mu.mod['rna'].obs['celltype'].unique()

n_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
print(f'SLURM allocated {n_cores} CPUs for this job')

ctrl_dic_path = '/projects/zhanglab/users/ana/multiome/results/ctar/final_eval/pbmc/pbmc_celltype_filt_5.5.5.5.1000/ctrl_dic'

df_ls = []
batch_size = 1000

for ct in celltypes:
    ct_cis_links_df = cis_links_df.copy()
    
    mini_mu = pbmc_mu[pbmc_mu.mod['rna'].obs['celltype'] == ct]
    print(f'{ct} shape', mini_mu.shape)
    if mini_mu.shape[0] < 10:
        print(f'skipping {ct}...')
        continue

    rna_sparse = mini_mu.mod['rna'].X
    atac_sparse = mini_mu.mod['atac'].X

    cis_coeff_dic = ctar.parallel.multiprocess_poisson_irls(
        links_dict=cis_links_dic,
        atac_sparse=atac_sparse,
        rna_sparse=rna_sparse,
        batch_size=batch_size,
        scheduler="threads",
        n_workers=n_cores,
        flag_se=True,
        flag_ll=False
    )

    cis_z_dic = {key: (arr[:,1] / arr[:,0]) for key, arr in cis_coeff_dic.items()}
    cis_se_dic = {key:arr[:,0] for key,arr in cis_coeff_dic.items()}
    cis_coeff_dic = {key:arr[:,1] for key,arr in cis_coeff_dic.items()}
    
    ct_cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, cis_coeff_dic, col_name=f'{ct}_poissonb')
    ct_cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, cis_se_dic, col_name=f'{ct}_poissonb_se')
    ct_cis_links_df = ctar.data_loader.map_dic_to_df(cis_links_df, cis_idx_dic, cis_z_dic, col_name=f'{ct}_poissonb_z')

    ctrl_coeff_dic = ctar.parallel.multiprocess_poisson_irls(
        links_dict=ctrl_links_dic,
        atac_sparse=atac_sparse,
        rna_sparse=rna_sparse,
        batch_size=batch_size,
        scheduler="threads",
        n_workers=n_cores,
        flag_se=True,
        flag_ll=False
    )

    with open(f'{ctrl_dic_path}/{ct}_ctrl_coeff_dic.pkl','wb') as f:
    	pickle.dump(ctrl_coeff_dic, f)

    ctrl_z_dic = {key: (arr[:,1] / arr[:,0]) for key, arr in ctrl_coeff_dic.items()}
    ctrl_se_dic = {key:arr[:,0] for key,arr in ctrl_coeff_dic.items()}
    ctrl_coeff_dic = {key:arr[:,1] for key,arr in ctrl_coeff_dic.items()}

    mcpval_dic, ppval_dic = ctar.method.binned_mcpval(cis_coeff_dic, ctrl_coeff_dic, b=1000)
    ct_cis_links_df = ctar.data_loader.map_dic_to_df(ct_cis_links_df, cis_idx_dic, mcpval_dic, col_name=f'{ct}_mcpval')
    ct_cis_links_df = ctar.data_loader.map_dic_to_df(ct_cis_links_df, cis_idx_dic, ppval_dic, col_name=f'{ct}_ppval')
    
    mcpval_z_dic, ppval_z_dic = ctar.method.binned_mcpval(cis_z_dic, ctrl_z_dic, b=1000)
    ct_cis_links_df = ctar.data_loader.map_dic_to_df(ct_cis_links_df, cis_idx_dic, mcpval_z_dic, col_name=f'{ct}_mcpval_z')
    ct_cis_links_df = ctar.data_loader.map_dic_to_df(ct_cis_links_df, cis_idx_dic, ppval_z_dic, col_name=f'{ct}_ppval_z')

    df_ls.append(ct_cis_links_df)
    print(f'{ct} done.')

final_df = pd.concat(df_ls,axis=1)
final_path = '/projects/zhanglab/users/ana/multiome/results/ctar/final_eval/pbmc/pbmc_celltype_filt_5.5.5.5.1000'
final_df.to_csv(f'{final_path}/ct_cis_links_df.csv')
