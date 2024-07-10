import muon as mu
import anndata as ad
import sklearn
from tqdm import tqdm
import numpy as np
from ctar.method import pearson_corr_sparse


def smooth_vals(rna,atac,k=50):
    
    '''Weighted average of atac.X and rna.X over k nearest neighbor graph of atac
    LSI (chromatin space).
    
    Parameters
    ----------
    rna : np.array
        Array of shape (#cells,#genes).
    atac : np.array
        Array of shape (#cells,#peaks).
    k : int
        Number of neighbors.
    
    Returns
    -------
    rna_smooth, atac_smooth : np.array
        Smoothed rna and atac values.
        
    '''

    # extract data
    x = atac.X.toarray()
    y = rna.X.toarray()
    # get lsi, create if dne
    try:
        lsi = atac.obsm['X_lsi']
    except:
        print('computing ATAC LSI')
        mu.atac.tl.lsi(atac) # slow
        lsi = atac.obsm['X_lsi']

    # get knn in chromatin space (knn 1, k=50)
    # cosine implemented by scarlink
    nn = sklearn.neighbors.NearestNeighbors(n_neighbors=k, n_jobs=-1, metric='cosine')
    nn.fit(lsi)
    dists, neighs = nn.kneighbors(lsi)
    
    # weighted average
    x_smooth = np.zeros(x.shape).astype(np.float32)
    y_smooth = np.zeros(y.shape).astype(np.float32)
    for i in range(lsi.shape[0]):
        # row-wise mean (mean of all k neighbors)
        x_smooth[i] = np.mean(x[neighs[i]], axis=0)
        y_smooth[i] = np.mean(y[neighs[i]], axis=0)

    atac.layers['X_smooth'] = x_smooth
    rna.layers['X_smooth'] = y_smooth
    
    return rna, atac


def simple_corr(rna, atac, links_df, corr_cutoff=0.1):

    ''' Initial x and y pairing with pearson corr as in SHAREseq method.

    Parameters
    ----------
    rna : np.array
        Array of shape (#cells,#genes).
    atac : np.array
        Array of shape (#cells,#peaks).
    links_df : pd.DataFrame
        DataFrame containing index_x and index_y columns, where index_x 
        indicate atac indices, index_y indicate rna indices.
    corr_cutoff : float
        Minimum |corr| required to consider a link.
    
    Returns
    -------
    rna_nlinks : np.array
        Array of shape (#cells,#QC'd links).
    atac_nlinks : np.array
        Array of shape (#cells,#QC'd links).
    
    '''

    # links_df can be generated with ctar.method.peak_to_gene
    # get corresponding values
    x_links = atac[:,links_df.peak].X
    y_links = rna[:,links_df.gene].X
    
    # correlate peak-gene links
    corrs = pearson_corr_sparse(x_links,y_links)
    links_df['corr'] = corrs.flatten()
    og_links = len(links_df)
    links_df = links_df[np.abs(links_df['corr']) >= corr_cutoff]
    print('retained',len(links_df),'/',f'{og_links} links')
    
    # update indices to indices of QC'd links
    # get corresponding values of QC'd links
    atac_nlinks = atac[:,links_df.peak].copy()
    rna_nlinks = rna[:,links_df.gene].copy()

    return rna_nlinks, atac_nlinks


def chrom_potential(rna, atac, links_df, corr_cutoff=0.1, k=10, embed='lsi'):
    
    '''Compute chromatin potential.
    References: Ma Cell 2020, Mitra Nat Gen 2024.
    https://github.com/snehamitra/SCARlink/blob/main/scarlink/src/chromatin_potential.py
    
    Parameters
    ----------
    rna : np.array
        Array of shape (#cells,#genes).
    atac : np.array
        Array of shape (#cells,#peaks).
    links_df : pd.DataFrame
        DataFrame containing index_x and index_y columns, where index_x 
        indicate atac indices, index_y indicate rna indices.
    corr_cutoff : float
        Minimum |corr| required to consider a link.
    
    Returns
    -------
    dij : np.array
        Matrix of shape (#cells,#cells) containing distances between
        cell_{atac,i} and cell_{rna,j} in chromatin space (LSI).
        
    '''
    
    # smoothed values, create if dne
    try:
        atac.layers['X_smooth'].astype(np.float32)
        rna.layers['X_smooth'].astype(np.float32)
    except: 
        print('smoothing values')
        rna, atac = smooth_vals(rna, atac)

    # initial correlation to generate aligned atac, rna
    rna, atac = simple_corr(rna, atac, links_df, corr_cutoff=corr_cutoff)

    # min max scaling of smoothed x,
    # over its neighbors in lsi space
    # as implemented by scarlink
    scaler = sklearn.preprocessing.MinMaxScaler()
    x_scaled = scaler.fit_transform(atac.layers['X_smooth'])
    y_scaled = scaler.fit_transform(rna.layers['X_smooth'])
    print('scaled rna,atac')

    # get knn of smoothed scaled profiles (knn 2, k=10)
    # x_scaled should be the same size as y_scaled
    nn = sklearn.neighbors.NearestNeighbors(n_neighbors=k, n_jobs=-1, metric='correlation') 
    nn.fit(x_scaled)
    # closest rna obs to atac obs
    dists, neighs = nn.kneighbors(y_scaled)
    print('obtained neighbors')

    return dists, neighs
    
