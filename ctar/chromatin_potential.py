import muon as mu
import anndata as ad
import sklearn
from tqdm import tqdm
import numpy as np
from ctar.method import pearson_corr_sparse
from ctar.data_loader import get_gene_coords


def smooth_vals(atac, y_obs, y_pred, atac_layer=None, yobs_layer=None,k=50):
    
    '''Weighted average of y_obs.X and y_pred.X over k nearest neighbor graph of atac
    LSI (chromatin space).
    
    Parameters
    ----------
    atac : ad.AnnData
        AnnData to compute LSI, of shape (#cells,#peaks).
    y_obs : ad.AnnData
        AnnData with RNA information, of shape (#cells,#genes).
    y_pred : ad.AnnData
        AnnData with RNA information predicted from atac, of 
        shape (#cells,#genes).
    k : int
        Number of neighbors.
    
    Returns
    -------
    y_obs, y_pred : ad.AnnData
        AnnData updated with X_smooth layer.
        
    '''

    # extract data
    if sp.sparse.issparse(y_obs.X):
        y_obs_arr = y_obs.X.toarray()
    else: y_obs_arr = y_obs.X
    if sp.sparse.issparse(y_pred.X):
        y_pred_arr = y_pred.X.toarray()
    else: y_pred_arr = y_pred.X

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
    y_obs_smooth = np.zeros(y_obs_arr.shape).astype(np.float32)
    y_pred_smooth = np.zeros(y_pred_arr.shape).astype(np.float32)
    for i in range(lsi.shape[0]):
        # row-wise mean (mean of all k neighbors)
        y_obs_smooth[i] = np.mean(y_obs_arr[neighs[i]], axis=0)
        y_pred_smooth[i] = np.mean(y_pred_arr[neighs[i]], axis=0)

    y_obs.layers['X_smooth'] = y_obs_smooth
    y_pred.layers['X_smooth'] = y_pred_smooth
    
    return y_obs, y_pred


def gene_activity(atac, genes_df, gene_id_type='ensembl_gene_id',gene_col='gene'):

    ''' Predict gene counts by summing ATAC fragments in gene region.
    Adapted from https://stuartlab.org/signac/reference/geneactivity

    Parameters
    ----------
    atac : ad.AnnData
        AnnData of shape (#cells,#peaks).
    genes_df : pd.DataFrame
        DataFrame containing gene names of length (#genes).
    gene_id_type,gene_col : str
        Passed into get_gene_coords.
    
    Returns
    -------
    y_pred : ad.AnnData
        AnnData with predicted RNA expression of shape (#cells,#genes).
    
    '''

    # note that count_fragments_features requires 'Chromosome' column
    # sometimes 10x + muon processed rna.var will have gene body already
    if 'interval' in genes_df.columns:
        genebed_df = genes_df.copy()
        genebed_df[['Chromosome','start','end']] = genebed_df.interval.str.split(':|-',expand=True)

    else:
        genebed_df = get_gene_coords(genes_df,col_names=['Chromosome','start','end'],gene_id_type=gene_id_type,gene_col=gene_col)
    # remove unsuccessful searches
    genebed_df = genebed_df[~genebed_df.start.isna()].copy()
    # chrM is not tolerated
    genebed_df = genebed_df[~(genebed_df.Chromosome == 'chrM')].copy()
    # expects start, end to be int
    genebed_df[['start','end']] = genebed_df[['start','end']].astype(int)

    # get fragments in region
    y_pred = mu.atac.tl.count_fragments_features(atac,features=genebed_df,extend_upstream=2000)
    # save raw data
    y_pred.X.raw = y_pred.X
    # normalize
    y_pred.X = y_pred.X.astype(np.float64)
    sc.pp.normalize_per_cell(y_pred)
    sc.pp.log1p(y_pred)

    return y_pred


def chrom_potential(y_obs, y_pred, k=10):
    
    '''Compute chromatin potential.
    References: Ma Cell 2020, Mitra Nat Gen 2024.
    https://github.com/snehamitra/SCARlink/blob/main/scarlink/src/chromatin_potential.py
    
    Parameters
    ----------
    y_obs : ad.AnnData
        AnnData with RNA information, of shape (#cells,#genes).
        Should contain y_obs.layers['X_smooth'].
    y_pred : ad.AnnData
        AnnData with RNA information predicted from atac, of shape 
        (#cells,#genes). Should contain y_pred.layers['X_smooth'].
    k : int
        Number of neighbors.
    
    Returns
    -------
    dists : np.array
        Matrices of shape (#cells,#cells) containing distances between
        cell_{yp,i} and cell_{yo,j}.
    dists : np.array
        Matrices of shape (#cells,k) containing closest k neighbors of
        LSI-smoothed yp and yo.
        
    '''

    # get smoothed values
    yo_smooth = y_obs.layers['X_smooth'].astype(np.float32)
    yp_smooth = y_pred.layers['X_smooth'].astype(np.float32)

    # min max scaling of smoothed x,
    # over its neighbors in lsi space
    # as implemented by scarlink
    scaler = sklearn.preprocessing.MinMaxScaler()
    yo_scaled = scaler.fit_transform(yo_smooth)
    yp_scaled = scaler.fit_transform(yp_smooth)
    print('scaled values')

    # get knn of smoothed scaled profiles (knn 2, k=10)
    # x_scaled should be the same size as y_scaled
    nn = sklearn.neighbors.NearestNeighbors(n_neighbors=k, n_jobs=-1, metric='correlation') 
    nn.fit(yp_scaled)
    # closest rna obs to atac obs
    dists, neighs = nn.kneighbors(yo_scaled)
    print('obtained neighbors')

    return dists, neighs
    

def embed_arrows(dists, neighs, adata, k=10, embed='lsi'):
    
    ''' Map chromatin potential onto embedding.
    Note that embed_projected do not depicted actual cells,
    but the average embedding values of k cells.
        
    '''
    
    # get embedding
    # lsi (n_cells, n_comps)
    # umap (n_cells, 2)
    embed_ = adata.obsm['X_'+embed]

    # distances in embedding space
    embed_projected = np.zeros(embed_.shape, dtype=np.float32)
    dij = np.zeros((adata.shape[0], adata.shape[0]), dtype=np.float32)
    for i in tqdm(range(adata.shape[0])):
        # average embedding position across k=10 neighbors in smoothed scaled x,y space
        embed_projected[i] = np.mean(embed_[neighs[i]], axis=0)
        if embed != 'umap':
            for j in range(adata.shape[0]):
                # cell_i x cell_j distances
                dij[i,j] = np.sqrt(np.sum((embed_[i]-embed_projected[j]) ** 2))
        
    # for visualization
    if embed == 'umap':

        # get knn of umap embedding (knn 3 (optional), k=15)
        nn = sklearn.neighbors.NearestNeighbors(n_neighbors=15, n_jobs=-1, metric='euclidean')
        nn.fit(embed_)
        dists, neighs = nn.kneighbors(embed_)
        print('smoothing in umap')
        
        embed_smooth = np.zeros(embed_.shape,dtype=np.float32)
        for i in tqdm(range(adata.shape[0])):
            embed_smooth[i] = embed_projected[neighs[i]].mean(0)
            for j in range(adata.shape[0]):
                # cell_i x cell_j distances
                dij[i,j] = np.sqrt(np.sum((embed_smooth[j] - embed_[i]) ** 2))

        embed_projected = embed_smooth
                
    print('obtained distances')
    
    return dij, embed_, embed_projected





############## corr ##############


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
