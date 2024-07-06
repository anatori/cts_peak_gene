from pybedtools import BedTool
import pybedtools
from biomart import BiomartServer
import numpy as np
import pandas as pd 
import muon as mu
import anndata as ad
import scanpy as sc


def peak_to_gene(peaks_df,genes_df,clean=True,split_peaks=True,distance=500000,col_names=['chr','start','end'],
                gene_col='gene',peak_col='peak',genome='hg38'):

    ''' Uses pybedtools to find the overlap between gene body +/- some distance and proximal peaks. 
    
    Parameters
    -------
    peaks_df : pd.DataFrame
        DataFrame of length (#peaks) containing peak_col describing unique peak names.

    genes_df : pd.DataFrame
        DataFrame of length (#genes) containing gene_col describing unique gene names.

    clean : bool
        If true, will remove peaks labelled with NA, remove non-standard chromosomes,
        and change start and end locations to integer types.
        
    split_peaks : bool
        If true, peaks must be in format chr:start-end and will be extracted from this
        string format into a bed-suitable format (split into chr,start,end columns).
        Start and end locations will also be changed into integer types.

    distance : int
        Distance away from gene body to assess for peak overlap.

    col_names : list
    	List with column names describing the chromosome, bp start position, and
    	bp end position in bed-suitable format (strictly in this order). These should
    	correspond to chromosome position columns in each of the peaks_df and 
    	genes_df dataframes.

    gene_col : str
    	The label of the column containing unique gene names.

	peak_col : str
		The label of the column containing unique peak names.

	genome : str
		The genome to use for creating distance windows around the gene body.

    Returns
    -------
    peak_gene_links : pd.DataFrame
        DataFrame containing matched peak-gene links according to +/- distance windows
        around the gene body. The index is set to a unique identifier in the format 
        'peak , gene'.
    
    '''

    if clean:
        genes_clean = genes_df[~genes_df[col_names[0]].isna()] # assumes first item in col_names is chr label
        genes_clean = genes_clean[genes_clean.chr.str.startswith('chr')] # removes non chr mapped genes
    else: genes_clean = genes_df

    if split_peaks:
        peaks_clean = peaks_df.copy()
        peaks_clean[col_names] = peaks_clean[peak_col].str.extract(r'(\w+):(\d+)-(\d+)')
    else: peaks_clean = peaks_df

    genes_clean[col_names[1:]] = genes_clean[col_names[1:]].astype(int)
    peaks_clean[col_names[1:]] = peaks_clean[col_names[1:]].astype(int)

    # add indices
    genes_clean['index_y'] = range(len(genes_clean))
    peaks_clean['index_x'] = range(len(peaks_clean))

    # creates temp bedtool files
    peaks_bed = BedTool.from_dataframe(peaks_clean[col_names+[peak_col]]).sort()
    genes_bed = BedTool.from_dataframe(genes_clean[col_names+[gene_col]]).sort()

    # add windows
    genes_bed_windowed = genes_bed.slop(genome=genome,b=distance).sort()
    
    # using bedtools closest, includes all distances from gene_bed (D='b')
    closest_links = peaks_bed.intersect(genes_bed_windowed,wao=True,sorted=True)

    # save as dataframe
    closest_links = closest_links.to_dataframe()
    closest_links.columns = ['peak_chr','peak_start','peak_end',peak_col,'window_chr','window_start','window_end',gene_col,'distance']

    # merge with original dataframe to obtain all information
    peak_gene_links = closest_links.merge(peaks_clean,on=peak_col)
    peak_gene_links = peak_gene_links.merge(genes_clean,on=gene_col)

    peak_gene_links = peak_gene_links[[gene_col,peak_col]+['distance']]
    
    # use unique indices
    peak_gene_links.index = peak_gene_links[peak_col].astype(str) + ' , ' + peak_gene_links[gene_col].astype(str)

    return peak_gene_links



def add_gene_positions(row,dictionary,gene_col='gene'):

    ''' Add pre-loaded gene positions to desired dataframe.
    
    Parameters
    -------
    row : pd.DataFrame row
        Row of dataframe fed in with df.apply.
        
    dictionary : dict
        Dictionary with keys containing the gene IDs in 'gene'.
        
    gene_id : str
        The name of the DataFrame column containing the gene IDs from which you
        will be converting.

    Returns
    -------
    [chrom,start,end] : list
        List to be expanded into pd.DataFrame columns representing the chromosome
        name, the start position, and the end position of each gene body. If there
        is no corresponding key, will return a list of Nones.
    
    '''
    
    try:
        info = dictionary[row[gene_col]]
    except: return [None]*3 # if the key does not exist in dict
    chrom = info['chromosome_name']
    if chrom.isnumeric() or (chrom=='X') or (chrom=='Y'):
        chrom = 'chr' + chrom
    if chrom == 'MT': # mitochondrial mapped in hg38 as chrM
        chrom = 'chrM'
    start = info['start_position']
    end = info['end_position']
    return [chrom,start,end]



def get_gene_coords(gene_df,gene_id_type='ensembl_gene_id',dataset='hsapiens_gene_ensembl',
                    attributes=['chromosome_name','start_position','end_position'],
                    col_names=['chr','start','end'],gene_col='gene'):
    
    ''' Uses BioMart API to insert gene coordinates based on a given gene_id_type.
    
    Parameters
    -------
    gene_df : pd.DataFrame
        DataFrame containing gene_col that you want to add coordinates to.

    gene_id_type : str
        The attribute that matches your existing gene_col IDs. Should exist in
        BiomartSever.dataset.show_attributes().
        
    dataset : str
        The dataset you wish to pull Biomart info from. Should exist in 
        BiomartSeve.show_datasets().
        
    attributes : list (str)
        The attributes you want to add to gene_df. Should exist in 
        BiomartSever.dataset.show_attributes().

    col_names : list (str)
        Name of columns you will be adding. Length must match length of attributes.

    gene_col : str
        The column in gene_df containing the gene_id_type IDs.

    Returns
    -------
    gene_df : pd.DataFrame
        Returns the original df with columns added corresponding to attributes, labelled
        as col_names.
    
    '''


    ### biomart ###
    
    # connect to the Biomart server
    server = BiomartServer("http://www.ensembl.org/biomart")
    # select the dataset
    dataset = server.datasets[dataset]
    # attributes to read
    attributes_all = [gene_id_type] + attributes
    
    # obtain attributes and read
    response = dataset.search({'attributes': attributes_all})
    responses = response.raw.data.decode('ascii')

    # develop a dictionary to map positions
    gene_positions = {}
    for line in responses.splitlines():                                              
        line = line.split('\t')
        gene_id = line[0] 
        gene_positions[gene_id] = {attribute:line[i+1] for i,attribute in enumerate(attributes)}

    # add attributes to dataframe
    gene_df[col_names] = gene_df.apply(add_gene_positions, axis=1, result_type='expand',
                                              args=(gene_positions,gene_col))
    
    return gene_df


def preprocess_mu(mdata,
    min_genes = 250,
    min_cells = 50,
    counts_per_cell_after = 1e4,
    n_cells_by_counts = 10,
    n_genes_by_counts_min = 2000,
    n_genes_by_counts_max = 15000,
    total_counts_min = 4000,
    total_counts_max = 40000
    ):

    '''Preprocess MuData. Adapted from ./experiments/job.mz/explore.ipynb

    Parameters
    ----------
    mdata : mu.MuData
        MuData object of shape (#cells,#peaks/#genes), containing RNA and ATAC modalities.
    ..parameters : int
        See Muon, scanpy functions.
    
    Returns
    ----------
    pp_mdata : mu.MuData
        Preprocessed MuData object of shape (#cells,#peaks/#genes), containing RNA and ATAC modalities.
    
    '''

    # Raw adata_rna & adata_atac
    adata_rna = mdata.mod['rna'].copy()
    adata_rna.X = adata_rna.X.astype(np.float32) # float32 to reduce memory cost
    adata_rna.var_names = [dic_id2sym[x] if x in dic_id2sym else x for x in adata_rna.var_names]
    adata_atac = mdata.mod['atac'].copy()
    adata_atac.X = adata_atac.X.astype(np.float32) # float32 to reduce memory cost
    print('Raw, adata_rna', adata_rna.shape, 'adata_atac', adata_atac.shape)

    # Filtering & normalization of RNA (using scanpy) (in log scale)
    sc.pp.filter_cells(adata_rna, min_genes = min_genes)
    sc.pp.filter_genes(adata_rna, min_cells = min_cells)
    # save raw data
    adata_rna.X.raw = adata_rna.X
    # normalize
    sc.pp.normalize_per_cell(adata_rna, counts_per_cell_after = counts_per_cell_after)
    sc.pp.log1p(adata_rna)
    print('Filtering & normalization of RNA', adata_rna.shape)

    # Filtering & normalization of ATAC (using mu) (in log scale)
    # Following tutorial https://muon-tutorials.readthedocs.io/en/latest/
    # single-cell-rna-atac/pbmc10k/2-Chromatin-Accessibility-Processing.html
    sc.pp.calculate_qc_metrics(adata_atac, percent_top=None, log1p=False, inplace=True)
    mu.pp.filter_var(adata_atac, 'n_cells_by_counts', lambda x: x >= n_cells_by_counts)
    mu.pp.filter_obs(adata_atac, 'n_genes_by_counts', lambda x: (x >= n_genes_by_counts_min) & (x <= n_genes_by_counts_max))
    mu.pp.filter_obs(adata_atac, 'total_counts', lambda x: (x >= total_counts_min) & (x <= total_counts_max))
    # save raw data
    adata_atac.X.raw = adata_atac.X
    # normalize
    sc.pp.normalize_per_cell(adata_atac, counts_per_cell_after = counts_per_cell_after)
    sc.pp.log1p(adata_atac)
    print('Filtering & normalization of ATAC', adata_atac.shape)

    # Align cells
    cell_list = [x for x in adata_rna.obs_names if x in set(adata_atac.obs_names)]
    adata_rna = adata_rna[cell_list, :].copy()
    adata_atac = adata_atac[cell_list, :].copy()
    print('Align cells, adata_rna', adata_rna.shape, 'adata_atac', adata_atac.shape)

    pp_mdata = mu.MuData({'atac':adata_atac,'rna':adata_rna})

    return pp_mdata



######## archived #############

def build_adata(mdata,gene_col='gene_name',peak_col='gene_ids',raw=False):
    
    '''Creates a new AnnData object for peak-gene links.

    Parameters
    ----------
    mdata : mu.MuData
        MuData object of shape (#cells,#peaks). Contains DataFrame under mdata.uns.peak_gene_pairs
        and adds columns ['index_x','index_y'] that correspond to peak and gene indices in atac.X
        and rna.X respectively. 'gene_col' should be the index of mdata['rna'].var, and 'peak_col'
        should be the index of mdata['atac'].var.
        If mdata.uns.control_peaks DNE, will create it.
    gene_col : str
        Label for gene ID column.
    peak_col : str
        Label for peak ID column.
    raw : bool
        If raw files are under mdata[modality].X, set to True. If False, will take values from 
        mdata[modality].X as anadata.layers[modality] and set mdata[modality].layers['counts'] as 
        anadata.layers[modality+'_raw'].
    
    Returns
    ----------
    anadata : an.AnnData
        New AnnData object of shape (#cells with same ct between layers,#peak-gene pairs) and layers atac, rna.
        Obs are labelled by cell_id. Var are labelled by peak-gene links.
    
    '''

    try:
        mdata.uns['peak_gene_pairs']
    except KeyError:
        print('Attempting to add peak-gene pairs.')
        find_peak_gene_pairs(mdata)

    # Add indices
    mdata['atac'].var['index_y'] = range(len(mdata['atac'].var))
    mdata['rna'].var['index_x'] = range(len(mdata['rna'].var))

    # Only take cells which match assigned celltypes between assays
    ct_mask = (mdata['rna'].obs['celltype'] == mdata['atac'].obs['celltype']).values

    # Initialize empty AnnData
    n = mdata[ct_mask,:].shape[0]
    m = len(mdata.uns['peak_gene_pairs'])
    anadata = ad.AnnData(np.zeros((n,m)))

    genes = mdata.uns['peak_gene_pairs'][gene_col].values
    peaks = mdata.uns['peak_gene_pairs'][peak_col].values

    if not raw:
        # Add aligned atac and rna layers. Should be CSC format.
        anadata.layers['atac'] = mdata['atac'][:,peaks].X[ct_mask,:]
        anadata.layers['rna'] = mdata['rna'][:,genes].X[ct_mask,:]

        # Add raw
        anadata.layers['atac_raw'] = mdata['atac'][:,peaks].layers['counts'][ct_mask,:]
        anadata.layers['rna_raw'] = mdata['rna'][:,genes].layers['counts'][ct_mask,:]

    else:
        anadata.layers['atac_raw'] = mdata['atac'][:,peaks].X[ct_mask,:]
        anadata.layers['rna_raw'] = mdata['rna'][:,genes].X[ct_mask,:]

    # Add peak-gene pair descriptions
    mdata.uns['peak_gene_pairs']['id'] = list(mdata.uns['peak_gene_pairs'][peak_col] + ',' + mdata.uns['peak_gene_pairs'][gene_col])
    anadata.var = mdata.uns['peak_gene_pairs'].set_index('id')

    # Add celltypes, which should be the same between layers
    anadata.obs = mdata[ct_mask,:]['atac'].obs

    return anadata



def cellranger_peak_gene_pairs(mdata):
    
    '''Adds dataframe containing peak-gene pairs from CellRanger.

    Parameters
    ----------
    mdata : mu.MuData
        MuData object of shape (#cells,#peaks). Must contain atac.var with peaks listed under 'gene_ids'.
        Must also contain atac.uns['atac']['peak_annotation'] with peaks listed under 'peak' and genes
        listed under 'gene_name'. See mu.initialise_default_files.
    
    Returns
    ----------
    mdata : mu.MuData
        Updates mdata input with mdata.uns['peak_gene_pairs'], a pd.DataFrame of len (#peak-gene pairs)
        containing columns ['index_x','index_y'] that correspond to peak and gene indices in atac.X
        and rna.X respectively.
    
    '''
    
    rna = mdata.mod['rna']
    atac = mdata.mod['atac']

    try:
        atac.uns['atac']['peak_annotation']
    except KeyError:
        print('Must provide atac.uns[atac][peak_annotation].')
        raise

    try:
        atac.uns['atac']['peak_annotation']['gene_name']
    except KeyError:
        atac.uns['atac']['peak_annotation'] = atac.uns['atac']['peak_annotation'].reset_index(names='gene_name')
        print('Using peak_annotation index as gene_name column.')

    try:
        atac.uns['atac']['peak_annotation']['gene_ids']
    except KeyError:
        atac.uns['atac']['peak_annotation'].rename(columns={'peak':'gene_ids'},inplace=True)
        print('Using peak as gene_ids column.')

    # Merge the peak annotations. Reset index to maintain original atac.X index for reference
    mdata['atac'].var['index'] = range(len(atac.var))
    peak_gene_labels = pd.merge(atac.var[['gene_ids','index']], \
                            atac.uns['atac']['peak_annotation'], \
                            how='left',on='gene_ids')

    try:
        rna.var['gene_name']
    except KeyError:
        rna.var = rna.var.reset_index(names='gene_name')
        print('Using rna.var index as gene_name column.')
    
    # Duplicates for multiple peaks per gene, but drops ones not observed in RNA
    # Reset index to maintain original rna.X index for reference
    mdata['rna'].var['index'] = range(len(rna.var))
    peak_gene_pairs = pd.merge(peak_gene_labels,rna.var[['gene_name','index']], \
                           how='left',on='gene_name').dropna()

    mdata.uns['peak_gene_pairs'] = peak_gene_pairs

    return mdata