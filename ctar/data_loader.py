
from pybedtools import BedTool
import pybedtools
from biomart import BiomartServer
import numpy as np
import pandas as pd 


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
        genes_clean[col_names[1:]] = genes_clean[col_names[1:]].astype(int)
    else: genes_clean = genes_df

    if split_peaks:
        peaks_clean = peaks_df.copy()
        peaks_clean[col_names] = peaks_clean[peak_col].str.extract(r'(\w+):(\d+)-(\d+)')
        peaks_clean[col_names[1:]] = peaks_clean[col_names[1:]].astype(int)
    else: peaks_clean = peaks_df

    # creates temp bedtool files
    peaks_bed = BedTool.from_dataframe(peaks_clean[col_names+[peak_col]]).sort()
    genes_bed = BedTool.from_dataframe(genes_clean[col_names+[gene_col]]).sort()

    # add windows
    genes_bed_windowed = genes_bed.slop(genome=genome,b=distance).sort()
    
    # using bedtools closest, includes all distances from gene_bed (D='b')
    closest_links = peaks_bed.intersect(genes_bed_windowed,wao=True,sorted=True)

    # save as dataframe
    closest_links = closest_links.to_dataframe()
    closest_links.columns = ['peak_chr','peak_start','peak_end','peak','window_chr','window_start','window_end','gene','distance']

    # merge with original dataframe to obtain all information
    peak_gene_links = closest_links.merge(peaks,on='peak')
    peak_gene_links = peak_gene_links.merge(genes_clean,on='gene')

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
        info = dictionary[row[gene]]
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
    genes[col_names] = genes.apply(add_gene_positions, axis=1, result_type='expand',
                                              args=(gene_positions,gene_col))
    
    return genes