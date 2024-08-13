import pandas as pd
import numpy as np
import pybedtools
from pybedtools import BedTool
import os


def preprocess_df(df):
    ''' Preprocessing for gentruth dfs.
    '''
    
    # add end if necessary
    df = df.copy()
    if 'end' not in df.columns:
        df['end'] = df['start']
    # remove nan values if present
    df = df.dropna(subset=['region'])
    # coerce start,end to int
    df[['start','end']] = df[['start','end']].astype(int)
    
    return df


def gentruth_intersect(query_file, reference_path, storage_path):

    ''' Intersects existing tsv files to obtain marginal and joint n_genes, n_snps, n_links.
    
    Parameters
    -------
    query_file : str 
        Path to file containing query dataset. Must be in .tsv.gz format.

    reference_path : str
        Path to all other files to intersect query_file with. Must be in .tsv.gz format.

    storage_path : str
        Path to save additional intersection information to. Must contain '_gene','_snp',
        '_links' files in .tsv format.
        
    label : str
        Name of query_file to label storage_file columns as.

    Returns
    -------
    
    '''
    
    # load query df
    query_df = pd.read_csv(query_file, sep='\t', compression='gzip',index_col=0)
    query_df = preprocess_df(query_df)
    querybed = BedTool.from_dataframe(query_df[['chr','start','end','gene']])

    # find ref files
    reference_files = [f for f in os.listdir(reference_path) if f.endswith('.tsv.gz')]
    reference_labels = [f.removesuffix('.tsv.gz') for f in reference_files]
    
    # store values in dictionaries
    # add marginal values
    query_label = os.path.basename(query_file).removesuffix('.tsv.gz')
    gene_dic = {query_label: len(set(query_df.gene))}
    snp_dic = {query_label: len(set(query_df.region))}
    links_dic = {query_label: query_df.drop_duplicates(subset=['gene', 'region']).shape[0]}
    
    # ref files are often large so we load one at a time
    for i,ref in enumerate(reference_files):

        # load ref df
        reference_df = pd.read_csv(reference_path + ref, sep='\t', compression='gzip',index_col=0)
        reference_df = preprocess_df(reference_df)
        refbed = BedTool.from_dataframe(reference_df[['chr','start','end','gene']])

        # intersect query bed with ref bed
        intersect_df = querybed.intersect(refbed,wa=True,wb=True).to_dataframe()
        intersect_df.columns = ['query_chr','query_start','query_end','query_gene',
                                   'ref_chr','ref_start','ref_end','ref_gene']
        intersect_df = intersect_df.drop_duplicates()

        ref_label = ref.removesuffix('.tsv.gz')
        print(intersect_df.shape[0], 'intersections with',ref_label, 'found.')

        # add joint values with ref
        if '_pos' not in query_label:
            gene_dic[ref_label] = len(set(reference_df.gene) & set(query_df.gene))
            snp_dic[ref_label] = intersect_df.shape[0] # refers to number of intersections between snps, note: contains duplicates if multiple intersections
        links_dic[ref_label] = sum(intersect_df.query_gene == intersect_df.ref_gene) # refers to number of links in agreement, ie regions overlap & same gene
    
    # remove some vars to save mem
    del query_df
    del reference_df
    del intersect_df

    zipped = list(zip(['a'], ['links.tsv']))
    if '_pos' not in query_label:
        zipped.append(list(zip([gene_dic, snp_dic], ['gene.tsv', 'snp.tsv'])))

    for dic, filename in zipped:
        df = pd.read_csv(storage_path + filename, sep='\t', index_col=0)
        df[query_label] = pd.Series(dic) # making into series will ensure that indices match, even if they're out of order
        df.loc[query_label] = pd.Series(dic)
        df.to_csv(os.path.join(storage_path, filename), sep='\t')
    