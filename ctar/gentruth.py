import pandas as pd
import numpy as np
import scipy as sp
from statsmodels import stats
import pybedtools
from pybedtools import BedTool
import os
import matplotlib.pyplot as plt
import matplotlib_venn as venn
from biomart import BiomartServer


def preprocess_df(df):
    ''' Preprocessing for gentruth dfs.
    '''
    
    # add end if necessary
    df = df.copy()
    if 'end' not in df.columns:
        df['end'] = df['start']
    # remove nan values if present
    df = df.dropna(subset=['region'])
    # remove non-ensembl ids
    df = df[df.gene.str.startswith('ENSG').astype(bool)].copy()
    # coerce start,end to int
    df[['start','end']] = df[['start','end']].astype(np.int32)
    return df


def intersect_beds(bed1,bed2,deduplicate=True,col_names = ['bed1_chr','bed1_start','bed1_end','bed1_id','bed2_chr','bed2_start','bed2_end','bed2_id']):
    ''' BedTools intersection from two BedTool objects and returns a DataFrame.
    '''

    intersect_df = bed1.intersect(bed2,wa=True,wb=True).to_dataframe()
    if intersect_df.empty:
        return pd.DataFrame(columns=col_names)
    intersect_df.columns = col_names
    if deduplicate:
        intersect_df = intersect_df.drop_duplicates()
    return intersect_df


def summarize_df(query_file, reference_path, storage_path):

    ''' Summarizes existing tsv files to obtain marginal and joint n_genes, n_snps, n_links.
    
    Parameters
    -------
    query_file : str
        Path to file containing query dataset in .tsv.gz format. Should have columns 
        ['chr','start','gene'].

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
    reference_files.remove(os.path.basename(query_file))
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
        col_names = ['query_chr','query_start','query_end','query_gene',
                                   'ref_chr','ref_start','ref_end','ref_gene']
        intersect_df = intersect_beds(querybed,refbed,col_names = col_names)

        ref_label = ref.removesuffix('.tsv.gz')
        print(intersect_df.shape[0], 'intersections with',ref_label, 'found.')

        # add joint values with ref
        if '_pos' not in ref_label:
            gene_dic[ref_label] = len(set(reference_df.gene) & set(query_df.gene))
            snp_dic[ref_label] = intersect_df.shape[0] # refers to number of intersections between snps, note: contains duplicates if multiple intersections
        links_dic[ref_label] = sum(intersect_df.query_gene == intersect_df.ref_gene) # refers to number of links in agreement, ie regions overlap & same gene
    
    # remove some vars to save mem
    del query_df
    del reference_df
    del intersect_df

    if '_pos' not in query_label:
        zipped = zip([gene_dic,snp_dic,links_dic],['gene.tsv','snp.tsv','links.tsv'])
    else: zipped = zip([links_dic],['links.tsv'])

    for dic, filename in zipped:
        df = pd.read_csv(storage_path + filename, sep='\t', index_col=0)
        df[query_label] = pd.Series(dic) # making into series will ensure that indices match, even if they're out of order
        df.loc[query_label] = pd.Series(dic)
        df.to_csv(os.path.join(storage_path, filename), sep='\t')


def plot_venn2(df,labels,pos=False):

    ''' Display 2 venn diagrams on intersecting (1) all links and (2) positive-labelled links.
    
    Parameters
    -------
    df : pd.DataFrame
        pd.DataFrame with labels in columns.
    labels : list
        List of strings

    Returns
    -------
    
    '''

    # all labels
    n3 = df.loc[labels[0],labels[1]] # intersection
    n2 = df.loc[labels[0],labels[0]] - n3 # label 0
    n4 = df.loc[labels[1],labels[1]] - n3 # label 1

    venn1 = venn.venn2([n2, n4, n3],set_labels=labels)
    plt.title('All links')
    plt.show()

    # positive only
    n3 = df.loc[labels[0]+'_pos',labels[1]+'_pos'] # intersection
    n2 = df.loc[labels[0]+'_pos',labels[1]] - n3 # pos label 0 in label 1 univ
    n4 = df.loc[labels[0],labels[1]+'_pos'] - n3 # pos label 1 in label 0 univ

    venn1 = venn.venn2([n2, n4, n3],set_labels=labels)

    plt.title('Positive links')
    plt.show()


def paired_stats_df(df_dic,labels):
    ''' Fetches marginal and joint stats for pair of labels.
    '''

    stats_df = pd.DataFrame(columns=labels + ['joint'],index=df_dic.keys(),data=0)

    for cat in df_dic.keys():
        stats_df.loc[cat,labels[0]] = df_dic[cat].loc[labels[0],labels[0]]
        stats_df.loc[cat,labels[1]] = df_dic[cat].loc[labels[1],labels[1]]
        stats_df.loc[cat,'joint'] = df_dic[cat].loc[labels[0],labels[1]]

    return stats_df


def odds_ratio_poslinks(df,ha_correction=True):
    ''' Provides odds ratio and chi^2 p-value with (+0.5) correction 
    for positive links.

    '''
    
    labels = [x for x in df.columns  if '_pos' not in x]
    odds_df = pd.DataFrame(index=labels,columns=labels,data=0.0)
    pval_df = pd.DataFrame(index=labels,columns=labels,data=1.0)
    contig_df = pd.DataFrame(index=labels,columns=labels,data=[])
    label_tup = [(a, b) for idx, a in enumerate(labels) for b in labels[idx + 1:]]
    
    for label_1, label_2 in label_tup:
        
        n1 = df.loc[label_1,label_2] # universe
        n3 = df.loc[label_1+'_pos',label_2+'_pos'] # intersection
        n2 = df.loc[label_1+'_pos',label_2] - n3 # pos label 0 in label 1 univ
        n4 = df.loc[label_1,label_2+'_pos'] - n3 # pos label 1 in label 0 univ
        
        contigency_mat = np.array([[n3, n4], [n2, n1]]) 
        odds_ratio = stats.contingency_tables.Table2x2(contigency_mat,shift_zeros=ha_correction).oddsratio # +0.5 haldane-anscombe / yates correction
        pval = sp.stats.chi2_contingency(contigency_mat,correction=ha_correction).pvalue
        
        odds_df.loc[label_1,label_2] = odds_ratio
        odds_df.loc[label_2,label_1] = odds_ratio
        
        pval_df.loc[label_1,label_2] = pval
        pval_df.loc[label_2,label_1] = pval

        contig_df.loc[label_1,label_2] = contigency_mat
        contig_df.loc[label_2,label_1] = contigency_mat
        
    return {'odds_ratio':odds_df,'pvals':pval_df,'contingency':contig_df}


def normalize_symm(mat):
    ''' Normalize a symmetric matrix by total elements (should be stored in diagonal).
    '''
    diag = np.diagonal(mat)
    # A-C+B for venn2([A,B,C]) = universal set
    univ = diag + diag.reshape(-1,1) - mat
    mat_norm = mat / univ
    return mat_norm


def fetch_attribute(attribute,query_attribute='ensembl_gene_id',dataset='hsapiens_gene_ensembl'):
    ''' Create dictionary mapping for genes to an attribute.
    
    Parameters
    -------
    attribute : str
        Biomart attribute you want to obtain, e.g. 'transcription_start_site', 'hgnc_symbol'.
    query_attribute : str
        Biomart attribute which defines input data.
    dataset : str
        Biomart dataset to use.

    Returns
    -------
    gene_dic : dict
        Dictionary with attribute as key and values as attribute.
    '''

    # call biomart
    server = BiomartServer("http://www.ensembl.org/biomart")
    dataset = server.datasets[dataset]
    attributes = [query_attribute,attribute]
    response = dataset.search({'attributes': attributes})
    responses = response.raw.data.decode('ascii')
    # store in dictionary
    gene_dic = {}
    for line in responses.splitlines():                                              
        line = line.split('\t')
        gene_id = line[0]
        gene_dic[gene_id] = line[1]

    return gene_dic

