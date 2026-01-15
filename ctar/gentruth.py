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
import subprocess
import tempfile


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


######################### overlap analysis #########################


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


######################### feature addition #########################


def annotate_with_references(reference_files,map_df,enh_gene_ls=['abc','scent','crisprflow'],map_col='name'):
    ''' Add boolean masks specifying whether map region is in any reference file region.
    
    Parameters
    -------
    reference_files : list of strings
        List of locations for reference .tsv files.

    map_df : pd.DataFrame
        Output of find_gene_distances, containing columns ['chr','start','end','region'].

    enh_gene_ls : list of strings
        List of ref_labels for enhancer-gene strategies in references.

    map_col : str
        Name of column containing linked ENSEMBL IDs.
    
    Returns
    -------
    result_df : pd.DataFrame
        DataFrame with original map_df columns + one boolean array indicating overlap with reference_file.

    '''

    result_df = map_df.copy()

    for ref in reference_files:
        # read in ref file
        reference_df = pd.read_csv(ref, sep='\t', compression='gzip',index_col=0)
        reference_df = preprocess_df(reference_df)
        ref_label = os.path.basename(ref).removesuffix('.tsv.gz')
        
        if ref_label in enh_gene_ls: # enhancer-gene dfs
            reference_bed = BedTool.from_dataframe(reference_df[['chr','start','end','region']])
            map_bed = BedTool.from_dataframe(map_df[['chr','start','end',map_col]])
            intersection_df = map_bed.intersect(reference_bed,wa=True,c=True,loj=True).to_dataframe() # loj to maintain original order
            result_df[ref_label] = intersection_df['score'] > 0 # score indicates number of overlaps with reference
            
        else: # snp-gene dfs
            reference_df['region'] = reference_df['chr'] + ':' + reference_df['start'].astype(str) # in case not in correct format
            result_df[ref_label] = map_df[map_col].isin(reference_df.region)

    return result_df


def find_gene_distances():
    ''' Returns dataframe contianing all gene body locations.
    '''

    # call biomart
    server = BiomartServer("http://www.ensembl.org/biomart")
    dataset = server.datasets['hsapiens_gene_ensembl']
    attributes = ['ensembl_gene_id','chromosome_name', 'start_position', 'end_position']
    response = dataset.search({'attributes': attributes})
    responses = response.raw.data.decode('ascii')
    
    # store results in dataframe
    gene_df = pd.DataFrame(columns=['gene','gene_chr','gene_start','gene_end'])
    for i,line in enumerate(responses.splitlines()):                                              
        line = line.split('\t')
        gene_df.loc[i] = line

    gene_df['gene_chr'] = 'chr' + gene_df['gene_chr'].astype(str)
    return gene_df


def add_gene_distances(query_df, gene_df, gene_col='gene'):
    ''' Adds gene body locations and calculates distances.
    
    Parameters
    -------
    query_df : pd.DataFrame
        Query dataset. Should contain columns ['chr','start','end',gene_col].

    gene_df : pd.DataFrame
        Output of find_gene_distances, containing columns ['gene','gene_chr','gene_start','gene_end'].

    gene_col : str
        Name of column containing linked ENSEMBL IDs.
    
    Returns
    -------
    intersect_df : pd.DataFrame
        DataFrame containing distances from assigned genes for query positions.

    '''

    # obtain gene locations
    intersect_df = query_df.merge(gene_df,how='left',left_on=gene_col,right_on='gene')
    
    if 'end' not in query_df.columns:
        query_df['end'] = query_df['start']

    # chooses positive distance, depending on whether enh is after or before the gene
    # will choose least negative distance if it is within the gene body
    intersect_df[['gene_start','gene_end','start','end']] = intersect_df[['gene_start','gene_end','start','end']].astype('Int32')
    intersect_df['distance'] = np.maximum(intersect_df.gene_start - intersect_df.end, \
                                    intersect_df.start - intersect_df.gene_end)
    intersect_df['same_chr'] = intersect_df['gene_chr'] == intersect_df['chr']

    return intersect_df


def find_gene_tss():
    """
    Fetch gene coordinates + strand from Ensembl BioMart and compute TSS.

    Returns
    -------
    gene_df : pd.DataFrame
        Columns: ['gene', 'gene_chr', 'tss', 'gene_start', 'gene_end', 'strand']
    """
    server = BiomartServer("http://www.ensembl.org/biomart")
    dataset = server.datasets['hsapiens_gene_ensembl']

    # More robust than relying on transcription_start_site attribute
    attributes = [
        'ensembl_gene_id',
        'chromosome_name',
        'start_position',
        'end_position',
        'strand'
    ]

    response = dataset.search({'attributes': attributes})
    text = response.raw.data.decode('ascii')

    rows = []
    for line in text.splitlines():
        ensg, chrom, start, end, strand = line.split('\t')
        # Skip weird contigs / missing
        if chrom in ["", "CHR_HSCHR6_MHC_QBL_CTG1", "HSCHR6_MHC_COX_CTG1", "HSCHR6_MHC_DBB_CTG1"]:
            continue
        rows.append((ensg, chrom, start, end, strand))

    gene_df = pd.DataFrame(rows, columns=["gene", "chromosome_name", "gene_start", "gene_end", "strand"])

    # Coerce types
    gene_df["gene_start"] = pd.to_numeric(gene_df["gene_start"], errors="coerce")
    gene_df["gene_end"]   = pd.to_numeric(gene_df["gene_end"], errors="coerce")
    gene_df["strand"]     = pd.to_numeric(gene_df["strand"], errors="coerce")

    gene_df = gene_df.dropna(subset=["gene_start", "gene_end", "strand"]).copy()

    # Add chr prefix (and drop non-standard chromosomes if you want)
    gene_df["gene_chr"] = "chr" + gene_df["chromosome_name"].astype(str)

    # Compute TSS from strand
    gene_df["tss"] = gene_df["gene_start"].where(gene_df["strand"] == 1, gene_df["gene_end"]).astype("Int64")

    # Keep only useful columns
    gene_df = gene_df[["gene", "gene_chr", "tss", "gene_start", "gene_end", "strand"]].copy()

    return gene_df


def add_tss_distances(query_df, gene_df, gene_col="gene"):
    """
    Adds gene TSS and calculates distance from query interval to TSS.

    Parameters
    ----------
    query_df : pd.DataFrame
        Must contain columns ['chr', 'start', 'end', gene_col] (end optional; will be created).

    gene_df : pd.DataFrame
        Output of find_gene_tss() containing columns ['gene','gene_chr','tss'] at minimum.

    Returns
    -------
    out : pd.DataFrame
        Adds columns: gene_chr, tss, same_chr, distance (signed), abs_distance
    """
    out = query_df.copy()
    if "end" not in out.columns:
        out["end"] = out["start"]

    out = out.merge(gene_df[["gene", "gene_chr", "tss"]], how="left", left_on=gene_col, right_on="gene")

    # same chromosome?
    out["same_chr"] = out["gene_chr"] == out["chr"]

    # distance from interval to point (TSS)
    # If TSS is inside [start,end], distance = 0
    # Else distance is signed to the nearest boundary:
    #   positive if peak is downstream of TSS, negative if upstream (based on genomic coordinate)
    out["start"] = pd.to_numeric(out["start"], errors="coerce")
    out["end"]   = pd.to_numeric(out["end"], errors="coerce")
    out["tss"]   = pd.to_numeric(out["tss"], errors="coerce")

    # signed distance: choose boundary that is closest to the point, with sign
    # Cases:
    #   end < tss  => peak entirely left of TSS => distance = end - tss (negative)
    #   start > tss => peak entirely right of TSS => distance = start - tss (positive)
    #   otherwise overlap => 0
    out["distance"] = np.where(
        out["end"] < out["tss"], out["end"] - out["tss"],
        np.where(out["start"] > out["tss"], out["start"] - out["tss"], 0)
    )

    out["abs_distance"] = np.abs(out["distance"])

    return out


def add_pli(query_df, pli_dict, gene_col='gene'):
    ''' Maps genes with pLI. pli_dict should be in format {'gene': pLI}.
    '''
    query_df['pli'] = query_df[gene_col].map(pli_dict)
    return query_df


def run_ucsc_tool(tool_location, input_file, bed, output_file):
    ''' Runs UCSC tools, such as bigWigAverageOverBed.

    Parameters
    -------
    tool_location : str
        UCSC tool path.

    input_file : str
        Input file 1.

    bed : str, pd.DataFrame
        Input file 2. Specify path to bed file otherwise, creates temp bed file.
    
    Returns
    -------
    result_df : pd.DataFrame
        DataFrame with results from the UCSC tool.

    '''

    if isinstance(bed,pd.DataFrame):
        # create temp bed
        bed_tmp = tempfile.NamedTemporaryFile(suffix='.bed')
        bed_file = bed_tmp.name
        bed[['start','end']] = bed[['start','end']].astype(int)
        bed[['chr','start','end','region']].to_csv(bed_file,sep='\t',header=False,index=False)

    else:
        bed_file = bed

    subprocess.run([tool_location,
                    '-tsv',
                    input_file,
                    bed_file,
                    output_file
                   ],stdout=subprocess.PIPE)
    result_df = pd.read_csv(output_file, sep='\t')
    
    if isinstance(bed, pd.DataFrame): # remove temp file
        os.remove(bed_file) 

    return result_df


def find_catlas(query,catlas_df,f=1e-9):
    ''' Add CATLAS-based features.
    
    Parameters
    -------
    query : str, pd.DataFrame
        Path to file containing query dataset in .tsv.gz format or query as dataframe. Should 
        contain columns ['chr','start','gene'].

    catlas_df : pd.DataFrame
        CATLAS DataFrame, downloaded from http://catlas.org/catlas_downloads/humantissues/.
        Should conttain columns ['#Chromosome','Start','End','id']. Note that more info can be
        added from Li et al. Science 2023, Supplementary tables.
        
    f : float
        Specify overlap necessary for bedtools. Default (1e-9) is 1bp.

    Returns
    -------
    intersect_df : pd.DataFrame
        DataFrame containing CATLAS database information for query positions.

    '''

    if isinstance(query,str):
        query_df = pd.read_csv(query, sep='\t', compression='gzip',index_col=0)
    else:
        query_df = query

    query_df['ind'] = range(len(query_df)) # add index for ref
    querybed = BedTool.from_dataframe(query_df[['chr','start','end','ind']])

    # format catlas enhancers
    catlas_df['id'] = range(len(catlas_df)) # add index for ref
    catlasbed = BedTool.from_dataframe(catlas_df[['#Chromosome','Start','End','id']])

    # intersect query bed with catlas bed
    col_names = ['query_chr','query_start','query_end','query_ind',
                    'catlas_chr','catlas_start','catlas_end','catlas_ind']
    intersect_df = querybed.intersect(catlasbed,wa=True,wb=True,f=f).to_dataframe()
    intersect_df.columns = col_names
    
    # return intersected catlas and query df
    intersect_df = intersect_df.merge(catlas_df,how='left',left_on='catlas_ind',right_on='id')
    intersect_df = intersect_df.drop(columns=['#Chromosome','Start','End','id'])
    intersect_df = intersect_df.drop_duplicates(subset=['query_ind','catlas_ind'])

    return intersect_df


def find_screen(query,screen_df,f=1e-9):
    ''' Add ENCODE SCREEN-based features.
    
    Parameters
    -------
    query : str, pd.DataFrame
        Path to file containing query dataset in .tsv.gz format or query as dataframe. Should 
        contain columns ['chr','start','gene'].

    screen_df : pd.DataFrame
        SCREEN DataFrame, downloaded from https://screen.wenglab.org/downloads, All Human cCREs.
        Should contain columns ['chr','start','end','type'].
        
    f : float
        Specify overlap necessary for bedtools. Default (1e-9) is 1bp.

    Returns
    -------
    intersect_df : pd.DataFrame
        DataFrame containing SCREEN database information for query positions.

    '''

    if isinstance(query,str):
        query_df = pd.read_csv(query, sep='\t', compression='gzip',index_col=0)
    else:
        query_df = query

    query_df['ind'] = range(len(query_df))
    querybed = BedTool.from_dataframe(query_df[['chr','start','end','ind']])

    # screen enhancers
    screenbed = BedTool.from_dataframe(screen_df[['chr','start','end','type']])

    # intersect query bed with catlas bed
    col_names = ['query_chr','query_start','query_end','query_ind',
                    'screen_chr','screen_start','screen_end','screen_type']
    intersect_df = querybed.intersect(screenbed,wa=True,wb=True,f=f).to_dataframe()
    intersect_df.columns = col_names

    # add dummy cols for screen types
    dummy_df = pd.get_dummies(intersect_df['screen_type'])
    intersect_df = intersect_df.assign(**dummy_df)
    
    return intersect_df


def annot_enhancer(query_df,enh_df,label='enh',return_intersect=False,sep=';',enh_cols=None):
    ''' Annotate a SNP query_df with an enhancer_df.
    Must contain 'label' column containing a unique identifier + whatever additional info you like,
    separated by sep.
    '''

    enh_label_length = enh_df['label'].str.count(sep).max() # num of add info
    querybed = BedTool.from_dataframe(query_df[['chr','start','end','label']])
    enhbed = BedTool.from_dataframe(enh_df[['chr','start','end','label']])

    # intersect query bed with catlas bed
    col_names = ['query_chr','query_start','query_end','query_label',
                    'enh_chr','enh_start','enh_end','enh_label_w_info']
    intersect_df = querybed.intersect(enhbed,wa=True,wb=True,loj=True).to_dataframe()
    intersect_df.columns = col_names

    intersect_df[label] = (intersect_df.enh_chr != '.')
    intersect_df.loc[~intersect_df[label],'enh_chr'] = ''
    
    if enh_cols == None:
        enh_col_names = [f'{label}_name'] + [f'{label}_{x + 1}' for x in range(enh_label_length)]
    else:
        enh_col_names = [f'{label}_name'] + [f'{label}_{s}' for s in enh_cols]

    intersect_df[enh_col_names] = intersect_df['enh_label_w_info'].str.split(';',n=enh_label_length,expand=True)

    if return_intersect:
        return intersect_df

    query_grpby = intersect_df.groupby('query_label')
    query_df[f'{label}_sum'] = query_grpby[label].sum().values
    
    # if there's max 1 enhancer match per query, there are no need to make lists!
    if query_df[f'{label}_sum'].max() == 1:
        query_df[enh_col_names] = intersect_df[enh_col_names]
        return query_df

    # warning: slow!
    query_df[f'{label}_hits'] = query_grpby[f'{label}_name'].agg(list).values

    # warning: super slow!
    if len(enh_col_names[1:]) == 0:
        for enh_col in enh_col_names[1:]: # excluding _hits
            query_df[f'{label}_{enh_col}'] = query_grpby[enh_col].agg(list).values 

    return query_df

