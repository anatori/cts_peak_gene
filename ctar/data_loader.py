from pybedtools import BedTool
import pybedtools
from biomart import BiomartServer
import numpy as np
import pandas as pd 
import muon as mu
import anndata as ad
import scanpy as sc
import pickle
import re
import os


def peak_to_gene(peaks_df,genes_df,clean=True,split_peaks=True,distance=500000,col_names=['chr','start','end'],
                gene_col='gene',peak_col='peak',genome='hg38',sep=';'):

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
        If true, peaks must be in format chr[:-]start-end and will be extracted from this
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
    sep : str
        Separator for peak and gene in index.

    Returns
    -------
    peak_gene_links : pd.DataFrame
        DataFrame containing matched peak-gene links according to +/- distance windows
        around the gene body. The index is set to a unique identifier in the format 
        'peak , gene'.
    
    '''

    # add indices
    genes_df['index_y'] = range(len(genes_df))
    peaks_df['index_x'] = range(len(peaks_df))

    if clean:
        genes_clean = genes_df[~genes_df[col_names[0]].isna()] # assumes first item in col_names is chr label
        genes_clean = genes_clean[genes_clean.chr.str.startswith('chr')] # removes non chr mapped genes
        genes_clean[col_names[1:]] = genes_clean[col_names[1:]].astype(int)
    else: genes_clean = genes_df

    if split_peaks:
        peaks_clean = peaks_df.copy()
        peaks_clean[col_names] = peaks_clean[peak_col].str.extract(r'(\w+)[-:](\d+)-(\d+)')
        peaks_clean[col_names[1:]] = peaks_clean[col_names[1:]].astype(int)
    else: peaks_clean = peaks_df

    # creates temp bedtool files
    peaks_bed = BedTool.from_dataframe(peaks_clean[col_names+[peak_col]]).sort()
    genes_bed = BedTool.from_dataframe(genes_clean[col_names+[gene_col]]).sort()

    # add windows
    genes_bed_windowed = genes_bed.slop(genome=genome,b=distance).sort()
    
    # using bedtools closest, includes all distances from gene_bed (D='b')
    closest_links = peaks_bed.intersect(genes_bed_windowed,wo=True,sorted=True)

    # save as dataframe
    closest_links = closest_links.to_dataframe()
    closest_links.columns = ['peak_chr','peak_start','peak_end',peak_col,'window_chr','window_start','window_end',gene_col,'distance']

    # merge with original dataframe to obtain all information
    peak_gene_links = closest_links.merge(peaks_clean,on=peak_col)
    peak_gene_links = peak_gene_links.merge(genes_clean,on=gene_col)

    peak_gene_links = peak_gene_links[[gene_col,peak_col]+['distance','index_x','index_y']]
    
    # use unique indices
    peak_gene_links.index = peak_gene_links[peak_col].astype(str) + sep + peak_gene_links[gene_col].astype(str)

    return peak_gene_links


def add_gene_positions(row,dictionary,gene_col='gene',attributes=None):

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
    
    if attributes is None:
        attributes = ['chromosome_name', 'start_position', 'end_position']
    try:
        info = dictionary[row[gene_col]]
    except Exception:
        return [None] * len(attributes)
    out = []
    for attr in attributes:
        val = info.get(attr, None)
        out.append(val)
    return out


def normalize_chr(chrom):
    if chrom is None or (isinstance(chrom, float) and np.isnan(chrom)):
        return None
    chrom = str(chrom)
    if chrom.isnumeric() or chrom in ('X','Y'):
        return 'chr' + chrom
    if chrom in ('MT', 'M'):
        return 'chrM'
    if chrom.startswith('chr'):
        return chrom
    return chrom


def canonicalize_peak(s):
    """
    Intakes Series and normalize peak strings to 'chr{chr}:{start}-{end}'.
    Handles '1:100-200', 'chr1-100-200', 'chr1:100-200', etc.
    Strips whitespace.
    """
    peak_regex = re.compile(r"^(?:chr)?(?P<chr>[0-9XYMT]+)[:\-](?P<start>\d+)-(?P<end>\d+)$", re.IGNORECASE)
    s = s.astype(str).str.strip()
    def _canon(x: str) -> str:
        m = peak_regex.match(x)
        if not m:
            return x
        chrom = m.group("chr").upper()
        return f"chr{chrom}:{m.group('start')}-{m.group('end')}"
    return s.map(_canon)


def get_gene_coords(
    gene_df,
    gene_id_type='ensembl_gene_id',
    dataset='hsapiens_gene_ensembl',
    attributes=['chromosome_name','start_position','end_position','strand'],
    col_names=['chr','start','end','strand'],
    gene_col='gene',
    add_tss=True,
    tss_col='tss'
):
    
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
    add_tss : bool
        Optionally add gene TSS.
    tss_col : 'tss'
        What to label TSS column.

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
                                              args=(gene_positions,gene_col,attributes))
    gene_df['chr'] = gene_df['chr'].apply(normalize_chr)

    for c in ('start','end'):
        if c in gene_df.columns:
            gene_df[c] = pd.to_numeric(gene_df[c], errors='coerce').astype('Int64')
    if 'strand' in gene_df.columns:
        gene_df['strand'] = pd.to_numeric(gene_df['strand'], errors='coerce').astype('Int64')

    if add_tss:
        if 'strand' not in gene_df.columns:
            raise ValueError("strand is required to compute TSS. Include 'strand' in attributes.")
        gene_df[tss_col] = gene_df['start'].where(gene_df['strand'] == 1,gene_df['end']).astype('Int64')

    return gene_df


def map_bin_col(links_df, bin_df, links_ref_col='peak', bin_col='mean_gc_bin', prefix='atac_', suffix='_5.5'):
    ''' Add binning columns.
    '''
    bins_dic = bin_df[bin_col].to_dict()
    links_df[f'{prefix}{bin_col}{suffix}'] = links_df[links_ref_col].map(bins_dic)
    return links_df


def combine_peak_gene_bins(
    links_df, 
    atac_bins_df, 
    rna_bins_df, 
    peak_col='peak', 
    gene_col='gene', 
    atac_type='mean_gc', 
    rna_type='mean_var', 
    atac_bins=[5,5], 
    rna_bins=[10,10]
):
    ''' Adds combined_bin column to links_df containing the binning information for each peak-gene pair.

    Parameters
    -------
    links_df : pd.DataFrame
        DataFrame of length (# peak-gene pairs) containing both peak_col and gene_col.
    atac_bins_df : pd.DataFrame
        Dataframe of length (# peaks) containing peak_col and atac_type bin labels.
    rna_bins_df : pd.DataFrame
        Dataframe of length (# genes) containing gene_col and rna_type bin labels.
    peak_col : str
        Label for column containing peak IDs in atac_bins_df.
    gene_col : str
        Label for column containing gene IDs in rna_bins_df.
    atac_type : str
        Metric atac binning was based on.
    rna_type : str
        Metric rna binning was based on on.
    atac_bins : int or list (int)
        Number of desired atac bins for groupings.
    rna_bins : int or list (int)
        Number of desired rna bins for groupings.

    Returns
    -------
    links_df : pd.DataFrame
        Returns the original df with added combined_bin column.
    '''
    for num_bins in [atac_bins, rna_bins]:
        if isinstance(num_bins, list):
            assert len(num_bins) <= 2, (
                'Maximum of 2 types of bins supported.'
                )
    if isinstance(atac_bins, int):
        atac_suffix = f'{atac_bins}'
    else:
        atac_suffix = f'{atac_bins[0]}.{atac_bins[1]}'
    rna_suffix = f'{rna_bins[0]}.{rna_bins[1]}'

    atac_bins_df.index = atac_bins_df[peak_col]
    rna_bins_df.index = rna_bins_df[gene_col]

    links_df = map_bin_col(
        links_df, 
        atac_bins_df, 
        links_ref_col=peak_col, 
        bin_col=f'{atac_type}_bin', 
        prefix='atac_', 
        suffix=f'_{atac_suffix}'
    )
    links_df = map_bin_col(
        links_df, 
        rna_bins_df, 
        links_ref_col=gene_col, 
        bin_col=f'{rna_type}_bin', 
        prefix='rna_', 
        suffix=f'_{rna_suffix}'
    )

    links_df[f'combined_bin_{atac_suffix}.{rna_suffix}'] = links_df[f'atac_{atac_type}_bin_{atac_suffix}'].astype(str) + \
        '_' + links_df[f'rna_{rna_type}_bin_{rna_suffix}'].astype(str)

    return links_df


def groupby_combined_bins(links_df, combined_bin_col='combined_bin_5.5.5.5', return_dic=False):
    ''' Group peak-gene pairs by a binning column. Add 'links_tuple' columns with an Nx2 array of pairs within each bin,
    and 'index_xy' column with original links_df indices for each pair. 
    '''
    links_df['links_tuple'] = np.array(zip(links_df.index_x.values, links_df.index_y.values))
    links_df['index_xy'] = range(len(links_df))
    links_groupby_df = links_df.groupby(combined_bin_col).agg({
        'index_xy': lambda x: list(x),
        'links_tuple': lambda x: np.stack(x),
    })
    if return_dic:
        return (links_groupby_df['links_tuple'].to_dict(),
            links_groupby_df['index_xy'].to_dict()
        )

    return links_groupby_df


def map_dic_to_df(links_df, idx_dic, results_dic, col_name='poissonb'):
    ''' Use an index dictionary to map results to a dataframe. Assumes order is preserved in all operations.
    Length of results_dic and idx_dic values with matching keys MUST be the same length.
    '''
    links_df[col_name] = np.nan
    col_idx = links_df.columns.get_loc(col_name)

    for bin_key, arr in results_dic.items():
        bin_idx = idx_dic[bin_key]
        links_df.iloc[bin_idx, col_idx] = arr

    return links_df


def map_df_to_dic(links_df, keys_col='combined_bin', values_col='poissonb'):
    ''' Return dictionary with groupby column as keys and col_name column as values.
    '''
    return links_df.groupby(keys_col)[values_col].apply(np.array).to_dict()


def read_to_frag(x):
    ''' Convert reads to fragments (Martens Nat Meth 2024).
    '''
    # round to nearest even: np.round(x/2) * 2
    # then divide by 2: np.round(x/2) * 2 / 2
    return np.round(x / 2).astype(int)


def filter_sparsity(adata, sparsity=0.01):
    ''' Filter by sparsity.
    '''
    adata.var['sparsity'] = adata.X.getnnz(axis=0) / adata.shape[0]
    mask = (adata.var['sparsity'] < sparsity)
    return adata[:,~mask].copy()


def preprocess_mu(mdata,
    min_genes = 250,
    min_cells = 50,
    counts_per_cell_after = 1e4,
    n_cells_by_counts = 10,
    n_genes_by_counts_min = 2000,
    n_genes_by_counts_max = 15000,
    total_counts_min = 4000,
    total_counts_max = 40000,
    sparsity=0.01,
    convert_read_to_frag=True,
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
    adata_rna.X = adata_rna.X.astype(np.int32) # int32 to reduce memory cost
    adata_atac = mdata.mod['atac'].copy()
    adata_atac.X = adata_atac.X.astype(np.int32) # int32 to reduce memory cost
    print('Raw, adata_rna', adata_rna.shape, 'adata_atac', adata_atac.shape)

    # Filtering of RNA (using scanpy)
    sc.pp.filter_cells(adata_rna, min_genes = min_genes)
    sc.pp.filter_genes(adata_rna, min_cells = min_cells)
    print('Filtering RNA', adata_rna.shape)
    if sparsity:
        adata_rna = filter_sparsity(adata_rna, sparsity=sparsity)
        print('Sparsity filter RNA', adata_rna.shape)


    # Filtering of ATAC (using mu)
    # Following tutorial https://muon-tutorials.readthedocs.io/en/latest/
    # single-cell-rna-atac/pbmc10k/2-Chromatin-Accessibility-Processing.html
    sc.pp.calculate_qc_metrics(adata_atac, percent_top=None, log1p=False, inplace=True)
    mu.pp.filter_var(adata_atac, 'n_cells_by_counts', lambda x: x >= n_cells_by_counts)
    mu.pp.filter_obs(adata_atac, 'n_genes_by_counts', lambda x: (x >= n_genes_by_counts_min) & (x <= n_genes_by_counts_max))
    mu.pp.filter_obs(adata_atac, 'total_counts', lambda x: (x >= total_counts_min) & (x <= total_counts_max))
    print('Filtering ATAC', adata_atac.shape)
    if sparsity:
        adata_atac = filter_sparsity(adata_atac, sparsity=sparsity)
        print('Sparsity filter ATAC', adata_atac.shape)
    if convert_read_to_frag:
        adata_atac.X = read_to_frag(adata_atac.X)
        print('Converted reads to fragments')

    # Align cells
    cell_list = [x for x in adata_rna.obs_names if x in set(adata_atac.obs_names)]
    adata_rna = adata_rna[cell_list, :].copy()
    adata_atac = adata_atac[cell_list, :].copy()
    print('Align cells, adata_rna', adata_rna.shape, 'adata_atac', adata_atac.shape)

    pp_mdata = mu.MuData({'atac':adata_atac,'rna':adata_rna})

    return pp_mdata


def extract_range(filename):
    ''' Helper function for consolidate_null. Matches results according to
    indices, rather than file order.
    '''
    match = re.search(r'(\d+)_(\d+)\.npy$', filename)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (0, 0)  # return if no match is found


def check_missing_intervals(sorted_filenames):
    ''' Helper function for consolidate_null. Check if any missing files
    according to sorted interval.
    '''
    missing_intervals = []
    prev_end = None
    for i, fname in enumerate(sorted_filenames):
        start, end = extract_range(fname)
        if prev_end is not None and start != prev_end:
            missing_intervals.append((prev_end, start, i))
        prev_end = end
    
    if missing_intervals:
        print("Missing intervals detected:")
        for interval in missing_intervals:
            print(f"  [{interval[2]}] Missing interval: {interval[0]} to {interval[1]}")

    return missing_intervals


def check_missing_bins(ctrl_path, corr_path, prefix = 'pearsonr_'):
    '''  Check if any results are missing from corr_path based on 
    available control bins in ctrl_path.
    '''

    corr_file = os.listdir(corr_path)
    ctrl_file = os.listdir(ctrl_path)

    missing_bins = []
    for f in ctrl_file:
        str_f = prefix + f
        if str_f not in corr_file:
            missing_bins.append(f)

    if missing_bins:
        print("Missing bins detected:")
        for bin_ in missing_bins:
            print(f"  Missing bin: {bin_}")

    return missing_bins


def consolidate_ranged_npy(path, startswith, reg=r".*_(\d+)_(\d+)\.npy$", allow_pickle=False, require_contiguous=True):
    """
    Loads all npy files in `path` starting with `startswith` and ending with _{start}_{end}.npy,
    sorts them by (start, end), and concatenates along axis 0.

    Parameters
    ----------
    path : str
        Path containing input files.
    startswith : str
        Prefix for all input files.

    Returns
    -------
    arr : np.ndarray
        Concatenated array.
    ranges : list[tuple[int,int,str]]
        (start, end, filename) for each chunk, in concatenation order.
    """
    _RANGE_RE = re.compile(reg)

    files = [f for f in os.listdir(path) if f.startswith(startswith) and f.endswith(".npy")]
    ranged = []
    for f in files:
        m = _RANGE_RE.match(f)
        if not m:
            continue
        s, e = int(m.group(1)), int(m.group(2))
        ranged.append((s, e, f))

    if not ranged:
        raise FileNotFoundError(f"No ranged files found for prefix {startswith} in {path}")

    ranged.sort(key=lambda x: (x[0], x[1]))

    # enforce contiguous, non-overlapping, increasing ranges
    if require_contiguous:
        for (s1, e1, f1), (s2, e2, f2) in zip(ranged[:-1], ranged[1:]):
            if e1 != s2:
                raise ValueError(f"Non-contiguous ranges: {f1} ends at {e1}, next {f2} starts at {s2}")

    arrays = []
    for s, e, f in ranged:
        arr = np.load(os.path.join(path, f), allow_pickle=allow_pickle)
        arrays.append(arr)

        # sanity check: first dimension should match (end-start)
        if arr.shape[0] != (e - s):
            raise ValueError(f"Chunk {f} has shape[0]={arr.shape[0]} but expected {e-s} from range {s}_{e}")

    out = np.concatenate(arrays, axis=0)
    return out, ranged


def consolidate_null_dic(path, startswith='ctrl_coeff_dic_', remove_empty=True, print_missing=False):
    '''Consolidate null dictionaries from batch job into single dictionary.

    Parameters
    ----------
    path :  str
        Path containing input files. 
    startswith : str
        Prefix for all input files. 
    remove_empty : bool
        Determines whether to skip empty dictionaries or not.
    print_missing : bool
        Whether to print missing intervals.
    
    Returns
    ----------
    consolidated_dic :  dict
        Dictionary with consolidated null values from all batch files.

    '''

    null_dics = [x for x in os.listdir(path) if x.startswith(startswith) and x.endswith('.pkl')]

    if len(null_dics) == 0:
        raise FileNotFoundError(f"No files starting with '{startswith}' and ending with '.pkl' found in {path}")

    # sort the filenames based on the extracted ranges
    sorted_filenames = sorted(null_dics, key=extract_range)
    
    if print_missing:
        missing_intervals = check_missing_intervals(sorted_filenames)

    consolidated_dic = {}
    for x in sorted_filenames:
        file_path = os.path.join(path, x)
        with open(file_path, 'rb') as f:
            dic = pickle.load(f)
        
        if remove_empty: 
            if len(dic) == 0:
                continue
        
        # merge dictionaries; keys should be unique across batches
        consolidated_dic.update(dic)
    
    print(f'Dictionary size: {len(consolidated_dic)} bins')

    return consolidated_dic


def load_validation_intersect_bed(path,
    validation_cols,
    score_type=float,
    flag_gene_match=True,
    flag_verbose=True
):
    ''' Process bedfile from bedtools intersection between (-a) validation data and (-b) candidate multiome links.
    Bedtools intersect documentation:
        https://bedtools.readthedocs.io/en/latest/content/tools/intersect.html
    4rd column should contain validation source information separated by `;`.
    8th column should contain candidate multiome peak then gene separated by `;`.

    Parameters
    ----------
    validation_cols :  List of str
        Column names for validation file wrt to original input (-a).
        Must contain `score`.
    score_type : dtype
        Type to coerce score column into, as bedtools only returns strings.
    flag_gene_match : bool
        Whether to match genes from validation data and multiome links.
        If True, be sure gene names use same convention (e.g. ENSEMBL ids).
    flag_verbose : bool
        Prints gene matching lengths.
    
    Returns
    ----------
    pd.DataFrame
        Dataframe with columns validation_cols, gene, bp_overlap.
    '''
    
    assert 'score' in validation_cols, "score not in {validation_cols}."

    df = pd.read_csv(path,sep='\t',header=None)
    df.columns = ['gt_chr','gt_start','gt_end','gt_id','chr','start','end','id','bp_overlap']

    gt_split_vals = df['gt_id'].str.split(';',expand=True).values
    len_gt_split_vals = len(gt_split_vals[0]) # sample split
    assert len(validation_cols) == len_gt_split_vals, \
        f"Length of provided validation_cols ({len(validation_cols)}) do not match number of splits ({len_gt_split_vals})"
    df[validation_cols] = gt_split_vals
    df[['peak','gene']] = df['id'].str.split(';',expand=True).values

    if score_type == bool:
        df['score'] = df['score'].map({'True':True,'False':False})
    else:
        df['score'] = df['score'].astype(score_type)

    if flag_gene_match:
        if flag_verbose:
            print('Initial overlap',df.shape)
        df = df[df.gt_gene == df.gene].copy()
        if flag_verbose:
            print('Gene-matched overlap',df.shape)

    return df[validation_cols + ['id','peak','gene','bp_overlap']]
