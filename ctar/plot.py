# Importing libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import warnings
import anndata as ad
import muon as mu
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as mpatches
import matplotlib as mpl
from scipy.stats import poisson, nbinom
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
from matplotlib.container import BarContainer
import math
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import re
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter1d



# custom func from https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)

    return newcmap


def rootogram(observed, expected, title=""):
    '''
    Creates a rootogram to compare observed and expected counts.

    Args:
        observed (np.array): Array of observed counts.
        expected (np.array): Array of expected counts (same length as observed).
        title (str): Optional title for the plot.
    '''

    obs_freq = np.bincount(observed)  # Count occurrences of each count value
    exp_freq = expected * len(observed) # Scale to total num

    # Ensure both arrays are the same length. Pad with zeros.
    max_len = max(len(obs_freq), len(exp_freq))
    obs_freq = np.pad(obs_freq, (0, max_len - len(obs_freq)))
    exp_freq = np.pad(exp_freq, (0, max_len - len(exp_freq)))

    # Calculate square roots
    sqrt_obs = np.sqrt(obs_freq)
    sqrt_exp = np.sqrt(exp_freq)

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.bar(np.arange(len(sqrt_obs)), sqrt_obs - sqrt_exp, width=1, color="blue", alpha=0.7, label="Observed - Expected")
    plt.axhline(0, color="black", linewidth=0.8)  # Line at y=0
    plt.xlabel("Count Value")
    plt.ylabel("√Observed Frequency - √Expected Frequency")
    plt.title(f"Rootogram ({title})")
    plt.legend()
    plt.show()


def format_pval(p):
    try:
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return ""
        p = float(p)
    except Exception:
        return str(p)
    if p <= 0:
        return "<1e-300"
    if p < 1e-4:
        return "<1e-4"
    if p < 0.001:
        return f"{p:.4f}"
    if p < 0.01:
        return f"{p:.3f}"
    if p < 0.1:
        return f"{p:.2f}"
    return f"{p:.2f}"


# line plot helper
def plot_line_collection(ax, x, y, colors):
    
    x_segments = np.column_stack([x[:-1], x[1:]]) # adjacent x vals
    y_segments = np.column_stack([np.zeros_like(y[:-1]), y[:-1]]) # from 0 to y value
    segments = np.stack([x_segments, y_segments], axis=-1)
    
    lc = mpl.collections.LineCollection(segments, colors=colors, linewidths=0.5)
    ax.add_collection(lc)

# sorting helper
def sortby_ct_adata(adata,obs_col='celltype'):
    
    color_axis = adata.obs.sort_values(obs_col) # assumes rna is the same as atac
    
    return adata[color_axis.index,:]


def multiome_trackplot(df, atac = None, rna = None, sortby = 'poiss_coeff', coeff_label = 'coeff', coeff = 'poiss_coeff', pval_label = 'pval', pval = 'mc_pval', 
                        ascending = False, top_n = 10, adata_sort = False, presorted = False, sort_cells = 'rna', obs_col='celltype',
                        height = 5, width = None, axlim = None):

    """Custom trackplot for showing ATAC and RNA information for top peak-gene pairs across all cells, sorted by
    celltype. Within celltype, both ATAC and RNA results are sorted by cells with the highest RNA expression.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing peak-gene links labelled with "gene" existing in 
        rna_adata index, and "peak" existing in atac_adata index
        for each peak-gene link. Contains column sortby.
    atac : ad.AnnData
        AnnData object. Should contain ATAC modality with var attribute indices 
        as "peak".
    rna : ad.AnnData
        AnnData object. Should contain RNA modality with var attribute indices 
        as "gene".
    sortby : str
        Column in df by which to order plot.
    top_n : int
        Describes how many of the top results will appear in plot.
    mdata_sort : bool
        If False, means mdata is already sorted by CT in desired order. If not,
        will call sortby_ct_mdata to sort it (takes longer).
    height : int
        Figure height passed into matplotlib.
    width : int | None
        Figure width passed into matplotlib. If None, width is scaled to the
        number of cell types and capped to avoid overly wide plots.

    Returns
    -------
    None :
        Does not return anything; automatically plots the trackplot when called.
        
    """
    
    required_cols = ['gene', 'peak', sortby, coeff, pval]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f'multiome_trackplot missing required columns: {missing_cols}')

    if sort_cells not in {'rna', 'atac', 'share_rna'}:
        raise ValueError("sort_cells must be one of {'rna', 'atac', 'share_rna'}")

    if obs_col not in rna.obs.columns:
        raise ValueError(f'{obs_col} not found in rna.obs')

    if atac is None or rna is None:
        raise ValueError('Both atac and rna AnnData objects are required.')

    # sort mdata if necessary
    if not adata_sort:
        atac_sorted = atac
        rna_sorted = rna
    else:
        atac_sorted = sortby_ct_adata(atac,obs_col=obs_col)
        rna_sorted = sortby_ct_adata(rna,obs_col=obs_col)

    ct_labels = rna_sorted.obs[obs_col].drop_duplicates().to_numpy()
    max_tracks = min(top_n, len(ct_labels))
    if width is None:
        width = min(18, max(10, 6 + 0.6 * len(ct_labels)))

    # sort df by top values of sortby arg
    if not presorted:
        sorted_df = df.sort_values(sortby,ascending=ascending).head(max_tracks).copy()
    else:
        sorted_df = df.head(max_tracks).copy()
    sorted_df = sorted_df.reset_index(drop=True)

    rna_feature_col = 'gene' if 'gene' in rna_sorted.var.columns else None
    atac_feature_col = 'peak' if 'peak' in atac_sorted.var.columns else None

    rna_feature_names = rna_sorted.var[rna_feature_col].astype(str) if rna_feature_col else pd.Series(rna_sorted.var_names.astype(str), index=rna_sorted.var_names)
    atac_feature_names = atac_sorted.var[atac_feature_col].astype(str) if atac_feature_col else pd.Series(atac_sorted.var_names.astype(str), index=atac_sorted.var_names)

    rna_lookup = pd.Series(np.arange(rna_sorted.n_vars), index=rna_feature_names.to_numpy())
    atac_lookup = pd.Series(np.arange(atac_sorted.n_vars), index=atac_feature_names.to_numpy())

    missing_genes = sorted_df.loc[~sorted_df['gene'].astype(str).isin(rna_lookup.index), 'gene'].astype(str).unique()
    missing_peaks = sorted_df.loc[~sorted_df['peak'].astype(str).isin(atac_lookup.index), 'peak'].astype(str).unique()
    if len(missing_genes) > 0:
        raise ValueError(f'Genes not found in rna features: {missing_genes[:10].tolist()}')
    if len(missing_peaks) > 0:
        raise ValueError(f'Peaks not found in atac features: {missing_peaks[:10].tolist()}')

    gene_indices = rna_lookup.loc[sorted_df['gene'].astype(str)].to_numpy(dtype=int)
    peak_indices = atac_lookup.loc[sorted_df['peak'].astype(str)].to_numpy(dtype=int)

    # create subplots, including a smaller plot at the bottom for the ct colormap
    nrows = len(sorted_df)
    fig,axs = plt.subplots(
        figsize=(width,height), nrows=nrows+1, ncols=2,
        sharex='col', sharey=False, gridspec_kw={'hspace':0,'wspace':0,'height_ratios':[2]*nrows + [0.5]}
        )
    axs = np.asarray(axs, dtype=object).reshape(nrows + 1, 2)

    # extract gene and peak ids
    y_axis_rna = sorted_df.gene.values
    y_axis_atac = sorted_df.peak.values
    x_axis = np.arange(atac_sorted.shape[0])

    # extract relevant raw gene and peak information
    rna_matrix = rna_sorted[:, gene_indices].X
    atac_matrix = atac_sorted[:, peak_indices].X
    rna_matrix = rna_matrix.toarray() if hasattr(rna_matrix, 'toarray') else np.asarray(rna_matrix)
    atac_matrix = atac_matrix.toarray() if hasattr(atac_matrix, 'toarray') else np.asarray(atac_matrix)

    # take ylims for plotting
    maxpos = max(np.max(rna_matrix),np.max(atac_matrix))
    maxneg = min(np.min(rna_matrix),np.min(atac_matrix))
    if axlim != None:
        maxpos, maxneg = axlim

    if atac_matrix.shape[0] != rna_matrix.shape[0]:
        raise ValueError('number of cell must be the same between modalities.')

    # get ct labels and number of cells
    cts = rna_sorted.obs[obs_col].reset_index(drop=True) # ct should be the same between groups so im using rna as ref
    ct_sizes = np.concatenate(([0], np.cumsum([(cts == ct).sum() for ct in ct_labels])))
    ct_centers = (ct_sizes[:-1] + ct_sizes[1:]) / 2

    color_length = len(ct_sizes)-1
    cmap = plt.get_cmap('tab20',color_length)
    
    for i in np.arange(len(sorted_df)):
        row = sorted_df.iloc[i]
        
        axs[i,0].tick_params(
            axis="y",
            labelsize="x-small",
            right=False,
            left=False,
            length=2,
            which="both",
            labelright=False,
            labelleft=False,
            labelbottom=False,
            direction="in",
        ) # modified from scanpy

        axs[i,0].tick_params(axis='x',labelleft=False,labelbottom=False)
        axs[i,0].set_ylabel(y_axis_rna[i], rotation=0, fontsize="small", ha="right", va="bottom")
        axs[i,0].yaxis.set_label_coords(-0.005, 0.1)

        # corr label
        axs[i,1].set_xlim(0,len(x_axis))
        axs[i,1].set_ylim(maxneg,maxpos)
        # peak names not listed, as they don't have much meaning

        axs[i,1].tick_params(
            axis="y",
            labelsize="x-small",
            right=False,
            left=False,
            length=2,
            which="both",
            labelright=False,
            labelleft=False,
            labelbottom=False,
            direction="in",
        ) # modified from scanpy

        axs[i,1].tick_params(axis='x',labelleft=False,labelbottom=False)
        axs[i,1].set_ylabel(coeff_label +': ' + str(round(row[coeff],3))+ \
                            '\n' + pval_label +': '+ f'{float(f"{row[pval]:.1g}"):g}',
                            rotation=0, fontsize="x-small", ha="left", va="bottom")
        axs[i,1].yaxis.set_label_coords(1.005, 0.1)

        # get info for particular link
        peak = atac_matrix[:, [i]]
        gene = rna_matrix[:, [i]]

        # rna gene label
        rna_max = np.max(gene)
        rna_min = np.min(gene)
        atac_max = np.max(peak)
        atac_min = np.min(peak)

        axs[i, 0].set_ylim(rna_min, rna_max)
        axs[i, 1].set_ylim(atac_min, atac_max)
        axs[i, 0].set_xlim(-0.5, len(x_axis) - 0.5)
        axs[i, 1].set_xlim(-0.5, len(x_axis) - 0.5)

        # for each ct
        for k in np.arange(len(ct_labels)):

            # plot data
            rnay_curve = gene[ct_sizes[k]:ct_sizes[k+1],:].flatten() # ct data for plot
            atacy_curve = peak[ct_sizes[k]:ct_sizes[k+1],:].flatten()

            # rna
            if sort_cells == 'rna':
                order = np.argsort(-rnay_curve)
            elif sort_cells == 'atac':
                order = np.argsort(-atacy_curve)
            elif sort_cells == 'share_rna':
                order = np.argsort(-rnay_curve)

            rnay_curve = rnay_curve[order]
            atacy_curve = atacy_curve[order]

            plot_line_collection(axs[i,0], x_axis[ct_sizes[k]:ct_sizes[k+1]], rnay_curve, cmap(k)) # line segments
            axs[i,0].axvline(ct_sizes[k+1],linestyle='dashed',color='grey',alpha=0.3) # dashed grey line to separate cts

            plot_line_collection(axs[i,1], x_axis[ct_sizes[k]:ct_sizes[k+1]], atacy_curve, cmap(k))
            axs[i,1].axvline(ct_sizes[k+1],linestyle='dashed',color='grey',alpha=0.3)

            if ct_sizes[k+1] == ct_sizes[-1]:
                axs[i,0].axvline(ct_sizes[k+1],color='black')

        # reference lines at y=0
        axs[i,0].axhline(0,color='grey') 
        axs[i,1].axhline(0,color='grey')

        # remove border
        
        axs[i,0].spines['top'].set_visible(False)
        axs[i,0].spines['right'].set_visible(False)
        axs[i,0].spines['bottom'].set_visible(False)
        axs[i,0].spines['left'].set_visible(False)

        axs[i,1].spines['top'].set_visible(False)
        axs[i,1].spines['right'].set_visible(False)
        axs[i,1].spines['bottom'].set_visible(False)
        axs[i,1].spines['left'].set_visible(False)

    # rna colormap
    norm = mpl.colors.BoundaryNorm(ct_sizes, cmap.N)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),cax=axs[len(sorted_df),0], orientation='horizontal',spacing='proportional',ticks=ct_sizes[:color_length])
    
    cbar.set_ticks([])
    cbar.ax.set_xlim(-0.5, len(x_axis) - 0.5)
    cbar.ax.set_xticks(ct_centers, minor=True)
    cbar.ax.set_xticklabels(ct_labels,fontsize=9,rotation=90,minor=True,ha='center')
    cbar.outline.set_visible(False)

    axs[len(sorted_df),0].tick_params(axis='x',which='major',bottom=False,top=False,labelbottom=False)
    axs[len(sorted_df),0].tick_params(axis='x',which='minor',bottom=False,top=False,labelbottom=True,pad=2)


    # atac colormap
    norm = mpl.colors.BoundaryNorm(ct_sizes, cmap.N)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),cax=axs[len(sorted_df),1], orientation='horizontal',spacing='proportional',ticks=ct_sizes[:color_length])
    
    cbar.set_ticks([])
    cbar.ax.set_xlim(-0.5, len(x_axis) - 0.5)
    cbar.ax.set_xticks(ct_centers, minor=True)
    cbar.ax.set_xticklabels(ct_labels,fontsize=9,rotation=90,minor=True,ha='center')
    cbar.outline.set_visible(False)

    axs[len(sorted_df),1].tick_params(axis='x',which='major',bottom=False,top=False,labelbottom=False)
    axs[len(sorted_df),1].tick_params(axis='x',which='minor',bottom=False,top=False,labelbottom=True,pad=2)

    # add modality labels on top
    axs[0,0].set_title('RNA',fontdict={'fontsize':10})
    axs[0,1].set_title('ATAC',fontdict={'fontsize':10})

    # fig.supylabel(sortby,x=0.93,rotation=270,fontsize=12) # label values
    plt.show();
    
    return


def get_bar_centers_grouped(ax, odds_data):
    """
    Corrects for patch order in matplotlib when using pandas bar plot with yerr.
    Returns: dict of method_name -> list of x-center positions per top_n group.
    """
    method_names = list(odds_data.columns)
    topn_groups = odds_data.index
    n_methods = len(method_names)
    n_groups = len(topn_groups)

    patches = ax.patches
    assert len(patches) == n_methods * n_groups, f"Expected {n_methods * n_groups} patches, got {len(patches)}"

    bar_centers = {method: [] for method in method_names}

    # patches are ordered: method_0 for all groups, then method_1 for all groups, etc.
    for m, method in enumerate(method_names):
        for g in range(n_groups):
            patch = patches[m * n_groups + g]
            x_center = patch.get_x() + patch.get_width() / 2
            bar_centers[method].append(x_center)

    return bar_centers


def plot_delta_or_with_significance(
    focal_method,
    selected_comparison_methods,
    all_delta_or_data,
    all_yerr,
    corrected_pval_lookup,
    nlinks_ls,
    selected_methods,
    title_type='OR',
    alpha_threshold=0.3,
    figsize=(15, 5),
    color_dic=None,
):
    """
    Plot delta odds ratios with error bars and FDR-corrected significance annotations.

    Parameters
    ----------
    focal_method: str
      The method used as the focal point for comparison.
    selected_comparison_methods: list of str
      Subset of methods to include in the comparison plot.
    all_delta_or_data: dict
      Output from bootstrap comparison function (focal -> delta OR array).
    all_yerr: dict
      Error bars dictionary (focal -> std deviations * 1.96).
    corrected_pval_lookup: dict
      (focal, nlink, comparison) -> (raw_pval, corrected_pval).
    nlinks_ls: list of int
      List of `nlinks` values for x-axis indexing.
    selected_methods: list of str
      Full list of methods, needed for column alignment.
    title_type: str, default 'OR'
      Fills in 'delta_{title_type}' for plot title.
    alpha_threshold: float, optional
      FDR threshold to annotate p-values. Default is 0.3.
    figsize: tuple, optional
      Size of the plot.
    color_dic: dictionary, optional
      Maps colors for each method.
    """

    comparison_methods = [m for m in selected_comparison_methods if m != focal_method]
    yerr = all_yerr[focal_method]

    delta_or_df = pd.DataFrame(
        data=all_delta_or_data[focal_method],
        index=nlinks_ls,
        columns=[m for m in selected_methods if m != focal_method]
    )

    if color_dic is not None:

        ax = delta_or_df[comparison_methods].plot(
            kind='bar',
            figsize=figsize,
            yerr=yerr,
            width=0.7,
            color=color_dic,
        )
    
    else:

        ax = delta_or_df[comparison_methods].plot(
        kind='bar',
        figsize=figsize,
        yerr=yerr,
        width=0.7
    )

    # Build bar labels from p-values
    labels = []
    for comp in comparison_methods:
        for key in nlinks_ls:
            raw, corr = corrected_pval_lookup.get((focal_method, key, comp), (np.nan, np.nan))
            if corr < alpha_threshold:
                labels.append(f"{raw:.3g}\n({corr:.3g})")
            else:
                labels.append("")

    # Annotate bars
    label_idx = 0
    for container in ax.containers:
        if isinstance(container, BarContainer):
            num_bars = len(container)
            if label_idx + num_bars <= len(labels):
                ax.bar_label(
                    container,
                    labels=labels[label_idx:label_idx + num_bars],
                    fmt='%s',
                    bbox=dict(boxstyle='round,pad=0.1',
                              facecolor='white', edgecolor='grey', linewidth=0.5),
                    label_type='edge',
                    fontsize=8
                )
                label_idx += num_bars

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin * 1.05, ymax * 1.05)
    ax.set_xlabel('top_nlinks')
    ax.set_ylabel(f'delta_{title_type}')
    ax.set_title(f'delta_{title_type} ({focal_method} - others)')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


def add_sig_vs_reference_staggered_global_fdr(
    ax, odds_data, yerr_upper, corrected_pval_lookup,
    focal_method, ref_idx, alpha=0.05,
    line_gap=0.05,  # spacing between lines
    text_gap=0.03,  # extra spacing above lines for the box
    fontsize=8,
    fontsize_gap=2
):
    """
    Annotates significance comparisons against a reference method with FDR-corrected p-values.

    Parameters
    ----------
        ax : matplotlib axis
        odds_data : pd.DataFrame [n_links x methods]
        yerr_upper : pd.DataFrame [n_links x methods]
        corrected_pval_lookup : dict[(focal_method, n_link, comparison_method)] = (raw_p, corr_p)
        focal_method : str
        ref_idx : int (index of focal method in columns)
        alpha : significance threshold
        line_gap : vertical space between stacked brackets
        text_gap : space between brackets and text box
    """
    num_groups = odds_data.shape[0]
    num_methods = odds_data.shape[1]
    ref_method = odds_data.columns[ref_idx]
    bar_centers = get_bar_centers_grouped(ax, odds_data)

    for i, n_link in enumerate(odds_data.index):  # x-axis group
        group_methods = odds_data.columns

        # 1. Find maximum bar height + error bar in this group
        tops = odds_data.loc[n_link] + yerr_upper.loc[n_link]
        base_y = tops.max()

        # 2. Initialize vertical stacking level
        current_y = base_y + line_gap

        for m, comp_method in enumerate(group_methods):
            if m == ref_idx:
                continue

            key = (focal_method, n_link, comp_method)
            if key not in corrected_pval_lookup:
                continue

            raw_p, corr_p = corrected_pval_lookup[key]
            if corr_p >= alpha:
                continue  # Not significant

            # x-positions for bracket
            x_center = i
            x1 = bar_centers[comp_method][i]
            x2 = bar_centers[ref_method][i]

            # Draw vertical bracket above current_y
            line_top = current_y + line_gap
            ax.plot([x1, x1, x2, x2], [current_y, line_top, line_top, current_y], c='grey', lw=1.2)

            # Draw text above the line
            text_y = line_top + text_gap
            ax.text(
                (x1 + x2) / 2, text_y,
                f"{raw_p:.3g}\n({corr_p:.3g})",
                ha='center', va='bottom', fontsize=fontsize,
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='grey', linewidth=0.5)
            )

            # Update current_y to stack the next one higher
            current_y = (text_y + fontsize_gap) + text_gap # rough height for 2-line text


def add_sig_vs_multiple_references_staggered_global_fdr(
    ax, odds_data, corrected_pval_lookup,
    focal_methods, alpha=0.2,
    line_gap=0.05,
    text_gap=0.03,
    fontsize=8,
    fontsize_gap=2,
    line_color='grey',
    add_legend=True
):
    """
    Annotates significance comparisons (FDR-corrected p-values) between multiple methods.

    Handles both standard enrichment (multi-nlinks) and AUC (single-row) data.

    Parameters
    ----------
    ax : matplotlib axis
    odds_data : pd.DataFrame [n_links x methods]
    corrected_pval_lookup : dict[(focal_method, n_link, comparison_method)] = (raw_p, corr_p)
    focal_methods : list of methods to use as references
    alpha : FDR threshold for significance annotation
    """

    def get_sig_stars(q):
        if q < 0.0001:
            return '****'
        elif q < 0.001:
            return '***'
        elif q < 0.01:
            return '**'
        elif q < 0.05:
            return '*'
        else:
            return None

    is_auc = odds_data.shape[0] == 1
    bar_centers = get_bar_centers_grouped(ax, odds_data)
    drawn_pairs = set()

    if is_auc:
        n_link = odds_data.index[0]
        i = 0
        tops = odds_data.loc[n_link]
        base_y = tops.max()
        current_y = base_y + line_gap

        for focal_method in focal_methods:
            for comp_method in odds_data.columns:
                if comp_method == focal_method:
                    continue

                pair_key = tuple(sorted([focal_method, comp_method]))
                if pair_key in drawn_pairs:
                    continue
                drawn_pairs.add(pair_key)

                key = (focal_method, n_link, comp_method)
                if key not in corrected_pval_lookup:
                    continue

                _, corr_p = corrected_pval_lookup[key]
                stars = get_sig_stars(corr_p)

                if not stars:
                    continue

                x1 = bar_centers[comp_method][i]
                x2 = bar_centers[focal_method][i]

                line_top = current_y + line_gap
                ax.plot([x1, x1, x2, x2], [current_y, line_top, line_top, current_y], c=line_color, lw=1.2)

                text_y = line_top + text_gap
                ax.text((x1 + x2) / 2, text_y, stars,
                        ha='center', va='bottom', fontsize=fontsize)

                current_y = text_y + fontsize_gap * 0.01 + text_gap

    else:
        for i, n_link in enumerate(odds_data.index):
            tops = odds_data.loc[n_link]
            base_y = tops.max()
            current_y = base_y + line_gap

            for focal_method in focal_methods:
                for comp_method in odds_data.columns:
                    if comp_method == focal_method:
                        continue

                    pair_key = tuple(sorted([focal_method, comp_method]))
                    pair_id = (n_link, pair_key)
                    if pair_id in drawn_pairs:
                        continue
                    drawn_pairs.add(pair_id)

                    key = (focal_method, n_link, comp_method)
                    if key not in corrected_pval_lookup:
                        continue

                    _, corr_p = corrected_pval_lookup[key]
                    stars = get_sig_stars(corr_p)

                    if not stars:
                        continue

                    x1 = bar_centers[comp_method][i]
                    x2 = bar_centers[focal_method][i]

                    line_top = current_y + line_gap
                    ax.plot([x1, x1, x2, x2], [current_y, line_top, line_top, current_y], c=line_color, lw=1.2)

                    text_y = line_top + text_gap
                    ax.text((x1 + x2) / 2, text_y, stars,
                            ha='center', va='bottom', fontsize=fontsize)

                    current_y = text_y + fontsize_gap * 0.01 + text_gap

    if add_legend:
        star_handles = [
            mpatches.Patch(facecolor='none', edgecolor='none', label='**** : FDR < 0.0001'),
            mpatches.Patch(facecolor='none', edgecolor='none', label='*** : FDR < 0.001'),
            mpatches.Patch(facecolor='none', edgecolor='none', label='** : FDR < 0.01'),
            mpatches.Patch(facecolor='none', edgecolor='none', label='* : FDR < 0.05'),
        ]
        return star_handles
    else:
        return []


def _ensure_method_column_inplace(df):
    """Ensure a 'method' column exists, even if it's currently the index."""
    if 'method' not in df.columns:
        # If index has a name, use it; otherwise use the generic 'index'
        idx_name = df.index.name if df.index.name else 'index'
        df.reset_index(inplace=True)
        df.rename(columns={idx_name: 'method'}, inplace=True)
    df['method'] = df['method'].astype(str)
    return df


def _format_p(p):
    if pd.isna(p): return ""
    p = float(p)
    if p < 1e-4: return "p<1e-4"
    elif p < 0.001: return f"p={p:.1e}"
    elif p < 0.1: return f"p={p:.3g}"
    else: return f"p={p:.2f}"


def forest_plot_multiple_labels(
    res_dic,
    labels,
    methods_order=None,
    estimate_col="estimate",
    ci_lower_col="ci_lower",
    ci_upper_col="ci_upper",
    pval_col="p_value",
    use_log_x=True,
    jitter=0.12,
    cmap_name="tab10",
    title=None,
    x_label=None,
    pval_label=None,         
    pval_position="by_point",
    pval_fontsize=9,
    figsize=(5,3),
    ax=None,
    add_legend=True,
    tight_layout=None,
    right_base_mult=1.02,
):
    # decide figure/axes
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    if tight_layout is None:
        tight_layout = created_fig

    # ensure labels exist
    missing = [lbl for lbl in labels if lbl not in res_dic]
    if missing:
        raise KeyError(f"Labels not found in res_dic: {missing}")

    # infer methods order from first label if not provided
    first_df = res_dic[labels[0]].copy()
    _ensure_method_column_inplace(first_df)
    if methods_order is None:
        methods_order = first_df['method'].astype(str).tolist()

    # base y positions
    y_base = np.arange(len(methods_order))[::-1]

    # colors per label
    cmap = plt.get_cmap(cmap_name)
    colors = {lbl: cmap(i % cmap.N) for i, lbl in enumerate(labels)}

    # y-axis
    ax.set_yticks(y_base)
    ax.set_yticklabels(methods_order)
    ax.invert_yaxis()

    # track global CI bounds to set x scale/ticks
    ci_mins = []
    ci_maxs = []

    # center offsets
    n_lbl = len(labels)
    center = (n_lbl - 1) / 2.0

    # keep plotted positions to place p-values later
    plotted = {}  # lbl -> dict(x, y, pvals_series)
    for j, lbl in enumerate(labels):
        df = res_dic[lbl].copy()
        _ensure_method_column_inplace(df)
        df = df[df['method'].isin(methods_order)]
        df = df.set_index('method').reindex(methods_order)

        # values
        for col in (estimate_col, ci_lower_col, ci_upper_col):
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in '{lbl}'. Columns: {list(df.columns)}")

        x = df[estimate_col].astype(float).to_numpy()
        ci_low = df[ci_lower_col].astype(float).to_numpy()
        ci_up = df[ci_upper_col].astype(float).to_numpy()

        # error bars
        left_err = x - ci_low
        right_err = ci_up - x
        left_err = np.where(np.isfinite(left_err) & (left_err >= 0), left_err, 0.0)
        right_err = np.where(np.isfinite(right_err) & (right_err >= 0), right_err, 0.0)

        # vertical offset for this label
        y = y_base + (j - center) * jitter

        ax.errorbar(
            x, y, xerr=[left_err, right_err],
            fmt='o', color=colors[lbl], ecolor=colors[lbl],
            capsize=4, markersize=6, linewidth=1.5,
            label=lbl
        )

        plotted[lbl] = {
            "x": x,
            "y": y,
            "pvals": df[pval_col] if pval_col in df.columns else None
        }

        # collect CI bounds
        finite_low = ci_low[np.isfinite(ci_low)]
        finite_up = ci_up[np.isfinite(ci_up)]
        if finite_low.size:
            ci_mins.append(np.nanmin(finite_low))
        if finite_up.size:
            ci_maxs.append(np.nanmax(finite_up))

    # reference line
    ax.axvline(1.0, color='k', linestyle='--', linewidth=1)

    # labels/title
    if x_label is None:
        x_label = "Average enrichment (95% CI)" if use_log_x else "Estimate (95% CI)"
    ax.set_xlabel(x_label)
    if title:
        ax.set_title(title)

    # x-scale formatting
    if use_log_x:
        ax.set_xscale('log')
        if ci_mins and ci_maxs:
            xmin = np.nanmin([v for v in ci_mins if np.isfinite(v) and v > 0])
            xmax = np.nanmax([v for v in ci_maxs if np.isfinite(v)])
            if np.isfinite(xmin) and xmin > 0 and np.isfinite(xmax):
                ex_min = math.floor(math.log2(max(xmin, 2**-8)))
                ex_max = math.ceil(math.log2(max(xmax, 1)))
                ticks = [2**e for e in range(ex_min, ex_max + 1)]
                if 1 not in ticks:
                    ticks.append(1)
                ax.set_xticks(sorted(set(ticks)))
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(axis='x', style='plain')
    else:
        if ci_mins and ci_maxs:
            left = np.nanmin(ci_mins)
            right = np.nanmax(ci_maxs)
            if np.isfinite(left) and np.isfinite(right):
                pad = 0.05 * (right - left)
                ax.set_xlim(max(0.0, left - pad), right + pad)

    # p-value annotations
    labels_to_annotate = [pval_label] if pval_label in labels else (labels if pval_label is None else [])
    if labels_to_annotate:
        xlim = ax.get_xlim()
        # spacing in data units for text placement
        if use_log_x:
            right_base = xlim[1]
            col_scale = 0.06  # separation between right-side columns (multiplicative)
        else:
            span = xlim[1] - xlim[0]
            pad = 0.02 * span
            right_base = xlim[1] + pad
            col_offset = 0.08 * span  # separation between right-side columns

        for k, lbl in enumerate(labels_to_annotate):
            info = plotted.get(lbl)
            if info is None or info["pvals"] is None:
                continue

            xs = info["x"]
            ys = info["y"]
            pvals = info["pvals"].to_numpy()

            if pval_position == "by_point":
                # annotate near each point
                for xi, yi, pv in zip(xs, ys, pvals):
                    if not np.isfinite(xi):
                        continue
                    if use_log_x:
                        # place 3% to the right multiplicatively
                        x_text = xi * 1.03 if xi > 0 else right_base * right_base_mult
                    else:
                        x_text = xi + pad
                    txt = _format_p(pv)
                    if txt:
                        ax.text(
                            x_text, yi, txt,
                            va='center', ha='left',
                            fontsize=pval_fontsize, color=colors[lbl], family='monospace'
                        )
            elif pval_position == "right":
                for yi, pv in zip(ys, pvals):
                    txt = _format_p(pv)
                    if txt:
                        ax.text(
                            right_base * right_base_mult, yi, txt,
                            va='center', ha='left',
                            fontsize=pval_fontsize, color=colors[lbl], family='monospace'
                        )
                # add margin for right columns (use this figure, not global plt)
                fig.subplots_adjust(right=0.82)

    ax.grid(axis='x', linestyle=':', linewidth=0.6, alpha=0.6)

    if add_legend:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=len(labels))

    if tight_layout:
        fig.tight_layout()

    return ax


def _stars(p, alpha=0.05):
    if p is None:
        return ""
    try:
        p = float(p)
    except Exception:
        return ""
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < alpha:
        return "*"
    return ""


def _draw_sig_bracket(ax, x1, x2, y, h, text, color='k', fontsize=9, lw=1.2):
    if ax.get_yscale() == 'log':
        y_top = y * (1.0 + h)
        ax.plot([x1, x1, x2, x2], [y, y_top, y_top, y], color=color, lw=lw)
        if text:
            ax.text((x1 + x2) / 2.0, y_top, text, ha='center', va='bottom', color=color, fontsize=fontsize)
    else:
        y_top = y + h
        ax.plot([x1, x1, x2, x2], [y, y_top, y_top, y], color=color, lw=lw)
        if text:
            ax.text((x1 + x2) / 2.0, y_top, text, ha='center', va='bottom', color=color, fontsize=fontsize)


def _normalize_pairwise_pvals(pairwise_pvals, method_renames):
    if pairwise_pvals is None:
        return None
    inverse_names = {}
    if method_renames:
        for internal, display in method_renames.items():
            if display != internal:
                inverse_names[display] = internal
    normalized = {}
    for lbl, pairs in pairwise_pvals.items():
        out = {}
        for key, p in pairs.items():
            if not isinstance(key, (tuple, list)) or len(key) != 2:
                continue
            a, b = key
            a_norm = inverse_names.get(a, a)
            b_norm = inverse_names.get(b, b)
            pair_norm = tuple(sorted((a_norm, b_norm)))
            if pair_norm in out:
                try:
                    out[pair_norm] = min(out[pair_norm], p)
                except Exception:
                    out[pair_norm] = p
            else:
                out[pair_norm] = p
        normalized[lbl] = out
    return normalized


def _normalize_vs_ref(pairwise_vs_ref, reference_method, method_renames):
    """
    pairwise_vs_ref format (per dataset), keys can be internal or display:
      { 'datasetA': {'scent': 0.01, 'scmm': 0.02, 'ctar_filt': None }, ... }
    reference_method can be internal or display; normalized to internal using method_renames.
    """
    if pairwise_vs_ref is None:
        return None

    # Build display->internal map
    inverse_names = {}
    if method_renames:
        for internal, display in method_renames.items():
            if display != internal:
                inverse_names[display] = internal

    # Normalize reference name
    ref_internal = inverse_names.get(reference_method, reference_method)

    normalized = {}
    for lbl, mp in pairwise_vs_ref.items():
        out = {}
        for m_key, p in mp.items():
            internal_m = inverse_names.get(m_key, m_key)
            if internal_m == ref_internal:
                continue  # skip self
            pair = tuple(sorted((internal_m, ref_internal)))
            out[pair] = p
        normalized[lbl] = out
    return normalized


def _map_to_internal(name, method_renames):
    if not method_renames:
        return name
    inv = {v: k for k, v in method_renames.items() if v != k}
    return inv.get(name, name)


def barplot_grouped_with_sig(
    res_dic,
    labels,
    metric=None,
    methods_order=None,
    metric_col="metric",
    estimate_col="estimate",
    ci_lower_col="ci_lower",
    ci_upper_col="ci_upper",
    show_ci=True,
    # Significance: read vs-reference p-values from a column in res_dic[label]
    pval_vs_ref_col=None,     # e.g., "p_value_vs_ctar_filt"
    reference_method="ctar_filt",
    alpha=0.05,
    show_brackets=True,
    fdr_correction=True,
    bracket_text="p",         # 'p', 'stars', or 'none'
    use_log_y=True,
    # Coloring
    method_colors=None,       # dict {internal_method -> color}
    palette=None,             # list-like (e.g., sns.color_palette('deep', n_colors=5))
    cmap_name="tab10",
    # Display renames
    label_renames=None,       # dict {original_label -> display_label}
    method_renames=None,      # dict {internal_method -> display_name}
    # Layout
    bar_width=0.1,
    bar_spacing_mult=1,
    group_pad=0.8,
    # Bracket spacing controls
    bracket_base_pad=0.04,    # start brackets this far above group max (fractional on log; fraction of max on linear)
    bracket_step=0.08,        # vertical step between brackets (fractional on log; fraction of max on linear)
    bracket_height=0.6,       # bracket height as a fraction of step
    title=None,
    y_label=None,
    figsize=(6,4),
    ax=None,
    add_legend=True,
    asterisk_fontsize=9,      # used only if bracket_text='stars'
    bracket_fontsize=9,
    bracket_linewidth=1.2,
    ci_legend_loc="upper right",
    ci_legend_fontsize=12,
    bbox_to_anchor=(0.5, -0.1),
    axhline=True,
    ncol=None,
):
    """
    Grouped bar plot of 'estimate' with CI error bars, colored by method, grouped by dataset label.
    Draws significance brackets per dataset between each method and the reference method, using p-values
    sourced from a column in res_dic[label] (pval_vs_ref_col).

    Bracket spacing:
      - On log scale: base = group_max * (1 + bracket_base_pad), each level y = base * (1 + level * bracket_step), height = bracket_step * bracket_height
      - On linear scale: base = group_max + bracket_base_pad * group_max, each level y = base + level * (bracket_step * group_max), height = (bracket_step * group_max) * bracket_height
    Increase bracket_step or bracket_base_pad to add space; reduce bracket_height to shrink bracket size.
    """

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    # Validate labels
    missing = [lbl for lbl in labels if lbl not in res_dic]
    if missing:
        raise KeyError(f"Labels not found in res_dic: {missing}")

    # Methods order
    first_df = res_dic[labels[0]].copy()
    _ensure_method_column_inplace(first_df)
    if methods_order is None:
        methods_order = first_df['method'].astype(str).tolist()
    M = len(methods_order)

    # Colors per method
    if method_colors is None:
        if palette is None:
            try:
                palette = list(sns.color_palette('deep', n_colors=max(M, 5)))
            except Exception:
                cmap = plt.get_cmap(cmap_name)
                palette = [cmap(i % cmap.N) for i in range(M)]
        else:
            palette = list(palette)
        method_colors = {m: palette[i % len(palette)] for i, m in enumerate(methods_order)}
    else:
        for i, m in enumerate(methods_order):
            if m not in method_colors:
                if palette is not None:
                    method_colors[m] = list(palette)[i % len(palette)]
                else:
                    cmap = plt.get_cmap(cmap_name)
                    method_colors[m] = cmap(i % cmap.N)

    # Normalize reference method to internal name if display provided
    ref_internal = _map_to_internal(reference_method, method_renames)

    # Positions
    bar_spacing = bar_width * bar_spacing_mult

    # AUTO group_pad to prevent overlap (unless user explicitly set it large enough)
    group_width = (M - 1) * bar_spacing + bar_width
    min_group_pad = group_width + 0.15  # add a little breathing room (tune 0.05–0.4)

    if group_pad is None:
        group_pad = min_group_pad
    else:
        group_pad = max(group_pad, min_group_pad)

    group_centers = np.arange(len(labels)) * group_pad
    within_offsets = np.array([(j - (M - 1) / 2.0) * bar_spacing for j in range(M)])

    positions = {}
    heights = {}
    err_up = {}
    err_low = {}

    # For reading per-dataset p-values vs reference from column
    per_label_vs_ref = {}  # lbl -> dict {internal_method -> pvalue_vs_ref}

    max_y_for_limits = []
    for i, lbl in enumerate(labels):
        df = res_dic[lbl].copy()
        if metric is not None:
            df = df[df[metric_col] == metric].copy()
        _ensure_method_column_inplace(df)
        df = df[df['method'].isin(methods_order)]
        df = df.set_index('method').reindex(methods_order)

        # Validate required columns
        if estimate_col not in df.columns:
            raise KeyError(f"Column '{estimate_col}' not found in '{lbl}'. Columns: {list(df.columns)}")

        # Validate CI columns only if we will draw CI
        if show_ci:
            for col in (ci_lower_col, ci_upper_col):
                if col not in df.columns:
                    raise KeyError(f"Column '{col}' not found in '{lbl}'. Columns: {list(df.columns)}")

        # Read p-values vs reference, if column provided (only matters if brackets are enabled)
        if pval_vs_ref_col is not None and pval_vs_ref_col in df.columns:
            per_label_vs_ref[lbl] = df[pval_vs_ref_col].to_dict()
        elif pval_vs_ref_col is not None and pval_vs_ref_col not in df.columns:
            raise KeyError(f"Column '{pval_vs_ref_col}' not found in '{lbl}'. Available: {list(df.columns)}")

        est = df[estimate_col].astype(float).to_numpy()

        if show_ci:
            lo = df[ci_lower_col].astype(float).to_numpy()
            hi = df[ci_upper_col].astype(float).to_numpy()
            e_low = np.clip(est - lo, 0, np.inf)
            e_up  = np.clip(hi - est, 0, np.inf)
        else:
            # CI disabled: set zero error so downstream logic (limits/brackets) still works
            e_low = np.zeros_like(est, dtype=float)
            e_up  = np.zeros_like(est, dtype=float)

        x_group = group_centers[i]
        xs = x_group + within_offsets

        for j, m in enumerate(methods_order):
            x = xs[j]
            y = est[j]
            positions[(lbl, m)] = x
            heights[(lbl, m)] = y
            err_up[(lbl, m)] = e_up[j]
            err_low[(lbl, m)] = e_low[j]

            color = method_colors[m]
            ax.bar(x, y, width=bar_width, color=color, edgecolor='none', alpha=0.9)

            if show_ci:
                ax.errorbar(
                    x, y,
                    yerr=[[e_low[j]], [e_up[j]]],
                    fmt='none', ecolor='black', elinewidth=1.2, capsize=4
                )

            max_y_for_limits.append(y + e_up[j])

    # Apply BH FDR correction within each label independently
    if fdr_correction and pval_vs_ref_col is not None and per_label_vs_ref:
        for lbl, pmap in per_label_vs_ref.items():
            entries = [(m, float(p)) for m, p in pmap.items()
                       if p is not None and np.isfinite(float(p))]
            if entries:
                ms, ps = zip(*entries)
                _, q_vals, _, _ = multipletests(ps, method='fdr_bh')
                for m, q in zip(ms, q_vals):
                    per_label_vs_ref[lbl][m] = q

    # X axis: dataset labels (renamed for display if provided)
    ax.set_xticks(group_centers)
    label_renames = label_renames or {}
    display_labels = [label_renames.get(lbl, lbl) for lbl in labels]
    ax.set_xticklabels(display_labels, rotation=0)

    # Y axis
    if use_log_y:
        ax.set_yscale('log')
        ax.get_yaxis().set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(axis='y', style='plain')

    # Reference baseline line (enrichment = 1.0)
    if axhline:
        ax.axhline(1.0, color='k', linestyle='--', linewidth=1)

    # Labels/title
    if y_label is None:
        y_label = "Weighted average enrichment (95% CI)" if show_ci else "Weighted average enrichment"
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)

    # Legend: method display names (+ optional CI legend)
    if add_legend:
        handles = []
        labels_legend = []
        for m in methods_order:
            patch = plt.Line2D(
                [0], [0],
                marker='s', color='w',
                markerfacecolor=method_colors[m],
                markersize=10,
                linestyle='None'
            )
            handles.append(patch)
            disp = method_renames.get(m, m) if method_renames else m
            labels_legend.append(disp)

        if ncol is None:
            ncol = len(methods_order)
        method_legend = ax.legend(
            handles, labels_legend,
            loc='upper center',
            bbox_to_anchor=bbox_to_anchor,
            frameon=False,
            ncol=ncol,
        )

        # Add CI legend only if CI is shown
        if show_ci:
            ci_handle = ax.errorbar(
                [np.nan], [np.nan],
                yerr=[[1], [1]],
                fmt='none',
                ecolor='k',
                elinewidth=1.4,
                capsize=4
            )

            ax.add_artist(
                ax.legend(
                    [ci_handle], ["Error bars = 95% CI"],
                    loc=ci_legend_loc,
                    frameon=True,
                    fontsize=ci_legend_fontsize
                )
            )

        ax.add_artist(method_legend)

    # Draw brackets vs reference using column-based p-values (OPTIONAL)
    if show_brackets and pval_vs_ref_col is not None and per_label_vs_ref:
        for i, lbl in enumerate(labels):
            pmap = per_label_vs_ref.get(lbl, {})
            if not pmap:
                continue

            # Ensure reference method is present in positions
            if (lbl, ref_internal) not in positions:
                continue

            group_max = max(
                heights[(lbl, m)] + err_up[(lbl, m)]
                for m in methods_order
                if (lbl, m) in heights
            )

            # Compute base, step, height with scale-aware semantics
            if ax.get_yscale() == 'log':
                base = group_max * (1.0 + bracket_base_pad)
                step_val = bracket_step
                height_val = step_val * bracket_height
            else:
                scale = max(group_max, 1.0)
                base = group_max + bracket_base_pad * scale
                step_val = bracket_step * scale
                height_val = step_val * bracket_height

            # Build pairs: every method vs reference (skip reference itself)
            pairs = []
            for m in methods_order:
                if m == ref_internal:
                    continue
                if (lbl, m) not in positions:
                    continue
                p = pmap.get(m, None)
                if p is None or np.isnan(p):
                    continue
                pairs.append(((m, ref_internal), p))

            # Sort by horizontal span to reduce overlap
            def span_len(pair):
                (m1, m2), _ = pair
                return abs(positions[(lbl, m1)] - positions[(lbl, m2)])
            pairs = sorted(pairs, key=span_len)

            level = 0
            for (m1, m2), p in pairs:
                if (lbl, m1) not in positions or (lbl, m2) not in positions:
                    continue

                x1 = positions[(lbl, m1)]
                x2 = positions[(lbl, m2)]

                if ax.get_yscale() == 'log':
                    y = base * (1.0 + level * step_val)
                    h = height_val
                else:
                    y = base + level * step_val
                    h = height_val

                if bracket_text == "p":
                    txt = _format_p(p)
                elif bracket_text == "stars":
                    txt = _stars(p, alpha=alpha)
                    # Skip drawing entirely if not significant (no stars)
                    if txt.strip() == "":
                        continue
                else:
                    txt = ""

                _draw_sig_bracket(
                    ax, x1, x2, y, h, txt,
                    color='k', fontsize=bracket_fontsize, lw=bracket_linewidth
                )
                level += 1

    ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.6)
    return ax


def residualize_raw(ctrl_df, target_cov, bins = ['atac_mean_bin', 'atac_gc_bin', 'rna_mean_bin', 'rna_var_bin'], corr="null_corr"):
    ''' Helper to residualize against other covariates, one target at a time.
    ctrl_df : Long dataframe containing one row per control statistic. Must contain bins in columns.
    target_cov : Focal covariate. Must be item in bins. Will automatically residualize against other covariates in bins.
    bins : Covariates by which to residualize control statistic against.
    corr : Column containing statistic to residualize.
    '''
    other_covs = [c for c in bins if c != target_cov]
    formula = f"{corr} ~ " + " + ".join(f"C({c})" for c in other_covs)
    resid = smf.ols(formula, data=ctrl_df).fit().resid
    ctrl_df = ctrl_df.copy()
    ctrl_df['resid'] = resid
    return ctrl_df


def plot_residual_distributions(ctrl_df, cov, xlabel='Residual null correlation', title='', corr="null_corr", figsize=(12,4)):
    """
    Plot CDF of residualized statistics.
    """
    fig, axes = plt.subplots(1, len(cov), figsize=figsize, sharey=True)

    for i, target_cov in enumerate(cov):
        resid_df = residualize_raw(ctrl_df, target_cov, corr=corr)
        levels = sorted(ctrl_df[target_cov].unique())
        # plot cdf
        for lvl in levels:
            vals = np.sort(resid_df.loc[resid_df[target_cov] == lvl, 'resid'].values)
            cdf = np.arange(1, len(vals)+1) / len(vals)
            axes[i].plot(vals, cdf, label=f'Bin {lvl}')
        axes[i].set_title(f'{target_cov}')
        axes[i].legend()

    axes[0].set_ylabel('CDF')
    fig.supxlabel(xlabel)
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()


def _parse_peak_coords(var, chrom_col=None, start_col=None, end_col=None):
    """Return chrom, start, end arrays for peaks in an AnnData .var table.

    If chrom_col/start_col/end_col are given, reads those columns directly.
    Otherwise tries to parse coordinates from the var index, e.g.
    'chr1:1000-2000' or 'chr1-1000-2000'.
    """
    if chrom_col and start_col and end_col:
        return (
            var[chrom_col].values,
            var[start_col].values.astype(int),
            var[end_col].values.astype(int),
        )
    pattern = re.compile(r'(chr[\w]+)[:\-](\d+)[:\-](\d+)')
    chroms, starts, ends = [], [], []
    for name in var.index:
        m = pattern.match(str(name))
        if not m:
            raise ValueError(
                f"Could not parse peak coordinates from '{name}'; "
                "pass chrom_col/start_col/end_col explicitly."
            )
        chroms.append(m.group(1))
        starts.append(int(m.group(2)))
        ends.append(int(m.group(3)))
    return np.array(chroms), np.array(starts), np.array(ends)


def _rna_pseudobulk(mdata, celltype_col, celltype_order, gene_name, rna_mod='rna', agg='mean'):
    """Pseudobulk a gene's RNA expression per celltype from mdata[rna_mod]."""
    rna_adata = mdata[rna_mod]
    if gene_name not in rna_adata.var_names:
        raise KeyError(f"'{gene_name}' not found in mdata['{rna_mod}'].var_names")
    gene_idx = rna_adata.var_names.get_loc(gene_name)
    X = rna_adata.X
    values = {}
    for ct in celltype_order:
        mask = (mdata[rna_mod].obs[celltype_col] == ct).values
        sub = X[mask][:, gene_idx]
        sub = np.asarray(sub.todense()).ravel() if hasattr(sub, 'todense') else np.asarray(sub).ravel()
        if len(sub) == 0:
            values[ct] = 0.0
        elif agg == 'mean':
            values[ct] = float(sub.mean())
        else:
            values[ct] = float(sub.sum())
    return values


def _smooth_track(win_starts, win_ends, vals, win_start, win_end, n_points=2000, smooth_sigma_frac=0.004):
    """Rasterize discrete peak heights onto a fine grid and Gaussian-smooth them
    into a continuous coverage-style curve (genome-browser look), rather than
    plotting each peak as a flat-topped bar.
    """
    grid = np.linspace(win_start, win_end, n_points)
    y = np.zeros(n_points)
    for s, e, v in zip(win_starts, win_ends, vals):
        mask = (grid >= s) & (grid <= e)
        if mask.any():
            y[mask] = np.maximum(y[mask], v)
    sigma_points = max(smooth_sigma_frac * n_points, 1)
    y_smooth = gaussian_filter1d(y, sigma=sigma_points)
    return grid, y_smooth



def plot_ctar_locus(
    mdata,
    celltype_col,
    gene_name,
    gene_chrom, gene_start, gene_end, gene_strand,
    peak_chrom, peak_start, peak_end,
    snp_chrom, snp_pos, snp_id,
    qvals,
    atac_mod='atac',
    chrom_col=None, start_col=None, end_col=None,
    celltype_order=None,
    celltype_colors=None,
    agg='sum',
    window_pad=5000,
    smooth=True,
    smooth_sigma_frac=0.004,
    rna_mod=None,
    rna_agg=None,
    pip_values=None,
    sig_style='lollipop',
    sig_threshold=0.05,
    method_colors=None,
    figsize=(12, 5.5),
    highlight_color='#D1495B',
    savepath=None,
):
    adata = mdata[atac_mod]
    chroms, starts, ends = _parse_peak_coords(adata.var, chrom_col, start_col, end_col)
 
    win_start = min(peak_start, gene_start, snp_pos) - window_pad
    win_end = max(peak_end, gene_end, snp_pos) + window_pad
    chrom = peak_chrom
 
    in_window = (chroms == chrom) & (ends >= win_start) & (starts <= win_end)
    win_starts, win_ends = starts[in_window], ends[in_window]
    peak_idx = np.where(in_window)[0]
    if len(peak_idx) == 0:
        raise ValueError("No peaks found in the requested window — check chrom naming/coords.")
 
    if celltype_order is None:
        celltype_order = sorted(mdata.obs[celltype_col].dropna().unique())
    n_ct = len(celltype_order)
 
    if celltype_colors is None:
        light = plt.get_cmap('tab20').colors[1::2]
        celltype_colors = {ct: light[i % len(light)] for i, ct in enumerate(celltype_order)}
 
    X = adata.X
    pseudobulk = {}
    for ct in celltype_order:
        mask = (mdata[atac_mod].obs[celltype_col] == ct).values
        sub = X[mask][:, peak_idx]
        vals = np.asarray(sub.sum(axis=0)).ravel()
        if agg == 'mean':
            n_cells = max(mask.sum(), 1)
            vals = vals / n_cells
        pseudobulk[ct] = vals
 
    global_max = max((v.max() if len(v) else 0) for v in pseudobulk.values())
    global_max = global_max if global_max > 0 else 1
 
    rna_values = None
    if rna_mod is not None and rna_mod in mdata.mod:
        rna_adata = mdata[rna_mod]
        if gene_name in rna_adata.var_names:
            rna_values = _rna_pseudobulk(
                mdata, celltype_col, celltype_order, gene_name,
                rna_mod=rna_mod, agg=rna_agg or agg,
            )
        else:
            print(f"Note: '{gene_name}' not found in mdata['{rna_mod}'].var_names — RNA panel skipped.")
 
    show_rna = rna_values is not None
 
    fig = plt.figure(figsize=figsize)
 
    # significance column made narrower (was 1.1 -> 0.7)
    if show_rna:
        outer_width_ratios = [3.7, 0.7]
    else:
        outer_width_ratios = [3, 0.7]
 
    HSPACE = 0.05
    outer_gs = fig.add_gridspec(
        n_ct + 2, 2,
        width_ratios=outer_width_ratios,
        height_ratios=[1] * n_ct + [0.55, 0.9],
        hspace=HSPACE, wspace=0.12,
    )
    sig_col = 1
 
    if show_rna:
        # IMPORTANT: pass the same hspace as outer_gs, otherwise this nested
        # subgridspec uses matplotlib's default hspace and its rows drift out
        # of vertical sync with outer_gs -> ax_sig's bottom no longer lines up
        # with the last track's bottom.
        main_gs = outer_gs[:n_ct + 2, 0].subgridspec(
            n_ct + 2, 2, width_ratios=[3, 0.7],
            height_ratios=[1] * n_ct + [0.55, 0.9],  # must match outer_gs's row heights
            wspace=0.03, hspace=HSPACE,
        )
        track_col, rna_col = 0, 1
    else:
        main_gs = outer_gs
        track_col, rna_col = 0, None
 
    track_axes = []
    for i, ct in enumerate(celltype_order):
        ax = fig.add_subplot(main_gs[i, track_col])
        track_axes.append(ax)
        ax.axhline(0, color='black', lw=0.8, zorder=1)
        vals = pseudobulk[ct] / global_max
        color = celltype_colors[ct]
 
        if smooth:
            grid, y_smooth = _smooth_track(win_starts, win_ends, vals, win_start, win_end,
                                            smooth_sigma_frac=smooth_sigma_frac)
            ax.fill_between(grid, 0, y_smooth, color=color, alpha=0.9, lw=0, zorder=2)
            ax.plot(grid, y_smooth, color=color, lw=1.2, zorder=3)
        else:
            min_w = (win_end - win_start) * 0.004
            for s, e, hh in zip(win_starts, win_ends, vals):
                width = max(e - s, min_w)
                center = (s + e) / 2
                ax.bar(center, hh, width=width, bottom=0, color=color, edgecolor='none', zorder=2)
 
        ax.set_xlim(win_start, win_end)
        ax.set_ylim(0, 1.08)
        ax.set_yticks([])
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if pip_values is not None and ct in pip_values:
            ax.text(-0.012, 0.66, ct, transform=ax.transAxes, ha='right', va='center', fontsize=9)
            ax.text(-0.012, 0.32, f'PIP: {pip_values[ct]:.2f}', transform=ax.transAxes,
                    ha='right', va='center', fontsize=7, color='#666666')
        else:
            ax.text(-0.012, 0.5, ct, transform=ax.transAxes, ha='right', va='center', fontsize=9)
 
    track_axes[0].set_title('Cell type ATAC tracks', fontsize=10, loc='left')
 
    ax_ann = fig.add_subplot(main_gs[n_ct:n_ct + 2, track_col])
    ax_ann.set_xlim(win_start, win_end)
    ax_ann.set_ylim(0, 1)
    ax_ann.axis('off')
 
    gene_rect = mpatches.Rectangle(
        (gene_start, 0.35), gene_end - gene_start, 0.3,
        facecolor='grey', alpha=0.6,
    )
    ax_ann.add_patch(gene_rect)
    strand_arrow = '\u2192' if gene_strand == '+' else '\u2190'
    ax_ann.text((gene_start + gene_end) / 2, 0.85, f'{gene_name} {strand_arrow}',
                ha='center', fontsize=9)
 
    ax_ann.plot([snp_pos, snp_pos], [0.35, 0.65], color=highlight_color, lw=2.5)
    ax_ann.text(snp_pos, 0.85, snp_id, ha='center', fontsize=8, color=highlight_color)
 
    ax_ann.text(
        -0.012, 0.12, chrom, transform=ax_ann.transAxes,
        ha='right', va='center', fontsize=9,
    )
    ax_ann.plot([win_start, win_end], [0.12, 0.12], color='black', lw=1.5, solid_capstyle='butt')
    tick_h = 0.04
    ax_ann.plot([win_start, win_start], [0.12 - tick_h, 0.12 + tick_h], color='black', lw=1.2)
    ax_ann.plot([win_end, win_end], [0.12 - tick_h, 0.12 + tick_h], color='black', lw=1.2)
    ax_ann.text(win_start, 0.0, f'{win_start:,}', ha='left', va='center', fontsize=7, color='#666666')
    ax_ann.text(win_end, 0.0, f'{win_end:,}', ha='right', va='center', fontsize=7, color='#666666')
 
    if show_rna:
        ax_rna = fig.add_subplot(main_gs[:n_ct, rna_col])
        y_positions = np.arange(n_ct)[::-1]
        rna_max = max(rna_values.get(ct, 0) for ct in celltype_order) or 1
        for y, ct in zip(y_positions, celltype_order):
            val = rna_values.get(ct, 0) / rna_max
            ax_rna.barh(y, val, height=0.6, color=celltype_colors[ct], edgecolor='none')
        ax_rna.set_xlim(0, 1.08)
        ax_rna.set_ylim(-0.5, n_ct - 0.5)
        ax_rna.set_yticks([])
        ax_rna.set_xticks([])
        for spine in ax_rna.spines.values():
            spine.set_visible(False)
        ax_rna.set_title(f'{gene_name}\nRNA', fontsize=9, loc='center')
 
    method_colors = method_colors or {}
    ax_sig = fig.add_subplot(outer_gs[:n_ct, sig_col])
    methods = list(qvals.keys())
    neglog_q = [-np.log10(q) if q and q > 0 else 0 for q in qvals.values()]
    x_pos = np.arange(len(methods))
    colors = [method_colors.get(m, '#888888') for m in methods]
 
    if sig_style == 'lollipop':
        ax_sig.vlines(x_pos, 0, neglog_q, color='#999999', lw=1.5, zorder=1)
        ax_sig.scatter(x_pos, neglog_q, color=colors, s=70, zorder=2,
                        edgecolor='black', linewidth=0.6)
        ax_sig.axhline(-np.log10(sig_threshold), color=highlight_color, ls='--', lw=1, zorder=0)
        ax_sig.set_xticks(x_pos)
        ax_sig.set_xticklabels(methods, rotation=45, ha='right')
        ax_sig.set_ylabel('$-log_{10}$(q-value)')
        for spine in ['top', 'right']:
            ax_sig.spines[spine].set_visible(False)
 
    elif sig_style == 'bar':
        ax_sig.bar(x_pos, neglog_q, color=colors, width=1.0)
        ax_sig.axhline(-np.log10(sig_threshold), color=highlight_color, ls='--', lw=1)
        ax_sig.set_xticks(x_pos)
        ax_sig.set_xticklabels(methods, rotation=90)
        ax_sig.set_ylabel('$-log_{10}$(q-value)')
        for spine in ['top', 'right']:
            ax_sig.spines[spine].set_visible(False)
 
    elif sig_style == 'dotmatrix':
        sizes = np.clip(np.array(neglog_q) * 40, 20, 400)
        alphas = [1.0 if q < sig_threshold else 0.35 for q in qvals.values()]
        for x, c, s, a in zip(x_pos, colors, sizes, alphas):
            ax_sig.scatter(x, 0, s=s, color=c, alpha=a, edgecolor='black', linewidth=0.6)
        ax_sig.set_xticks(x_pos)
        ax_sig.set_xticklabels(methods, rotation=45, ha='right')
        ax_sig.set_yticks([])
        ax_sig.set_ylim(-0.5, 0.5)
        for spine in ax_sig.spines.values():
            spine.set_visible(False)
 
    fig.suptitle(f'{gene_name} \u2014 {snp_id}', fontsize=11, x=0.06, ha='left')
 
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
    return fig

 
def _parse_peak_midpoint(peak_str):
    """Parse 'chr1:1000000-1002000' (or 'chr1-1000000-1002000') -> midpoint bp."""
    m = re.match(r"[^:\-]+[:\-](\d+)[\-](\d+)", str(peak_str))
    if m:
        start, end = int(m.group(1)), int(m.group(2))
        return (start + end) / 2
    nums = re.findall(r"\d+", str(peak_str))
    if len(nums) >= 2:
        return (int(nums[0]) + int(nums[1])) / 2
    raise ValueError(f"Could not parse peak position from: {peak_str!r}")


def _as_gene_series(gene_df):
    """Coerce gene_df (Series or 1-col DataFrame) into a clean Series: gene_id -> tss.
    Drops duplicate index entries (keeps first) since duplicated gene ids with
    different TSS values silently break exact-match lookups downstream."""
    if isinstance(gene_df, pd.DataFrame):
        s = gene_df.iloc[:, 0]
    else:
        s = gene_df
    s = s.copy()
    s.index.name = "gene"
    if s.index.duplicated().any():
        warnings.warn(
            "gene_df has duplicate gene ids; keeping the first occurrence of each."
        )
        s = s[~s.index.duplicated(keep="first")]
    return s


def _format_q(q):
    if pd.isna(q):
        return "n.s."
    if q == 0:
        return "0"
    if q < 1e-3 or q >= 1e4:
        return f"{q:.1e}"
    return f"{q:.3g}"


def _bezier_arc(x0, x1, y0, height):
    """Quadratic bezier arc from (x0,y0) to (x1,y0) peaking at y0+height."""
    xm = (x0 + x1) / 2
    verts = [(x0, y0), (xm, y0 + height), (x1, y0)]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
    return Path(verts, codes)


def _label_for(gene_id, gene_symbols):
    """Map a gene id to a display symbol if a mapping is provided and has a hit."""
    if gene_symbols is None:
        return gene_id
    if gene_id in gene_symbols.index:
        val = gene_symbols.loc[gene_id]
        if isinstance(val, pd.Series):  # duplicate index entries
            val = val.iloc[0]
        if pd.notna(val) and str(val).strip():
            return val
    return gene_id


def _span_ellipse_w(win_lo, win_hi, frac=0.02):
    return max((win_hi - win_lo) * frac, 1)


def _fade(hex_color, alpha):
    """Blend a hex color toward white by `alpha` (0=white, 1=full color) so
    non-winning arcs read as a muted version of the method's own color rather
    than a flat generic gray."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r = int(r * alpha + 255 * (1 - alpha))
    g = int(g * alpha + 255 * (1 - alpha))
    b = int(b * alpha + 255 * (1 - alpha))
    return f"#{r:02x}{g:02x}{b:02x}"


def plot_locus_arcs(
    locus_res_df,
    gene_df,
    snp_id=None,
    methods=None,
    method_labels=None,
    method_colors=None,
    gene_symbols=None,
    flank=15000,
    max_arc_height=1.0,
    default_color="#9e9e9e",
    fade_alpha=0.35,
    figsize=None,
    label_every=None,
    row_height=1.3,
    hspace=0.15,
    hide_unmapped_genes=False,
):
    """
    Parameters
    ----------
    locus_res_df : DataFrame filtered (or filterable) to one locus/SNP.
    gene_df : Series/DataFrame gene_id -> TSS bp position.
    snp_id : if locus_res_df has multiple loci, subsets to this snp_id first.
    methods : optional explicit ordering of methods (rows), using the RAW
              method names as they appear in locus_res_df['method'].
              Defaults to sorted unique values.
    method_labels : dict, raw method name -> display name for the row label.
    method_colors : dict, raw method name -> color (hex/name) for that
                     method's arcs. Methods not in the dict fall back to
                     `default_color`.
    gene_symbols : optional Series, gene_id -> gene symbol, used only for
                    display text (gene track labels + title). All position
                    lookups and gold-gene matching still use the original
                    gene_df/locus_res_df ids, so this is purely cosmetic.
    flank : bp padding added on each side of the plotted window.
    max_arc_height : cap on arc height (axis units, 0-1 per method panel).
    fade_alpha : how strongly non-winning arcs are faded toward white,
                 relative to that method's own color (0=white, 1=full color).
    label_every : if set, label every Nth gene in the top track (by position)
                  instead of the automatic overlap-avoiding spacing.
    row_height : inches of vertical space allotted per track when `figsize`
                 isn't given explicitly. Lower = more compressed. Default 1.3;
                 try 0.8-0.9 for a tight stack.
    hspace : gridspec vertical gap between panels, as a fraction of average
             panel height. Lower = tracks sit closer together. Default 0.15;
             try 0.02-0.05 for minimal gaps (panels still won't overlap since
             sharex keeps baselines aligned).
    hide_unmapped_genes : if True, drop genes from the top gene track whose
             display name still starts with "ENSG" after applying
             `gene_symbols` (i.e. the symbol lookup had no hit for that gene).
             Only meaningful when `gene_symbols` is also passed. The gold
             gene is always kept regardless of this flag, even if unmapped,
             so it can't silently disappear the way it used to.
 
    Returns
    -------
    matplotlib.figure.Figure
    """
    df = locus_res_df.copy()
    if snp_id is not None:
        df = df[df["snp_id"] == snp_id]
    if df.empty:
        raise ValueError("No rows for this locus after filtering.")
 
    genes = _as_gene_series(gene_df)
    method_labels = method_labels or {}
    method_colors = method_colors or {}
 
    if methods is None:
        methods = sorted(df["method"].unique())
 
    peak_pos = _parse_peak_midpoint(df["peak"].mode().iat[0])
 
    # min q-value per (method, nominated_gene) across datasets
    grp = (
        df.groupby(["method", "nominated_gene"], as_index=False)
        .agg(
            min_qval=("nominated_qval", "min"),
            is_gold=("is_gold", "max"),
        )
    )
 
    # --- gold/target gene -------------------------------------------------
    gold_candidates = df.loc[df["is_gold"] == True, "nominated_gene"].unique()  # noqa: E712
    gold_gene = df["snp_gene"].iloc[0]
 
    # Resolve the gold gene's TSS directly from gene_df, independent of
    # whether it happens to be a "nominated_gene" for any method. This is
    # the fix for the gold gene sometimes not appearing/highlighting: the
    # old version only ever considered genes that were nominated by some
    # method when building the plotting window, so a gold gene that no
    # method actually nominated as a candidate (or that fell outside that
    # induced window) silently never made it onto the canvas.
    gold_pos = genes.get(gold_gene, np.nan)
    if pd.isna(gold_pos):
        warnings.warn(
            f"Gold gene {gold_gene!r} was not found in gene_df (id mismatch?) "
            "so it cannot be drawn or highlighted. Check gene id conventions "
            "match between locus_res_df and gene_df."
        )
 
    # --- plotting window ----------------------------------------------------
    genes_in_play = grp["nominated_gene"].unique()
    tss_positions = genes.reindex(genes_in_play).dropna()
    if tss_positions.empty and pd.isna(gold_pos):
        raise ValueError("None of the nominated genes (or the gold gene) were found in gene_df.")
 
    relevant_positions = list(tss_positions.values) + [peak_pos]
    if not pd.isna(gold_pos):
        relevant_positions.append(gold_pos)
 
    win_lo = min(relevant_positions) - flank
    win_hi = max(relevant_positions) + flank
 
    window_genes = genes[(genes >= win_lo) & (genes <= win_hi)].sort_values()
 
    # Guarantee the gold gene is present in the window even if some
    # upstream filtering/rounding would otherwise have excluded it.
    if not pd.isna(gold_pos) and gold_gene not in window_genes.index:
        window_genes = pd.concat(
            [window_genes, pd.Series([gold_pos], index=[gold_gene])]
        ).sort_values()
 
    if hide_unmapped_genes:
        if gene_symbols is None:
            warnings.warn(
                "hide_unmapped_genes=True has no effect without gene_symbols."
            )
        else:
            mapped_mask = pd.Series(
                [
                    (name == gold_gene) or not str(_label_for(name, gene_symbols)).startswith("ENSG")
                    for name in window_genes.index
                ],
                index=window_genes.index,
            )
            window_genes = window_genes[mapped_mask]
 
    # ---------------------------------------------------------------
    # figure / axes layout
    # ---------------------------------------------------------------
    n_methods = len(methods)
    height_ratios = [1.1] + [1.0] * n_methods
    if figsize is None:
        figsize = (10, row_height * (n_methods + 1) + 1.4)
 
    fig, axes = plt.subplots(
        n_methods + 1,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios, "hspace": hspace},
    )
    gene_ax = axes[0]
    method_axes = axes[1:]
 
    # ---------------------------------------------------------------
    # top panel: gene track
    # ---------------------------------------------------------------
    gene_ax.axhline(0, color="black", lw=1.2, zorder=1)
    gene_ax.set_ylim(-1, 2.1)
    gene_ax.set_yticks([])
    for spine in ("top", "right", "left"):
        gene_ax.spines[spine].set_visible(False)
    gene_ax.spines["bottom"].set_visible(False)
 
    positions = window_genes.values
    names = window_genes.index.values
    highlight_color = method_colors.get("_gold", "#d62728")
 
    if label_every is not None:
        label_mask = np.arange(len(positions)) % label_every == 0
    else:
        span = max(win_hi - win_lo, 1)
        min_gap = span * 0.06
        label_mask = np.zeros(len(positions), dtype=bool)
        last_labeled = -np.inf
        for i, p in enumerate(positions):
            if p - last_labeled >= min_gap:
                label_mask[i] = True
                last_labeled = p
    gold_hits = np.where(names == gold_gene)[0]
    label_mask[gold_hits] = True  # always label the gold gene
 
    for i, (name, pos) in enumerate(zip(names, positions)):
        is_gold = name == gold_gene
        display_name = _label_for(name, gene_symbols)
        gene_ax.plot(
            [pos, pos], [-0.15, 0.15],
            color=highlight_color if is_gold else "black",
            lw=2.2 if is_gold else 1.2,
            zorder=3,
        )
        if is_gold:
            gene_ax.scatter(
                [pos], [0.55], marker="*", s=260,
                color=highlight_color, edgecolor="black", linewidth=0.6,
                zorder=5,
            )
            gene_ax.add_patch(
                mpatches.Ellipse(
                    (pos, 0.0), width=_span_ellipse_w(win_lo, win_hi), height=0.9,
                    fill=False, edgecolor=highlight_color, lw=1.6, zorder=4,
                )
            )
        if label_mask[i]:
            gene_ax.text(
                pos, 0.85 if not is_gold else 1.15, display_name,
                rotation=45, ha="left", va="bottom",
                fontsize=7.5,
                color=highlight_color if is_gold else "#333333",
                fontweight="bold" if is_gold else "normal",
            )
 
    gene_ax.annotate(
        "peak", xy=(peak_pos, -0.15), xytext=(peak_pos, -0.9),
        ha="center", va="top", fontsize=8, color="#1f77b4",
        arrowprops=dict(arrowstyle="-", color="#1f77b4", lw=1.2),
    )
 
    # ---------------------------------------------------------------
    # per-method arc panels
    # ---------------------------------------------------------------
    for ax, method in zip(method_axes, methods):
        sub = grp[grp["method"] == method].copy()
        sub = sub[sub["nominated_gene"].isin(genes.index)]
        sub["tss"] = genes.reindex(sub["nominated_gene"]).values
 
        method_color = method_colors.get(method, default_color)
        faded_color = _fade(method_color, fade_alpha) if method_color.startswith("#") else method_color
        display_method = method_labels.get(method, method)
 
        ax.axhline(0, color="black", lw=1.0, zorder=1)
        ax.set_yticks([])
        ax.set_ylim(-0.9, max_arc_height + 0.55)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_ylabel(display_method, rotation=0, ha="right", va="center", fontsize=10,
                       fontweight="bold", labelpad=25, color=method_color)
 
        if sub.empty:
            continue
 
        best_idx = sub["min_qval"].idxmin()
        span = max(win_hi - win_lo, 1)
 
        for idx, row in sub.iterrows():
            dist = abs(row["tss"] - peak_pos)
            height = 0.15 + max_arc_height * min(dist / span, 1.0)
            is_best = idx == best_idx
            color = method_color if is_best else faded_color
            lw = 2.6 if is_best else 1.4
            zorder = 5 if is_best else 2
 
            path = _bezier_arc(peak_pos, row["tss"], 0.0, height)
            patch = mpatches.PathPatch(
                path, facecolor="none", edgecolor=color, lw=lw, zorder=zorder
            )
            ax.add_patch(patch)
 
            ax.plot(
                row["tss"], 0.0, marker="v", color=color,
                markersize=6 if is_best else 4.5, zorder=zorder + 1,
            )
 
            # extra marker if this method's top pick is also the true gold gene
            if is_best and row["nominated_gene"] == gold_gene:
                ax.scatter(
                    [row["tss"]], [height + 0.12], marker="*", s=90,
                    color=method_color, edgecolor="black", linewidth=0.4,
                    zorder=zorder + 2,
                )
 
            ax.text(
                row["tss"], -0.35, _format_q(row["min_qval"]),
                ha="center", va="top", fontsize=8,
                color=color, fontweight="bold" if is_best else "normal",
            )
 
        ax.plot(peak_pos, 0.0, marker="o", color="#1f77b4", markersize=5, zorder=6)
 
    method_axes[-1].set_xlabel("genomic position (bp)", fontsize=9)
    for ax in list(method_axes) + [gene_ax]:
        ax.ticklabel_format(style="plain", axis="x")
 
    # Title as a figure-level suptitle (not an axes title) so it never
    # collides with the rotated gene-name labels sitting just below it,
    # and reserve headroom for it via subplots_adjust rather than tight_layout.
    target_display = _label_for(gold_gene, gene_symbols)
    title_snp = snp_id if snp_id is not None else df["snp_id"].iloc[0]
    fig.suptitle(
        f"Locus: {title_snp}  |  target gene: {target_display}",
        x=0.01, ha="left", fontsize=11, fontweight="bold", y=0.995,
    )
    fig.subplots_adjust(top=1 - (0.55 / figsize[1]))
 
    fig.align_ylabels(method_axes)
    return fig