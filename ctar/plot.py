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
from matplotlib.container import BarContainer
import math
from matplotlib.ticker import ScalarFormatter


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
                        height = 5, width = 18, axlim = None):

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
    width : int
        Figure width passed into matplotlib.

    Returns
    -------
    None :
        Does not return anything; automatically plots the trackplot when called.
        
    """
    
    # to do: fix names appearing on side

    # sort mdata if necessary
    if not adata_sort:
        atac_sorted = atac
        rna_sorted = rna
    else:
        atac_sorted = sortby_ct_adata(atac,obs_col=obs_col)
        rna_sorted = sortby_ct_adata(rna,obs_col=obs_col)

    # sort df by top values of sortby arg
    if not presorted:
        sorted_df = df.sort_values(sortby,ascending=ascending)[:top_n]
    else:
        sorted_df = df

    # create subplots, including a smaller plot at the bottom for the ct colormap
    nrows = len(sorted_df)
    fig,axs = plt.subplots(
        figsize=(width,height), nrows=nrows+1, ncols=2,
        sharex=False, sharey=False, gridspec_kw={'hspace':0,'wspace':0,'height_ratios':[2]*nrows + [0.5]}
        )

    # extract gene and peak ids
    y_axis_rna = sorted_df.gene.values
    y_axis_atac = sorted_df.peak.values
    x_axis = np.arange(atac_sorted.shape[0])

    # extract relevant raw gene and peak information
    rna = rna_sorted[:,rna_sorted.var.gene.isin(y_axis_rna.tolist())].X.toarray()
    atac = atac_sorted[:,atac_sorted.var.peak.isin(y_axis_atac.tolist())].X.toarray()

    # take ylims for plotting
    maxpos = max(np.max(rna),np.max(atac))
    maxneg = min(np.min(rna),np.min(atac))
    if axlim != None:
        maxpos, maxneg = axlim

    print(maxpos,maxneg)

    if atac.shape[0] != rna.shape[0]:
        raise ValueError('number of cell must be the same between modalities.')

    # get ct labels and number of cells
    cts = rna_sorted.obs[obs_col] # ct should be the same between groups so im using rna as ref
    ct_sizes = [0]+[np.sum(cts == ct) for ct in cts.unique()] # get group lengths
    ct_sizes = np.cumsum(np.array(ct_sizes))

    color_length = len(ct_sizes)-1
    # if color_length < 20: 
    cmap = plt.get_cmap('tab20',color_length)
    # else:
    #     colors = list(mpl.colors._colors_full_map.values()) # expanded color map
    #     while len(colors) < color_length: 
    #         colors += colors
    #     colors = colors[:color_length]
    #     cmap = mpl.colors.ListedColormap(colors)
    idx = []
    
    for i in np.arange(len(sorted_df)):
        
        # axs[i,0].set_xlim(0,len(x_axis))
        # axs[i,0].set_ylim(maxneg,maxpos)
        
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
        axs[i,1].set_ylabel(coeff_label +': ' + str(round(sorted_df.iloc[i][coeff],3))+ \
                            '\n' + pval_label +': '+ f'{float(f"{sorted_df.iloc[i][pval]:.1g}"):g}',
                            rotation=0, fontsize="x-small", ha="left", va="bottom")
        axs[i,1].yaxis.set_label_coords(1.005, 0.1)

        # get info for particular link
        peak = atac[:,[i]]
        gene = rna[:,[i]]

        # rna gene label
        rna_max = np.max(gene)
        rna_min = np.min(gene)
        atac_max = np.max(peak)
        atac_min = np.min(peak)

        axs[i, 0].set_ylim(rna_min, rna_max)
        axs[i, 1].set_ylim(atac_min, atac_max)

        # for each ct
        for k in np.arange(cts.unique().shape[0]):

            # plot data
            rnay_curve = gene[ct_sizes[k]:ct_sizes[k+1],:].flatten() # ct data for plot
            atacy_curve = peak[ct_sizes[k]:ct_sizes[k+1],:].flatten()

            # rna
            if sort_cells == 'rna':
                rnay_curve = -np.sort(-rnay_curve) # sorted in reverse order
            elif sort_cells == 'atac':
                rnay_curve = rnay_curve[-np.argsort(-atacy_curve)]
                # since the cells are in the same order, we can sort by rna's idx to align
            elif sort_cells == 'share_rna':
                rnay_curve = -np.sort(-rnay_curve)
                idx += [-np.argsort(-rnay_curve)]

            plot_line_collection(axs[i,0], x_axis[ct_sizes[k]:ct_sizes[k+1]], rnay_curve, cmap(k)) # line segments
            axs[i,0].axvline(ct_sizes[k+1],linestyle='dashed',color='grey',alpha=0.3) # dashed grey line to separate cts

            # atac
            if sort_cells == 'rna':
                atacy_curve = atacy_curve[-np.argsort(-rnay_curve)] 
            elif sort_cells == 'atac':
                atacy_curve = -np.sort(-atacy_curve)
            elif sort_cells == 'share_rna':
                rnay_curve = atacy_curve[idx[-1]]

   
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
    
    cbar.ax.set_xticklabels('')
    cbar.ax.set_xticks([((ct_sizes[x]+ct_sizes[x+1])/2) for x in np.arange(color_length)],minor=True)
    cbar.ax.set_xticklabels(cts.unique(),fontsize=9,rotation=90,minor=True)
    cbar.outline.set_visible(False)

    axs[len(sorted_df),0].tick_params(axis='x',which='minor',bottom=False,top=False,labelbottom=True)


    # atac colormap
    norm = mpl.colors.BoundaryNorm(ct_sizes, cmap.N)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),cax=axs[len(sorted_df),1], orientation='horizontal',spacing='proportional',ticks=ct_sizes[:color_length])
    
    cbar.ax.set_xticklabels('')
    cbar.ax.set_xticks([((ct_sizes[x]+ct_sizes[x+1])/2) for x in np.arange(color_length)],minor=True)
    cbar.ax.set_xticklabels(cts.unique(),fontsize=9,rotation=90,minor=True)
    cbar.outline.set_visible(False)

    axs[len(sorted_df),1].tick_params(axis='x',which='minor',bottom=False,top=False,labelbottom=True)

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
    methods_order=None,
    estimate_col="estimate",
    ci_lower_col="ci_lower",
    ci_upper_col="ci_upper",
    # Significance: read vs-reference p-values from a column in res_dic[label]
    pval_vs_ref_col=None,     # e.g., "p_value_vs_ctar_filt"
    reference_method="ctar_filt",
    alpha=0.05,
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
    bar_width=0.18,
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
        _ensure_method_column_inplace(df)
        df = df[df['method'].isin(methods_order)]
        df = df.set_index('method').reindex(methods_order)

        # Validate columns
        for col in (estimate_col, ci_lower_col, ci_upper_col):
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in '{lbl}'. Columns: {list(df.columns)}")

        # Read p-values vs reference, if column provided
        if pval_vs_ref_col is not None and pval_vs_ref_col in df.columns:
            per_label_vs_ref[lbl] = df[pval_vs_ref_col].to_dict()
        elif pval_vs_ref_col is not None and pval_vs_ref_col not in df.columns:
            raise KeyError(f"Column '{pval_vs_ref_col}' not found in '{lbl}'. Available: {list(df.columns)}")

        est = df[estimate_col].astype(float).to_numpy()
        lo  = df[ci_lower_col].astype(float).to_numpy()
        hi  = df[ci_upper_col].astype(float).to_numpy()

        e_low = np.clip(est - lo, 0, np.inf)
        e_up  = np.clip(hi - est, 0, np.inf)

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
            ax.errorbar(x, y, yerr=[[e_low[j]], [e_up[j]]], fmt='none', ecolor=color, elinewidth=1.2, capsize=4)

            max_y_for_limits.append(y + e_up[j])

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
    ax.axhline(1.0, color='k', linestyle='--', linewidth=1)

    # Labels/title
    if y_label is None:
        y_label = "Weighted average enrichment (95% CI)"
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)

    # Legend: method display names
    if add_legend:
        handles = []
        labels_legend = []
        for m in methods_order:
            patch = plt.Line2D([0], [0], marker='s', color='w',
                               markerfacecolor=method_colors[m], markersize=10, linestyle='None')
            handles.append(patch)
            disp = method_renames.get(m, m) if method_renames else m
            labels_legend.append(disp)
        ax.legend(handles, labels_legend, loc='upper center', bbox_to_anchor=(0.5, -0.1),
                  frameon=False, ncol=min(len(methods_order), 4))

    # Draw brackets vs reference using column-based p-values
    if per_label_vs_ref:
        for i, lbl in enumerate(labels):
            pmap = per_label_vs_ref.get(lbl, {})
            if not pmap:
                continue

            # Ensure reference method is present in positions
            if (lbl, ref_internal) not in positions:
                continue

            group_max = max(heights[(lbl, m)] + err_up[(lbl, m)]
                            for m in methods_order if (lbl, m) in heights)

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
                p = pmap.get(m, None)
                if p is None:
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
                else:
                    txt = ""

                _draw_sig_bracket(ax, x1, x2, y, h, txt, color='k', fontsize=bracket_fontsize, lw=bracket_linewidth)
                level += 1

    ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.6)

    if created_fig:
        fig.tight_layout()

    return ax