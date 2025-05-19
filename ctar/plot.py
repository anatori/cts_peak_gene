
# Importing libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import warnings
import anndata as ad
import muon as mu
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl
from scipy.stats import poisson, nbinom


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
        figsize=(width,height), nrows=nrows, ncols=2,
        sharex=False, sharey=False, gridspec_kw={'hspace':0,'wspace':0,'height_ratios':[2]*nrows}
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
    if color_length < 20: cmap = plt.get_cmap('tab20',color_length)
    else:
        colors = list(mpl.colors._colors_full_map.values()) # expanded color map
        while len(colors) < color_length: 
            colors += colors
        colors = colors[:color_length]
        cmap = mpl.colors.ListedColormap(colors)
    idx = []
    
    for i in np.arange(len(sorted_df)):
        
        # rna gene label
        axs[i,0].set_xlim(0,len(x_axis))
        axs[i,0].set_ylim(maxneg,maxpos)
        
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

    Parameters:
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
    focal_methods, alpha=0.05,
    line_gap=0.05,  # spacing between lines
    text_gap=0.03,  # extra spacing above lines for the box
    fontsize=8,
    fontsize_gap=2,
    line_color='grey'
):
    """
    Annotates significance comparisons for multiple focal methods with FDR-corrected p-values.

    Parameters:
        ax : matplotlib axis
        odds_data : pd.DataFrame [n_links x methods]
        yerr_upper : pd.DataFrame [n_links x methods]
        corrected_pval_lookup : dict[(focal_method, n_link, comparison_method)] = (raw_p, corr_p)
        focal_methods : list of methods to use as references
        alpha : significance threshold
    """
    num_groups = odds_data.shape[0]
    bar_centers = get_bar_centers_grouped(ax, odds_data)
    drawn_pairs = set()

    for i, n_link in enumerate(odds_data.index):  # x-axis group
        # Find max bar + error in this group
        tops = odds_data.loc[n_link] # + yerr_upper.loc[n_link]
        base_y = tops.max()
        current_y = base_y + line_gap

        for focal_method in focal_methods:
            for comp_method in odds_data.columns:
                if comp_method == focal_method:
                    continue

                # Skip already-drawn symmetrical pair
                pair_key = tuple(sorted([focal_method, comp_method]))
                pair_id = (n_link, pair_key)
                if pair_id in drawn_pairs:
                    continue
                drawn_pairs.add(pair_id)

                key = (focal_method, n_link, comp_method)
                if key not in corrected_pval_lookup:
                    continue

                raw_p, corr_p = corrected_pval_lookup[key]
                if corr_p >= alpha:
                    continue  # not significant

                x1 = bar_centers[comp_method][i]
                x2 = bar_centers[focal_method][i]

                # Draw bracket
                line_top = current_y + line_gap
                ax.plot([x1, x1, x2, x2], [current_y, line_top, line_top, current_y], c=line_color, lw=1.2)

                # Add label
                text_y = line_top + text_gap
                ax.text(
                    (x1 + x2) / 2, text_y,
                    f"{raw_p:.3g}\n({corr_p:.3g})",
                    ha='center', va='bottom', fontsize=fontsize,
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor=line_color, linewidth=0.5)
                )

                current_y = text_y + fontsize_gap + text_gap

