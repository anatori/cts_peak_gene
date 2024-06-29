
# Importing libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import warnings
import anndata as ad
import muon as mu
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl



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





def multiome_trackplot(df, adata, sortby = 'poiss_coeff', coeff_label = 'coeff', coeff = 'poiss_coeff', pval = 'mc_pval', 
                        ascending = False, top_n = 10, adata_sort = False, presorted = False, sort_cells = 'rna', obs_col='celltype',
                        height = 5, width = 18, axlim = None):

    """Custom trackplot for showing ATAC and RNA information for top peak-gene pairs across all cells, sorted by
    celltype. Within celltype, both ATAC and RNA results are sorted by cells with the highest RNA expression.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing peak-gene links labelled with "gene" existing in 
        mdata.mod['RNA'] index, and "peak" existing in mdata.mod['ATAC'] index
        for each peak-gene link. Contains column sortby.
    adata : ad.AnnData
        AnnData object. Should contain ATAC modality and RNA modality with var
        attribute indices as "gene" and "peak" respectively.
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
        adata_sorted = adata
    else:
        adata_sorted = sortby_ct_adata(adata)

    # sort df by top values of sortby arg
    if not presorted:
        sorted_df = df.sort_values(sortby,ascending=ascending)[:top_n]
    else:
        sorted_df = df

    # create subplots, including a smaller plot at the bottom for the ct colormap
    fig,axs = plt.subplots(
        figsize=(width,height), nrows=len(sorted_df)+1, ncols=2,
        sharex=False, sharey=False, gridspec_kw={'hspace':0,'wspace':0,'height_ratios':[2]*len(sorted_df)+[1]}
        )

    # extract gene and peak ids
    y_axis_rna = sorted_df.gene.values
    y_axis_atac = sorted_df.peak.values
    x_axis = np.arange(adata_sorted.shape[0])

    # extract relevant raw gene and peak information
    rna = adata_sorted[:,adata_sorted.var.gene.isin(y_axis_rna.tolist())].layers['rna_raw'].A
    atac = adata_sorted[:,adata_sorted.var.peak.isin(y_axis_atac.tolist())].layers['atac_raw'].A

    # take ylims for plotting
    maxpos = max(np.max(rna),np.max(atac))
    maxneg = min(np.min(rna),np.min(atac))
    if axlim != None:
        maxpos, maxneg = axlim

    print(maxpos,maxneg)

    if atac.shape[0] != rna.shape[0]:
        raise ValueError('number of cell must be the same between modalities.')

    # get ct labels and number of cells
    cts = adata_sorted.obs[obs_col] # ct should be the same between groups so im using rna as ref
    ct_sizes = [0]+[np.sum(cts == ct) for ct in cts.unique()] # get group lengths
    ct_sizes = np.cumsum(np.array(ct_sizes))
    
    cmap = plt.get_cmap('tab20',len(ct_sizes)-1) # expanded color map
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
                            '\n' + 'pval: '+ f'{float(f"{sorted_df.iloc[i][pval]:.1g}"):g}',
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
    cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),cax=axs[len(sorted_df),0], orientation='horizontal',spacing='proportional',ticks=ct_sizes[:len(ct_sizes)-1])
    
    cbar.ax.set_xticklabels('')
    cbar.ax.set_xticks([((ct_sizes[x]+ct_sizes[x+1])/2) for x in np.arange(len(ct_sizes)-1)],minor=True)
    cbar.ax.set_xticklabels(cts.unique(),fontsize=9,rotation=90,minor=True)
    cbar.outline.set_visible(False)

    axs[len(sorted_df),0].tick_params(axis='x',which='minor',bottom=False,top=False,labelbottom=True)
    
    
    # atac colormap
    norm = mpl.colors.BoundaryNorm(ct_sizes, cmap.N)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),cax=axs[len(sorted_df),1], orientation='horizontal',spacing='proportional',ticks=ct_sizes[:len(ct_sizes)-1])
    
    cbar.ax.set_xticklabels('')
    cbar.ax.set_xticks([((ct_sizes[x]+ct_sizes[x+1])/2) for x in np.arange(len(ct_sizes)-1)],minor=True)
    cbar.ax.set_xticklabels(cts.unique(),fontsize=9,rotation=90,minor=True)
    cbar.outline.set_visible(False)

    axs[len(sorted_df),1].tick_params(axis='x',which='minor',bottom=False,top=False,labelbottom=True)

    # add modality labels on top
    axs[0,0].set_title('RNA',fontdict={'fontsize':10})
    axs[0,1].set_title('ATAC',fontdict={'fontsize':10})

    # fig.supylabel(sortby,x=0.93,rotation=270,fontsize=12) # label values
    plt.show();
    
    return