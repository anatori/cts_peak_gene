
# Importing libraries
import numpy as np
import pandas as pd 
from statsmodels.graphics.gofplots import qqplot_2samples
import statsmodels.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import math
import warnings
import time
import random
from tqdm import tqdm

import anndata as ad
import scanpy as sc
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
def sortby_ct_mdata(mdata):
    
    color_axis = mdata.mod['rna'].obs.sort_values('celltype') # assumes rna is the same as atac
    
    return mdata[color_axis.index,:]



def multiome_trackplot(df, mdata, sortby = 'theta_0', top_n = 10, mdata_sort = False, height = 5, width = 18):

    """Custom trackplot for showing ATAC and RNA information for top peak-gene pairs across all cells, sorted by
    celltype. Within celltype, both ATAC and RNA results are sorted by cells with the highest RNA expression.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing peak-gene links labelled with "gene_name" existing in 
        mdata.mod['RNA'] index, and "gene_ids" existing in mdata.mod['ATAC'] index
        for each peak-gene link. Contains column sortby.
    mdata : mu.MuData
        MuData object. Should contain ATAC modality and RNA modality with var
        attribute indices as "gene_name" and "gene_ids" respectively.
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
    if mdata_sort:
        mdata_sorted = mdata
    else:
        mdata_sorted = sortby_ct_mdata(mdata)

    # sort df by top values of sortby arg
    sorted_df = df.sort_values(sortby,ascending=False)[:top_n]

    # create subplots, including a smaller plot at the bottom for the ct colormap
    fig,axs = plt.subplots(
        figsize=(width,height), nrows=len(sorted_df)+1, ncols=2,
        sharex=False, sharey=False, gridspec_kw={'hspace':0,'wspace':0,'height_ratios':[2]*len(sorted_df)+[1]}
        )

    # extract gene and peak ids
    y_axis_rna = sorted_df.gene_name.values
    y_axis_atac = sorted_df.gene_ids.values
    x_axis = np.arange(mdata_sorted.shape[0])

    # extract relevant raw gene and peak information
    rna = mdata_sorted.mod['rna'][:,y_axis_rna.tolist()].X.A
    atac = mdata_sorted.mod['atac'][:,y_axis_atac.tolist()].X.A

    # take ylims for plotting
    maxpos = max(np.max(rna),np.max(atac))
    maxneg = min(np.min(rna),np.min(atac))

    if atac.shape[0] != rna.shape[0]:
        raise ValueError('number of cell must be the same between modalities.')

    # get ct labels and number of cells
    cts = mdata_sorted.mod['rna'].obs.celltype # ct should be the same between groups so im using rna as ref
    ct_sizes = [0]+[np.sum(cts == ct) for ct in cts.unique()] # get group lengths
    ct_sizes = np.cumsum(np.array(ct_sizes))
    
    cmap = plt.get_cmap('tab20',len(ct_sizes)-1) # expanded color map

    
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
        axs[i,1].set_ylabel(round(sorted_df.iloc[i][sortby],3), rotation=0, fontsize="small", ha="left", va="bottom")
        axs[i,1].yaxis.set_label_coords(1.005, 0.1)

        # get info for particular link
        peak = atac[:,[i]]
        gene = rna[:,[i]]

        # for each ct
        for k in np.arange(cts.unique().shape[0]):

            # rna data
            rnay_curve = gene[ct_sizes[k]:ct_sizes[k+1],:].flatten() # ct data for plot
            rnay_curve = -np.sort(-rnay_curve) # sorted in reverse order
            plot_line_collection(axs[i,0], x_axis[ct_sizes[k]:ct_sizes[k+1]], rnay_curve, cmap(k)) # line segments
            axs[i,0].axvline(ct_sizes[k+1],linestyle='dashed',color='grey',alpha=0.3) # dashed grey line to separate cts

            # atac data
            atacy_curve = peak[ct_sizes[k]:ct_sizes[k+1],:].flatten()
            atacy_curve = atacy_curve[-np.argsort(-rnay_curve)]
            plot_line_collection(axs[i,1], x_axis[ct_sizes[k]:ct_sizes[k+1]], atacy_curve, cmap(k))
            axs[i,1].axvline(ct_sizes[k+1],linestyle='dashed',color='grey',alpha=0.3)

        # reference lines at y=0
        axs[i,0].axhline(0,color='grey') 
        axs[i,1].axhline(0,color='grey')


    
    # rna colormap
    norm = mpl.colors.BoundaryNorm(ct_sizes, cmap.N)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),cax=axs[len(sorted_df),0], orientation='horizontal',spacing='proportional',ticks=ct_sizes[:len(ct_sizes)-1])
    
    cbar.ax.set_xticklabels('')
    cbar.ax.set_xticks([((ct_sizes[x]+ct_sizes[x+1])/2) for x in np.arange(len(ct_sizes)-1)],minor=True)
    cbar.ax.set_xticklabels(cts.unique(),fontsize=9,rotation=90,minor=True)

    axs[len(sorted_df),0].tick_params(axis='x',which='minor',bottom=False,top=False,labelbottom=True)
    
    
    # atac colormap
    norm = mpl.colors.BoundaryNorm(ct_sizes, cmap.N)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),cax=axs[len(sorted_df),1], orientation='horizontal',spacing='proportional',ticks=ct_sizes[:len(ct_sizes)-1])
    
    cbar.ax.set_xticklabels('')
    cbar.ax.set_xticks([((ct_sizes[x]+ct_sizes[x+1])/2) for x in np.arange(len(ct_sizes)-1)],minor=True)
    cbar.ax.set_xticklabels(cts.unique(),fontsize=9,rotation=90,minor=True)

    axs[len(sorted_df),1].tick_params(axis='x',which='minor',bottom=False,top=False,labelbottom=True)

    # add modality labels on top
    axs[0,0].set_title('RNA',fontdict={'fontsize':10})
    axs[0,1].set_title('ATAC',fontdict={'fontsize':10})

    fig.supylabel(sortby,x=0.93,rotation=270,fontsize=12) # label values
    plt.show();
    
    return
