library(GenomeInfoDb)
library(Seurat)
library(EnsDb.Hsapiens.v86)
library(BSgenome.Hsapiens.UCSC.hg38)

library(GenomicRanges)
library(Signac)
library(Matrix)

directory <- '/projects/zhanglab/users/ana/GSE178707_all/GSM5396329/'
cells <- read.csv(file = paste(directory,'cd4t/GSM5396332_CD4cells.csv',sep=''),
                    header=T,sep=',')
rna <- readRDS(file = paste(directory,'cd4t/GSM5396333_CD4_RNA_counts.rds',sep=''))
atac <- readRDS(file = paste(directory,'cd4t/GSM5396336_CD4_Peak_matrix.rds',sep=''))
atac <- t(atac)

# get gene annotations for hg38
# using this rather than custom peak-gene links
annotation <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v86)
seqlevels(annotation) <- paste0('chr', seqlevels(annotation))

# create a Seurat object containing the RNA adata
neat <- CreateSeuratObject(
  counts = rna,
  assay = "RNA"
)

# normalizedata is used by SCENT
# but SCTransform is suggested by signac vignette?
DefaultAssay(neat) <- "RNA"
neat <- NormalizeData(neat)


lane1_frags <- CreateFragmentObject(path = paste(directory,'GSM5396332_lane1_atac_fragments.tsv.gz',sep=''))
lane2_frags <- CreateFragmentObject(path = paste(directory,'GSM5396336_lane2_atac_fragments.tsv.gz',sep=''))


# create ATAC assay and add it to the object
neat[['ATAC']] <- CreateChromatinAssay(
  counts = atac,
  sep = c(":", "-"),
  fragments = list(lane1_frags,lane2_frags),
  annotation = annotation
)


DefaultAssay(neat) <- "ATAC"
# first compute the GC content for each peak
neat <- RegionStats(neat, genome = BSgenome.Hsapiens.UCSC.hg38)


start.time <- Sys.time()
# link peaks to genes
neat <- LinkPeaks(
  object = neat,
  peak.assay = "ATAC",
  expression.assay = "RNA",
  gene.id = TRUE, # since this dataset uses ensemblIDs
  pvalue_cutoff = 1, # max pvalue
  score_cutoff = 0 # min pearson corr
)
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken # test


l = Links(neat)
write.csv(l, file = "signac_neatseq_links.csv")