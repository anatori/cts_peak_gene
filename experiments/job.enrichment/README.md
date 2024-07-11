# Analysis and enrichment experiments
### Compute enrichment with CRISPRi-FlowFISH
1. Using `load-data.ipynb`,
   - Load NEATseq dataset found in `/projects/zhanglab/users/ana/multiome/raw/neatseq/`
   - Load CRISPRi-FlowFISH data found in `/projects/zhanglab/users/ana/multiome/validation/`
   - Compute NEATseq CTAR peak-gene links or use precomputed results at `/projects/zhanglab/users/ana/multiome/results/ctar/`
2. Using `signac-scent.ipynb`, compute NEATseq SCENT and Signac peak-gene links or use precomputed results at `./scent` or `./signac`.
3. Using `crispr-enrich.ipynb`, compute enrichment with CRISPRi-FlowFISH ([Nasser Nat Gen 2019](https://www.nature.com/articles/s41588-019-0538-0)) links.
4. Using `caviar-enrich.ipynb`, compute enrichment with CAVIAR GTEx eQTL([Hormozdiari Genetics 2014](https://academic.oup.com/genetics/article/198/2/497/5935918)).