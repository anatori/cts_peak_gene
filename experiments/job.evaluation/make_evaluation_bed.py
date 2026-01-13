import argparse
import os
import pandas as pd

def main(args):

	EVAL_FILE_PATH = args.eval_file_path
	SCREEN_PATH = args.screen_path
	BED_PATH = args.bed_path


	###########################################################################################
	######                                  Data Loading                                 ######
	###########################################################################################

	# gtex
	gtex_path = EVAL_FILE_PATH + '/gtex_v8_susie'
	gtex_df = pd.read_csv(f'{gtex_path}/GTEx_49tissues_release1.tsv.bgz',compression='gzip',sep='\t')
	print('GTEX shape:',gtex_df.shape)

	# abc
	abc_df = pd.read_csv(f'{EVAL_FILE_PATH}/V4-hg38.Gene-Links.Computational-Methods_abc_only_edited.txt.gz',header=None,compression='gzip',sep='\t')
	abc_df.columns = ['cCRE ID','Gene ID','Common Gene Name','Gene Type','Assay Type','Location','Experiment ID','Biosample','Score']
	print('ABC shape:',abc_df.shape)

	# crispr
	crispr_path = EVAL_FILE_PATH + '/epcrispr_2026'
	crispr_other = pd.read_csv(f'{crispr_path}/EPCrisprBenchmark_combined_data.heldout_5_cell_types.GRCh38.tsv.gz',compression='gzip',sep='\t')
	crispr_k562 = pd.read_csv(f'{crispr_path}/EPCrisprBenchmark_combined_data.training_K562.GRCh38.tsv.gz',compression='gzip',sep='\t')
	crispr_df = pd.concat([crispr_k562,crispr_other])
	print('CRISPR shape:',crispr_df.shape)

	# hic
	hic_df = pd.read_csv(f'{EVAL_FILE_PATH}/V4-hg38.Gene-Links.3D-Chromatin.txt',header=None,sep='\t')
	hic_df.columns = ['cCRE ID','Gene ID','Common Gene Name','Gene Type','Assay Type','Experiment ID','Biosample','Score','P-value']
	print('3D Chrom shape:',hic_df.shape)

	# select susie mapping
	gtex_df = gtex_df[gtex_df['method'] == 'SUSIE'].copy()

	# add coordinates for hic
	screen_df = pd.read_csv(f'{SCREEN_PATH}/GRCh38-cCREs.bed',sep='\t',header=None)
	screen_df.columns = ['chr','start','end','d_id','e_id','type']

	print(f'Merging 3D Chrom with {SCREEN_PATH} coordinates...')
	print('Pre-merge shape:',hic_df.shape)
	hic_df = hic_df.merge(screen_df,how='left',left_on='cCRE ID',right_on='e_id')
	hic_df[['start','end']] = hic_df[['start','end']].astype(int)
	print('Post-merge shape:',hic_df.shape)



	###########################################################################################
	######                               Max across tissues                              ######
	###########################################################################################

	# take max across tissues
	gtex_df = gtex_df.sort_values('pip',ascending=False).drop_duplicates(subset='variant_hg38').copy()
	abc_df = abc_df.sort_values('Score',ascending=False).drop_duplicates(subset=['cCRE ID','Gene ID']).copy()
	crispr_df = crispr_df.sort_values('Regulated',ascending=False).drop_duplicates('name').copy()

	# separate out hic assays and take max
	rnachia_df = hic_df[hic_df['Assay Type'] == 'RNAPII-ChIAPET'].copy()
	rnachia_df = rnachia_df.sort_values('Score').drop_duplicates(subset=['cCRE ID','Gene ID']).copy()
	ctcfchia_df = hic_df[hic_df['Assay Type'] == 'CTCF-ChIAPET'].copy()
	ctcfchia_df = ctcfchia_df.sort_values('Score').drop_duplicates(subset=['cCRE ID','Gene ID']).copy()
	intact_df = hic_df[hic_df['Assay Type'] == 'Intact-HiC'].copy()
	intact_df = intact_df.sort_values('Score').drop_duplicates(subset=['cCRE ID','Gene ID']).copy()



	###########################################################################################
	######                                  Save bedfiles                                ######
	###########################################################################################

	# gtex
	gtex_df['gene_standard'] = gtex_df['gene'].str.split('.').str[0]
	gtex_df[['chr','end']] = gtex_df['variant_hg38'].str.split('_',expand=True)[[0,1]]
	gtex_df['end'] = gtex_df['end'].astype(int)
	gtex_df['start'] = gtex_df['end'] - 1
	gtex_df[['start','end']] = gtex_df[['start','end']].astype(int)
	gtex_df['final_col'] = gtex_df['variant_hg38'] + ';' + gtex_df['pip'].astype(str) + ';' + gtex_df['gene_standard']
	gtex_df[['chr','start','end','final_col']].to_csv(f'{BED_PATH}/gtex.bed',header=False,index=False,sep='\t')

	# abc
	abc_df[['chr','start','end']] = abc_df['Location'].str.split('_',expand=True)
	abc_df[['start','end']] = abc_df[['start','end']].astype(int)
	abc_df['final_col'] = abc_df['cCRE ID'] + ';' + abc_df['Experiment ID'] + ';' + abc_df['Assay Type'] + ';' + abc_df['Score'].astype(str) + ';' + abc_df['Gene ID'].astype(str)
	abc_df[['chr','start','end','final_col']].to_csv(f'{BED_PATH}/abc.bed',header=False,index=False,sep='\t')

	# crispr
	crispr_df[['chromStart','chromEnd']] = crispr_df[['chromStart','chromEnd']].astype(int)
	crispr_df['final_col'] = crispr_df['name'] + ';' + crispr_df['Regulated'].astype(str) + ';' + crispr_df['measuredGeneEnsemblId']
	crispr_df[['chrom','chromStart','chromEnd','final_col']].to_csv(f'{BED_PATH}/crispr.bed',header=False,index=False,sep='\t')

	# hic
	rnachia_df['final_col'] = rnachia_df['cCRE ID'] + ';' + rnachia_df['Experiment ID'] + ';' + rnachia_df['Score'].astype(str) + ';' + rnachia_df['Gene ID']
	rnachia_df[['chr','start','end','final_col']].to_csv(f'{BED_PATH}/rnap2_chiapet.bed',header=False,index=False,sep='\t')
	ctcfchia_df['final_col'] = ctcfchia_df['cCRE ID'] + ';' + ctcfchia_df['Experiment ID']  + ';' + ctcfchia_df['Score'].astype(str) + ';' + ctcfchia_df['Gene ID']
	ctcfchia_df[['chr','start','end','final_col']].to_csv(f'{BED_PATH}/ctcf_chiapet.bed',header=False,index=False,sep='\t')
	intact_df['final_col'] = intact_df['cCRE ID'] + ';' + intact_df['Experiment ID'] + ';' + intact_df['P-value'].astype(str) + ';' + intact_df['Score'].astype(str) + ';' + intact_df['Gene ID'].astype(str)
	intact_df[['chr','start','end','final_col']].to_csv(f'{BED_PATH}/intact_hic.bed',header=False,index=False,sep='\t')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_file_path", type=str, default='/projects/zhanglab/users/ana/multiome/validation/originals')
    parser.add_argument("--screen_path", type=str, default='/projects/zhanglab/users/ana/multiome/validation/gentruth/auxiliary/screen')
    parser.add_argument("--bed_path", type=str, default='/projects/zhanglab/users/ana/bedtools2/ana_bedfiles/validation/tissue_max')

    args = parser.parse_args()
    main(args)