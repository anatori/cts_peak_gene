import argparse
import os
import pandas as pd


'''
TODO:
- tissue matching for hic
'''

def main(args):

	EVAL_FILE_PATH = args.eval_file_path
	SCREEN_PATH = args.screen_path
	BED_PATH = args.bed_path


	###########################################################################################
	######                                  Data Loading                                 ######
	###########################################################################################

	# onek1k
	onek1k_path = EVAL_FILE_PATH + '/susie_finemap_onek1k'
	onek1k_df = pd.read_csv(f'{onek1k_path}/susie_onek1k_allcells.tsv',sep='\t')
	print('OneK1K shape:',onek1k_df.shape)

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

	# split up hic assays
	rnachia_df = hic_df[hic_df['Assay Type'] == 'RNAPII-ChIAPET'].copy()
	ctcfchia_df = hic_df[hic_df['Assay Type'] == 'CTCF-ChIAPET'].copy()
	intact_df = hic_df[hic_df['Assay Type'] == 'Intact-HiC'].copy()


	###########################################################################################
	######                                  Clean for bed                                ######
	###########################################################################################

	# onek1k
	onek1k_df['end'] = onek1k_df['pos'].astype(int)
	onek1k_df['start'] = onek1k_df['end'].astype(int) - 1
	onek1k_df[['start','end']] = onek1k_df[['start','end']].astype(int)
	onek1k_df['final_col'] = onek1k_df['snp'] + ';' + onek1k_df['pip'].astype(str) + ';' + onek1k_df['gene']

	# gtex
	gtex_df['gene_standard'] = gtex_df['gene'].str.split('.').str[0]
	gtex_df[['chr','end']] = gtex_df['variant_hg38'].str.split('_',expand=True)[[0,1]]
	gtex_df['end'] = gtex_df['end'].astype(int)
	gtex_df['start'] = gtex_df['end'] - 1
	gtex_df[['start','end']] = gtex_df[['start','end']].astype(int)
	gtex_df['final_col'] = gtex_df['variant_hg38'] + ';' + gtex_df['pip'].astype(str) + ';' + gtex_df['gene_standard']

	# abc
	abc_df[['chr','start','end']] = abc_df['Location'].str.split('_',expand=True)
	abc_df[['start','end']] = abc_df[['start','end']].astype(int)
	abc_df['final_col'] = abc_df['cCRE ID'] + ';' + abc_df['Experiment ID'] + ';' + abc_df['Assay Type'] + ';' + abc_df['Score'].astype(str) + ';' + abc_df['Gene ID'].astype(str)

	# crispr
	crispr_df[['chromStart','chromEnd']] = crispr_df[['chromStart','chromEnd']].astype(int)
	crispr_df['final_col'] = crispr_df['name'] + ';' + crispr_df['Regulated'].astype(str) + ';' + crispr_df['measuredGeneEnsemblId']

	# hic
	rnachia_df['final_col'] = rnachia_df['cCRE ID'] + ';' + rnachia_df['Experiment ID'] + ';' + rnachia_df['Score'].astype(str) + ';' + rnachia_df['Gene ID']
	ctcfchia_df['final_col'] = ctcfchia_df['cCRE ID'] + ';' + ctcfchia_df['Experiment ID']  + ';' + ctcfchia_df['Score'].astype(str) + ';' + ctcfchia_df['Gene ID']
	intact_df['final_col'] = intact_df['cCRE ID'] + ';' + intact_df['Experiment ID'] + ';' + intact_df['P-value'].astype(str) + ';' + intact_df['Score'].astype(str) + ';' + intact_df['Gene ID'].astype(str)


	###########################################################################################
	######                       Max within matched tissues                              ######
	###########################################################################################

	# tissue matching
	neat_gtex = ["Whole_Blood","Cells_EBV-transformed_lymphocytes","Spleen"] # bmmc and neat have same tissue matches
	brain_gtex = ["Brain_Cerebellum","Brain_Cerebellar_Hemisphere"]
	pbmc_gtex = ["Whole_Blood"]

	neat_abc = [
	    "CD4-positive,_alpha-beta_T_cell",
	    "CD4-positive,_alpha-beta_memory_T_cell",
	    "activated_CD4-positive,_alpha-beta_T_cell",
	    "activated_CD4-positive,_alpha-beta_memory_T_cell",
	    "activated_naive_CD4-positive,_alpha-beta_T_cell",
	    "stimulated_activated_CD4-positive,_alpha-beta_T_cell",
	    "activated_effector_memory_CD4-positive,_alpha-beta_T_cell",
	    "central_memory_CD4-positive,_alpha-beta_T_cell",
	    "effector_memory_CD4-positive,_alpha-beta_T_cell",
	    "T-cell",
	    "activated_T-cell",
	    "T-helper_1_cell",
	    "T-helper_2_cell",
	    "T-helper_17_cell",
	    "T_follicular_helper_cell"
	]

	brain_abc = [
	    "cerebellum",
	    "cerebellar_cortex",
	    "astrocyte_of_the_cerebellum"
	]

	bmmc_abc = [
		"hematopoietic_multipotent_progenitor_cell",
	    "common_myeloid_progenitor,_CD34-positive",
	    "stromal_cell_of_bone_marrow",
	]

	pbmc_abc = neat_abc + [
		'B_cell',
		'naive_B_cell',
		'memory_B_cell',
		'activated_B_cell',
		'stimulated_activated_naive_B_cell',
		'natural_killer_cell',
		'immature_natural_killer_cell',
		'CD14-positive_monocyte',
		'CD1c-positive_myeloid_dendritic_cell',
	]

	shareseq_abc = ['K562']

	# take max across tissues
	print('deduplicating...')
	print('onek1k_df total',onek1k_df.shape)
	onek1k_df = onek1k_df.sort_values('pip',ascending=False).drop_duplicates(subset=['snp','gene']).copy()
	print('onek1k_df after',onek1k_df.shape)
	print('')

	print('gtex_df total',gtex_df.shape)
	gtex_neat_df = gtex_df[gtex_df["tissue"].isin(neat_gtex)].copy()
	print('gtex_neat_df before',gtex_neat_df.shape)
	gtex_neat_df = gtex_neat_df.sort_values('pip',ascending=False).drop_duplicates(subset=['variant_hg38','gene']).copy()
	print('gtex_neat_df after',gtex_neat_df.shape)
	gtex_brain_df = gtex_df[gtex_df["tissue"].isin(brain_gtex)].copy()
	print('gtex_brain_df before',gtex_brain_df.shape)
	gtex_brain_df = gtex_brain_df.sort_values('pip',ascending=False).drop_duplicates(subset=['variant_hg38','gene']).copy()
	print('gtex_brain_df after',gtex_brain_df.shape)
	gtex_pbmc_df = gtex_df[gtex_df["tissue"].isin(pbmc_gtex)].copy()
	print('gtex_pbmc_df before',gtex_brain_df.shape)
	gtex_pbmc_df = gtex_pbmc_df.sort_values('pip',ascending=False).drop_duplicates(subset=['variant_hg38','gene']).copy()
	print('gtex_pbmc_df after',gtex_pbmc_df.shape)
	print('')

	print('abc_df total',abc_df.shape)
	abc_neat_df = abc_df[abc_df["Biosample"].isin(neat_abc)].copy()
	print('abc_neat_df before',abc_neat_df.shape)
	abc_neat_df = abc_neat_df.sort_values('Score',ascending=False).drop_duplicates(subset=['cCRE ID','Gene ID']).copy()
	print('abc_neat_df after',abc_neat_df.shape)
	abc_brain_df = abc_df[abc_df["Biosample"].isin(brain_abc)].copy()
	print('abc_brain_df before',abc_brain_df.shape)
	abc_brain_df = abc_brain_df.sort_values('Score',ascending=False).drop_duplicates(subset=['cCRE ID','Gene ID']).copy()
	print('abc_brain_df after',abc_brain_df.shape)
	abc_bmmc_df = abc_df[abc_df["Biosample"].isin(bmmc_abc)].copy()
	print('abc_bmmc_df before',abc_bmmc_df.shape)
	abc_bmmc_df = abc_bmmc_df.sort_values('Score',ascending=False).drop_duplicates(subset=['cCRE ID','Gene ID']).copy()
	print('abc_bmmc_df after',abc_bmmc_df.shape)
	abc_pbmc_df = abc_df[abc_df["Biosample"].isin(pbmc_abc)].copy()
	print('abc_pbmc_df before',abc_pbmc_df.shape)
	abc_pbmc_df = abc_pbmc_df.sort_values('Score',ascending=False).drop_duplicates(subset=['cCRE ID','Gene ID']).copy()
	print('abc_pbmc_df after',abc_pbmc_df.shape)
	abc_shareseq_df = abc_df[abc_df["Biosample"].isin(shareseq_abc)].copy()
	print('abc_shareseq_df before',abc_shareseq_df.shape)
	abc_shareseq_df = abc_shareseq_df.sort_values('Score',ascending=False).drop_duplicates(subset=['cCRE ID','Gene ID']).copy()
	print('abc_shareseq_df after',abc_shareseq_df.shape)
	print('')

	# there should be no duplicates in crispr	
	# separate out hic assays and take max
	print('rnachia_df before',rnachia_df.shape)
	rnachia_df = rnachia_df.sort_values('Score').drop_duplicates(subset=['cCRE ID','Gene ID']).copy()
	print('rnachia_df after',rnachia_df.shape)
	print('ctcfchia_df before',ctcfchia_df.shape)
	ctcfchia_df = ctcfchia_df.sort_values('Score').drop_duplicates(subset=['cCRE ID','Gene ID']).copy()
	print('ctcfchia_df after',ctcfchia_df.shape)
	print('intact_df before',intact_df.shape)
	intact_df = intact_df.sort_values('Score').drop_duplicates(subset=['cCRE ID','Gene ID']).copy()
	print('intact_df after',intact_df.shape)



	###########################################################################################
	######                                  Save bedfiles                                ######
	###########################################################################################

	# onek1k
	onek1k_df[['chrom','start','end','final_col']].to_csv(f'{BED_PATH}/onek1k_pbmc.bed',header=False,index=False,sep='\t')

	# gtex
	gtex_neat_df[['chr','start','end','final_col']].to_csv(f'{BED_PATH}/gtex_neat.bed',header=False,index=False,sep='\t')
	gtex_neat_df[['chr','start','end','final_col']].to_csv(f'{BED_PATH}/gtex_bmmc.bed',header=False,index=False,sep='\t')
	gtex_brain_df[['chr','start','end','final_col']].to_csv(f'{BED_PATH}/gtex_brain.bed',header=False,index=False,sep='\t')
	gtex_pbmc_df[['chr','start','end','final_col']].to_csv(f'{BED_PATH}/gtex_pbmc.bed',header=False,index=False,sep='\t')

	# abc
	abc_neat_df[['chr','start','end','final_col']].to_csv(f'{BED_PATH}/abc_neat.bed',header=False,index=False,sep='\t')
	abc_brain_df[['chr','start','end','final_col']].to_csv(f'{BED_PATH}/abc_brain.bed',header=False,index=False,sep='\t')
	abc_bmmc_df[['chr','start','end','final_col']].to_csv(f'{BED_PATH}/abc_bmmc.bed',header=False,index=False,sep='\t')
	abc_pbmc_df[['chr','start','end','final_col']].to_csv(f'{BED_PATH}/abc_pbmc.bed',header=False,index=False,sep='\t')
	abc_shareseq_df[['chr','start','end','final_col']].to_csv(f'{BED_PATH}/abc_shareseq.bed',header=False,index=False,sep='\t')

	# crispr - save bulk and per celltype and per k562 experiment
	crispr_df[['chrom','chromStart','chromEnd','final_col']].to_csv(f'{BED_PATH}/crispr.bed',header=False,index=False,sep='\t')
	for ct in crispr_df.CellType.unique():
		crispr_df.loc[(crispr_df.CellType == ct),['chrom','chromStart','chromEnd','final_col']].to_csv(f'{BED_PATH}/crispr_{ct}.bed',header=False,index=False,sep='\t')
	crispr_df.loc[(crispr_df.Reference == 'Gasperini et al., 2019'),['chrom','chromStart','chromEnd','final_col']].to_csv(f'{BED_PATH}/crispr_gasperini_k562.bed',header=False,index=False,sep='\t')
	crispr_df.loc[(crispr_df.CellType == 'Nasser et al., 2021 from Fulco et al., 2019'),['chrom','chromStart','chromEnd','final_col']].to_csv(f'{BED_PATH}/crispr_fulco_k562.bed',header=False,index=False,sep='\t')

	# hic
	rnachia_df[['chr','start','end','final_col']].to_csv(f'{BED_PATH}/rnap2_chiapet.bed',header=False,index=False,sep='\t')
	ctcfchia_df[['chr','start','end','final_col']].to_csv(f'{BED_PATH}/ctcf_chiapet.bed',header=False,index=False,sep='\t')
	intact_df[['chr','start','end','final_col']].to_csv(f'{BED_PATH}/intact_hic.bed',header=False,index=False,sep='\t')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_file_path", type=str, default='/projects/zhanglab/users/ana/multiome/validation/originals')
    parser.add_argument("--screen_path", type=str, default='/projects/zhanglab/users/ana/multiome/validation/gentruth/auxiliary/screen')
    parser.add_argument("--bed_path", type=str, default='/projects/zhanglab/users/ana/bedtools2/ana_bedfiles/validation/tissue_max')

    args = parser.parse_args()
    main(args)