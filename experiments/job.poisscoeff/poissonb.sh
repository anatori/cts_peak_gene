#!/bin/sh

# FILE: /projects/zhanglab/users/ana/cts_peak_gene/experiments/job.poisscoeff/poissonb.sh

MULTIOME_FILE=/projects/zhanglab/users/ana/multiome/processed/neurips_bmmc/bmmc.h5mu
TARGET_PATH=/projects/zhanglab/users/ana/multiome/simulations/bin_analysis
GENOME_FILE=/projects/zhanglab/users/ana/bedtools2/ana_bedfiles/ref/GRCh38.p14.genome.fa.bgz
BIN_CONFIG=$1
BIN_TYPE='mean_var'
PYBEDTOOLS_PATH=/projects/zhanglab/users/ana/bedtools2/bin
BATCH_SIZE=50


# echo "Beginning $BIN_CONFIG create_ctrl job..."
# # Submit create_ctrl job
# sbatch -p mzhang,pool1 -t 1-00:00:00 --mem=64Gb -o /projects/zhanglab/users/ana/logs/create_ctrl_%J.err -J "ctrl_$BIN_CONFIG" --wrap " \
# source ~/.bashrc && \
# conda activate ctar && \
# python /projects/zhanglab/users/ana/cts_peak_gene/CLI_ctar.py \
#     --job create_ctrl \
#     --multiome_file $MULTIOME_FILE \
#     --links_file $LINKS_FILE \
#     --target_path $TARGET_PATH \
#     --genome_file $GENOME_FILE \
#     --binning_config $BIN_CONFIG \
#     --binning_type $BIN_TYPE \
#     --pybedtools_path $PYBEDTOOLS_PATH \
#     --method pr" 


# echo "Beginning compute_cis_pairs job..."
# # Submit create_ctrl job
# sbatch -p mzhang,pool1 -t 1-00:00:00 -x compute-1-1 --mem-per-cpu 32Gb -n 1 -c 2 \
#   -o /projects/zhanglab/users/ana/logs/compute_cis_pairs_%J.err -J "pr_cis_pairs" --wrap " \
#   source ~/.bashrc && \
#   conda activate ctar && \
#   python /projects/zhanglab/users/ana/cts_peak_gene/CLI_ctar.py \
#       --job compute_cis_pairs \
#       --multiome_file $MULTIOME_FILE \
#       --links_file $LINKS_FILE \
#       --pybedtools_path $PYBEDTOOLS_PATH \
#       --batch_size $BATCH_SIZE \
#       --method pr" 


echo "Submitting $BIN_CONFIG compute_pr job..."
# Submit compute_corr job
sbatch -p mzhang,pool1 -t 1-00:00:00 -x compute-1-1 --mem-per-cpu 4Gb -n 1 -c 16 --mail-type=END --mail-user=asprieto@andrew.cmu.edu \
  -o /projects/zhanglab/users/ana/logs/compute_pr_%J.err -J "pr_$BIN_CONFIG" --wrap " \
  source ~/.bashrc && \
  conda activate ctar && \
  python /projects/zhanglab/users/ana/cts_peak_gene/CLI_ctar.py \
      --job compute_pr \
      --multiome_file $MULTIOME_FILE \
      --batch_size $BATCH_SIZE \
      --links_file $LINKS_FILE \
      --target_path $TARGET_PATH \
      --binning_config $BIN_CONFIG \
      --binning_type $BIN_TYPE \
      --pybedtools_path $PYBEDTOOLS_PATH"


# echo "Beginning $BIN_CONFIG compute_pval job..."
# # Submit compute_pval job
# sbatch -p mzhang,pool1 -t 1-00:00:00 --mem=82Gb -o /projects/zhanglab/users/ana/logs/compute_pval_%J.err -J "pval_$BIN_CONFIG" --wrap " \
# source ~/.bashrc && \
# conda activate ctar && \
# python /projects/zhanglab/users/ana/cts_peak_gene/CLI_ctar.py \
#     --job compute_pval \
#     --links_file $LINKS_FILE \
#     --target_path $TARGET_PATH \
#     --binning_config $BIN_CONFIG \
#     --binning_type $BIN_TYPE \
#     --method pr" 