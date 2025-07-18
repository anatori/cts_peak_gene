#!/bin/sh

# FILE: /projects/zhanglab/users/ana/cts_peak_gene/experiments/job.poisscoeff/poissonb.sh

MULTIOME_FILE=/projects/zhanglab/users/ana/multiome/processed/neurips_bmmc/bmmc.h5mu
LINKS_FILE=/projects/zhanglab/users/ana/multiome/simulations/bin_analysis/cis_gene/cis_gene.csv
TARGET_PATH=/projects/zhanglab/users/ana/multiome/simulations/bin_analysis/
GENOME_FILE=/projects/zhanglab/users/ana/bedtools2/ana_bedfiles/ref/GRCh38.p14.genome.fa.bgz
BIN_CONFIG=$1
BIN_TYPE='mean_var'
PYBEDTOOLS_PATH=/projects/zhanglab/users/ana/bedtools2/bin
BATCH_SIZE=200

# echo "Beginning $BIN_CONFIG create_ctrl job..."
# # Submit create_ctrl job
# sbatch -p mzhang,pool1 -t 1-00:00:00 --mem=82Gb -o /home/asprieto/logs/create_ctrl_%J.err -J "ctrl_$BIN_CONFIG" --wrap " \
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

# # Count control link files
# TOTAL_NUM_BIN=$(find "$TARGET_PATH/ctrl_peaks/ctrl_links_$BIN_CONFIG" -type f | wc -l)

# # Check if control files < batch size or requires link-based chunks
# if [[ $BIN_CONFIG =~ "inf" ]]; then 
#   NUM_LINKS=$(wc -l < "$LINKS_FILE")
#   TOTAL_NUM_BATCHES=$(( (NUM_LINKS + BATCH_SIZE - 1) / BATCH_SIZE ))
#   if [ "$TOTAL_NUM_BATCHES" -eq 0 ]; then
#     ARRAY_OPT="--array=0"
#   else
#     ARRAY_OPT="--array=0-$((TOTAL_NUM_BATCHES - 1))%20"
#   fi
# else
#   if [ "$TOTAL_NUM_BIN" -lt "$BATCH_SIZE" ]; then
#     TOTAL_NUM_BATCHES=1
#     ARRAY_OPT="--array=0"
#   else
#     TOTAL_NUM_BATCHES=$(( (TOTAL_NUM_BIN + BATCH_SIZE - 1) / BATCH_SIZE ))
#     if [ "$TOTAL_NUM_BATCHES" -eq 0 ]; then
#       ARRAY_OPT="--array=0"
#     else
#       ARRAY_OPT="--array=0-$((TOTAL_NUM_BATCHES - 1))%20"
#     fi
#   fi
# fi

echo "Found $TOTAL_NUM_BIN control files. Submitting $TOTAL_NUM_BATCHES batch job(s)."

echo "Submitting $BIN_CONFIG compute_pr job..."
# Submit compute_corr job
sbatch -p mzhang,pool1 -t 1-00:00:00 -x compute-1-1 --mem=32Gb --array=0 --mail-type=END --mail-user=asprieto@andrew.cmu.edu \
  -o /home/asprieto/logs/compute_pr_%A_%a.err -J "pr_$BIN_CONFIG" --wrap " \
  source ~/.bashrc && \
  conda activate ctar && \
  python /projects/zhanglab/users/ana/cts_peak_gene/CLI_ctar.071725.py \
      --job compute_pr \
      --multiome_file $MULTIOME_FILE \
      --batch_size $BATCH_SIZE \
      --batch_id \$SLURM_ARRAY_TASK_ID \
      --links_file $LINKS_FILE \
      --target_path $TARGET_PATH \
      --binning_config $BIN_CONFIG \
      --binning_type $BIN_TYPE \
      --pybedtools_path $PYBEDTOOLS_PATH"


# echo "Beginning $BIN_CONFIG compute_pval job..."
# # Submit compute_pval job
# sbatch -p mzhang,pool1 -t 1-00:00:00 --mem=82Gb -o /home/asprieto/logs/compute_pval_%J.err -J "pval_$BIN_CONFIG" --wrap " \
# source ~/.bashrc && \
# conda activate ctar && \
# python /projects/zhanglab/users/ana/cts_peak_gene/CLI_ctar.py \
#     --job compute_pval \
#     --links_file $LINKS_FILE \
#     --target_path $TARGET_PATH \
#     --binning_config $BIN_CONFIG \
#     --binning_type $BIN_TYPE \
#     --method pr" 