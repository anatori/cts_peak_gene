#!/bin/sh

# FILE: /projects/zhanglab/users/ana/cts_peak_gene/experiments/job.pipeline/cli.sh

MULTIOME_FILE=/projects/zhanglab/users/ana/multiome/processed/neurips_bmmc/bmmc.h5mu
TARGET_PATH=/projects/zhanglab/users/ana/multiome/simulations/bin_analysis
GENOME_FILE=/projects/zhanglab/users/ana/bedtools2/ana_bedfiles/ref/GRCh38.p14.genome.fa.bgz
BIN_CONFIG=$1
BIN_TYPE='mean_var'
PYBEDTOOLS_PATH=/projects/zhanglab/users/ana/bedtools2/bin
BATCH_SIZE=50
RESULTS_PATH=/projects/zhanglab/users/ana/multiome/simulations/bin_analysis/5055088_results
COVAR_FILE=/projects/zhanglab/users/ana/multiome/processed/neurips_bmmc/covars.npy


echo "Submitting $BIN_CONFIG compute_cis_only job..."
# Submit compute_cis_only job
sbatch -p mzhang,pool1 -t 1-00:00:00 -x compute-1-1 --mem-per-cpu 4Gb -n 1 -c 16 --mail-type=END --mail-user=asprieto@andrew.cmu.edu \
  -o /projects/zhanglab/users/ana/logs/compute_%J.err -J "pr_$BIN_CONFIG" --wrap " \
  source ~/.bashrc && \
  conda activate ctar && \
  python /projects/zhanglab/users/ana/cts_peak_gene/CLI_ctar.py \
      --job compute_cis_only \
      --job_id \$SLURM_JOB_ID \
      --multiome_file $MULTIOME_FILE \
      --batch_size $BATCH_SIZE \
      --target_path $TARGET_PATH \
      --binning_config $BIN_CONFIG \
      --binning_type $BIN_TYPE \
      --genome_file $GENOME_FILE \
      --covar_file $COVAR_FILE \
      --pybedtools_path $PYBEDTOOLS_PATH"


# echo "Submitting $BIN_CONFIG compute_ctrl_only job..."
# # Submit compute_ctrl_only job
# sbatch -p mzhang,pool1 -t 3-00:00:00 -x compute-1-1 --mem-per-cpu 4Gb -n 1 -c 32 --mail-type=END --mail-user=asprieto@andrew.cmu.edu \
#   -o /projects/zhanglab/users/ana/logs/compute_%J.err -J "pr_$BIN_CONFIG" --wrap " \
#   source ~/.bashrc && \
#   conda activate ctar && \
#   python /projects/zhanglab/users/ana/cts_peak_gene/CLI_ctar.py \
#       --job compute_ctrl_only \
#       --job_id \$SLURM_JOB_ID \
#       --multiome_file $MULTIOME_FILE \
#       --batch_size $BATCH_SIZE \
#       --target_path $TARGET_PATH \
#       --results_path $RESULTS_PATH \
#       --binning_config $BIN_CONFIG \
#       --binning_type $BIN_TYPE \
#       --genome_file $GENOME_FILE \
#       --covar_file $COVAR_FILE \
#       --pybedtools_path $PYBEDTOOLS_PATH"