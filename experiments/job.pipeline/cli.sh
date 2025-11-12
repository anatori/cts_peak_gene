#!/bin/sh

# FILE: /projects/zhanglab/users/ana/cts_peak_gene/experiments/job.pipeline/cli.sh

MULTIOME_FILE=/projects/zhanglab/users/ana/multiome/processed/neatseq/neat_filt.h5mu
TARGET_PATH=/projects/zhanglab/users/ana/multiome/simulations/bin_analysis/neatseq/studentized/filtered/inf
GENOME_FILE=/projects/zhanglab/users/ana/bedtools2/ana_bedfiles/ref/GRCh38.p14.genome.fa.bgz
BIN_CONFIG=1.1.1.1.10000
BIN_TYPE='mean_var'
PYBEDTOOLS_PATH=/projects/zhanglab/users/ana/bedtools2/bin
BATCH_SIZE=1000
RESULTS_PATH=/projects/zhanglab/users/ana/multiome/simulations/bin_analysis/neatseq/studentized/filtered/inf
# COVAR_FILE=/projects/zhanglab/users/ana/multiome/processed/neurips_bmmc/covars.npy

IFS='.' read -r b1 b2 b3 b4 rest <<< "$BIN_CONFIG"
NUM_LINKS=$(( b1 * b2 * b3 * b4 ))
TOTAL_NUM_BATCHES=$(( (NUM_LINKS + BATCH_SIZE - 1) / BATCH_SIZE ))
ARRAY_OPT="--array=0-$((TOTAL_NUM_BATCHES - 1))%10"

# echo "Submitting $BIN_CONFIG generate_links job..."
# # Submit generate_links job
# sbatch -p mzhang,pool1 -t 1-00:00:00 -x compute-1-1 --mem 24Gb \
#   -o /projects/zhanglab/users/ana/logs/compute_%A_%a.err -J "links_$BIN_CONFIG" --wrap " \
#   source ~/.bashrc && \
#   conda activate ctar && \
#   python /projects/zhanglab/users/ana/cts_peak_gene/CLI_ctar.py \
#       --job generate_links \
#       --job_id \$SLURM_JOB_ID \
#       --multiome_file $MULTIOME_FILE \
#       --batch_size $BATCH_SIZE \
#       --binning_config $BIN_CONFIG \
#       --binning_type $BIN_TYPE \
#       --genome_file $GENOME_FILE \
#       --target_path $TARGET_PATH \
#       --pybedtools_path $PYBEDTOOLS_PATH"

# echo "Submitting $BIN_CONFIG compute_cis_only job with $TOTAL_NUM_BATCHES batches..."
# # Submit compute_cis_only job
# sbatch -p mzhang,pool1 -t 3-00:00:00 -x compute-1-1 --mem-per-cpu 8Gb -n 1 -c 16 --mail-type=END --mail-user=asprieto@andrew.cmu.edu \
#   -o /projects/zhanglab/users/ana/logs/compute_%A_%a.err -J "pr_$BIN_CONFIG" --wrap " \
#   source ~/.bashrc && \
#   conda activate ctar && \
#   python /projects/zhanglab/users/ana/cts_peak_gene/CLI_ctar.py \
#       --job compute_cis_only \
#       --job_id \$SLURM_JOB_ID \
#       --multiome_file $MULTIOME_FILE \
#       --batch_size $BATCH_SIZE \
#       --binning_config $BIN_CONFIG \
#       --binning_type $BIN_TYPE \
#       --genome_file $GENOME_FILE \
#       --target_path $TARGET_PATH \
#       --results_path $RESULTS_PATH \
#       --pybedtools_path $PYBEDTOOLS_PATH"

# echo "Submitting $BIN_CONFIG generate_controls job..."
# # Submit generate_controls job
# sbatch -p mzhang,pool1 -t 1-00:00:00 -x compute-1-1 --mem 48Gb \
#   -o /projects/zhanglab/users/ana/logs/compute_%A_%a.err -J "ctrl_$BIN_CONFIG" --wrap " \
#   source ~/.bashrc && \
#   conda activate ctar && \
#   python /projects/zhanglab/users/ana/cts_peak_gene/CLI_ctar.py \
#       --job generate_controls \
#       --job_id \$SLURM_JOB_ID \
#       --multiome_file $MULTIOME_FILE \
#       --batch_size $BATCH_SIZE \
#       --binning_config $BIN_CONFIG \
#       --binning_type $BIN_TYPE \
#       --genome_file $GENOME_FILE \
#       --target_path $TARGET_PATH \
#       --results_path $RESULTS_PATH \
#       --pybedtools_path $PYBEDTOOLS_PATH"


echo "Submitting $BIN_CONFIG compute_ctrl_only job with $TOTAL_NUM_BATCHES batches..."
# Submit compute_ctrl_only job
sbatch -p mzhang,pool1 -t 3-00:00:00 -x compute-1-1 --array=0-389%10 --mem-per-cpu 4Gb -n 1 -c 16 --mail-type=END --mail-user=asprieto@andrew.cmu.edu \
  -o /projects/zhanglab/users/ana/logs/compute_%A_%a.err -J "pr_$BIN_CONFIG" --wrap " \
  source ~/.bashrc && \
  conda activate ctar && \
  python /projects/zhanglab/users/ana/cts_peak_gene/CLI_ctar.py \
      --job compute_ctrl_only \
      --job_id \$SLURM_JOB_ID \
      --array_idx \$SLURM_ARRAY_TASK_ID \
      --multiome_file $MULTIOME_FILE \
      --batch_size $BATCH_SIZE \
      --target_path $TARGET_PATH \
      --results_path $RESULTS_PATH \
      --binning_config $BIN_CONFIG \
      --binning_type $BIN_TYPE \
      --genome_file $GENOME_FILE \
      --pybedtools_path $PYBEDTOOLS_PATH"