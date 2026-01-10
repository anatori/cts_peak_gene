#!/bin/sh

# FILE: /projects/zhanglab/users/ana/cts_peak_gene/experiments/job.pipeline/cli.sh

MULTIOME_FILE=/projects/zhanglab/users/ana/multiome/processed/neatseq/neat_filt.h5mu
TARGET_PATH=/projects/zhanglab/users/ana/multiome/results/ctar/final_eval
GENOME_FILE=/projects/zhanglab/users/ana/bedtools2/ana_bedfiles/ref/GRCh38.p14.genome.fa.bgz
BIN_CONFIG=5.5.inf.inf.200
BIN_TYPE='mean_var'
PYBEDTOOLS_PATH=/projects/zhanglab/users/ana/bedtools2/bin
BATCH_SIZE=200
RESULTS_PATH=/projects/zhanglab/users/ana/multiome/results/ctar/final_eval

# Parse BIN_CONFIG
IFS='.' read -r b1 b2 b3 b4 rest <<< "$BIN_CONFIG"

# Check if using inf.inf mode
if [[ "$b3" == "inf" && "$b4" == "inf" ]]; then
    echo "Using inf.inf mode (no RNA binning)"
    USE_INF_MODE=true
    # For inf.inf mode, we don't pre-compute number of batches
    # because it depends on the number of cis-links (unknown until runtime)
else
    echo "Using standard binned mode"
    USE_INF_MODE=false
    # Calculate number of bins for standard mode
    NUM_LINKS=$(( b1 * b2 * b3 * b4 ))
    TOTAL_NUM_BATCHES=$(( (NUM_LINKS + BATCH_SIZE - 1) / BATCH_SIZE ))
    ARRAY_OPT="--array=0-$((TOTAL_NUM_BATCHES - 1))%10"
fi

echo "Submitting $BIN_CONFIG generate_links job..."
# Submit generate_links job (no array job needed for either mode)
sbatch -p mzhang,pool1 -t 1-00:00:00 -x compute-1-1 --mem 24Gb \
  -o /projects/zhanglab/users/ana/logs/compute_%A_%a.err -J "links_$BIN_CONFIG" --wrap " \
  source ~/.bashrc && \
  conda activate ctar && \
  python /projects/zhanglab/users/ana/cts_peak_gene/CLI_ctar.py \
      --job generate_links \
      --job_id \$SLURM_JOB_ID \
      --multiome_file $MULTIOME_FILE \
      --batch_size $BATCH_SIZE \
      --binning_config $BIN_CONFIG \
      --binning_type $BIN_TYPE \
      --genome_file $GENOME_FILE \
      --target_path $TARGET_PATH \
      --pybedtools_path $PYBEDTOOLS_PATH"

# Uncomment to run compute_cis_only
# if [ "$USE_INF_MODE" = true ]; then
#     echo "Submitting $BIN_CONFIG compute_cis_only job (inf.inf mode - single job)..."
#     # In inf.inf mode, no array job needed (processes all cis-links in one job)
#     sbatch -p mzhang,pool1 -t 3-00:00:00 -x compute-1-1 --mem-per-cpu 8Gb -n 1 -c 16 --mail-type=END --mail-user=asprieto@andrew.cmu.edu \
#       -o /projects/zhanglab/users/ana/logs/compute_%A_%a.err -J "cis_$BIN_CONFIG" --wrap " \
#       source ~/.bashrc && \
#       conda activate ctar && \
#       python /projects/zhanglab/users/ana/cts_peak_gene/CLI_ctar.py \
#           --job compute_cis_only \
#           --job_id \$SLURM_JOB_ID \
#           --multiome_file $MULTIOME_FILE \
#           --batch_size $BATCH_SIZE \
#           --binning_config $BIN_CONFIG \
#           --binning_type $BIN_TYPE \
#           --genome_file $GENOME_FILE \
#           --target_path $TARGET_PATH \
#           --results_path $RESULTS_PATH \
#           --pybedtools_path $PYBEDTOOLS_PATH"
# else
#     echo "Submitting $BIN_CONFIG compute_cis_only job (standard mode - no array job)..."
#     # In standard mode, compute_cis_only doesn't use array jobs in this script
#     sbatch -p mzhang,pool1 -t 3-00:00:00 -x compute-1-1 --mem-per-cpu 8Gb -n 1 -c 16 --mail-type=END --mail-user=asprieto@andrew.cmu.edu \
#       -o /projects/zhanglab/users/ana/logs/compute_%A_%a.err -J "cis_$BIN_CONFIG" --wrap " \
#       source ~/.bashrc && \
#       conda activate ctar && \
#       python /projects/zhanglab/users/ana/cts_peak_gene/CLI_ctar.py \
#           --job compute_cis_only \
#           --job_id \$SLURM_JOB_ID \
#           --multiome_file $MULTIOME_FILE \
#           --batch_size $BATCH_SIZE \
#           --binning_config $BIN_CONFIG \
#           --binning_type $BIN_TYPE \
#           --genome_file $GENOME_FILE \
#           --target_path $TARGET_PATH \
#           --results_path $RESULTS_PATH \
#           --pybedtools_path $PYBEDTOOLS_PATH"
# fi

# Uncomment to run generate_controls
# echo "Submitting $BIN_CONFIG generate_controls job..."
# # Submit generate_controls job (no array job needed for either mode)
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

# Uncomment to run compute_ctrl_only
# if [ "$USE_INF_MODE" = true ]; then
#     # For inf.inf mode, we need to determine the number of cis-links from the CSV
#     # Wait for generate_links to complete, then count lines in CSV
#     echo "For inf.inf mode compute_ctrl_only with array jobs:"
#     echo "  1. Wait for generate_links job to complete"
#     echo "  2. Count cis-links:  NUM_CIS_LINKS=\$(wc -l < \$TARGET_PATH/{JOB_ID}_results/cis_links_df.csv)"
#     echo "  3. Calculate batches:  TOTAL_NUM_BATCHES=\$(( (NUM_CIS_LINKS + BATCH_SIZE - 1) / BATCH_SIZE ))"
#     echo "  4. Then submit with --array=0-\$((TOTAL_NUM_BATCHES - 1))"
#     echo ""
#     echo "Example command to run after generate_links completes:"
#     echo "NUM_CIS_LINKS=\$(tail -n +2 $TARGET_PATH/\${JOB_ID}_results/cis_links_df.csv | wc -l)"
#     echo "TOTAL_NUM_BATCHES=\$(( (NUM_CIS_LINKS + $BATCH_SIZE - 1) / $BATCH_SIZE ))"
#     echo 'sbatch -p mzhang,pool1 -t 3-00:00:00 --array=0-$((TOTAL_NUM_BATCHES - 1))%10 .. .'
# else
#     echo "Submitting $BIN_CONFIG compute_ctrl_only job with $TOTAL_NUM_BATCHES batches..."
#     # Submit compute_ctrl_only job with array
#     sbatch -p mzhang,pool1 -t 3-00:00:00 -x compute-1-1 $ARRAY_OPT --mem-per-cpu 4Gb -n 1 -c 16 --mail-type=END --mail-user=asprieto@andrew.cmu.edu \
#       -o /projects/zhanglab/users/ana/logs/compute_%A_%a.err -J "ctrl_$BIN_CONFIG" --wrap " \
#       source ~/. bashrc && \
#       conda activate ctar && \
#       python /projects/zhanglab/users/ana/cts_peak_gene/CLI_ctar. py \
#           --job compute_ctrl_only \
#           --job_id \$SLURM_JOB_ID \
#           --array_idx \$SLURM_ARRAY_TASK_ID \
#           --multiome_file $MULTIOME_FILE \
#           --batch_size $BATCH_SIZE \
#           --target_path $TARGET_PATH \
#           --results_path $RESULTS_PATH \
#           --binning_config $BIN_CONFIG \
#           --binning_type $BIN_TYPE \
#           --genome_file $GENOME_FILE \
#           --pybedtools_path $PYBEDTOOLS_PATH"
# fi