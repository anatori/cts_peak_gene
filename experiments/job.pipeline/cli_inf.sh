#!/bin/sh

# FILE: /projects/zhanglab/users/ana/cts_peak_gene/experiments/job.pipeline/cli_inf.sh

MULTIOME_DATASET=pbmc_male_chrom_ctrl
MULTIOME_FILE=/projects/zhanglab/users/ana/multiome/processed/pbmc_male_chrom_ctrl/pbmc_male_chrom_ctrl.h5mu
BIN_CONFIG=4.4.inf.inf.1000
TARGET_PATH=/projects/zhanglab/users/ana/multiome/results/ctar/benchmark/${MULTIOME_DATASET}/${BIN_CONFIG}
GENOME_FILE=/projects/zhanglab/users/ana/bedtools2/ana_bedfiles/ref/GRCh38.p14.genome.fa.bgz
BIN_TYPE='mean_var'
PYBEDTOOLS_PATH=/projects/zhanglab/users/ana/bedtools2/bin
BATCH_SIZE=2500
DIC_BATCH_SIZE=4096
RESULTS_PATH=/projects/zhanglab/users/ana/multiome/results/ctar/benchmark/${MULTIOME_DATASET}/${BIN_CONFIG}
PARTITION="mzhang,statgen,pool1"
MAX_CONCURRENT=10
FLAG_LL='False'
FLAG_SE='True'
FLAG_CORR='True'
N_CORES=64
RANDOM_SEED=777
DONOR_COL=''
N_LENGTH_BINS=4
FDR_BIN_TYPE='dist_rmean_rvar'
N_DIST_BINS=4
TSS_FILE=/projects/zhanglab/users/ana/bedtools2/ana_bedfiles/ref/gencode.v47.annotation.gtf.gz
RNA_LAYER='counts'
N_FDR_RNA_BINS=4
FDR_MIN_BIN_COUNT=25000
USE_CIS_ARRAY=true
FLAG_TFIDF='True'
EXCLUDE='--exclude=lanec2-0-16,lanec2-0-36'
MAX_ARRAY_SIZE=1000

IFS='.' read -r b1 b2 b3 b4 rest <<< "$BIN_CONFIG"
if [[ "$b3" == "inf" && "$b4" == "inf" ]]; then
    echo "Using inf.inf mode (no RNA binning)"
    USE_INF_MODE=true

    # cis_links_df.csv is a required pre-existing input for compute_cis_only
    CSV_FILE=$RESULTS_PATH/cis_links_df.csv
    if [ ! -f "$CSV_FILE" ]; then
        echo "ERROR: $CSV_FILE not found (expected a pre-existing cis_links_df.csv; generate_links is skipped in this script)"
        exit 1
    fi
    NUM_CIS_LINKS=$(tail -n +2 "$CSV_FILE" | wc -l)
    TOTAL_NUM_BATCHES=$(( (NUM_CIS_LINKS + BATCH_SIZE - 1) / BATCH_SIZE ))
    echo "Found $NUM_CIS_LINKS cis-links -> $TOTAL_NUM_BATCHES array tasks (batch_size=$BATCH_SIZE, dic_batch_size=$DIC_BATCH_SIZE)"

    if [ $TOTAL_NUM_BATCHES -gt $MAX_ARRAY_SIZE ]; then
        echo "ERROR: Need $TOTAL_NUM_BATCHES jobs but max array size is $MAX_ARRAY_SIZE"
        echo "Increase BATCH_SIZE (links per array task) or use a parallel-arrays approach"
        exit 1
    fi
    ARRAY_OPT="--array=0-$((TOTAL_NUM_BATCHES - 1))%${MAX_CONCURRENT}"
else
    echo "Using standard binned mode"
    USE_INF_MODE=false
    NUM_LINKS=$(( b1 * b2 * b3 * b4 * N_LENGTH_BINS ))
    TOTAL_NUM_BATCHES=$(( (NUM_LINKS + BATCH_SIZE - 1) / BATCH_SIZE ))
    ARRAY_OPT="--array=0-$((TOTAL_NUM_BATCHES - 1))%${MAX_CONCURRENT}"
fi


# ###########################################################################################
# ######                                  Generate Links                               ######
# ###########################################################################################

# if [ "$USE_CIS_ARRAY" = true ]; then
#     echo "Submitting $BIN_CONFIG generate_links job..."
#     GEN_LINKS_JOB=$(sbatch --parsable -p $PARTITION -t 1-00:00:00 \
#       -A bio240051p \
#       --ntasks-per-node=32 \
#       -o /projects/zhanglab/users/ana/logs/links_%j.err -J "links_$BIN_CONFIG" \
#       --wrap "source ~/.bashrc && conda activate ctar && python /projects/zhanglab/users/ana/cts_peak_gene/CLI_ctar.py \
#               --job generate_links --multiome_file $MULTIOME_FILE --batch_size $BATCH_SIZE \
#               --binning_config $BIN_CONFIG --binning_type $BIN_TYPE --genome_file $GENOME_FILE --target_path $TARGET_PATH \
#               --pybedtools_path $PYBEDTOOLS_PATH --results_path $RESULTS_PATH --random_seed $RANDOM_SEED \
#               --n_length_bins $N_LENGTH_BINS")
#     echo "Generate links job ID: $GEN_LINKS_JOB"
# fi

###########################################################################################
######                              Generate Controls                              ######
###########################################################################################

echo "Submitting $BIN_CONFIG generate_controls job..."
GEN_CTRL_JOB=$(sbatch --parsable -p $PARTITION -t 1-00:00:00 \
  --ntasks-per-node=32 $EXCLUDE --mem=128Gb \
  -o /projects/zhanglab/users/ana/logs/gen_ctrl_%j.err -J "gen_ctrl_$BIN_CONFIG" \
  --wrap "source ~/.bashrc && conda activate ctar && python /projects/zhanglab/users/ana/cts_peak_gene/CLI_ctar.py \
          --job generate_controls --multiome_file $MULTIOME_FILE --batch_size $BATCH_SIZE \
          --binning_config $BIN_CONFIG --binning_type $BIN_TYPE --genome_file $GENOME_FILE --target_path $TARGET_PATH \
          --pybedtools_path $PYBEDTOOLS_PATH --results_path $RESULTS_PATH --random_seed $RANDOM_SEED \
          --n_length_bins $N_LENGTH_BINS")
echo "Generate controls job ID: $GEN_CTRL_JOB"

# ###########################################################################################
# ######                            Compute cis coefficients                           ######
# ###########################################################################################

# if [ "$USE_CIS_ARRAY" = true ]; then
#     echo "Submitting $BIN_CONFIG compute_cis_only array job with $TOTAL_NUM_BATCHES tasks..."
#     CIS_JOB=$(sbatch --parsable -p $PARTITION -t 3-00:00:00 \
#         $ARRAY_OPT $EXCLUDE \
#         --ntasks-per-node=$N_CORES \
#         -o /projects/zhanglab/users/ana/logs/cis_%A_%a.err \
#         -J "cis_$BIN_CONFIG" \
#         --wrap "source ~/.bashrc && conda activate ctar && python /projects/zhanglab/users/ana/cts_peak_gene/CLI_ctar.py \
#                 --job compute_cis_only --multiome_file $MULTIOME_FILE --batch_size $BATCH_SIZE --dic_batch_size $DIC_BATCH_SIZE \
#                 --binning_config $BIN_CONFIG --binning_type $BIN_TYPE --genome_file $GENOME_FILE --target_path $TARGET_PATH \
#                 --results_path $RESULTS_PATH --pybedtools_path $PYBEDTOOLS_PATH --n_cores $N_CORES \
#                 --flag_ll $FLAG_LL --flag_se $FLAG_SE --flag_corr $FLAG_CORR --flag_tfidf $FLAG_TFIDF --random_seed $RANDOM_SEED \
#                 --n_length_bins $N_LENGTH_BINS \
#                 --array_idx \$SLURM_ARRAY_TASK_ID \
#                 ${DONOR_COL:+--donor_col $DONOR_COL}")
#     echo "Cis job ID: $CIS_JOB"
# else
#     echo "Submitting $BIN_CONFIG compute_cis_only job..."
#     CIS_JOB=$(sbatch --parsable -p $PARTITION -t 3-00:00:00 \
#       --ntasks-per-node=$N_CORES $EXCLUDE \
#       -o /projects/zhanglab/users/ana/logs/cis_%j.err -J "cis_$BIN_CONFIG" \
#       --wrap "source ~/.bashrc && conda activate ctar && python /projects/zhanglab/users/ana/cts_peak_gene/CLI_ctar.py \
#               --job compute_cis_only --multiome_file $MULTIOME_FILE --batch_size $BATCH_SIZE --dic_batch_size $DIC_BATCH_SIZE \
#               --binning_config $BIN_CONFIG --binning_type $BIN_TYPE --genome_file $GENOME_FILE --target_path $TARGET_PATH \
#               --results_path $RESULTS_PATH --pybedtools_path $PYBEDTOOLS_PATH --n_cores $N_CORES \
#               --flag_ll $FLAG_LL --flag_se $FLAG_SE --flag_corr $FLAG_CORR --flag_tfidf $FLAG_TFIDF --random_seed $RANDOM_SEED \
#               --n_length_bins $N_LENGTH_BINS \
#               ${DONOR_COL:+--donor_col $DONOR_COL}")
#     echo "Cis job ID: $CIS_JOB"
# fi

###########################################################################################
######                            Compute ctrl coefficients                          ######
###########################################################################################

echo "Submitting $BIN_CONFIG compute_ctrl_only array job with $TOTAL_NUM_BATCHES tasks (afterok:$GEN_CTRL_JOB)..."
CTRL_JOB=$(sbatch --parsable -p $PARTITION -t 3-00:00:00 \
    $ARRAY_OPT $EXCLUDE --mem=128Gb \
    --dependency=afterok:$GEN_CTRL_JOB \
    --ntasks-per-node=$N_CORES \
    -o /projects/zhanglab/users/ana/logs/ctrl_%A_%a.err \
    -J "ctrl_$BIN_CONFIG" \
    --wrap "source ~/.bashrc && conda activate ctar && python /projects/zhanglab/users/ana/cts_peak_gene/CLI_ctar.py \
            --job compute_ctrl_only --multiome_file $MULTIOME_FILE --array_idx \$SLURM_ARRAY_TASK_ID \
            --batch_size $BATCH_SIZE --dic_batch_size $DIC_BATCH_SIZE --target_path $TARGET_PATH --results_path $RESULTS_PATH \
            --binning_config $BIN_CONFIG --binning_type $BIN_TYPE --genome_file $GENOME_FILE --pybedtools_path $PYBEDTOOLS_PATH \
            --n_cores $N_CORES --flag_ll $FLAG_LL --flag_se $FLAG_SE --flag_corr $FLAG_CORR --flag_tfidf $FLAG_TFIDF --random_seed $RANDOM_SEED \
            --n_length_bins $N_LENGTH_BINS \
            ${DONOR_COL:+--donor_col $DONOR_COL}")
echo "Ctrl job ID: $CTRL_JOB"

###########################################################################################
######                               Compute P-values                                ######
###########################################################################################

echo "Submitting $BIN_CONFIG compute_pval job..."
PVAL_JOB=$(sbatch --parsable -p $PARTITION -t 04:00:00 \
  --dependency=afterok:${CTRL_JOB} $EXCLUDE \
  --ntasks-per-node=32 \
  --mail-type=END --mail-user=asprieto@andrew.cmu.edu \
  -o /projects/zhanglab/users/ana/logs/pval_%j.err -J "pval_$BIN_CONFIG" \
  --wrap "source ~/.bashrc && conda activate ctar && python /projects/zhanglab/users/ana/cts_peak_gene/CLI_ctar.py \
          --job compute_pval --binning_config $BIN_CONFIG --binning_type $BIN_TYPE --results_path $RESULTS_PATH --flag_corr $FLAG_CORR \
          --multiome_file $MULTIOME_FILE --genome_file $GENOME_FILE --pybedtools_path $PYBEDTOOLS_PATH \
          --fdr_bin_type $FDR_BIN_TYPE --tss_file $TSS_FILE --rna_layer $RNA_LAYER \
          --n_fdr_rna_bins $N_FDR_RNA_BINS --fdr_min_bin_count $FDR_MIN_BIN_COUNT")
echo "P-value job ID: $PVAL_JOB"


echo ""
echo "================================"
echo "Pipeline submitted successfully!"
echo "================================"
echo "Gen ctrl job: $GEN_CTRL_JOB"
echo "Cis job:      $CIS_JOB"
echo "Ctrl job:     $CTRL_JOB"
echo "P-value job:  $PVAL_JOB"
echo ""
echo "Used $MULTIOME_FILE."
echo "Results in: $RESULTS_PATH/cis_links_df.csv"
