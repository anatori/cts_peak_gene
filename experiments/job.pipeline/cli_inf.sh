#!/bin/sh

# FILE: /projects/zhanglab/users/ana/cts_peak_gene/experiments/job.pipeline/cli_inf.sh

MULTIOME_DATASET=bmmc
MULTIOME_FILE=/projects/zhanglab/users/ana/multiome/processed/neurips_bmmc/bmmc.h5mu
BIN_CONFIG=4.4.inf.inf.1000
TARGET_PATH=/projects/zhanglab/users/ana/multiome/results/ctar/corr_eval/${MULTIOME_DATASET}/${MULTIOME_DATASET}_unfiltered_len_donor_${BIN_CONFIG}
GENOME_FILE=/projects/zhanglab/users/ana/bedtools2/ana_bedfiles/ref/GRCh38.p14.genome.fa.bgz
BIN_TYPE='mean_var'
PYBEDTOOLS_PATH=/projects/zhanglab/users/ana/bedtools2/bin
BATCH_SIZE=5000
RESULTS_PATH=/projects/zhanglab/users/ana/multiome/results/ctar/corr_eval/${MULTIOME_DATASET}/${MULTIOME_DATASET}_unfiltered_len_donor_${BIN_CONFIG}
PARTITION="mzhang,statgen,pool1"
MAX_CONCURRENT=50
FLAG_LL='False'
FLAG_SE='True'
FLAG_CORR='True'
N_CORES=32
RANDOM_SEED=777
DONOR_COL='DonorID'
N_LENGTH_BINS=4

IFS='.' read -r b1 b2 b3 b4 rest <<< "$BIN_CONFIG"
if [[ "$b3" == "inf" && "$b4" == "inf" ]]; then
    echo "Using inf.inf mode (no RNA binning)"
    USE_INF_MODE=true
else
    echo "Using standard binned mode"
    USE_INF_MODE=false
    NUM_LINKS=$(( b1 * b2 * b3 * b4 ))
    TOTAL_NUM_BATCHES=$(( (NUM_LINKS + BATCH_SIZE - 1) / BATCH_SIZE ))
    ARRAY_OPT="--array=0-$((TOTAL_NUM_BATCHES - 1))%${MAX_CONCURRENT}"
fi


###########################################################################################
######                              Generate Controls                              ######
###########################################################################################

echo "Submitting $BIN_CONFIG generate_controls job..."
GEN_CTRL_JOB=$(sbatch --parsable -p $PARTITION -t 1-00:00:00 \
  -c 16 --mem 128Gb \
  -o /projects/zhanglab/users/ana/logs/gen_ctrl_%j.err -J "gen_ctrl_$BIN_CONFIG" \
  --wrap "source ~/.bashrc && conda activate ctar && python /projects/zhanglab/users/ana/cts_peak_gene/CLI_ctar.py \
          --job generate_controls --multiome_file $MULTIOME_FILE --batch_size $BATCH_SIZE \
          --binning_config $BIN_CONFIG --binning_type $BIN_TYPE --genome_file $GENOME_FILE --target_path $TARGET_PATH \
          --pybedtools_path $PYBEDTOOLS_PATH --results_path $RESULTS_PATH --random_seed $RANDOM_SEED \
          --n_length_bins $N_LENGTH_BINS")
echo "Generate controls job ID: $GEN_CTRL_JOB"

###########################################################################################
######                            Compute cis coefficients                           ######
###########################################################################################

if [ "$USE_INF_MODE" = true ]; then
    CSV_FILE=$RESULTS_PATH/cis_links_df.csv

    if [ ! -f "$CSV_FILE" ]; then
        echo "ERROR: $CSV_FILE not found (generate_controls must run first)"
        exit 1
    fi

    NUM_CIS_LINKS=$(tail -n +2 "$CSV_FILE" | wc -l)
    TOTAL_NUM_BATCHES=$(( (NUM_CIS_LINKS + BATCH_SIZE - 1) / BATCH_SIZE ))
    echo "Found $NUM_CIS_LINKS cis-links, need $TOTAL_NUM_BATCHES array jobs"

    MAX_ARRAY_SIZE=1000
    if [ $TOTAL_NUM_BATCHES -gt $MAX_ARRAY_SIZE ]; then
        echo "ERROR: Need $TOTAL_NUM_BATCHES jobs but max array size is $MAX_ARRAY_SIZE"
        echo "Increase BATCH_SIZE"
        exit 1
    fi

    echo "Submitting $BIN_CONFIG compute_cis_only array job with $TOTAL_NUM_BATCHES tasks..."
    CIS_JOB=$(sbatch --parsable -p $PARTITION -t 1-00:00:00 \
        --dependency=afterok:${GEN_CTRL_JOB} \
        -c $N_CORES --mem 128Gb \
        --array=0-$((TOTAL_NUM_BATCHES - 1))%${MAX_CONCURRENT} \
        -o /projects/zhanglab/users/ana/logs/cis_%A_%a.err \
        -J "cis_$BIN_CONFIG" \
        --wrap "source ~/.bashrc && conda activate ctar && python /projects/zhanglab/users/ana/cts_peak_gene/CLI_ctar.py \
                --job compute_cis_only --multiome_file $MULTIOME_FILE --batch_size $BATCH_SIZE \
                --binning_config $BIN_CONFIG --binning_type $BIN_TYPE --genome_file $GENOME_FILE --target_path $TARGET_PATH \
                --results_path $RESULTS_PATH --pybedtools_path $PYBEDTOOLS_PATH --n_cores $N_CORES \
                --flag_ll $FLAG_LL --flag_se $FLAG_SE --flag_corr $FLAG_CORR --random_seed $RANDOM_SEED \
                --n_length_bins $N_LENGTH_BINS \
                --array_idx \$SLURM_ARRAY_TASK_ID \
                ${DONOR_COL:+--donor_col $DONOR_COL}")
    echo "Cis job ID: $CIS_JOB"

else
    echo "Submitting $BIN_CONFIG compute_cis_only job..."
    CIS_JOB=$(sbatch --parsable -p $PARTITION -t 1-00:00:00 \
      -c $N_CORES --mem 128Gb \
      -o /projects/zhanglab/users/ana/logs/cis_%j.err -J "cis_$BIN_CONFIG" \
      --wrap "source ~/.bashrc && conda activate ctar && python /projects/zhanglab/users/ana/cts_peak_gene/CLI_ctar.py \
              --job compute_cis_only --multiome_file $MULTIOME_FILE --batch_size $BATCH_SIZE \
              --binning_config $BIN_CONFIG --binning_type $BIN_TYPE --genome_file $GENOME_FILE --target_path $TARGET_PATH \
              --results_path $RESULTS_PATH --pybedtools_path $PYBEDTOOLS_PATH --n_cores $N_CORES \
              --flag_ll $FLAG_LL --flag_se $FLAG_SE --flag_corr $FLAG_CORR --random_seed $RANDOM_SEED \
              --n_length_bins $N_LENGTH_BINS \
              ${DONOR_COL:+--donor_col $DONOR_COL}")
    echo "Cis job ID: $CIS_JOB"
fi

###########################################################################################
######                              Compute ctrl coefficients                          ######
###########################################################################################

if [ "$USE_INF_MODE" = true ]; then
    echo "Submitting $BIN_CONFIG compute_ctrl_only array job with $TOTAL_NUM_BATCHES tasks..."
    CTRL_JOB=$(sbatch --parsable -p $PARTITION -t 2-00:00:00 \
        --dependency=afterok:${GEN_CTRL_JOB} \
        -c $N_CORES --mem 128Gb \
        --array=0-$((TOTAL_NUM_BATCHES - 1))%${MAX_CONCURRENT} \
        -o /projects/zhanglab/users/ana/logs/ctrl_%A_%a.err \
        -J "ctrl_$BIN_CONFIG" \
        --wrap "source ~/.bashrc && conda activate ctar && python /projects/zhanglab/users/ana/cts_peak_gene/CLI_ctar.py \
                --job compute_ctrl_only --multiome_file $MULTIOME_FILE \
                --batch_size $BATCH_SIZE --target_path $TARGET_PATH --results_path $RESULTS_PATH --array_idx \$SLURM_ARRAY_TASK_ID \
                --binning_config $BIN_CONFIG --binning_type $BIN_TYPE --genome_file $GENOME_FILE --pybedtools_path $PYBEDTOOLS_PATH \
                --n_cores $N_CORES --flag_ll $FLAG_LL --flag_se $FLAG_SE --flag_corr $FLAG_CORR --random_seed $RANDOM_SEED \
                --n_length_bins $N_LENGTH_BINS \
                ${DONOR_COL:+--donor_col $DONOR_COL}")
    echo "Ctrl job ID: $CTRL_JOB"

else
    echo "Submitting $BIN_CONFIG compute_ctrl_only job with $TOTAL_NUM_BATCHES batches..."
    CTRL_JOB=$(sbatch --parsable -p $PARTITION -t 1-00:00:00 $ARRAY_OPT \
      -c $N_CORES --mem 128Gb \
      -o /projects/zhanglab/users/ana/logs/ctrl_%A_%a.err -J "ctrl_$BIN_CONFIG" \
      --wrap "source ~/.bashrc && conda activate ctar && python /projects/zhanglab/users/ana/cts_peak_gene/CLI_ctar.py \
              --job compute_ctrl_only --multiome_file $MULTIOME_FILE \
              --batch_size $BATCH_SIZE --target_path $TARGET_PATH --results_path $RESULTS_PATH \
              --binning_config $BIN_CONFIG --binning_type $BIN_TYPE --genome_file $GENOME_FILE --pybedtools_path $PYBEDTOOLS_PATH \
              --n_cores $N_CORES --flag_ll $FLAG_LL --flag_se $FLAG_SE --flag_corr $FLAG_CORR --random_seed $RANDOM_SEED \
              --n_length_bins $N_LENGTH_BINS \
              ${DONOR_COL:+--donor_col $DONOR_COL}")
    echo "Ctrl job ID: $CTRL_JOB"
fi

###########################################################################################
######                               Compute P-values                                ######
###########################################################################################

echo "Submitting $BIN_CONFIG compute_pval job..."
PVAL_JOB=$(sbatch --parsable -p $PARTITION -t 0-00:30:00 \
  -c 16 --mem 128Gb \
  --dependency=afterok:${CTRL_JOB}:${CIS_JOB} \
  --mail-type=END --mail-user=asprieto@andrew.cmu.edu \
  -o /projects/zhanglab/users/ana/logs/pval_%j.err -J "pval_$BIN_CONFIG" \
  --wrap "source ~/.bashrc && conda activate ctar && python /projects/zhanglab/users/ana/cts_peak_gene/CLI_ctar.py \
          --job compute_pval --binning_config $BIN_CONFIG --binning_type $BIN_TYPE --n_length_bins $N_LENGTH_BINS --results_path $RESULTS_PATH --flag_corr $FLAG_CORR")
echo "P-value job ID: $PVAL_JOB"


echo ""
echo "================================"
echo "Pipeline submitted successfully!"
echo "================================"
echo "Gen ctrl job:  $GEN_CTRL_JOB"
echo "Cis job:      $CIS_JOB"
echo "Ctrl job:     $CTRL_JOB"
echo "P-value job:  $PVAL_JOB"
echo ""
echo "Used $MULTIOME_FILE."
echo "Results in: $RESULTS_PATH/cis_links_df.csv"