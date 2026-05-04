#!/bin/sh

# FILE: /ocean/projects/bio240051p/aprieto/cts_peak_gene/experiments/job.null/null_cli.sh

MULTIOME_FILE=/ocean/projects/bio240051p/aprieto/multiome/processed/pbmc_sorted_10k/pbmc.h5mu
BIN_PREFIX=3.3.3.3
BIN_CONFIG=${BIN_PREFIX}.10000
TARGET_PATH=/ocean/projects/bio240051p/aprieto/multiome/simulations/null_pbmc/celltype_specificity/corr/${BIN_PREFIX}
GENOME_FILE=/ocean/projects/bio240051p/aprieto/bedfiles/ref/GRCh38.p14.genome.fixed.fa
BIN_TYPE='mean_var'
PYBEDTOOLS_PATH=/opt/packages/bedtools/2.30.0
BATCH_SIZE=100000
RESULTS_PATH=/ocean/projects/bio240051p/aprieto/multiome/simulations/null_pbmc/celltype_specificity/corr/${BIN_PREFIX}
PARTITION="RM-shared"
MAX_CONCURRENT=200
FLAG_LL='False'
FLAG_SE='True'
FLAG_CORR='True'
N_CORES=64
RANDOM_SEED=777
DONOR_COL=''

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


# ###########################################################################################
# ######                                  Generate Links                               ######
# ###########################################################################################

# echo "Submitting $BIN_CONFIG generate_links job..."
# LINKS_JOB=$(sbatch --parsable -p $PARTITION -t 1-00:00:00 \
#   -A bio240051p \
#   --ntasks-per-node=32 \
#   -o /ocean/projects/bio240051p/aprieto/logs/links_%j.err -J "links_$BIN_CONFIG" \
#   --wrap "source ~/.bashrc && conda activate ctar && python /ocean/projects/bio240051p/aprieto/cts_peak_gene/CLI_ctar.py \
#           --job generate_links --job_id \$SLURM_JOB_ID --multiome_file $MULTIOME_FILE --batch_size $BATCH_SIZE \
#           --binning_config $BIN_CONFIG --binning_type $BIN_TYPE --genome_file $GENOME_FILE --target_path $TARGET_PATH \
#           --pybedtools_path $PYBEDTOOLS_PATH --results_path $RESULTS_PATH --random_seed $RANDOM_SEED")
# echo "Links job ID: $LINKS_JOB"

###########################################################################################
######                            Compute cis coefficients                           ######
###########################################################################################

echo "Submitting $BIN_CONFIG compute_cis_only job..."
CIS_JOB=$(sbatch --parsable -p $PARTITION -t 1-00:00:00 \
  -A bio240051p \
  --ntasks-per-node=$N_CORES \
  -o /ocean/projects/bio240051p/aprieto/logs/cis_%j.err -J "cis_$BIN_CONFIG" \
  --wrap "source ~/.bashrc && conda activate ctar && python /ocean/projects/bio240051p/aprieto/cts_peak_gene/CLI_ctar.py \
          --job compute_cis_only --multiome_file $MULTIOME_FILE --batch_size $BATCH_SIZE \
          --binning_config $BIN_CONFIG --binning_type $BIN_TYPE --genome_file $GENOME_FILE --target_path $TARGET_PATH \
          --results_path $RESULTS_PATH --pybedtools_path $PYBEDTOOLS_PATH --n_cores $N_CORES \
          --flag_ll $FLAG_LL --flag_se $FLAG_SE --flag_corr $FLAG_CORR --random_seed $RANDOM_SEED \
          ${DONOR_COL:+--donor_col $DONOR_COL}")
echo "Cis job ID: $CIS_JOB"

###########################################################################################
######                    Wait for Links Job and Count Cis-Links                     ######
###########################################################################################

if [ "$USE_INF_MODE" = true ]; then
    echo ""
    echo "Waiting for links job to complete to count cis-links..."
    echo "(Script will wait here until job $LINKS_JOB finishes)"
    
    # Wait for the links job to complete
    TIMEOUT=7200  # 2 hours timeout
    ELAPSED=0
    while [ $ELAPSED -lt $TIMEOUT ]; do
        JOB_STATE=$(squeue -j $LINKS_JOB -h -o %T 2>/dev/null)
        if [ -z "$JOB_STATE" ]; then
            # Job no longer in queue, check if it succeeded
            sacct -j $LINKS_JOB -o State -n | head -1 | grep -q COMPLETED
            if [ $? -eq 0 ]; then
                echo "Links job completed successfully!"
                break
            else
                echo "ERROR: Links job failed or was cancelled"
                exit 1
            fi
        fi
        echo "  Links job status: $JOB_STATE (checking again in 30s)"
        sleep 30
        ELAPSED=$((ELAPSED + 30))
    done
    
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "ERROR:  Timeout waiting for links job"
        exit 1
    fi
    
    # Now count the cis-links
    CSV_FILE=$RESULTS_PATH/cis_links_df.csv
    
    if [ ! -f "$CSV_FILE" ]; then
        echo "ERROR: $CSV_FILE not found"
        exit 1
    fi
    
    NUM_CIS_LINKS=$(tail -n +2 "$CSV_FILE" | wc -l)
    TOTAL_NUM_BATCHES=$(( (NUM_CIS_LINKS + BATCH_SIZE - 1) / BATCH_SIZE ))
    
    echo "Found $NUM_CIS_LINKS cis-links"
    echo "Need $TOTAL_NUM_BATCHES array jobs"
    
    # Check if we exceed array limit
    MAX_ARRAY_SIZE=1000
    
    if [ $TOTAL_NUM_BATCHES -le $MAX_ARRAY_SIZE ]; then
        echo "Submitting single array job with $TOTAL_NUM_BATCHES tasks..."
        
        CTRL_JOB=$(sbatch --parsable -p $PARTITION -t 2-00:00:00 \
            -A bio240051p \
            --ntasks-per-node=$N_CORES \
            --array=0-$((TOTAL_NUM_BATCHES - 1))%${MAX_CONCURRENT} \
            -o /ocean/projects/bio240051p/aprieto/logs/ctrl_%A_%a.err \
            -J "ctrl_$BIN_CONFIG" \
            --wrap "source ~/.bashrc && conda activate ctar && python /ocean/projects/bio240051p/aprieto/cts_peak_gene/CLI_ctar.py \
                    --job compute_ctrl_only --multiome_file $MULTIOME_FILE \
                    --batch_size $BATCH_SIZE --target_path $TARGET_PATH --results_path $RESULTS_PATH \
                    --binning_config $BIN_CONFIG --binning_type $BIN_TYPE --genome_file $GENOME_FILE --pybedtools_path $PYBEDTOOLS_PATH \
                    --n_cores $N_CORES --flag_ll $FLAG_LL --flag_se $FLAG_SE --flag_corr $FLAG_CORR --random_seed $RANDOM_SEED \
                    ${DONOR_COL:+--donor_col $DONOR_COL}")
        echo "Ctrl job ID: $CTRL_JOB"
        CTRL_JOB_FOR_DEPENDENCY=$CTRL_JOB

    else
        echo "ERROR: Need $TOTAL_NUM_BATCHES jobs but max array size is $MAX_ARRAY_SIZE"
        echo "Increase BATCH_SIZE or use parallel arrays approach"
        exit 1
    fi
    
else
    echo "Submitting $BIN_CONFIG compute_ctrl_only job with $TOTAL_NUM_BATCHES batches..."
    CTRL_JOB=$(sbatch --parsable -p $PARTITION -t 1-00:00:00 $ARRAY_OPT \
      -A bio240051p \
      --ntasks-per-node=$N_CORES \
      -o /ocean/projects/bio240051p/aprieto/logs/ctrl_%A_%a.err -J "ctrl_$BIN_CONFIG" \
      --wrap "source ~/.bashrc && conda activate ctar && python /ocean/projects/bio240051p/aprieto/cts_peak_gene/CLI_ctar.py \
              --job compute_ctrl_only --multiome_file $MULTIOME_FILE \
              --batch_size $BATCH_SIZE --target_path $TARGET_PATH --results_path $RESULTS_PATH \
              --binning_config $BIN_CONFIG --binning_type $BIN_TYPE --genome_file $GENOME_FILE --pybedtools_path $PYBEDTOOLS_PATH \
              --n_cores $N_CORES --flag_ll $FLAG_LL --flag_se $FLAG_SE --flag_corr $FLAG_CORR --random_seed $RANDOM_SEED \
              ${DONOR_COL:+--donor_col $DONOR_COL}")
    echo "Ctrl job ID: $CTRL_JOB"
    CTRL_JOB_FOR_DEPENDENCY=$CTRL_JOB
fi

###########################################################################################
######                               Compute P-values                                ######
###########################################################################################

echo "Submitting $BIN_CONFIG compute_pval job..."
PVAL_JOB=$(sbatch --parsable -p $PARTITION -t 00:30:00 \
  -A bio240051p \
  --dependency=afterok:${CIS_JOB}:${CTRL_JOB} \
  --ntasks-per-node=12 \
  --mail-type=END --mail-user=asprieto@andrew.cmu.edu \
  -o /ocean/projects/bio240051p/aprieto/logs/pval_%j.err -J "pval_$BIN_CONFIG" \
  --wrap "source ~/.bashrc && conda activate ctar && python /ocean/projects/bio240051p/aprieto/cts_peak_gene/CLI_ctar.py \
          --job compute_pval --binning_config $BIN_CONFIG --binning_type $BIN_TYPE --results_path $RESULTS_PATH --flag_corr $FLAG_CORR")
echo "P-value job ID: $PVAL_JOB"


echo ""
echo "================================"
echo "Pipeline submitted successfully!"
echo "================================"
echo "Links job:     $LINKS_JOB"
echo "Cis job:      $CIS_JOB"
echo "Ctrl job:     $CTRL_JOB"
echo "P-value job:  $PVAL_JOB"
echo ""
echo "Used $MULTIOME_FILE."
echo "Results in: $RESULTS_PATH/cis_links_df.csv"