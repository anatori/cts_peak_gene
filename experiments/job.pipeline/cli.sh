#!/bin/sh

# FILE: /projects/zhanglab/users/ana/cts_peak_gene/experiments/job.pipeline/cli.sh

MULTIOME_FILE=/projects/zhanglab/users/ana/multiome/processed/brain_3k/brain_filt.h5mu
TARGET_PATH=/projects/zhanglab/users/ana/multiome/results/ctar/final_eval/brain/brain_filtered_5.5.5.5.10000
GENOME_FILE=/projects/zhanglab/users/ana/bedtools2/ana_bedfiles/ref/GRCh38.p14.genome.fa.bgz
BIN_CONFIG=5.5.5.5.10000
BIN_TYPE='mean_var'
PYBEDTOOLS_PATH=/projects/zhanglab/users/ana/bedtools2/bin
BATCH_SIZE=1800
RESULTS_PATH=/projects/zhanglab/users/ana/multiome/results/ctar/final_eval/brain/brain_filtered_5.5.5.5.10000
PARTITION="mzhang,pool1,statgen"
MAX_CONCURRENT=10
FLAG_LL='False'
FLAG_SE='True'

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
######                                  Generate Links                               ######
###########################################################################################

echo "Submitting $BIN_CONFIG generate_links job..."
LINKS_JOB=$(sbatch --parsable -p $PARTITION -t 1-00:00:00 --mem 24Gb \
  -o /projects/zhanglab/users/ana/logs/links_%j.err -J "links_$BIN_CONFIG" \
  --wrap "source ~/.bashrc && conda activate ctar && python /projects/zhanglab/users/ana/cts_peak_gene/CLI_ctar.py \
          --job generate_links --job_id \$SLURM_JOB_ID --multiome_file $MULTIOME_FILE --batch_size $BATCH_SIZE \
          --binning_config $BIN_CONFIG --binning_type $BIN_TYPE --genome_file $GENOME_FILE --target_path $TARGET_PATH \
          --pybedtools_path $PYBEDTOOLS_PATH")
echo "Links job ID: $LINKS_JOB"

###########################################################################################
######                            Compute cis coefficients                           ######
###########################################################################################

echo "Submitting $BIN_CONFIG compute_cis_only job..."
CIS_JOB=$(sbatch --parsable -p $PARTITION -t 6-00:00:00 \
  --dependency=afterok:${LINKS_JOB} \
  -c 16 --mem 128Gb \
  -o /projects/zhanglab/users/ana/logs/cis_%j.err -J "cis_$BIN_CONFIG" \
  --wrap "source ~/.bashrc && conda activate ctar && python /projects/zhanglab/users/ana/cts_peak_gene/CLI_ctar.py \
          --job compute_cis_only --job_id $LINKS_JOB --multiome_file $MULTIOME_FILE --batch_size $BATCH_SIZE \
          --binning_config $BIN_CONFIG --binning_type $BIN_TYPE --genome_file $GENOME_FILE --target_path $TARGET_PATH \
          --results_path $TARGET_PATH/${LINKS_JOB}_results --pybedtools_path $PYBEDTOOLS_PATH \
          --flag_ll $FLAG_LL --flag_se $FLAG_SE")
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
    RESULTS_DIR=$TARGET_PATH/${LINKS_JOB}_results
    CSV_FILE=$RESULTS_DIR/cis_links_df.csv
    
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
        
        CTRL_JOB=$(sbatch --parsable -p $PARTITION -t 6-00:00:00 \
          --array=0-$((TOTAL_NUM_BATCHES - 1))%${MAX_CONCURRENT} \
          -c 16 --mem 64Gb \
          -o /projects/zhanglab/users/ana/logs/ctrl_%A_%a.err \
          -J "ctrl_$BIN_CONFIG" \
          --wrap "source ~/.bashrc && conda activate ctar && python /projects/zhanglab/users/ana/cts_peak_gene/CLI_ctar.py \
                  --job compute_ctrl_only --job_id $LINKS_JOB --array_idx \$SLURM_ARRAY_TASK_ID --multiome_file $MULTIOME_FILE \
                  --batch_size $BATCH_SIZE --target_path $TARGET_PATH --results_path $TARGET_PATH/${LINKS_JOB}_results \
                  --binning_config $BIN_CONFIG --binning_type $BIN_TYPE --genome_file $GENOME_FILE --pybedtools_path $PYBEDTOOLS_PATH \
                  --flag_ll $FLAG_LL --flag_se $FLAG_SE")
        echo "Ctrl job ID: $CTRL_JOB"
        CTRL_JOB_FOR_DEPENDENCY=$CTRL_JOB
        
    else
        echo "ERROR: Need $TOTAL_NUM_BATCHES jobs but max array size is $MAX_ARRAY_SIZE"
        echo "Increase BATCH_SIZE or use parallel arrays approach"
        exit 1
    fi
    
else
    echo "Submitting $BIN_CONFIG compute_ctrl_only job with $TOTAL_NUM_BATCHES batches..."
    CTRL_JOB=$(sbatch --parsable -p $PARTITION -t 6-00:00:00 $ARRAY_OPT \
      --dependency=afterok:${LINKS_JOB} \
      -c 16 --mem 64Gb \
      -o /projects/zhanglab/users/ana/logs/ctrl_%A_%a.err -J "ctrl_$BIN_CONFIG" \
      --wrap "source ~/.bashrc && conda activate ctar && python /projects/zhanglab/users/ana/cts_peak_gene/CLI_ctar.py \
              --job compute_ctrl_only --job_id $LINKS_JOB --array_idx \$SLURM_ARRAY_TASK_ID --multiome_file $MULTIOME_FILE \
              --batch_size $BATCH_SIZE --target_path $TARGET_PATH --results_path $TARGET_PATH/${LINKS_JOB}_results \
              --binning_config $BIN_CONFIG --binning_type $BIN_TYPE --genome_file $GENOME_FILE --pybedtools_path $PYBEDTOOLS_PATH \
              --flag_ll $FLAG_LL --flag_se $FLAG_SE")
    echo "Ctrl job ID: $CTRL_JOB"
    CTRL_JOB_FOR_DEPENDENCY=$CTRL_JOB
fi

###########################################################################################
######                               Compute P-values                                ######
###########################################################################################


echo "Submitting $BIN_CONFIG compute_pval job..."
PVAL_JOB=$(sbatch --parsable -p $PARTITION -t 00:30:00 --mem 16Gb \
  --dependency=afterok:${CTRL_JOB}:${CIS_JOB} \
  --mail-type=END --mail-user=asprieto@andrew.cmu.edu \
  -o /projects/zhanglab/users/ana/logs/pval_%j.err -J "pval_$BIN_CONFIG" \
  --wrap "source ~/.bashrc && conda activate ctar && python /projects/zhanglab/users/ana/cts_peak_gene/CLI_ctar.py \
          --job compute_pval --binning_config $BIN_CONFIG --binning_type $BIN_TYPE --results_path $TARGET_PATH/${LINKS_JOB}_results")
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
echo "Results in: $TARGET_PATH/${LINKS_JOB}_results/cis_links_df.csv"