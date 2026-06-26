#!/bin/bash

# FILE: /ocean/projects/bio240051p/aprieto/cts_peak_gene/experiments/job.pipeline/cli_cts_all.sh

declare -A CELLTYPES
CELLTYPES[brain]="astrocyte microglia neurons oligodendrocyte"
CELLTYPES[pbmc]="bcell dendritic myeloid tcell"
CELLTYPES[bmmc]="bcell dendritic myeloid tcell erythroid progenitor"

declare -A MULTIOME_PATHS
MULTIOME_PATHS[pbmc]=/ocean/projects/bio240051p/aprieto/multiome/processed/pbmc_sorted_10k/unfilt_broad_celltypes
MULTIOME_PATHS[brain]=/ocean/projects/bio240051p/aprieto/multiome/processed/brain_3k/unfilt_broad_celltypes
MULTIOME_PATHS[bmmc]=/ocean/projects/bio240051p/aprieto/multiome/processed/neurips_bmmc/unfilt_broad_celltypes

BIN_CONFIG=5.5.5.5.10000
BIN_TYPE='mean_var'
PYBEDTOOLS_PATH=/opt/packages/bedtools/2.30.0
BATCH_SIZE=100000
PARTITION="RM-shared"
MAX_CONCURRENT=200
FLAG_LL='False'
FLAG_SE='True'
FLAG_CORR='True'
N_CORES=64
RANDOM_SEED=777
DONOR_COL=''
N_LENGTH_BINS=1
GENOME_FILE=/ocean/projects/bio240051p/aprieto/bedfiles/ref/GRCh38.p14.genome.fixed.fa

IFS='.' read -r b1 b2 b3 b4 rest <<< "$BIN_CONFIG"
NUM_LINKS=$(( b1 * b2 * b3 * b4 * N_LENGTH_BINS ))
TOTAL_NUM_BATCHES=$(( (NUM_LINKS + BATCH_SIZE - 1) / BATCH_SIZE ))
ARRAY_OPT="--array=0-$((TOTAL_NUM_BATCHES - 1))%${MAX_CONCURRENT}"

for MULTIOME_DATASET in "${!CELLTYPES[@]}"; do
    for CELLTYPE in ${CELLTYPES[$MULTIOME_DATASET]}; do

        MULTIOME_FILE=${MULTIOME_PATHS[$MULTIOME_DATASET]}/${CELLTYPE}.h5mu
        TARGET_PATH=/ocean/projects/bio240051p/aprieto/multiome/results/ctar/corr_eval/${MULTIOME_DATASET}/${MULTIOME_DATASET}_broad_celltype_unfilt_${BIN_CONFIG}/${CELLTYPE}
        RESULTS_PATH=${TARGET_PATH}

        echo "--- Submitting ${MULTIOME_DATASET}/${CELLTYPE} ---"

        CIS_JOB=$(sbatch --parsable -p $PARTITION -t 1-00:00:00 \
          -A bio240051p \
          --ntasks-per-node=$N_CORES \
          -o /ocean/projects/bio240051p/aprieto/logs/cis_${MULTIOME_DATASET}_${CELLTYPE}_%j.err \
          -J "cis_${MULTIOME_DATASET}_${CELLTYPE}" \
          --wrap "source ~/.bashrc && conda activate ctar && python /ocean/projects/bio240051p/aprieto/cts_peak_gene/CLI_ctar.py \
                  --job compute_cis_only --multiome_file $MULTIOME_FILE --batch_size $BATCH_SIZE \
                  --binning_config $BIN_CONFIG --binning_type $BIN_TYPE --genome_file $GENOME_FILE \
                  --target_path $TARGET_PATH --results_path $RESULTS_PATH \
                  --pybedtools_path $PYBEDTOOLS_PATH --n_cores $N_CORES \
                  --flag_ll $FLAG_LL --flag_se $FLAG_SE --flag_corr $FLAG_CORR --random_seed $RANDOM_SEED \
                  --n_length_bins $N_LENGTH_BINS \
                  ${DONOR_COL:+--donor_col $DONOR_COL}")
        echo "  Cis job:  $CIS_JOB"

        CTRL_JOB=$(sbatch --parsable -p $PARTITION -t 1-00:00:00 $ARRAY_OPT \
          -A bio240051p \
          --ntasks-per-node=$N_CORES \
          -o /ocean/projects/bio240051p/aprieto/logs/ctrl_${MULTIOME_DATASET}_${CELLTYPE}_%A_%a.err \
          -J "ctrl_${MULTIOME_DATASET}_${CELLTYPE}" \
          --wrap "source ~/.bashrc && conda activate ctar && python /ocean/projects/bio240051p/aprieto/cts_peak_gene/CLI_ctar.py \
                  --job compute_ctrl_only --multiome_file $MULTIOME_FILE \
                  --batch_size $BATCH_SIZE --target_path $TARGET_PATH --results_path $RESULTS_PATH \
                  --binning_config $BIN_CONFIG --binning_type $BIN_TYPE --genome_file $GENOME_FILE \
                  --pybedtools_path $PYBEDTOOLS_PATH --n_cores $N_CORES \
                  --flag_ll $FLAG_LL --flag_se $FLAG_SE --flag_corr $FLAG_CORR --random_seed $RANDOM_SEED \
                  --n_length_bins $N_LENGTH_BINS \
                  ${DONOR_COL:+--donor_col $DONOR_COL}")
        echo "  Ctrl job: $CTRL_JOB"

        PVAL_JOB=$(sbatch --parsable -p $PARTITION -t 00:30:00 \
          -A bio240051p \
          --dependency=afterok:${CIS_JOB}:${CTRL_JOB} \
          --ntasks-per-node=32 \
          --mail-type=END --mail-user=asprieto@andrew.cmu.edu \
          -o /ocean/projects/bio240051p/aprieto/logs/pval_${MULTIOME_DATASET}_${CELLTYPE}_%j.err \
          -J "pval_${MULTIOME_DATASET}_${CELLTYPE}" \
          --wrap "source ~/.bashrc && conda activate ctar && python /ocean/projects/bio240051p/aprieto/cts_peak_gene/CLI_ctar.py \
                  --job compute_pval --binning_config $BIN_CONFIG --binning_type $BIN_TYPE \
                  --n_length_bins $N_LENGTH_BINS --results_path $RESULTS_PATH --flag_corr $FLAG_CORR")
        echo "  Pval job: $PVAL_JOB"
        echo ""

    done
done
