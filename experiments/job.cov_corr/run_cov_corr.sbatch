#!/bin/bash
#SBATCH --job-name=cov_corr
#SBATCH --output=/ocean/projects/bio240051p/aprieto/logs/cov_corr_%j.out
#SBATCH --error=/ocean/projects/bio240051p/aprieto/logs/cov_corr_%j.err
#SBATCH --time=1-00:00:00
#SBATCH -A bio250028p
#SBATCH -p RM-shared
#SBATCH --ntasks-per-node=128


MULTIOME_FILE="/ocean/projects/bio240051p/aprieto/multiome/processed/neurips_bmmc/bmmc.h5mu"
OUTPUT_DIR="/ocean/projects/bio240051p/aprieto/multiome/results/ctar/corr_eval/bmmc/bmmc_unfiltered_len_donor_covcorrect_4.4.4.4.1000/"
COV_COLS="DonorID,GEX_pct_counts_mt"
OUTPUT_PATH="/ocean/projects/bio240051p/aprieto/multiome/results/ctar/corr_eval/bmmc/bmmc_unfiltered_len_donor_covcorrect_4.4.4.4.1000/"
CONDA_ENV="ctar"
N_WORKERS=128

source activate ${CONDA_ENV}
python run_cov_corr.py \
    --multiome_file   "${MULTIOME_FILE}" \
    --ctar_output_dir "${OUTPUT_DIR}" \
    --cov_cols        "${COV_COLS}" \
    --output_path     "${OUTPUT_PATH}" \
    --n_workers       ${N_WORKERS} \
    --batch_size      50