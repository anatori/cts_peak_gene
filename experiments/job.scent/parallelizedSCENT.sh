#!/bin/bash

#SBATCH -p pool1,mzhang
#SBATCH --time 6-00:00:00
#SBATCH --job-name SCENT_neat
#SBATCH --mem-per-cpu 10Gb
#SBATCH --chdir /home/asprieto
#SBATCH -o Output_%A_%a.out  #output file
#SBATCH -e Error_%A_%a.err   #error file
#SBATCH --array=1-10

# remember to start at 1 since R is 1-indexed.
# max array size = 1001

source activate r-environment

Rscript SCENT_parallelization.R $SLURM_ARRAY_TASK_ID 1 'SCENT_neatseq_paper.rds' 'CD4' 'poisson' TRUE scent-neat-seq-mt/


# source activate r-environment
# Rscript SCENT_parallelization.R 1 2 'SCENT_obj.rds' 'CD4+ naïve T' 'poisson' TRUE outputs/