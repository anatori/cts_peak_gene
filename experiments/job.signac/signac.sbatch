#!/bin/bash

#SBATCH -p pool1,mzhang
#SBATCH --time 3-00:00:00
#SBATCH --job-name signac_neat
#SBATCH --mem-per-cpu 50Gb
#SBATCH --chdir /home/asprieto
#SBATCH -o Output_%A_%a.out  #output file
#SBATCH -e Error_%A_%a.err   #error file

source activate r-environment

Rscript neatseq_signac.R