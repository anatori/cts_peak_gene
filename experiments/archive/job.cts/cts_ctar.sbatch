#!/bin/bash

#SBATCH -N 1
#SBATCH -p mzhang,statgen
#SBATCH -t 6-00:00:00
#SBATCH --job-name ctar_cts
#SBATCH -o /projects/zhanglab/users/ana/logs/cts_%A_%a.out  #output file
#SBATCH -e /projects/zhanglab/users/ana/logs/cts_%A_%a.err   #error file
#SBATCH --mail-type=END
#SBATCH --mail-user=asprieto@andrew.cmu.edu
#SBATCH --mem 64Gb
#SBATCH -c 16

source activate ctar
srun python /projects/zhanglab/users/ana/cts_peak_gene/experiments/job.cts/cts_ctar.py