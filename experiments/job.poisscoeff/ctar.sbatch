#!/bin/sh
#SBATCH -p mzhang,pool1
#SBATCH --time 1-00:00:00
#SBATCH --job-name poiss_neat
#SBATCH --mem=20Gb
#SBATCH --error Error_%A_%a.err
#SBATCH --array 1-50

# activate your conda env
source activate py39

srun python -u ctar-controls.py $SLURM_ARRAY_TASK_ID
