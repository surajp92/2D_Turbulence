#!/bin/bash
#SBATCH -p batch
#SBATCH -t 0:30:00
#SBATCH --job-name="kt_trial"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
##SBATCH --mail-user=supawar@okstate.edu
##SBATCH --mail-type=end

module load cuda
module load anaconda3/2020.07
source activate python36
cd /scratch/suraj/KT/PGML
python kraichnan_turbulence_clean.py config/input.yaml > logs/trial.log
