#!/bin/bash
#SBATCH -p batch
#SBATCH -t 48:00:00
#SBATCH --job-name="kt_1024_128_8000"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
##SBATCH --mail-user=supawar@okstate.edu
##SBATCH --mail-type=end

module load cuda/11.0
#module load anaconda3/2020.07
#source activate python36
cd /scratch/suraj/KT/PGML
python kraichnan_turbulence_clean_TF2.py config/input_1024_8000.yaml 1 > logs/input_1024_128_8000.log
