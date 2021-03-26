#!/bin/bash
#SBATCH -p batch
#SBATCH -t 6:00:00
#SBATCH --job-name="k2d_da_v0"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
##SBATCH --mail-user=supawar@okstate.edu
##SBATCH --mail-type=end

module load anaconda3/2020.07
source activate python35
python kraichnan_turbulence_enkf.py 
