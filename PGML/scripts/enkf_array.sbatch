#!/bin/bash
#SBATCH -p batch
#SBATCH -t 12:00:00
#SBATCH --job-name="k2d_da_cnn"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
##SBATCH --mail-user=supawar@okstate.edu
##SBATCH --mail-type=end

#SBATCH --array=1-30

# pad the task ID with leading zeros (to get 001, 002, etc.)
CASE_NUM=`printf %03d $SLURM_ARRAY_TASK_ID`

cd v_$CASE_NUM

module load anaconda3/2020.07
source activate python35
python kraichnan_turbulence_enkf.py 
