#!/bin/bash
#SBATCH -p batch
#SBATCH -t 8:00:00
#SBATCH --job-name="kt_cnn_1024_128_8000"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mail-user=supawar@okstate.edu
##SBATCH --mail-type=end

module load cuda
module load anaconda3/2020.07

python DHIT_CNN_apriori_sgs_TF2.py 
