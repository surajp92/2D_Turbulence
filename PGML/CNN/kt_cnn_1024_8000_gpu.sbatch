#!/bin/bash
#SBATCH -p bullet
#SBATCH -t 4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --gres=gpu:1
#SBATCH --mail-user=supawar@okstate.edu

# Load any modules needed.  E.g.: module load amber
module load cuda/11.0

# Insert commands for the software.  E.g.:  pmemd.cuda -O -i md.in -o md.out -p topology -c input.crd -ref ref.crd -r output.crd
python DHIT_CNN_apriori_sgs_TF2.py 
