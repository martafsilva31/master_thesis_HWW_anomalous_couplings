#!/bin/bash

#SBATCH -p lipq
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=50G
export NUMEXPR_NUM_THREADS=6
export OMP_NUM_THREADS=6
# Transfering the python script to the execution node
# INPUT = 5a_alices_training.py


module load python/3.9.12

python3 /lstore/titan/martafsilva/master_thesis/master_thesis_HWW_anomalous_couplings/2D_high_pT/5a_alices_training.py --augment --config_file /lstore/titan/martafsilva/master_thesis/master_thesis_HWW_anomalous_couplings/2D_high_pT/config_2D_high_pT.yaml > /lstore/titan/martafsilva/master_thesis/master_thesis_output/2D_high_pT/output_logs/augmentation/output_alices_augmentation_gaussian_prior_0_0.4_0_0.3_10000_thetas_wh_signalWithBSMAndBackgrounds_2D_high_pT.txt 2>&1