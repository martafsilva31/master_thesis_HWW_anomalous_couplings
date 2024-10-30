#!/bin/bash

#SBATCH -p lipq
#SBATCH --mem=48G 
#SBATCH --ntasks=8
# Transfering the python script to the execution node
# INPUT = 5a_alices_training.py


module load python/3.9.12

python3 5a_alices_training.py --augment --config_file config.yaml > /lstore/titan/martafsilva/master_thesis/batch_scripts/logs/detector_level_pythia_and_delphes/output_alices_augmentation_10000_thetas_gaussian_0_0.4_wh_signalWithBSMAndBackgrounds.txt 2>&1