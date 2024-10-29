# -*- coding: utf-8 -*-

"""
analysis_sample_combiner.py

Combines analyzed samples from the output of the analysis scripts.

Marta Silva (LIP/IST/CERN-ATLAS), 09/02/2024
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import os, shutil, sys
import argparse 
import yaml
from madminer.core import MadMiner
from madminer.lhe import LHEReader
from madminer.sampling import combine_and_shuffle

# MadMiner output
logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.INFO
)

# Output of all other modules (e.g. matplotlib)
for key in logging.Logger.manager.loggerDict:
    if "madminer" not in key:
        logging.getLogger(key).setLevel(logging.WARNING)


def combine_SM_and_BSM(full_proc_dir):
   
    # generated SM samples + generated BSM samples
    combine_and_shuffle([
        f'{full_proc_dir}/ud_wph_mu_smeftsim_SM_lhe.h5',
        f'{full_proc_dir}/ud_wph_mu_smeftsim_BSM_lhe.h5'],
        f'{full_proc_dir}/ud_wph_mu_smeftsim_lhe.h5'
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combines and shuffles different samples, depending on the purposes.',
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--config_file', help='Path to the YAML configuration file', default='config.yaml')

    args = parser.parse_args()

    # Read configuration parameters from the YAML file
    with open(args.config_file, 'r') as config_file:
        config = yaml.safe_load(config_file)

        main_dir = config['main_dir']
        observable_set = config['observable_set']


    full_proc_dir = f'{main_dir}/{observable_set}/'

    # combining positive and negative charge channels and all the flavors
    
    combine_SM_and_BSM(full_proc_dir)

