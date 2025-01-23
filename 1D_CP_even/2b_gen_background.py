# -*- coding: utf-8 -*-

"""
gen_background.py

Generates background events: W + b b~, t t~, single top t b
- divided by W decay channel and charge (250k events for each combination)

Marta Silva (LIP/IST/CERN-ATLAS), 08/02/2024
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import os
import argparse 
import yaml
import math
from madminer.core import MadMiner
from madminer.lhe import LHEReader



# MadMiner output
logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.DEBUG #INFO 
)

# Output of all other modules (e.g. matplotlib)
for key in logging.Logger.manager.loggerDict:
    if "madminer" not in key:
        logging.getLogger(key).setLevel(logging.WARNING)


def gen_background(main_dir, setup_file, do_pythia, pythia_card, prepare_scripts, mg_dir, launch_SLURM_jobs,cards_folder_name):
    
    """
    Generates background events: W + b b~, t t~, single top t b - divided by W decay channel and charge.
    """
    
    # Load morphing setup file
    miner = MadMiner()
    miner.load(f'{main_dir}/{setup_file}.h5')
    lhe = LHEReader(f'{main_dir}/{setup_file}.h5')

    # LIP Madgraph specifics
    init_command="export LD_LIBRARY_PATH=/cvmfs/sw.el7/gcc63/madgraph/3.3.1/b01/HEPTools/pythia8/lib:$LD_LIBRARY_PATH; module load gcc63/madgraph/3.3.1; module unload gcc63/pythia/8.2.40",

    factor=math.ceil(args.nevents/1e6)
    # W + b b~, divided by W decay channel and charge

    channels = ['wpbb_mu', 'wpbb_e', 'wmbb_mu', 'wmbb_e']

    for channel in channels:
        miner.run_multiple(
            mg_directory=mg_dir,
            log_directory=f'{main_dir}/logs/{channel}_background',
            mg_process_directory=f'{main_dir}/background_samples/{channel}_background',
            proc_card_file=f'cards/background_processes/proc_card_{channel}.dat',
            param_card_template_file='cards/param_card_template_SMEFTsim3_MwScheme.dat',
            pythia8_card_file='cards/pythia8_card.dat' if do_pythia else None,
            run_card_files=['cards/run_card_250k_WHMadminerCuts.dat' for _ in range(factor)],
            sample_benchmarks=['sm'],
            initial_command=init_command,
            is_background=True,
            only_prepare_script=prepare_scripts
        )


    # t b production, divided by top (W) charge and W decay channel

    channels = ['tpb_mu', 'tpb_e', 'tmb_mu', 'tmb_e']

    for channel in channels:
        miner.run_multiple(
            mg_directory=mg_dir,
            log_directory=f'{main_dir}/logs/{channel}_background',
            mg_process_directory=f'{main_dir}/background_samples/{channel}_background',
            proc_card_file=f'cards/background_processes/proc_card_{channel}.dat',
            param_card_template_file='cards/param_card_template_SMEFTsim3_MwScheme.dat',
            pythia8_card_file='cards/pythia8_card.dat' if do_pythia else None,
            run_card_files=['cards/run_card_250k_WHMadminerCuts.dat' for _ in range(factor)],
            sample_benchmarks=['sm'],
            initial_command=init_command,
            is_background=True,
            only_prepare_script=prepare_scripts
        )


    # Semi-leptonic ttbar production, divided in possible decay channels

    channels = ['tt_mupjj', 'tt_epjj', 'tt_mumjj', 'tt_emjj']

    for channel in channels:
        miner.run_multiple(
            mg_directory=mg_dir,
            log_directory=f'{main_dir}/logs/{channel}_background',
            mg_process_directory=f'{main_dir}/background_samples/{channel}_background',
            proc_card_file=f'cards/background_processes/proc_card_{channel}.dat',
            param_card_template_file='cards/param_card_template_SMEFTsim3_MwScheme.dat',
            pythia8_card_file='cards/pythia8_card.dat' if do_pythia else None,
            run_card_files=['cards/run_card_250k_WHMadminerCuts.dat' for _ in range(factor)],
            sample_benchmarks=['sm'],
            initial_command=init_command,
            is_background=True,
            only_prepare_script=prepare_scripts
        )


    # Launch gen jobs to SLURM # LIP specifics
    if args.launch_SLURM_jobs and args.prepare_scripts:
        logging.info("Launching SLURM generation jobs")
        cmd = f'find {main_dir}/background_samples/*/madminer -name "run.sh" -exec sbatch -p lipq --mem=4G {{}} \;'
        os.popen(cmd)

    os.remove('/tmp/generate.mg5')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generates background events: W + b b~, t t~, single top t b - divided by W decay channel and charge', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config_file', help='Path to the YAML configuration file', default='config_1D_CP_even.yaml')

    parser.add_argument('--do_pythia', help='whether or not to run Pythia after Madgraph', action='store_true',default=False)

    parser.add_argument('--prepare_scripts', help='Prepares only run scripts to e.g. submit to a batch system separately', action='store_true', default=False)

    parser.add_argument('--launch_SLURM_jobs', help='If SLURM jobs are to be launched immediately after preparation of scripts', action="store_true", default=False)
    
    parser.add_argument('--nevents',help='number of total hard scattering events to generate (Madgraph-level)',type=float,default=10e6)

    args = parser.parse_args()

    # Read configuration parameters from the YAML file
    with open(args.config_file, 'r') as config_file:
        config = yaml.safe_load(config_file)

        main_dir = config['main_dir']
        setup_file = config['setup_file']
        pythia_card = config['pythia_card']
        mg_dir = config['mg_dir']
        cards_folder_name = config['cards_folder_name']

    # Generate signal
    gen_background(main_dir, setup_file, args.do_pythia, pythia_card, args.prepare_scripts, mg_dir, args.launch_SLURM_jobs,cards_folder_name)

