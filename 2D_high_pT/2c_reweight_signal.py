 # -*- coding: utf-8 -*-

"""
Reweights WH signal events WH(->l v b b~), divided by W decay channel and charge (250k events/submission)
- to be ran after signal generation if you want to run it multicore, since reweighting doesn't work in multi-core mode

Marta Silva (LIP/IST/CERN-ATLAS), 23/01/2024
"""

import logging
import os
import math

from madminer.core import MadMiner
from madminer.lhe import LHEReader

import argparse as ap
import yaml


# MadMiner output
logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.DEBUG
)

# Output of all other modules (e.g. matplotlib)
for key in logging.Logger.manager.loggerDict:
    if "madminer" not in key:
        logging.getLogger(key).setLevel(logging.WARNING)

if __name__ == "__main__":

    parser = ap.ArgumentParser(description='Reweights WH signal events WH(->l v b b~), divided by W decay channel and charge. Run after generation.',
                                formatter_class=ap.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config_file', help='Path to the YAML configuration file', default='config_2D_high_pT.yaml')
    
    parser.add_argument('-s','--do_signal',help='reweight signal samples', default=False, action='store_true')
    
    parser.add_argument('-bsm','--do_bsm',help='reweight bsm samples', default=False, action='store_true')

    parser.add_argument('--auto_widths',help='Use parameter card with automatic width calculation',action='store_true',default=False)

    args=parser.parse_args()

    with open(args.config_file, 'r') as config_file:
        config = yaml.safe_load(config_file)

        main_dir = config['main_dir']
        setup_file = config['setup_file']
        cards_folder_name = config['cards_folder_name']

    # Load morphing setup file
    miner = MadMiner()
    miner.load(f'{main_dir}/{setup_file}.h5')
    lhe = LHEReader(f'{main_dir}/{setup_file}.h5')

    # List of benchmarks - SM + 2 BSM benchmarks (from Madminer)
    list_benchmarks = lhe.benchmark_names_phys

    # auto width calculation
    # NB: Madgraph+SMEFTsim include terms up to quadratic order in the automatic width calculation, even when the ME^2 is truncated at the SM+interference term
    if args.auto_widths:
        param_card_template_file=f'{cards_folder_name}/param_card_template_SMEFTsim3_MwScheme_autoWidths.dat'
    else:
        param_card_template_file=f'{cards_folder_name}/param_card_template_SMEFTsim3_MwScheme.dat'
        
    channels = ['wph_mu', 'wph_e', 'wmh_mu', 'wmh_e']
    
    if args.do_signal:
        for channel in channels:
            for run in os.listdir(f'{main_dir}/signal_samples/{channel}_smeftsim_SM/Events'):
                miner.reweight_existing_sample(
                    mg_process_directory=f'{main_dir}/signal_samples/{channel}_smeftsim_SM',
                    run_name=run,
                    sample_benchmark='sm',
                    reweight_benchmarks= ['bench_1', 'bench_2', 'bench_3','bench_4', 'bench_5'], 
                    param_card_template_file=param_card_template_file,
                    log_directory=f'{main_dir}/logs/{channel}_smeftsim_SM_reweight/{run}',
                )

    if args.do_bsm:
        for channel in channels:
            for run in os.listdir(f'{main_dir}/signal_samples/{channel}_smeftsim_bench_1/Events'):
                miner.reweight_existing_sample(
                    mg_process_directory=f'{main_dir}/signal_samples/{channel}_smeftsim_bench_1',
                    run_name=run,
                    sample_benchmark='bench_1',
                    reweight_benchmarks= ['sm', 'bench_2', 'bench_3','bench_4', 'bench_5'], 
                    param_card_template_file=param_card_template_file,
                    log_directory=f'{main_dir}/logs/{channel}_smeftsim_bench_1_reweight/{run}',
                )

            for run in os.listdir(f'{main_dir}/signal_samples/{channel}_smeftsim_bench_2/Events'):
                miner.reweight_existing_sample(
                    mg_process_directory=f'{main_dir}/signal_samples/{channel}_smeftsim_bench_2',
                    run_name=run,
                    sample_benchmark='bench_2',
                    reweight_benchmarks= ['sm', 'bench_1', 'bench_3','bench_4', 'bench_5'], 
                    param_card_template_file=param_card_template_file,
                    log_directory=f'{main_dir}/logs/{channel}_smeftsim_bench_2_reweight/{run}',
                )

            for run in os.listdir(f'{main_dir}/signal_samples/{channel}_smeftsim_bench_3/Events'):
                miner.reweight_existing_sample(
                    mg_process_directory=f'{main_dir}/signal_samples/{channel}_smeftsim_bench_3',
                    run_name=run,
                    sample_benchmark='bench_3',
                    reweight_benchmarks= ['sm', 'bench_1', 'bench_2','bench_4', 'bench_5'], 
                    param_card_template_file=param_card_template_file,
                    log_directory=f'{main_dir}/logs/{channel}_smeftsim_bench_3_reweight/{run}',
                )

            for run in os.listdir(f'{main_dir}/signal_samples/{channel}_smeftsim_bench_4/Events'):
                miner.reweight_existing_sample(
                    mg_process_directory=f'{main_dir}/signal_samples/{channel}_smeftsim_bench_4',
                    run_name=run,
                    sample_benchmark='bench_4',
                    reweight_benchmarks= ['sm', 'bench_1', 'bench_2','bench_3', 'bench_5'], 
                    param_card_template_file=param_card_template_file,
                    log_directory=f'{main_dir}/logs/{channel}_smeftsim_bench_4_reweight/{run}',
                )

            for run in os.listdir(f'{main_dir}/signal_samples/{channel}_smeftsim_bench_5/Events'):
                miner.reweight_existing_sample(
                    mg_process_directory=f'{main_dir}/signal_samples/{channel}_smeftsim_bench_5',
                    run_name=run,
                    sample_benchmark='bench_5',
                    reweight_benchmarks= ['sm', 'bench_1', 'bench_2','bench_3', 'bench_4'], 
                    param_card_template_file=param_card_template_file,
                    log_directory=f'{main_dir}/logs/{channel}_smeftsim_bench_5_reweight/{run}',
                )
