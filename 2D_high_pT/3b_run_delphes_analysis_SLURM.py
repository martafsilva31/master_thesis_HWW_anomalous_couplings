# -*- coding: utf-8 -*-

import argparse as ap
import os, subprocess, sys, shutil
import time
import yaml

def run_delphes_analysis_SLURM(sample_folder, main_dir,setup_file, do_delphes, delphes_card='', benchmark='sm'):

    sample_folder=f'{sample_folder}/Events' 
    runs = sorted(os.listdir(sample_folder))
    for run in runs:
        full_path=f'{sample_folder}/{run}/'
        sbatch_env=f'--export=ALL,SAMPLE_DIR={full_path},MAIN_DIR={main_dir}, BENCHMARK={benchmark},SETUP_FILE={setup_file}'
        if do_delphes:
            sbatch_env+=f',DO_DELPHES=True,DELPHES_CARD={delphes_card}'
        subprocess.run(['sbatch',sbatch_env,'./3c_run_delphes_analysis_SLURM.sh'])
        

if __name__ == '__main__':

    parser = ap.ArgumentParser(description='Run Delphes + MadMiner analysis on generated files.')

    parser.add_argument('--config_file', help='Path to the YAML configuration file', default='config_2D_high_pT.yaml')
  
    parser.add_argument('-b','--do_backgrounds',help='run over background samples', default=False, action='store_true')
    
    parser.add_argument('-s','--do_signal',help='run over signal samples', default=False, action='store_true')

    parser.add_argument('-bsm','--do_bsm',help='run over bsm samples', default=False, action='store_true')

    parser.add_argument('-d','--do_delphes',help='run Delphes', default=False, action='store_true')

    args = parser.parse_args()

    # Read configuration parameters from the YAML file
    with open(args.config_file, 'r') as config_file:

        config = yaml.safe_load(config_file)
        main_dir = config['main_dir']
        setup_file = config['setup_file']
        delphes_card = config['delphes_card']


    signal_samples=['wph_mu', 'wph_e', 'wmh_mu', 'wmh_e']

    background_samples=['wpbb_mu','wpbb_e','wmbb_mu','wmbb_e'] # W + (b-)jets
    background_samples+=['tpb_mu','tpb_e','tmb_mu','tmb_e'] # single top production (tb channel)
    background_samples+=['tt_mupjj','tt_epjj','tt_mumjj','tt_emjj'] # semi-leptonic ttbar

    benchmarks = ['bench_1', 'bench_2', 'bench_3', 'bench_4', 'bench_5']
    
    if args.do_signal:
        for sample in signal_samples:
            sample_folder=f'{args.main_dir}/signal_samples/{sample}_smeftsim_SM/'
            run_delphes_analysis_SLURM(sample_folder,main_dir,setup_file,args.do_delphes,delphes_card,'sm')
    

    if args.do_backgrounds:
        for sample in background_samples:
            sample_folder=f'{args.main_dir}/background_samples/{sample}_background/'
            run_delphes_analysis_SLURM(sample_folder,main_dir,setup_file,args.do_delphes,delphes_card, 'sm')


    if args.do_bsm:
        for sample in signal_samples:
            for bench in benchmarks:
                sample_folder=f'{args.main_dir}/signal_samples/{sample}_smeftsim_{bench}/'
                run_delphes_analysis_SLURM(sample_folder,main_dir,setup_file,args.do_delphes,delphes_card, bench)
