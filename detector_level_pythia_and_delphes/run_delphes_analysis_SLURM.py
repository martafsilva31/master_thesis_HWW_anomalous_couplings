# -*- coding: utf-8 -*-

import argparse as ap
import os, subprocess, sys, shutil
import time

def run_delphes_analysis_SLURM(sample_folder, main_dir, do_delphes, delphes_card='cards/delphes_card_ATLAS.tcl'):

    sample_folder=f'{sample_folder}/Events' 
    runs = sorted(os.listdir(sample_folder))
    for run in runs:
        full_path=f'{sample_folder}/{run}/'
        sbatch_env=f'--export=ALL,SAMPLE_DIR={full_path},MAIN_DIR={main_dir}'
        if do_delphes:
            sbatch_env+=f',DO_DELPHES=True,DELPHES_CARD={delphes_card}'
        subprocess.run(['sbatch',sbatch_env,'./run_delphes_analysis_SLURM.sh'])
        
    # full_path=f'{sample_folder}//Events/run_22/'
    # sbatch_env=f'--export=ALL,SAMPLE_DIR={full_path},MAIN_DIR={main_dir}'
    # if do_delphes:
    #     sbatch_env+=f',DO_DELPHES=True,DELPHES_CARD={delphes_card}'
    # subprocess.run(['sbatch',sbatch_env,'./run_delphes_analysis_SLURM.sh'])

if __name__ == '__main__':

    parser = ap.ArgumentParser(description='Run Delphes + MadMiner analysis on generated files.')

    parser.add_argument('--main_dir',help='folder where to keep everything for MadMiner WH studies, on which we store Madgraph background_samples and all .h5 files (setup, analyzed events, ...)',required=True)

    parser.add_argument('-b','--do_backgrounds',help='run over background samples', default=False, action='store_true')
    
    parser.add_argument('-s','--do_signal',help='run over signal samples', default=False, action='store_true')

    parser.add_argument('-d','--do_delphes',help='run Delphes', default=False, action='store_true')

    parser.add_argument('-bsm','--do_bsm',help='run over bsm samples', default=False, action='store_true')

    parser.add_argument('--delphes_card',help='path to Delphes card',default='cards/delphes_card_ATLAS.tcl')
    
    args = parser.parse_args()
 
    signal_samples=['wmh_mu','wmh_e']#'wph_mu','wph_e','wmh_mu','wmh_e']

    background_samples=['wpbb_mu','wpbb_e','wmbb_mu','wmbb_e'] # W + (b-)jets
    background_samples+=['tpb_mu','tpb_e','tmb_mu','tmb_e'] # single top production (tb channel)
    background_samples+=['tt_mupjj','tt_epjj','tt_mumjj','tt_emjj'] # semi-leptonic ttbar

    BSM_benchmarks = ['pos_chwtil', 'neg_chwtil']
    
    # if args.do_signal:
    #     for sample in signal_samples:
    #         sample_folder=f'{args.main_dir}/signal_samples/{sample}_smeftsim_SM/'
    #         run_delphes_analysis_SLURM(sample_folder,args.main_dir,args.do_delphes,args.delphes_card)
    
    # if args.do_signal:
        
    #     sample_folder=f'{args.main_dir}/signal_samples/wph_mu_smeftsim_SM/'
    #     run_delphes_analysis_SLURM(sample_folder,args.main_dir,args.do_delphes,args.delphes_card)

    # if args.do_backgrounds:
    #     for sample in background_samples:
    #         sample_folder=f'{args.main_dir}/background_samples/{sample}_background/'
    #         run_delphes_analysis_SLURM(sample_folder,args.main_dir,args.do_delphes,args.delphes_card)

    signal_samples=['wph_mu','wph_e','wmh_mu','wmh_e']

    if args.do_bsm:
        for sample in signal_samples:
            bench = "bench_5" #change the benchmark name in the delphes_analysis.py file
            sample_folder=f'{args.main_dir}/signal_samples/{sample}_smeftsim_{bench}/'
            run_delphes_analysis_SLURM(sample_folder,args.main_dir,args.do_delphes,args.delphes_card)
