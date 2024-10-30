# -*- coding: utf-8 -*-

import argparse as ap
import os, subprocess

if __name__ == '__main__':

    parser = ap.ArgumentParser(description='Description of your script')

    parser.add_argument('--main_dir',help='folder where to keep everything for MadMiner WH studies, on which we store Madgraph background_samples and all .h5 files (setup, analyzed events, ...)',required=True)

    args = parser.parse_args()
 
    signal_samples=['wph_mu','wph_e','wmh_mu','wmh_e']

    background_samples=['wpbb_mu','wpbb_e','wmbb_mu','wmbb_e'] # W + (b-)jets
    background_samples+=['tpb_mu','tpb_e','tmb_mu','tmb_e'] # single top production (tb channel)
    background_samples+=['tt_mupjj','tt_epjj','tt_mumjj','tt_emjj'] # semi-leptonic ttbar

    # for sample in signal_samples:
    #     event_folder=f'{args.main_dir}/signal_samples/{sample}_smeftsim_SM/Events'
    #     for run in os.listdir(event_folder):
    #         full_path=f'{event_folder}/{run}/'
    #         subprocess.run(['sbatch',f'--export=SAMPLE_DIR={full_path}','./run_delphes_SLURM.sh'])
    
    for sample in background_samples:
        event_folder=f'{args.main_dir}/background_samples/{sample}_background/Events'
        for run in os.listdir(event_folder):
            full_path=f'{event_folder}/{run}/'
            subprocess.run(['sbatch',f'--export=SAMPLE_DIR={full_path}','./run_delphes_SLURM.sh'])


    # for sample in signal_samples:
    #     event_folder=f'{args.main_dir}/signal_samples/{sample}_smeftsim_neg_chwtil/Events'
    #     for run in os.listdir(event_folder):
    #         full_path=f'{event_folder}/{run}/'
    #         subprocess.run(['sbatch',f'--export=SAMPLE_DIR={full_path}','./run_delphes_SLURM.sh'])
