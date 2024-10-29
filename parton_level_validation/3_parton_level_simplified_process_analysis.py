# -*- coding: utf-8 -*-

"""
parton_level_analysis.py

Parton-level analysis of signal events.

Marta Silva (LIP/IST/CERN-ATLAS), 04/03/2024
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import yaml
import os
import argparse 
from madminer.lhe import LHEReader

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
        
        
def choose_observables(observable_set):
        
    if observable_set == 'all':   

        #m_tot is not in the list because all its values were NaN

        #Elements in the lists will be given as input to the add_observable function

        observable_names = [
            'b1_px', 'b1_py', 'b1_pz', 'b1_e',
            'b2_px', 'b2_py', 'b2_pz', 'b2_e',
            'l_px', 'l_py', 'l_pz', 'l_e',
            'v_px', 'v_py', 'v_pz', 'v_e',
            'pt_b1', 'pt_b2', 'pt_l1', 'pt_l2', 'pt_w', 'pt_h',
            'eta_b1', 'eta_b2', 'eta_l', 'eta_v', 'eta_w', 'eta_h',
            'phi_b1', 'phi_b2', 'phi_l', 'phi_v', 'phi_w', 'phi_h',
            'theta_b1', 'theta_b2', 'theta_l', 'theta_v', 'theta_w', 'theta_h',
            'dphi_bb', 'dphi_lv', 'dphi_wh',
            'm_bb', 'm_lv',
            'q_l', 'q_v', 'q_b1', 'q_b2',
            'dphi_lb1', 'dphi_lb2', 'dphi_vb1', 'dphi_vb2',
            'dR_bb', 'dR_lv', 'dR_lb1', 'dR_lb2', 'dR_vb1', 'dR_vb2'
        ]

        list_of_observables = [
            'j[0].px', 'j[0].py', 'j[0].pz', 'j[0].e',
            'j[1].px', 'j[1].py', 'j[1].pz', 'j[1].e',
            'l[0].px', 'l[0].py', 'l[0].pz', 'l[0].e',
            'v[0].px', 'v[0].py', 'v[0].pz', 'v[0].e',
            'j[0].pt', 'j[1].pt', 'l[0].pt', 'v[0].pt', '(l[0] + v[0]).pt', '(j[0] + j[1]).pt',
            'j[0].eta', 'j[1].eta', 'l[0].eta', 'v[0].eta', '(l[0] + v[0]).eta', '(j[0] + j[1]).eta',
            'j[0].phi', 'j[1].phi', 'l[0].phi', 'v[0].phi', '(l[0] + v[0]).phi', '(j[0] + j[1]).phi',
            'j[0].theta', 'j[1].theta', 'l[0].theta', 'v[0].theta', '(l[0] + v[0]).theta', '(j[0] + j[1]).theta',
            'j[0].deltaphi(j[1])', 'l[0].deltaphi(v[0])', '(l[0] + v[0]).deltaphi(j[0] + j[1])',
            '(j[0] + j[1]).m', '(l[0] + v[0]).m',
            'l[0].charge', 'v[0].charge', 'j[0].charge', 'j[1].charge',
            'l[0].deltaphi(j[0])', 'l[0].deltaphi(j[1])', 'v[0].deltaphi(j[0])', 'v[0].deltaphi(j[1])',
            'j[0].deltaR(j[1])', 'l[0].deltaR(v[0])', 'l[0].deltaR(j[0])', 'l[0].deltaR(j[1])', 'v[0].deltaR(j[0])', 'v[0].deltaR(j[1])',
        ]
    else:
        raise ValueError(f"Unknown observable_set: {observable_set}. Please choose 'all' as the observable set.")

    return observable_names, list_of_observables
        

# Function to process the events, run on each separate data sample
def process_events(observable_set,event_path, setup_file_path,output_file_path,is_background_process=False,is_SM=True):

    # Load Madminer setup
    lhe = LHEReader(setup_file_path)

    # Add events
    if(is_SM):
        lhe.add_sample(
            f'{event_path}/Events/run_01/unweighted_events.lhe.gz',
            sampled_from_benchmark='sm',
            is_background=is_background_process
        )
    else:
        list_BSM_benchmarks = [x for x in lhe.benchmark_names_phys if x != 'sm']
        for i,benchmark in enumerate(list_BSM_benchmarks,start=1):
            run_str = str(i)
            if len(run_str) < 2:
                run_str = '0' + run_str
            lhe.add_sample(
                f'{event_path}/Events/run_{run_str}/unweighted_events.lhe.gz',
                sampled_from_benchmark=benchmark,
                is_background=is_background_process
            )
    
    # Choosing observables
    observable_names, list_of_observables = choose_observables(observable_set)

    # Adding observables
    for i, name in enumerate(observable_names):
        lhe.add_observable( name, list_of_observables[i], required=True )
    
    # Analyse samples and save the processed events as an .h5 file for later use
    lhe.analyse_samples()

    lhe.save(output_file_path)

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser(description='Parton-level analysis of signal events with a complete set of observables (including leading neutrino 4-vector).',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--config_file', help='Path to the YAML configuration file', default='config.yaml')

  parser.add_argument('--do_signal',help='analyze signal events', action='store_true',default=True)

  parser.add_argument('--do_BSM',help='analyze samples generated at the BSM benchmarks', action='store_true',default=False)

  args=parser.parse_args()

  # Read configuration parameters from the YAML file
  with open(args.config_file, 'r') as config_file:
    config = yaml.safe_load(config_file)

    main_dir = config['main_dir']
    setup_file = config['setup_file']
    observable_set = config['observable_set']

  output_dir=f'{main_dir}/{observable_set}'

############## Signal ###############
    


  if args.do_signal:

    os.makedirs(f'{output_dir}/signal/',exist_ok=True)

    process_events(observable_set,
        event_path=f'{main_dir}/signal_samples/ud_wph_mu_smeftsim_SM',
        setup_file_path=f'{main_dir}/{setup_file}.h5',
        is_background_process=False,
        is_SM=True,
        output_file_path=f'{output_dir}/signal/ud_wph_mu_smeftsim_SM_lhe.h5',
    )

    if args.do_BSM:
        process_events(observable_set,
            event_path=f'{main_dir}/signal_samples/ud_wph_mu_smeftsim_BSM',
            setup_file_path=f'{main_dir}/{setup_file}.h5',
            is_background_process=False,
            is_SM=False,
            output_file_path=f'{output_dir}/signal/ud_wph_mu_smeftsim_BSM_lhe.h5',

        )