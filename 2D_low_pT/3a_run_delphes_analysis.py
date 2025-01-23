from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from madminer.delphes import DelphesReader
import argparse as ap
import os
import yaml
import utils.analysis as parton

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

# redefining here since for Delphes files, the particles input variable (existent in LHE files) is not used anymore
def get_neutrino_pz(leptons=[],photons=[],jets=[],met=None,debug=False):
    
    return parton.get_neutrino_pz([],leptons,photons,jets,met,debug)

def get_cos_deltaPlus(leptons=[],photons=[],jets=[],met=None,debug=False):

    return parton.get_cos_deltaPlus([],leptons,photons,jets,met,debug)

def get_ql_cos_deltaPlus(leptons=[],photons=[],jets=[],met=None,debug=False):

    return parton.get_ql_cos_deltaPlus([],leptons,photons,jets,met,debug)

def get_cos_deltaMinus(leptons=[],photons=[],jets=[],met=None,debug=False):

    return parton.get_cos_deltaMinus([],leptons,photons,jets,met,debug)

def get_ql_cos_deltaMinus(leptons=[],photons=[],jets=[],met=None,debug=False):

    return parton.get_ql_cos_deltaMinus([],leptons,photons,jets,met,debug)

def process_events(event_path, setup_file_path, is_background_process=False, k_factor=1.0, do_delphes=True, delphes_card='', benchmark='sm'):
    
    reader=DelphesReader(setup_file_path)
    
    logging.info(f'event_path: {event_path}, is_background: {is_background_process}')

    reader.add_sample(hepmc_filename=f'{event_path}/tag_1_pythia8_events.hepmc.gz',
                    sampled_from_benchmark=benchmark,
                    is_background=is_background_process,
                    lhe_filename=f'{event_path}/unweighted_events.lhe.gz',
                    delphes_filename=None if do_delphes else f'{event_path}/delphes_events.root',
                    k_factor=k_factor,
                    weights='lhe')

    if do_delphes:
        if os.path.exists(event_path+'/tag_1_pythia8_events_delphes.root'):
            logging.warning(f'Delphes file in {event_path} already exists !')
        reader.run_delphes('/cvmfs/sw.el7/gcc63/madgraph/3.3.1/b01/Delphes/', delphes_card, initial_command='module load gcc63/madgraph/3.3.1; source /cvmfs/sw.el7/gcc63/madgraph/3.3.1/b01/Delphes/DelphesEnv.sh', log_file=event_path+'/do_delphes.log')

    if os.path.exists(event_path+'/analysed_events.h5'):
        logging.warning(f'analysed (.h5) file in {event_path} already exists !')

    # this will have to be changed if I am to change 
    for i, name in enumerate(parton.observable_names):
        reader.add_observable( name, parton.list_of_observables[i], required=True )

    reader.add_observable_from_function('pz_nu', get_neutrino_pz,required=True)
    reader.add_observable_from_function('cos_deltaPlus', get_cos_deltaPlus,required=True)
    reader.add_observable_from_function('ql_cos_deltaPlus', get_ql_cos_deltaPlus,required=True)
    reader.add_observable_from_function('cos_deltaMinus', get_cos_deltaMinus,required=True)
    reader.add_observable_from_function('ql_cos_deltaMinus', get_ql_cos_deltaMinus,required=True)

    # requiring the two leading jets to be b-tagged
    reader.add_cut('j[0].b_tag',required=True)
    reader.add_cut('j[1].b_tag',required=True)
    

    reader.analyse_delphes_samples(delete_delphes_files=True)

    reader.save(f'{event_path}/analysed_events.h5')

if __name__ == '__main__':

    parser = ap.ArgumentParser(description='Detector-level analysis of signal and background events (with Delphes). Includes the computation of the pZ of the neutrino and several angular observables',formatter_class=ap.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config_file', help='Path to the YAML configuration file', default='config_1D_CP_even.yaml')

    parser.add_argument('--sample_dir',help='folder where the individual sample is', required=True)

    parser.add_argument('--do_delphes',help='run Delphes before analysis code', default=True, action="store_true")
   
    parser.add_argument('--benchmark',help='benchmark from which the sample is originally generated', default=True, action="store_true")
   
    args=parser.parse_args()

    # Read configuration parameters from the YAML file
    with open(args.config_file, 'r') as config_file:

        config = yaml.safe_load(config_file)
        main_dir = config['main_dir']
        setup_file = config['setup_file']
        delphes_card = config['delphes_card']
        
    if 'background' in args.sample_dir:
        process_events(f'{args.sample_dir}',f'{setup_file}',is_background_process=True,k_factor=1.0,do_delphes=args.do_delphes, delphes_card=delphes_card, benchmark = args.benchmark)
    else:
        process_events(f'{args.sample_dir}',f'{setup_file}',is_background_process=False,k_factor=1.0,do_delphes=args.do_delphes, delphes_card=delphes_card, benchmark = args.benchmark)