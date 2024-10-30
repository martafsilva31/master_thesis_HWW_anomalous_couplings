"""
Madminer parameter and morphing setup code for a WH signal.

Includes the CP-even operator(oHW), with morphing done up to 2nd order (SM + SM-EFT interference term + EFT^2 term).

WARNING: events should ALWAYS be generated with proc cards which have at least the same order in the ME^2 as in the morphing (best: same order) - see gen_signal.py and gen_background.py

Marta Silva (LIP/IST/CERN-ATLAS), 08/02/2024


"""

import os
import logging
import argparse 
import yaml
from madminer import MadMiner
from madminer.plotting import plot_1d_morphing_basis

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


def setup_madminer(main_dir,plot_dir):
    """
    Sets up the MadMiner instance for WH signal, with only the CP-odd operator (oHWtil), and morphing up to 2nd order.
    """

    # Instance of MadMiner core class
    miner = MadMiner()


    miner.add_parameter(
        lha_block='smeft',
        lha_id=7,
        parameter_name='cHW',
        morphing_max_power=2, # interference + squared terms
        parameter_range=(-1.0,1.0),
        param_card_transform='1.0*theta' # mandatory to avoid a crash due to a bug
    )

    # Only want the SM benchmark specifically - let Madminer choose the others
    miner.add_benchmark({'cHW':0.00},'sm')


    # Morphing - automatic optimization to avoid large weights
    miner.set_morphing(max_overall_power=2,include_existing_benchmarks=True)

    miner.save(f'{main_dir}/setup_CP_even.h5')

    morphing_basis=plot_1d_morphing_basis(miner.morpher,xlabel=r'$c_{HW}$',xrange=(-1.0,1.0))
    morphing_basis.savefig(f'{plot_dir}/morphing_basis_CP_even.png')

    return miner
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Creates MadMiner parameter and morphing setup file for a WH signal,  with the CP-even operator(oHW), \
                               morphing up to second order (SM + SM-EFT interference + EFT^2 term).',
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config_file', help='Path to the YAML configuration file', default='config_with_CP_even.yaml')
    args = parser.parse_args()

    # Read main_dir and plot_dir from the YAML configuration file
    with open(args.config_file, 'r') as config_file:
        config = yaml.safe_load(config_file)
        main_dir = config['main_dir']
        plot_dir = config['plot_dir']

    # MadMiner setup function
    setup_madminer(main_dir, plot_dir)