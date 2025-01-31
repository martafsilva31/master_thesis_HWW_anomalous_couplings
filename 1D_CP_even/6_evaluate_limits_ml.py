from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import os,sys
import argparse as ap
from madminer.limits import AsymptoticLimits
from madminer import sampling
from madminer.plotting import plot_histograms
from madminer.utils.histo import Histo
import numpy as np
import matplotlib
import yaml
import matplotlib.pyplot as plt
import argparse 
matplotlib.use('Agg') 
from madminer.sampling import combine_and_shuffle

#from operator import 
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


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "1")



def evaluate_and_save_llr_individual(config):

  """Evaluates and saves the log likelihood ratio for each estimator in the ensemble"""
  
  filename = f"{config['main_dir']}/{config['sample_name']}.h5"

  base_model_path = f"{config['main_dir']}/models/{config['limits']['prior']}/{config['limits']['observables']}/{config['limits']['model']}/{config['limits']['method']}_ensemble_{config['sample_name']}"

  limits_file=AsymptoticLimits(filename)

  for i in range(5):
  
    model_path = base_model_path + f"/estimator_{i}"

    parameter_grid,p_values,index_best_point,llr_kin,llr_rate,(histos, observed, observed_weights)=limits_file.expected_limits(
    mode=config['limits']['mode'],theta_true=[0.0], include_xsec=not config['limits']['shape_only'],
    model_file = model_path if( 'ml' == config['limits']['mode'] or 'sally' == config['limits']['mode'] ) else None,
    # hist_vars=hist_vars if 'histo' == config['limits']['mode'] else None,
    # hist_bins=hist_bins,
    luminosity=config['limits']['lumi']*1000.0,
    return_asimov=True,test_split=0.2,n_histo_toys=None,
    grid_ranges=[config['limits']['grid_ranges']],grid_resolutions=[config['limits']['grid_resolutions']])


    os.makedirs(f"{config['main_dir']}/llr_fits/{config['limits']['prior']}/{config['limits']['observables']}/{config['limits']['model']}/{config['sample_name']}/range_{config['limits']['grid_ranges']}_resolutions_{config['limits']['grid_resolutions']}",exist_ok=True)
    save_dir = f"{config['main_dir']}/llr_fits/{config['limits']['prior']}/{config['limits']['observables']}/{config['limits']['model']}/{config['sample_name']}/range_{config['limits']['grid_ranges']}_resolutions_{config['limits']['grid_resolutions']}"

    np.savez(f"{save_dir}/estimator_{i}_data.npz", parameter_grid=parameter_grid, p_values=p_values,
                index_best_point=index_best_point, llr_kin=llr_kin, llr_rate=llr_rate)

def evaluate_and_save_llr_ensemble(config):
    
    """Evaluates and saves the log likelihood ratio for the ensemble of estimators"""

    filename = f"{config['main_dir']}/{config['sample_name']}.h5"

    model_path = f"{config['main_dir']}/models/{config['limits']['prior']}/{config['limits']['observables']}/{config['limits']['model']}/{config['limits']['method']}_ensemble_{config['sample_name']}"

    limits_file=AsymptoticLimits(filename)


    parameter_grid,p_values,index_best_point,llr_kin,llr_rate,(histos, observed, observed_weights)=limits_file.expected_limits(
    mode=config['limits']['mode'],theta_true=[0.0], include_xsec=not config['limits']['shape_only'],
    model_file = model_path if( 'ml' == config['limits']['mode'] or 'sally' == config['limits']['mode'] )  else None,
    # hist_vars=hist_vars if 'histo' == config['limits']['mode'] else None,
    # hist_bins=hist_bins,
    luminosity=config['limits']['lumi']*1000.0,
    return_asimov=True,test_split=0.2,n_histo_toys=None,
    grid_ranges=[config['limits']['grid_ranges']],grid_resolutions=[config['limits']['grid_resolutions']])

    os.makedirs(f"{config['main_dir']}/llr_fits/{config['limits']['prior']}/{config['limits']['observables']}/{config['limits']['model']}/{config['sample_name']}/range_{config['limits']['grid_ranges']}_resolutions_{config['limits']['grid_resolutions']}",exist_ok=True)
    save_dir = f"{config['main_dir']}/llr_fits/{config['limits']['prior']}/{config['limits']['observables']}/{config['limits']['model']}/{config['sample_name']}/range_{config['limits']['grid_ranges']}_resolutions_{config['limits']['grid_resolutions']}"

    np.savez(f"{save_dir}/ensemble_data.npz", parameter_grid=parameter_grid, p_values=p_values,
                index_best_point=index_best_point, llr_kin=llr_kin, llr_rate=llr_rate)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plots log likelihood ratio evaluate for all estimators or with and ensemble', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config_file', help='Path to the YAML configuration file', default='config_1D_CP_even.yaml')

    parser.add_argument('--evaluate', help='Evaluates and saves llr for each estimator (individual) or using an ensemble (ensemble)', choices=['individual', 'ensemble'])

    args = parser.parse_args()

    # Read configuration parameters from the YAML file
    with open(args.config_file, 'r') as config_file:
      config = yaml.safe_load(config_file)

    os.makedirs(f"{config['plot_dir']}/llr_fits/",exist_ok=True)

    if args.evaluate == 'individual':
        evaluate_and_save_llr_individual(config)

    if args.evaluate == 'ensemble':
        evaluate_and_save_llr_ensemble(config)
    