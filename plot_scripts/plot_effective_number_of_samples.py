"""
alices_training.py

Handles extraction of joint score + likelihood ratio from event samples and training of ALICES method

Marta Silva (LIP/IST/CERN-ATLAS), 29/02/2024
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import yaml
import os
from time import strftime, sleep, time
import argparse 
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from madminer.plotting.distributions import *
from madminer.utils.interfaces.hdf5 import load_madminer_settings
from madminer.sampling import SampleAugmenter
from madminer import sampling
from madminer.ml import ParameterizedRatioEstimator, Ensemble
import psutil  # Import psutil for system monitoring
import multiprocessing
import matplotlib
matplotlib.use('Agg')


# MadMiner output
logging.basicConfig(
  format='%(asctime)-5.5s %(funcName)-20.20s %(levelname)-7.7s %(message)s',
  datefmt='%H:%M',
  level=logging.INFO
)

# Output of all other modules (e.g. matplotlib)
for key in logging.Logger.manager.loggerDict:
  if "madminer" not in key:
    logging.getLogger(key).setLevel(logging.WARNING)

# Choose the GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "0")
#os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "1")


# timestamp for model saving
timestamp = strftime("%d%m%y")



def alices_augmentation(config):
  """ Creates augmented training samples for the ALICES method  """

    # access to the .h5 file with MadMiner settings
  madminer_settings=load_madminer_settings(f"{config['main_dir']}/{config['sample_name']}.h5", include_nuisance_benchmarks=False)

  if config['alices']['augmentation']['n_samples'] == -1:
    nsamples = madminer_settings[6]

  else:
    nsamples = config['alices']['augmentation']['n_samples']
  logging.info(
      f'sample_name: {config["sample_name"]}; '
      f'training observables: {config["alices"]["training"]["observables"]}; '
      f'nsamples: {nsamples}'
  )


  ######### Outputting training variable index for training step ##########
  observable_dict=madminer_settings[5]

  for i_obs, obs_name in enumerate(observable_dict):
    logging.info(f'index: {i_obs}; name: {obs_name};') # this way we can easily see all the features 

  ########## Sample Augmentation ###########

  # object to create the augmented training samples
  sampler=SampleAugmenter( f'{config["main_dir"]}/{config["sample_name"]}.h5')

  # Creates a set of training data (as many as the number of estimators) - centered around the SM


  x, theta0, theta1, y, r_xz, t_xz, n_effective = sampler.sample_train_ratio(
  #theta0=sampling.random_morphing_points(config["alices"]["augmentation"]["n_thetas"], [config["alices"]["augmentation"]["priors"]]),
  theta0=sampling.random_morphing_points(10000, [('gaussian', 0., 0.4), ('gaussian', 0., 0.3)]), # Running 2D sampling
  theta1=sampling.benchmark("sm"),
  n_samples=int(nsamples),
  folder=f'{config["main_dir"]}/training_samples/alices_{config["alices"]["augmentation"]["prior_name"]}',
  filename=f'train_ratio_{config["sample_name"]}',
  sample_only_from_closest_benchmark=True,
  return_individual_n_effective=True,
  n_processes = 6
  )

  np.save("x.npy", x)
  np.save("theta0",theta0)
  np.save("theta1",theta1)
  np.save("y.npy", y)
  np.save("r_xz",r_xz)
  np.save("t_xz",t_xz)
  np.save("n_effective.npy", n_effective)

  logging.info(f'effective number of samples: {n_effective}')

  cmin, cmax = 10., 10000.

  cut = (y.flatten() == 0)

  fig, ax = plt.subplots(figsize=(5, 4))  # Create figure and axes

  sc = ax.scatter(theta0[cut][:, 0], theta0[cut][:, 1], c=n_effective[cut],
                  s=30., cmap='viridis',
                  norm=matplotlib.colors.LogNorm(vmin=cmin, vmax=cmax),
                  marker='o')

  cb = fig.colorbar(sc, ax=ax)
  cb.set_label('Effective number of samples', size=14)

  ax.set_xlabel(r'$c_{H\tildeW}$', size=14)
  ax.set_ylabel(r'$c_{HW}$', size=14)
  cb.ax.tick_params(labelsize=14)

  ax.set_xlim(-1.2, 1.2)
  ax.set_ylim(-1.0, 1.0)
  fig.tight_layout()

  output = "/lstore/titan/martafsilva/master_thesis/master_thesis_output/detector_level_pythia_and_delphes/plots"

  fig.savefig(f"{output}/effective_number_of_samples.pdf")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Creates augmented (unweighted) training samples for the Approximate likelihood with improved cross-entropy estimator and score method (ALICES). Trains an ensemble of NNs as estimators.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config_file', help='Path to the YAML configuration file', default='config_2D.yaml')


    args=parser.parse_args()
  

    # Read configuration parameters from the YAML file
    with open(args.config_file, 'r') as config_file:
        config = yaml.safe_load(config_file)

    logging.info(f'sample type: {config["sample_name"]}')

    

    alices_augmentation(config)
 