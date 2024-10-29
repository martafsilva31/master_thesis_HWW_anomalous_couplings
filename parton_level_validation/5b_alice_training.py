
"""
alice_training.py

Handles training of ALICE method

Marta Silva (LIP/IST/CERN-ATLAS), 20/02/2024


"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import yaml
import os
from time import strftime
import argparse 
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from madminer.plotting.distributions import *
from madminer.utils.interfaces.hdf5 import load_madminer_settings
from madminer.sampling import SampleAugmenter
from madminer import sampling
from madminer.ml import ParameterizedRatioEstimator, Ensemble

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
#os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "1")


# timestamp for model saving
timestamp = strftime("%d%m%y")

def alice_training(config):
  
  """ Trains an ensemble of NNs for the ALICE method """
      
  model_name = f"alice_hidden_{config['alice']['training']['n_hidden']}_{config['alice']['training']['activation']}_epochs_{config['alice']['training']['n_epochs']}_bs_{config['alice']['training']['batch_size']}"

  # access to the .h5 file with MadMiner settings
  madminer_settings=load_madminer_settings(f"{config['main_dir']}/{config['observable_set']}/{config['sample_name']}.h5", include_nuisance_benchmarks=False)

  if config['alice']['training']['n_samples'] == -1:
    nsamples = madminer_settings[6]

    logging.info(
        f'sample_name: {config["sample_name"]}; '
        f'observable set: {config["observable_set"]}; '
        f'training observables: {config["alice"]["training"]["observables"]}; '
        f'nsamples: {nsamples}'
    )

  ######### Outputting training variable index for training step ##########
  observable_dict=madminer_settings[5]

  for i_obs, obs_name in enumerate(observable_dict):
    logging.info(f'index: {i_obs}; name: {obs_name};') # this way we can easily see all the features 

  ########## Training ###########
  my_features = None

  # Choose which features to train on 
  # If 'all', we use all of them (None), otherwise we select the correct indices
  if config["observable_set"] ==  "all":
    if config["alice"]["training"]["observables"] == 'kinematic_only':
       my_features = None

  # Other options still to be implemented
  
  #Create a list of ParameterizedRatioEstimator objects to add to the ensemble
  estimators = [ ParameterizedRatioEstimator(features=my_features, n_hidden=config["alice"]["training"]["n_hidden"],activation=config["alice"]["training"]["activation"]) for _ in range(config['alice']['training']['nestimators']) ]
  ensemble = Ensemble(estimators)

  # Run the training of the ensemble
  # result is a list of N tuples, where N is the number of estimators,
  # and each tuple contains two arrays, the first with the training losses, the second with the validation losses
  result = ensemble.train_all(method='alice',
    theta=[f"{config['main_dir']}/{config['observable_set']}/training_samples/alices_{config['alice']['training']['training_samples_name']}/theta0_train_ratio_{config['sample_name']}_{i_estimator}.npy"for i_estimator in range(config['alice']['training']['nestimators'])],
    x=[f"{config['main_dir']}/{config['observable_set']}/training_samples/alices_{config['alice']['training']['training_samples_name']}/x_train_ratio_{config['sample_name']}_{i_estimator}.npy"for i_estimator in range(config['alice']['training']['nestimators'])],
    y=[f"{config['main_dir']}/{config['observable_set']}/training_samples/alices_{config['alice']['training']['training_samples_name']}/y_train_ratio_{config['sample_name']}_{i_estimator}.npy" for i_estimator in range(config['alice']['training']['nestimators'])],
    r_xz=[f"{config['main_dir']}/{config['observable_set']}/training_samples/alices_{config['alice']['training']['training_samples_name']}/r_xz_train_ratio_{config['sample_name']}_{i_estimator}.npy" for i_estimator in range(config['alice']['training']['nestimators'])],
    memmap=True,verbose="all",n_workers=config["alice"]["training"]["n_workers"],limit_samplesize=nsamples,n_epochs=config["alice"]["training"]["n_epochs"],batch_size=config["alice"]["training"]["batch_size"],
  )    
  
  # saving ensemble state dict and training and validation losses
  os.makedirs(f"{config['main_dir']}/{config['observable_set']}/models/{config['alice']['training']['training_samples_name']}/{config['alice']['training']['observables']}/{model_name}", exist_ok=True)
  ensemble.save(f"{config['main_dir']}/{config['observable_set']}/models/{config['alice']['training']['training_samples_name']}/{config['alice']['training']['observables']}/{model_name}/alice_ensemble_{config['sample_name']}")
  np.savez(f"{config['main_dir']}/{config['observable_set']}/models/{config['alice']['training']['training_samples_name']}/{config['alice']['training']['observables']}/{model_name}/alice_losses_{config['sample_name']}",result)

  results = np.load(f"{config['main_dir']}/{config['observable_set']}/models/{config['alice']['training']['training_samples_name']}/{config['alice']['training']['observables']}/{model_name}/alice_losses_{config['sample_name']}.npz")

  fig=plt.figure()

  for i_estimator,(train_loss,val_loss) in enumerate(results['arr_0']):
      plt.plot(train_loss,lw=1.5,ls='solid',label=f'Estimator {i_estimator+1} (Train)',color=list(mcolors.TABLEAU_COLORS.keys())[i_estimator])
      plt.plot(val_loss,lw=1.5,ls='dashed',label=f'Estimator {i_estimator+1} (Validation)',color=list(mcolors.TABLEAU_COLORS.keys())[i_estimator])
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.tight_layout()


  os.makedirs(f"{config['plot_dir']}/{config['observable_set']}/losses/{config['alice']['training']['training_samples_name']}/{config['alice']['training']['observables']}/{model_name}", exist_ok=True)

  # Save the plot #Tenho que por aqui o path coreeto
  fig.savefig(f"{config['plot_dir']}/{config['observable_set']}/losses/{config['alice']['training']['training_samples_name']}/{config['alice']['training']['observables']}/{model_name}/alice_losses_{config['sample_name']}.png")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Trains an ensemble of NNs as estimators for the ALICE method.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config_file', help='Path to the YAML configuration file', default='config.yaml')


    args=parser.parse_args()

    # Read configuration parameters from the YAML file
    with open(args.config_file, 'r') as config_file:
        config = yaml.safe_load(config_file)

                
    logging.info(f'observable set: {config["observable_set"]}; sample type: {config["sample_name"]}')
    
    
    alice_training(config)



    