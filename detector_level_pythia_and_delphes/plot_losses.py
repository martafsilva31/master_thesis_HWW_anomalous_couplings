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

config_file = 'config.yaml'

# Read configuration parameters from the YAML file
with open(config_file, 'r') as config_file:
    config = yaml.safe_load(config_file)

model_name = f"alices_hidden_{config['alices']['training']['n_hidden']}_{config['alices']['training']['activation']}_alpha_{config['alices']['training']['alpha']}_epochs_{config['alices']['training']['n_epochs']}_bs_{config['alices']['training']['batch_size']}"


results = np.load(f"{config['main_dir']}/models/{config['alices']['training']['training_samples_name']}/{config['alices']['training']['observables']}/{model_name}/alices_losses_{config['sample_name']}.npz")

fig=plt.figure()

for i_estimator,(train_loss,val_loss) in enumerate(results['arr_0']):
      plt.plot(train_loss,lw=1.5,ls='solid',label=f'Estimator {i_estimator+1} (Train)',color=list(mcolors.TABLEAU_COLORS.keys())[i_estimator])
      plt.plot(val_loss,lw=1.5,ls='dashed',label=f'Estimator {i_estimator+1} (Validation)',color=list(mcolors.TABLEAU_COLORS.keys())[i_estimator])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()


os.makedirs(f"{config['plot_dir']}/losses/{config['alices']['training']['training_samples_name']}/{config['alices']['training']['observables']}/{model_name}", exist_ok=True)

  # Save the plot #Tenho que por aqui o path coreeto
fig.savefig(f"{config['plot_dir']}/losses/{config['alices']['training']['training_samples_name']}/{config['alices']['training']['observables']}/{model_name}/alices_losses_{config['sample_name']}.png")
