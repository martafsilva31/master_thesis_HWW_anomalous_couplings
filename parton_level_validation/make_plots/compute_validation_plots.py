from __future__ import absolute_import, division, print_function, unicode_literals
import logging

import sys
import os
from time import strftime
import argparse
import numpy as np
import yaml
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors

from madminer.utils.interfaces.hdf5 import load_madminer_settings
from madminer.sampling import SampleAugmenter
from madminer import sampling
from madminer.ml import ParameterizedRatioEstimator,ScoreEstimator, Ensemble
from scipy.stats import gaussian_kde
from scipy.interpolate import interpn

from matplotlib import cm
from matplotlib.colors import Normalize 

from matplotlib.cm import ScalarMappable
from matplotlib import colors


# MadMiner output
logging.basicConfig(
  format='%(asctime)-5.5s %(funcName)-20.20s %(levelname)-7.7s %(message)s',
  datefmt='%H:%M',
  level=logging.INFO)

# Output of all other modules (e.g. matplotlib)
for key in logging.Logger.manager.loggerDict:
  if "madminer" not in key:
    logging.getLogger(key).setLevel(logging.WARNING)

# Choose the GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "1")
#os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "0")

def augment_test(config):
    
  """  
  Extracts the joint likelihood ratio and the joint score for the test partition with the ALICES or SALLY method
  """


  if args.model == 'alices' or args.model == 'alice':
    # access to the .h5 file with MadMiner settings
    madminer_settings=load_madminer_settings(f"{config['main_dir']}/{config['observable_set']}/{config['sample_name']}.h5", include_nuisance_benchmarks=False)

    if config['alices']['testing']['n_samples'] == -1:
      nsamples=madminer_settings[6]

    else: 
      nsamples = config['alices']['testing']['n_samples']

      logging.info(
          f'sample_name: {config["sample_name"]}; '
          f'observable set: {config["observable_set"]}; '
          f'nsamples: {nsamples}'
      )
    ########## Sample Augmentation ###########
    # # object to create the augmented training samples
    sampler=SampleAugmenter( f'{config["main_dir"]}/{config["observable_set"]}/{config["sample_name"]}.h5')

    # Creates a set of testing data - centered around the SM
    
    x, theta0, theta1, y, r_xz, t_xz, n_effective = sampler.sample_train_ratio(
    theta0=sampling.random_morphing_points(config["alices"]["testing"]["n_thetas"], [config["alices"]["testing"]["priors"]]),
    theta1=sampling.benchmark("sm"),
    n_samples=int(nsamples),
    folder=f'{config["main_dir"]}/{config["observable_set"]}/testing_samples/alices_{config["alices"]["testing"]["prior_name"]}',
    filename=f'test_ratio_{config["sample_name"]}',
    sample_only_from_closest_benchmark=True,
    return_individual_n_effective=True,
    partition = "test",
    n_processes = config["alices"]["testing"]["n_processes"]
    )

    logging.info(f'effective number of samples: {n_effective}')
  
  if args.model == 'sally':
       
    # access to the .h5 file with MadMiner settings
    madminer_settings=load_madminer_settings(f"{config['main_dir']}/{config['observable_set']}/{config['sample_name']}.h5", include_nuisance_benchmarks=False)

    if config['sally']['testing']['n_samples'] == -1:
      nsamples=madminer_settings[6]

    else: 
      nsamples = config['sally']['testing']['n_samples']
      logging.info(
          f'sample_name: {config["sample_name"]}; '
          f'observable set: {config["observable_set"]}; '
          f'training observables: {config["sally"]["training"]["observables"]}; '
          f'nsamples: {nsamples}'
      )

    ######### Outputting training variable index for training step ##########
    observable_dict=madminer_settings[5]

    for i_obs, obs_name in enumerate(observable_dict):
      logging.info(f'index: {i_obs}; name: {obs_name};') # this way we can easily see all the features 

    ########## Sample Augmentation ###########

    # object to create the augmented training samples
    sampler=SampleAugmenter( f'{config["main_dir"]}/{config["observable_set"]}/{config["sample_name"]}.h5')

    # Creates a set of training data (as many as the number of estimators) - centered around the SM

    
      
    _,_,_,eff_n_samples = sampler.sample_train_local(theta=sampling.benchmark('sm'),
                                        n_samples=int(nsamples),
                                        folder=f'{config["main_dir"]}/{config["observable_set"]}/testing_samples/sally',
                                        filename=f'test_score_{config["sample_name"]}',
                                        partition = "test",
                                        sample_only_from_closest_benchmark=False)
    


def evaluate_and_save_llr(config,args):
    
    if args.model == 'alices':
      
      model_name = f"alices_hidden_{config['alices']['training']['n_hidden']}_{config['alices']['training']['activation']}_alpha_{config['alices']['training']['alpha']}_epochs_{config['alices']['training']['n_epochs']}_bs_{config['alices']['training']['batch_size']}"

      model_path = f"{config['main_dir']}/{config['observable_set']}/models/{config['alices']['training']['training_samples_name']}/{config['alices']['training']['observables']}/{model_name}/alices_ensemble_{config['sample_name']}"

      thetas = np.load(f"{config['main_dir']}/{config['observable_set']}/testing_samples/alices_{config['alices']['testing']['prior_name']}/theta0_test_ratio_{config['sample_name']}.npy")
      x = np.load(f"{config['main_dir']}/{config['observable_set']}/testing_samples/alices_{config['alices']['testing']['prior_name']}/x_test_ratio_{config['sample_name']}.npy")

      for i in range(5):
        model_path_i = model_path + f"/estimator_{i}"
        alices = ParameterizedRatioEstimator()
        alices.load(model_path_i)
        log_r_hat = alices.evaluate_log_likelihood_ratio(x=x, theta = thetas,test_all_combinations = False)[0]
        
        os.makedirs(f"{config['main_dir']}/{config['observable_set']}/validation/{config['alices']['testing']['prior_name']}/{config['alices']['testing']['observables']}/{model_name}/{config['sample_name']}/",exist_ok=True)
        save_dir = f"{config['main_dir']}/{config['observable_set']}/validation/{config['alices']['testing']['prior_name']}/{config['alices']['testing']['observables']}/{model_name}/{config['sample_name']}/"

        np.savez(f"{save_dir}/estimator_{i}_log_r_hat.npz", log_r_hat = log_r_hat)
    
  

    if args.model== 'alice':
      
      model_name = f"alice_hidden_{config['alice']['training']['n_hidden']}_{config['alice']['training']['activation']}_epochs_{config['alice']['training']['n_epochs']}_bs_{config['alice']['training']['batch_size']}"

      model_path = f"{config['main_dir']}/{config['observable_set']}/models/{config['alice']['training']['training_samples_name']}/{config['alice']['training']['observables']}/{model_name}/alice_ensemble_{config['sample_name']}"

      thetas = np.load(f"{config['main_dir']}/{config['observable_set']}/testing_samples/alices_{config['alice']['testing']['prior_name']}/theta0_test_ratio_{config['sample_name']}.npy")
      x = np.load(f"{config['main_dir']}/{config['observable_set']}/testing_samples/alices_{config['alice']['testing']['prior_name']}/x_test_ratio_{config['sample_name']}.npy")


      for i in range(5):
        model_path_i = model_path + f"/estimator_{i}"
        alice = ParameterizedRatioEstimator()
        alice.load(model_path_i)
        log_r_hat = alice.evaluate_log_likelihood_ratio(x=x, theta = thetas,test_all_combinations = False)[0]
        
        os.makedirs(f"{config['main_dir']}/{config['observable_set']}/validation/{config['alice']['testing']['prior_name']}/{config['alice']['testing']['observables']}/{model_name}/{config['sample_name']}/",exist_ok=True)
        save_dir = f"{config['main_dir']}/{config['observable_set']}/validation/{config['alice']['testing']['prior_name']}/{config['alice']['testing']['observables']}/{model_name}/{config['sample_name']}/"

        np.savez(f"{save_dir}/estimator_{i}_log_r_hat.npz", log_r_hat = log_r_hat)


  
  
    
    if args.model== 'sally':
      sally = ScoreEstimator()

      model_name = f"sally_hidden_{config['sally']['training']['n_hidden']}_{config['sally']['training']['activation']}_epochs_{config['sally']['training']['n_epochs']}_bs_{config['sally']['training']['batch_size']}"

      model_path = f"{config['main_dir']}/{config['observable_set']}/models/sally/{config['sally']['training']['observables']}/{model_name}/sally_ensemble_{config['sample_name']}"

      thetas = np.load(f"{config['main_dir']}/{config['observable_set']}/testing_samples/sally/theta_test_score_{config['sample_name']}.npy")
      x = np.load(f"{config['main_dir']}/{config['observable_set']}/testing_samples/sally/x_test_score_{config['sample_name']}.npy")

      for i in range(5):
        model_path_i = model_path + f"/estimator_{i}"
        sally = ScoreEstimator()
        sally.load(model_path_i)
        log_r_hat = sally.evaluate_score(x=x, theta = thetas)
        
        os.makedirs(f"{config['main_dir']}/{config['observable_set']}/validation/{config['sally']['testing']['prior_name']}/{config['sally']['testing']['observables']}/{model_name}/{config['sample_name']}/",exist_ok=True)
        save_dir = f"{config['main_dir']}/{config['observable_set']}/validation/{config['sally']['testing']['prior_name']}/{config['sally']['testing']['observables']}/{model_name}/{config['sample_name']}/"

        np.savez(f"{save_dir}/estimator_{i}_t_hat.npz", log_r_hat = log_r_hat)



def simple_plot(config,args):
    
    if args.model == 'alices':
        
        model_name = f"alices_hidden_{config['alices']['training']['n_hidden']}_{config['alices']['training']['activation']}_alpha_{config['alices']['training']['alpha']}_epochs_{config['alices']['training']['n_epochs']}_bs_{config['alices']['training']['batch_size']}"

        joint_likelihood_ratio = np.load(f"{config['main_dir']}/{config['observable_set']}/testing_samples/alices_{config['alices']['testing']['prior_name']}/r_xz_test_ratio_{config['sample_name']}.npy")
        joint_score = np.load(f"{config['main_dir']}/{config['observable_set']}/testing_samples/alices_{config['alices']['testing']['prior_name']}/t_xz_test_ratio_{config['sample_name']}.npy")

        joint_likelihood_ratio_log = np.squeeze(np.log(joint_likelihood_ratio))

        predictions = []
        for i in range(5):
          log_r_hat = np.load(f"{config['main_dir']}/{config['observable_set']}/validation/{config['alices']['testing']['prior_name']}/{config['alices']['testing']['observables']}/{model_name}/{config['sample_name']}/estimator_{i}_log_r_hat.npz")['log_r_hat']
          predictions.append(log_r_hat)

        log_r_hat = np.average(predictions, axis=0)
        
        std_deviation = -2 * np.std(predictions, axis=0)
        
        mse = np.mean((joint_likelihood_ratio_log - log_r_hat)**2)

        fig=plt.figure()
        
        plt.errorbar( -2 * joint_likelihood_ratio_log,-2 * log_r_hat, yerr=std_deviation, fmt='o',markersize=5, color='blue',ecolor='black', alpha = 0.5, linewidth=1, label = "ALICES")

        # Calculate min and max values
        max_val = max(np.max(np.abs(-2 * log_r_hat)), np.max(np.abs(-2 * joint_likelihood_ratio_log)))
        
        # Add diagonal line
        plt.plot([-max_val-1 , max_val+1], [-max_val -1,max_val+1 ], linestyle="--", color="grey", lw=1)

        # Set limits for x and y axes
        plt.xlim(-max_val-1 , max_val+1)
        plt.ylim(-max_val -1, max_val+1)

        plt.xlabel(r'True log likelihood ratio: log r(x)')
        plt.ylabel(r'Estimated log likelihood ratio: log $\hat{r}(x)$ ')
        plt.legend(title =f'MSE: {mse:.4f}')
        plt.tight_layout()
        
      
        os.makedirs(f"{config['plot_dir']}/{config['observable_set']}/validation/{config['alices']['testing']['prior_name']}/{config['alices']['testing']['observables']}/{model_name}/", exist_ok=True)

        plt.savefig(f"{config['plot_dir']}/{config['observable_set']}/validation/{config['alices']['testing']['prior_name']}/{config['alices']['testing']['observables']}/{model_name}/{args.model}_validation_llr_simple_{config['sample_name']}.pdf")
            
    

    if args.model== 'alice':
        
        model_name = f"alice_hidden_{config['alice']['training']['n_hidden']}_{config['alice']['training']['activation']}_epochs_{config['alice']['training']['n_epochs']}_bs_{config['alice']['training']['batch_size']}"

        joint_likelihood_ratio = np.load(f"{config['main_dir']}/{config['observable_set']}/testing_samples/alices_{config['alice']['testing']['prior_name']}/r_xz_test_ratio_{config['sample_name']}.npy")
        joint_score = np.load(f"{config['main_dir']}/{config['observable_set']}/testing_samples/alices_{config['alice']['testing']['prior_name']}/t_xz_test_ratio_{config['sample_name']}.npy")

        joint_likelihood_ratio_log = np.squeeze(np.log(joint_likelihood_ratio))

        predictions = []
        for i in range(5):
          log_r_hat = np.load(f"{config['main_dir']}/{config['observable_set']}/validation/{config['alice']['testing']['prior_name']}/{config['alice']['testing']['observables']}/{model_name}/{config['sample_name']}/estimator_{i}_log_r_hat.npz")['log_r_hat']
          predictions.append(log_r_hat)

        log_r_hat = np.average(predictions, axis=0)
        
        std_deviation = -2 * np.std(predictions, axis=0)
        
        mse = np.mean((joint_likelihood_ratio_log - log_r_hat)**2)
        fig=plt.figure()
        
        plt.errorbar( -2 * joint_likelihood_ratio_log,-2 * log_r_hat, yerr=std_deviation, fmt='o',markersize=5, color='darkgreen',ecolor='black', alpha = 0.5, linewidth=1, label = "ALICE")

        # Calculate min and max values
        max_val = max(np.max(np.abs(-2 * log_r_hat)), np.max(np.abs(-2 * joint_likelihood_ratio_log)))
        
        # Add diagonal line
        plt.plot([-max_val-1 , max_val+1], [-max_val -1,max_val+1 ], linestyle="--", color="grey", lw=1)

        # Set limits for x and y axes
        plt.xlim(-max_val-1 , max_val+1)
        plt.ylim(-max_val -1, max_val+1)

        plt.xlabel(r'True log likelihood ratio: log r(x)')
        plt.ylabel(r'Estimated log likelihood ratio: log $\hat{r}(x)$ ')
        plt.legend(title = f'MSE: {mse:.4f}')
        plt.tight_layout()
        
        os.makedirs(f"{config['plot_dir']}/{config['observable_set']}/validation/{config['alice']['testing']['prior_name']}/{config['alice']['testing']['observables']}/{model_name}/", exist_ok=True)

        plt.savefig(f"{config['plot_dir']}/{config['observable_set']}/validation/{config['alice']['testing']['prior_name']}/{config['alice']['testing']['observables']}/{model_name}/{args.model}_validation_llr_simple_{config['sample_name']}.pdf")
            
    
    
    if args.model== 'sally':

        model_name = f"sally_hidden_{config['sally']['training']['n_hidden']}_{config['sally']['training']['activation']}_epochs_{config['sally']['training']['n_epochs']}_bs_{config['sally']['training']['batch_size']}"
        
        joint_score = np.load(f"{config['main_dir']}/{config['observable_set']}/testing_samples/{config['sally']['testing']['prior_name']}/t_xz_test_score_{config['sample_name']}.npy")
        joint_score = np.squeeze(joint_score)

        predictions = []
        for i in range(5):
          t_hat = np.load(f"{config['main_dir']}/{config['observable_set']}/validation/{config['sally']['testing']['prior_name']}/{config['sally']['testing']['observables']}/{model_name}/{config['sample_name']}/estimator_{i}_t_hat.npz")['log_r_hat']
          predictions.append(t_hat)

        t_hat = np.average(predictions, axis=0)
        t_hat = np.squeeze(t_hat)
        
        std_deviation = -2 * np.std(predictions, axis=0)
        std_deviation= np.squeeze(std_deviation)
        mse = np.mean((joint_score - t_hat)**2)
        fig=plt.figure()
        
        plt.errorbar( -2 * joint_score,-2 * t_hat, yerr=std_deviation, fmt='o',markersize=5, color='mediumvioletred',ecolor='black', alpha = 0.5, linewidth=1, label = "SALLY")

        # Calculate min and max values
        max_val = max(np.max(np.abs(-2 * t_hat)), np.max(np.abs(-2 * joint_score)))
        
        # Add diagonal line
        plt.plot([-max_val-1 , max_val+1], [-max_val -1,max_val+1 ], linestyle="--", color="grey", lw=1)

        # Set limits for x and y axes
        plt.xlim(-max_val-1 , max_val+1)
        plt.ylim(-max_val -1, max_val+1)

        #plt.title(f'MSE: {mse:.4f}')
        plt.xlabel(r'True score: t(x)')
        plt.ylabel(r'Estimated score: $\hat{t}(x)$')
        plt.legend(title = f'MSE: {mse:.4f}')
        plt.tight_layout()
        
        os.makedirs(f"{config['plot_dir']}/{config['observable_set']}/validation/sally/{config['sally']['testing']['observables']}/{model_name}/", exist_ok=True)

        plt.savefig(f"{config['plot_dir']}/{config['observable_set']}/validation/sally/{config['sally']['testing']['observables']}/{model_name}/{args.model}_validation_score_simple_{config['sample_name']}.pdf")
        
    
   
def color_plot(config,args):
    
    if args.model == 'alices':
        
        
        model_name = f"alices_hidden_{config['alices']['training']['n_hidden']}_{config['alices']['training']['activation']}_alpha_{config['alices']['training']['alpha']}_epochs_{config['alices']['training']['n_epochs']}_bs_{config['alices']['training']['batch_size']}"

        joint_likelihood_ratio = np.load(f"{config['main_dir']}/{config['observable_set']}/testing_samples/alices_{config['alices']['testing']['prior_name']}/r_xz_test_ratio_{config['sample_name']}.npy")
        joint_score = np.load(f"{config['main_dir']}/{config['observable_set']}/testing_samples/alices_{config['alices']['testing']['prior_name']}/t_xz_test_ratio_{config['sample_name']}.npy")

        joint_likelihood_ratio_log = np.squeeze(np.log(joint_likelihood_ratio))

        predictions = []
        for i in range(5):
          log_r_hat = np.load(f"{config['main_dir']}/{config['observable_set']}/validation/{config['alices']['testing']['prior_name']}/{config['alices']['testing']['observables']}/{model_name}/{config['sample_name']}/estimator_{i}_log_r_hat.npz")['log_r_hat']
          predictions.append(log_r_hat)

        log_r_hat = np.average(predictions, axis=0)
        mse = np.mean((joint_likelihood_ratio_log - log_r_hat)**2)

        
        std_deviation =  np.std(2*predictions, axis=0)
        x = -2*joint_likelihood_ratio_log
        y= -2*log_r_hat

        residuals = (x - y) / std_deviation

        bins = [70,70]
        ranges = [[-10, 10], [-10, 10]]

        norm=colors.LogNorm(vmin = 10e-2, vmax = 1)
        cmap=colors.LinearSegmentedColormap.from_list("", ['#111A44',"#009988",'whitesmoke'])
        

        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(7, 7))

        hist = ax1.hist2d(-2*joint_likelihood_ratio_log, -2*log_r_hat, bins=bins, range=ranges, norm=norm,cmap=cmap ,density=True)

        plt.subplots_adjust(right=0.8, hspace=0.05)


        cbar_ax = fig.add_axes([0.82, 0.28, 0.03, 0.6])  # [left, bottom, width, height]
        cbar = fig.colorbar(hist[3], cax=cbar_ax, orientation='vertical')
        cbar.set_label('Frequency', fontsize=13)


        # Calculate min and max values
        max_val = max(np.max(np.abs(-2 * log_r_hat)), np.max(np.abs(-2 * joint_likelihood_ratio_log)))
        
        # Add diagonal line
        ax1.plot([-max_val-1 , max_val+1], [-max_val -1,max_val+1 ], linestyle="--", color="grey", lw=1)

        # Set limits for x and y axes
        ax1.set_xlim(-max_val-1 , max_val+1)
        ax1.set_ylim(-max_val -1, max_val+1)
        legend = ax1.legend(title=f'\n  ALICES\n  N = 1000\n  MSE: {mse:.4f}', frameon=False)
        title_text = legend.get_title()

        # Set the font size
        title_text.set_fontsize(12)  # Adjust the size as needed

        # Remove ticks from the first subplot
        ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        # Adjust spacing between subplots
        plt.subplots_adjust(hspace=0.05)  # Change the value to reduce or increase the gap

        value = 2

        # Get the RGBA color corresponding to the value
        rgba_color = cmap(value)
        
        
        ax1.set_ylabel(r'Estimated log likelihood ratio: log $\hat{r}(x)$ ', fontsize=13)

        ax2.scatter(-2 * joint_likelihood_ratio_log, residuals, marker='o', s=10, color=rgba_color)
        ax2.set_ylabel(r'$ \frac{\mathrm{log}\  r(x)- \mathrm{log} \ \hat{r}(x)}{\sigma_{\hat{r}(x)}}$', fontsize=15, labelpad=4)

        ax2.set_xlabel(r'True log likelihood ratio: log r(x)', fontsize=13, labelpad=10)
        ax2.axhline(0, color='grey', linestyle='--',lw =1)  # Add horizontal line at y = 0
        ax2.set_xlim(-max_val-1 , max_val+1)
        
      
        os.makedirs(f"{config['plot_dir']}/{config['observable_set']}/validation/{config['alices']['testing']['prior_name']}/{config['alices']['testing']['observables']}/{model_name}/", exist_ok=True)

        plt.savefig(f"{config['plot_dir']}/{config['observable_set']}/validation/{config['alices']['testing']['prior_name']}/{config['alices']['testing']['observables']}/{model_name}/{args.model}_validation_llr_color_{config['sample_name']}.pdf")
            
    

    if args.model== 'alice':
         
        model_name = f"alice_hidden_{config['alice']['training']['n_hidden']}_{config['alice']['training']['activation']}_epochs_{config['alice']['training']['n_epochs']}_bs_{config['alice']['training']['batch_size']}"

        joint_likelihood_ratio = np.load(f"{config['main_dir']}/{config['observable_set']}/testing_samples/alices_{config['alice']['testing']['prior_name']}/r_xz_test_ratio_{config['sample_name']}.npy")
        joint_score = np.load(f"{config['main_dir']}/{config['observable_set']}/testing_samples/alices_{config['alice']['testing']['prior_name']}/t_xz_test_ratio_{config['sample_name']}.npy")

        joint_likelihood_ratio_log = np.squeeze(np.log(joint_likelihood_ratio))

        predictions = []
        for i in range(5):
          log_r_hat = np.load(f"{config['main_dir']}/{config['observable_set']}/validation/{config['alice']['testing']['prior_name']}/{config['alice']['testing']['observables']}/{model_name}/{config['sample_name']}/estimator_{i}_log_r_hat.npz")['log_r_hat']
          predictions.append(log_r_hat)

        log_r_hat = np.average(predictions, axis=0)
        mse = np.mean((joint_likelihood_ratio_log - log_r_hat)**2)

        
        std_deviation =  np.std(2*predictions, axis=0)
        x = -2*joint_likelihood_ratio_log
        y= -2*log_r_hat

        residuals = (x - y) / std_deviation

        bins = [70,70]
        ranges = [[-10, 10], [-10, 10]]

        norm=colors.LogNorm(vmin = 10e-2, vmax = 1)
        cmap=colors.LinearSegmentedColormap.from_list("", ['#111A44',"#009988",'whitesmoke'])
        

        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(7, 7))

        hist = ax1.hist2d(-2*joint_likelihood_ratio_log, -2*log_r_hat, bins=bins, range=ranges, norm=norm,cmap=cmap ,density=True)

        plt.subplots_adjust(right=0.8, hspace=0.05)


        cbar_ax = fig.add_axes([0.82, 0.28, 0.03, 0.6])  # [left, bottom, width, height]
        cbar = fig.colorbar(hist[3], cax=cbar_ax, orientation='vertical')
        cbar.set_label('Frequency', fontsize=13)


        # Calculate min and max values
        max_val = max(np.max(np.abs(-2 * log_r_hat)), np.max(np.abs(-2 * joint_likelihood_ratio_log)))
        
        # Add diagonal line
        ax1.plot([-max_val-1 , max_val+1], [-max_val -1,max_val+1 ], linestyle="--", color="grey", lw=1)

        # Set limits for x and y axes
        ax1.set_xlim(-max_val-1 , max_val+1)
        ax1.set_ylim(-max_val -1, max_val+1)
        legend = ax1.legend(title=f'\n  ALICE\n  N = 1000\n  MSE: {mse:.4f}', frameon=False)
        title_text = legend.get_title()

        # Set the font size
        title_text.set_fontsize(12)  # Adjust the size as needed

        # Remove ticks from the first subplot
        ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        # Adjust spacing between subplots
        plt.subplots_adjust(hspace=0.05)  # Change the value to reduce or increase the gap

        value = 2

        # Get the RGBA color corresponding to the value
        rgba_color = cmap(value)
        
        
        ax1.set_ylabel(r'Estimated log likelihood ratio: log $\hat{r}(x)$ ', fontsize=13)

        ax2.scatter(-2 * joint_likelihood_ratio_log, residuals, marker='o', s=10, color=rgba_color)
        ax2.set_ylabel(r'$ \frac{\mathrm{log}\  r(x)- \mathrm{log} \ \hat{r}(x)}{\sigma_{\hat{r}(x)}}$', fontsize=15, labelpad=4)

        ax2.set_xlabel(r'True log likelihood ratio: log r(x)', fontsize=13, labelpad=10)
        ax2.axhline(0, color='grey', linestyle='--',lw =1)  # Add horizontal line at y = 0
        ax2.set_xlim(-max_val-1 , max_val+1)
        
      
        os.makedirs(f"{config['plot_dir']}/{config['observable_set']}/validation/{config['alice']['testing']['prior_name']}/{config['alice']['testing']['observables']}/{model_name}/", exist_ok=True)

        plt.savefig(f"{config['plot_dir']}/{config['observable_set']}/validation/{config['alice']['testing']['prior_name']}/{config['alice']['testing']['observables']}/{model_name}/{args.model}_validation_llr_color_{config['sample_name']}.pdf")
            
    
    
    if args.model== 'sally':


        model_name = f"sally_hidden_{config['sally']['training']['n_hidden']}_{config['sally']['training']['activation']}_epochs_{config['sally']['training']['n_epochs']}_bs_{config['sally']['training']['batch_size']}"

        joint_score = np.load(f"{config['main_dir']}/{config['observable_set']}/testing_samples/{config['sally']['testing']['prior_name']}/t_xz_test_score_{config['sample_name']}.npy")
        joint_score = np.squeeze(joint_score)

        predictions = []
        for i in range(5):
          t_hat = np.load(f"{config['main_dir']}/{config['observable_set']}/validation/{config['sally']['testing']['prior_name']}/{config['sally']['testing']['observables']}/{model_name}/{config['sample_name']}/estimator_{i}_t_hat.npz")['log_r_hat']
          predictions.append(t_hat)

        t_hat = np.average(predictions, axis=0)
        t_hat = np.squeeze(t_hat)
        
        std_deviation = -2 * np.std(predictions, axis=0)
        std_deviation= np.squeeze(std_deviation)
        mse = np.mean((joint_score - t_hat)**2)
     
        x = -2*joint_score
        y= -2*t_hat

        residuals = (x - y) / std_deviation

        bins = [70,70]
        ranges = [[-10, 10], [-10, 10]]

        norm=colors.LogNorm(vmin = 10e-2, vmax = 1)
        cmap=colors.LinearSegmentedColormap.from_list("", ['#111A44',"#009988",'whitesmoke'])
        

        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(7, 7))

        hist = ax1.hist2d(-2*joint_score, -2*t_hat, bins=bins, range=ranges, norm=norm,cmap=cmap ,density=True)

        plt.subplots_adjust(right=0.8, hspace=0.05)


        cbar_ax = fig.add_axes([0.82, 0.28, 0.03, 0.6])  # [left, bottom, width, height]
        cbar = fig.colorbar(hist[3], cax=cbar_ax, orientation='vertical')
        cbar.set_label('Frequency', fontsize=13)


        # Calculate min and max values
        max_val = max(np.max(np.abs(-2 * t_hat)), np.max(np.abs(-2 * joint_score)))
        
        # Add diagonal line
        ax1.plot([-max_val-1 , max_val+1], [-max_val -1,max_val+1 ], linestyle="--", color="grey", lw=1)

        # Set limits for x and y axes
        ax1.set_xlim(-max_val-1 , max_val+1)
        ax1.set_ylim(-max_val -1, max_val+1)
        legend = ax1.legend(title=f'\n  SALLY\n  N = 1000\n  MSE: {mse:.4f}', frameon=False)
        title_text = legend.get_title()

        # Set the font size
        title_text.set_fontsize(12)  # Adjust the size as needed

        # Remove ticks from the first subplot
        ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        # Adjust spacing between subplots
        plt.subplots_adjust(hspace=0.05)  # Change the value to reduce or increase the gap

        value = 2

        # Get the RGBA color corresponding to the value
        rgba_color = cmap(value)
        
        
        ax1.set_ylabel(r'Estimated score: $\hat{t}(x)$', fontsize=13)

        ax2.scatter(-2 * joint_score, residuals, marker='o', s=10, color=rgba_color)
        ax2.set_ylabel(r'$ \frac{\mathrm{log}\  t(x)- \mathrm{log} \ \hat{t}(x)}{\sigma_{\hat{t}(x)}}$', fontsize=15, labelpad=4)

        ax2.set_xlabel(r'True score: t(x)', fontsize=13, labelpad=10)
        ax2.axhline(0, color='grey', linestyle='--',lw =1)  # Add horizontal line at y = 0
        ax2.set_xlim(-max_val-1 , max_val+1)

      
        os.makedirs(f"{config['plot_dir']}/{config['observable_set']}/validation/{config['sally']['testing']['prior_name']}/{config['sally']['testing']['observables']}/{model_name}/", exist_ok=True)

        plt.savefig(f"{config['plot_dir']}/{config['observable_set']}/validation/{config['sally']['testing']['prior_name']}/{config['sally']['testing']['observables']}/{model_name}/{args.model}_validation_llr_color_{config['sample_name']}.pdf")
         


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Computes distributions of different variables for signal and backgrounds.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--config_file', help='Path to the YAML configuration file', default='../config.yaml')

    parser.add_argument('--augment_test',help="creates testing samples with the ALICES or SALLY method;",action='store_true',  default =False)

    parser.add_argument('--evaluate',help="Evaluates and saves llr with ALICE, ALICES or SALLY ;",action='store_true',  default =False)
    
    parser.add_argument('-m','--model',help='which model to plot validation plot: SALLY, ALICE, ALICES' , choices=['sally','alice','alices'], default='alices')

    parser.add_argument('--simple_plot',help='wheter to plot the simple validation plot (without color map, only predicted vs actual)',action='store_true',default=False )

    parser.add_argument('--color_plot',help='wheter to plot the colored density validation plot ',action='store_true',default=False )
    
    args=parser.parse_args()
    
    # Read configuration parameters from the YAML file
    with open(args.config_file, 'r') as config_file:
        config = yaml.safe_load(config_file)

                
    if args.augment_test:
        augment_test(config)

    if args.evaluate:
        evaluate_and_save_llr(config,args)
        

    if args.simple_plot:
        simple_plot(config,args)

    if args.color_plot:
        color_plot(config,args)

    