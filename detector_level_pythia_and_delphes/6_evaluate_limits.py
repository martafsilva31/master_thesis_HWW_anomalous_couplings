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
  #   hist_vars=hist_vars if 'histo' == config['limits']['mode'] else None,
  #   hist_bins=hist_bins,
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
#   hist_vars=hist_vars if 'histo' == config['limits']['mode'] else None,
#   hist_bins=hist_bins,
    luminosity=config['limits']['lumi']*1000.0,
    return_asimov=True,test_split=0.2,n_histo_toys=None,
    grid_ranges=[config['limits']['grid_ranges']],grid_resolutions=[config['limits']['grid_resolutions']])

    os.makedirs(f"{config['main_dir']}/llr_fits/{config['limits']['prior']}/{config['limits']['observables']}/{config['limits']['model']}/{config['sample_name']}/range_{config['limits']['grid_ranges']}_resolutions_{config['limits']['grid_resolutions']}",exist_ok=True)
    save_dir = f"{config['main_dir']}/llr_fits/{config['limits']['prior']}/{config['limits']['observables']}/{config['limits']['model']}/{config['sample_name']}/range_{config['limits']['grid_ranges']}_resolutions_{config['limits']['grid_resolutions']}"

    np.savez(f"{save_dir}/ensemble_data.npz", parameter_grid=parameter_grid, p_values=p_values,
                index_best_point=index_best_point, llr_kin=llr_kin, llr_rate=llr_rate)

def plot_llr_individual(config):

    plt.figure(figsize=(7,6))

    load_dir = f"{config['main_dir']}/llr_fits/{config['limits']['prior']}/{config['limits']['observables']}/{config['limits']['model']}/{config['sample_name']}/range_{config['limits']['grid_ranges']}_resolutions_{config['limits']['grid_resolutions']}"
    fig_name = f"llr_fit_individual_{config['sample_name']}_range_{config['limits']['grid_ranges']}_resolutions_{config['limits']['grid_resolutions']}"

    os.makedirs(f"{config['plot_dir']}/llr_fits/{config['limits']['prior']}/{config['limits']['observables']}/{config['limits']['model']}/",exist_ok=True)
    save_dir = f"{config['plot_dir']}/llr_fits/{config['limits']['prior']}/{config['limits']['observables']}/{config['limits']['model']}/"
    all_parameter_points_to_plot = [] 



    for i in range(5):
    
      estimator_number = i + 1
      dir =  load_dir + f"/estimator_{i}_data.npz"

      data = np.load(dir)

      parameter_grid = data['parameter_grid']
      llr_kin = data['llr_kin']
      llr_rate = data['llr_rate']
      index_best_point = data['index_best_point']

      rescaled_log_r = llr_kin+llr_rate
      rescaled_log_r = -2.0*(rescaled_log_r[:] - rescaled_log_r[index_best_point])    

      if config['limits']['method'] == 'sally':
         title = 'SALLY'

      if config['limits']['method'] == 'alices':
         title = 'ALICES'

      if config['limits']['method'] == 'alice':
         title = 'ALICE'


      if args.CL: 
         lw = 1
      else:
         lw = 0.5

      plt.plot(parameter_grid,rescaled_log_r,linewidth=lw, label=f"Estimator {estimator_number}")    

      if args.CL: 
            # Store parameter points for this estimator
            parameter_points_to_plot = [parameter_grid[i] for i, llr_val in enumerate(rescaled_log_r) if llr_val < 3.28]
            all_parameter_points_to_plot.extend(parameter_points_to_plot)

    if args.CL:

      fig_name = f"{fig_name}_CL.pdf"
      # Set x-axis and y-axis limits using all parameter points
      abs_parameter_points = [abs(point) for point in all_parameter_points_to_plot]
      plt.axhline(y=3.84,linestyle='-.',linewidth=lw,color='grey',label='95%CL')
      plt.axhline(y=1.0,linestyle=':',linewidth=lw,color='grey',label='68%CL')
      plt.xlim(-max(abs_parameter_points), max(abs_parameter_points))
      plt.ylim(-1, 5)

    else:
      fig_name = f"{fig_name}.pdf"

    plt.xlabel(r"$c_{H\tildeW}$")
    plt.ylabel(r"$q'(\theta)$") 
    plt.legend( title=  title)
    plt.savefig(f"{save_dir}/{fig_name}")


def calculate_std(config):
    
    load_dir = f"{config['main_dir']}/llr_fits/{config['limits']['prior']}/{config['limits']['observables']}/{config['limits']['model']}/{config['sample_name']}/range_{config['limits']['grid_ranges']}_resolutions_{config['limits']['grid_resolutions']}"
    

    predictions = []

    for i in range(5):
    
      estimator_number = i + 1
      dir =  load_dir + f"/estimator_{i}_data.npz"

      data = np.load(dir)

      parameter_grid = data['parameter_grid']
      llr_kin = data['llr_kin']
      llr_rate = data['llr_rate']
      index_best_point = data['index_best_point']

      rescaled_log_r = llr_kin+llr_rate
      rescaled_log_r = -2.0*(rescaled_log_r[:] - rescaled_log_r[index_best_point])    
      predictions.append(rescaled_log_r)

    std_deviation = np.std(predictions, axis=0)

    return std_deviation

def plot_llr_ensemble(config):
    
    load_dir = f"{config['main_dir']}/llr_fits/{config['limits']['prior']}/{config['limits']['observables']}/{config['limits']['model']}/{config['sample_name']}/range_{config['limits']['grid_ranges']}_resolutions_{config['limits']['grid_resolutions']}"

    fig_name = f"llr_fit_ensemble_{config['sample_name']}_range_{config['limits']['grid_ranges']}_resolutions_{config['limits']['grid_resolutions']}"

    os.makedirs(f"{config['plot_dir']}/llr_fits/{config['limits']['prior']}/{config['limits']['observables']}/{config['limits']['model']}/",exist_ok=True)
    save_dir = f"{config['plot_dir']}/llr_fits/{config['limits']['prior']}/{config['limits']['observables']}/{config['limits']['model']}/"
    
    data = np.load(f"{load_dir}/ensemble_data.npz")

    parameter_grid = data['parameter_grid']
    llr_kin = data['llr_kin']
    llr_rate = data['llr_rate']
    index_best_point = data['index_best_point']

    if config['limits']['method'] == 'sally':
      title = 'SALLY'
      color = 'mediumvioletred'

    if config['limits']['method'] == 'alices':
      title = 'ALICES'
      color = 'blue'

    if config['limits']['method'] == 'alice':
      title = 'ALICE'
      color = 'darkgreen'


    plt.figure(figsize=(7, 6))

    rescaled_log_r = llr_kin+llr_rate
    rescaled_log_r = -2.0*(rescaled_log_r[:] - rescaled_log_r[index_best_point])    

    std = calculate_std(config)
    std = np.squeeze(std)
    parameter_grid = np.squeeze(parameter_grid)
 

    plt.plot(parameter_grid,rescaled_log_r,lw=1.5, color = color, label = title)    
    plt.fill_between(parameter_grid, rescaled_log_r - std, rescaled_log_r + std, color=color, alpha=0.1)



    if args.CL:
      fig_name = f"{fig_name}_CL.pdf"
      # Set x-axis and y-axis limits using all parameter points
      parameter_points_to_plot=[parameter_grid[i] for i,llr_val in enumerate(rescaled_log_r) if llr_val<3.28]
      plt.axhline(y=3.84,lw = 1.5, linestyle='-.',color='grey',label='95%CL')
      plt.axhline(y=1.0,lw=1.5, linestyle=':',color='grey',label='68%CL')
      plt.xlim(-max(abs(parameter_points_to_plot[0]),abs(parameter_points_to_plot[-1])),+max(abs(parameter_points_to_plot[0]),abs(parameter_points_to_plot[-1])))
      plt.ylim(-1, 5)

    else:
      fig_name = f"{fig_name}.pdf"

    plt.xlabel(r"$c_{H\tildeW}$")
    plt.ylabel(r"$q'(\theta)$") 
    plt.legend()
    plt.savefig(f"{save_dir}/{fig_name}")


def extract_limits_single_parameter(parameter_grid,p_values,index_central_point):
  
  n_parameters=len(parameter_grid[0])
  list_points_inside_68_cl=[parameter_grid[i][0] if n_parameters==1 else parameter_grid[i] for i in range(len(parameter_grid)) if p_values[i] > 0.32 ]
  list_points_inside_95_cl=[parameter_grid[i][0] if n_parameters==1 else parameter_grid[i] for i in range(len(parameter_grid)) if p_values[i] > 0.05 ]  
  return parameter_grid[index_central_point],[round(list_points_inside_68_cl[0],5),round(list_points_inside_68_cl[-1],5)],[round(list_points_inside_95_cl[0],5),round(list_points_inside_95_cl[-1],5)]




def save_limits_ensemble(config):
   
    os.makedirs(f"{config['plot_dir']}/llr_fits/",exist_ok=True)

    log_file_path = f"{config['plot_dir']}/llr_fits/limits_ensemble.csv"
    
    # Check if the file exists
    if not os.path.exists(log_file_path):
        # If the file doesn't exist, create it and write the header
        with open(log_file_path, 'w') as log_file:
            log_file.write("sample name, prior name, observables, model name, grid range, resolutions, central value, 68% CL, 95% CL\n")


    with open(log_file_path, 'a') as log_file:
        load_dir_ensemble = f"{config['main_dir']}/llr_fits/{config['limits']['prior']}/{config['limits']['observables']}/{config['limits']['model']}/{config['sample_name']}/range_{config['limits']['grid_ranges']}_resolutions_{config['limits']['grid_resolutions']}"

        data_ensemble = np.load(f"{load_dir_ensemble}/ensemble_data.npz")

        parameter_grid = data_ensemble['parameter_grid']
        index_best_point = data_ensemble['index_best_point']
        p_values = data_ensemble['p_values']

        central_value,cl_68,cl_95=extract_limits_single_parameter(parameter_grid,p_values,index_best_point)

        log_file.write(f"{config['sample_name']}, {config['limits']['prior']}, {config['limits']['model']}, {config['limits']['observables']},  [{str(config['limits']['grid_ranges'][0])} {str(config['limits']['grid_ranges'][1])}], {config['limits']['grid_resolutions']}, {str(central_value[0])}, {str(cl_68).replace(',',' ')}, {str(cl_95).replace(',',' ')} \n") 

    log_file.close()

def save_limits_individual(config):

    os.makedirs(f"{config['plot_dir']}/llr_fits/",exist_ok=True)
   
    log_file_path = f"{config['plot_dir']}/llr_fits/limits_individual.csv"
    
    # Check if the file exists
    if not os.path.exists(log_file_path):
        # If the file doesn't exist, create it and write the header
        with open(log_file_path, 'w') as log_file:
           log_file.write("sample name, prior name, observables, model name, grid range , resolutions, estimator, central value, 68% CL, 95% CL \n")

    with open(log_file_path, 'a') as log_file:
      load_dir_individual = f"{config['main_dir']}/llr_fits/{config['limits']['prior']}/{config['limits']['observables']}/{config['limits']['model']}/{config['sample_name']}/range_{config['limits']['grid_ranges']}_resolutions_{config['limits']['grid_resolutions']}"
    

      for i in range(5):
      
        estimator_number = i + 1
        dir =  load_dir_individual + f"/estimator_{i}_data.npz"

        data = np.load(dir)

        parameter_grid = data['parameter_grid']
        p_values = data['p_values']
        index_best_point = data['index_best_point']
        central_value,cl_68,cl_95=extract_limits_single_parameter(parameter_grid,p_values,index_best_point)
          
        log_file.write(f"{config['sample_name']}, {config['limits']['prior']}, {config['limits']['model']}, {config['limits']['observables']},  [{str(config['limits']['grid_ranges'][0])} {str(config['limits']['grid_ranges'][1])}], {config['limits']['grid_resolutions']}, {estimator_number}, {str(central_value[0])}, {str(cl_68).replace(',',' ')}, {str(cl_95).replace(',',' ')} \n") 

    log_file.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plots log likelihood ratio evaluate for all estimators or with and ensemble', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config_file', help='Path to the YAML configuration file', default='config_CP_even.yaml')

    parser.add_argument('--evaluate', help='Evaluates and saves llr for each estimator (individual) or using an ensemble (ensemble)', choices=['individual', 'ensemble'])

    parser.add_argument('--plot', help='Plots llr for each estimator (individual) or using an ensemble (ensemble)', choices=['individual', 'ensemble'])

    parser.add_argument('--CL', help='Wether to plot the CLs lines', action='store_true')

    parser.add_argument('--limits', help='Calculates and saves limits for each estimator (individual) or using an ensemble (ensemble)', choices=['individual', 'ensemble'])


    args = parser.parse_args()

    # Read configuration parameters from the YAML file
    with open(args.config_file, 'r') as config_file:
      config = yaml.safe_load(config_file)

    os.makedirs(f"{config['plot_dir']}/llr_fits/",exist_ok=True)

    if args.evaluate == 'individual':
        evaluate_and_save_llr_individual(config)

    if args.evaluate == 'ensemble':
        evaluate_and_save_llr_ensemble(config)
    
    if args.plot == 'individual':
        plot_llr_individual(config)

    if args.plot == 'ensemble':
        plot_llr_ensemble(config)

    if args.limits == 'individual':
        save_limits_individual(config)

    if args.limits == 'ensemble':
        save_limits_ensemble(config)