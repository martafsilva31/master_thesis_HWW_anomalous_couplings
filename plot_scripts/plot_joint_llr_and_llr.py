import numpy as np
import logging
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import os
from madminer.ml import ParameterizedRatioEstimator,ScoreEstimator, Ensemble
from scipy.stats import gaussian_kde
from madminer.ml import  Ensemble
from madminer.plotting import plot_histograms
from madminer.sampling import SampleAugmenter
from madminer.limits import AsymptoticLimits
import logging
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from madminer.sampling import SampleAugmenter
from madminer import sampling
from madminer.ml import ParameterizedRatioEstimator
import logging
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from madminer.sampling import SampleAugmenter
from madminer import sampling
from madminer.ml import ParameterizedRatioEstimator

logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
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


# filename = "/lstore/titan/martafsilva/master_thesis/master_thesis_output/detector_level_pythia_and_delphes/wh_signalWithBSM.h5"
# # # thetas = np.load('/lstore/titan/martafsilva/master_thesis/master_thesis_output/detector_level_pythia_and_delphes/training_samples/alices_alices_gaussian_prior_0_03_10000_thetas_5000000_samples_CP_even/theta0_train_ratio_wh_signalWithBSM_CP_even_4.npy') 
# # # joint_llr = np.load('/lstore/titan/martafsilva/master_thesis/master_thesis_output/detector_level_pythia_and_delphes/training_samples/alices_alices_gaussian_prior_0_03_10000_thetas_5000000_samples_CP_even/r_xz_train_ratio_wh_signalWithBSM_CP_even_4.npy') 
# # # x = np.load('/lstore/titan/martafsilva/master_thesis/master_thesis_output/detector_level_pythia_and_delphes/training_samples/alices_alices_gaussian_prior_0_03_10000_thetas_5000000_samples_CP_even/x_train_ratio_wh_signalWithBSM_CP_even_4.npy')

# sampler = SampleAugmenter(f'{filename}')

# _ = sampler.sample_test(
#     theta=sampling.benchmark('sm'),
#     n_samples=1000,
#     folder='./lstore/titan/martafsilva/master_thesis/master_thesis_code/detector_level_pythia_and_delphes/test_samples',
#     filename='test_CP_even_signal_only'
# )

model_path = '/lstore/titan/martafsilva/master_thesis/master_thesis_output/detector_level_pythia_and_delphes/models/alices_gaussian_prior_0_0.4_10000_thetas_5000000_samples/kinematic_only/alices_hidden_[50]_relu_alpha_5_epochs_100_bs_128/alices_ensemble_wh_signalWithBSM'
alices = Ensemble()
alices.load(model_path)
#theta_each = np.linspace(-1.0,1.0,50)
theta_each = np.linspace(-0.1,0.1,50)
theta_grid = np.array([theta_each]).T

log_r_hat, _ = alices.evaluate_log_likelihood_ratio(
    theta=theta_grid,
    x='./lstore/titan/martafsilva/master_thesis/master_thesis_code/detector_level_pythia_and_delphes/test_samples/x_test_CP_odd_signal_only.npy',
    evaluate_score=False,
    test_all_combinations = True
)
expected_llr =log_r_hat
expected_llr = np.mean(log_r_hat,axis=0)

# Create histogram
#plt.hist(log_r_hat, bins=30, edgecolor='black', density=True, alpha=0.7)
bins = np.linspace(-0.03,0.03,30)
plt.hist(expected_llr, bins=bins, histtype='step', color='teal', density=True,linewidth=2)
plt.hist(expected_llr, bins=bins, color='teal', density=True, alpha=0.2)
plt.plot(-0.5,0.5,label = "Signal Only", linewidth = 2, color='teal')

model_path = '/lstore/titan/martafsilva/master_thesis/master_thesis_output/detector_level_pythia_and_delphes/models/alices_gaussian_prior_0_0.4_10000_thetas_10e7_samples/kinematic_only/alices_hidden_[100, 100]_tanh_alpha_5_epochs_50_bs_128/alices_ensemble_wh_signalWithBSMAndBackgrounds'
alices = Ensemble()
alices.load(model_path)
#theta_each = np.linspace(-1.0,1.0,50)
theta_each = np.linspace(-0.1,0.1,50)
theta_grid = np.array([theta_each]).T

log_r_hat, _ = alices.evaluate_log_likelihood_ratio(
    theta=theta_grid,
    x='./lstore/titan/martafsilva/master_thesis/master_thesis_code/detector_level_pythia_and_delphes/test_samples/x_test_CP_odd_signal_and_backgrounds.npy',
    evaluate_score=False,
    test_all_combinations = True
)
expected_llr =log_r_hat
expected_llr = np.mean(log_r_hat,axis=0)



bins = np.linspace(-0.03,0.03,30)
plt.hist(expected_llr, bins=bins, histtype='step', color='mediumblue', density=True,linewidth=2)
plt.hist(expected_llr, bins=bins, color='mediumblue', density=True, alpha=0.2)
plt.plot(-0.5,0.5,label = "Signal + Backgrounds", linewidth = 2, color = "mediumblue")
plt.xlim(-0.03,0.03)
# plt.xlabel(r'$r(x,z)$')
plt.xlabel(r'$\log \ \hat{r}(x|\theta_0,\theta_1)$', size=14)
plt.ylabel('Normalized distribution', size=14)
plt.legend(frameon=False, fontsize=11)
plt.savefig("llr_hist_cp_odd.pdf", dpi=600,bbox_inches='tight')


# thetas = np.load('/lstore/titan/martafsilva/master_thesis/master_thesis_output/detector_level_pythia_and_delphes/training_samples/alices_alices_gaussian_prior_0_0.4_10000_thetas_5000000_samples/theta0_train_ratio_wh_signalWithBSM_0.npy') 
# joint_llr = np.load('/lstore/titan/martafsilva/master_thesis/master_thesis_output/detector_level_pythia_and_delphes/training_samples/alices_alices_gaussian_prior_0_0.4_10000_thetas_5000000_samples/r_xz_train_ratio_wh_signalWithBSM_0.npy') 

# filtered_indices =  np.where((thetas >= -0.1) & (thetas <= 0.1))

# filtered_joint_llr = joint_llr[filtered_indices]

# log_filtered_joint_llr=np.log(joint_llr)

# bins = np.linspace(-0.03, 0.03, 101)  
# plt.hist(log_filtered_joint_llr, bins=bins, histtype='step', color='teal', density=True,linewidth=2)
# plt.hist(log_filtered_joint_llr, bins=bins, color='teal', density=True, alpha=0.2)
# plt.plot(-0.5,0.5,label = "Signal Only", linewidth = 2, color = "teal")

# thetas = np.load('/lstore/titan/martafsilva/master_thesis/master_thesis_output/detector_level_pythia_and_delphes/training_samples/alices_alices_gaussian_prior_0_0.4_10000_thetas_5000000_samples_CP_even/theta0_train_ratio_wh_signalWithBSM_CP_even_0.npy') 
# joint_llr = np.load('/lstore/titan/martafsilva/master_thesis/master_thesis_output/detector_level_pythia_and_delphes/training_samples/alices_alices_gaussian_prior_0_0.4_10000_thetas_5000000_samples_CP_even/r_xz_train_ratio_wh_signalWithBSM_CP_even_0.npy') 

# filtered_indices =  np.where((thetas >= -0.1) & (thetas <= 0.1))

# filtered_joint_llr = joint_llr[filtered_indices]
# thetas = thetas[filtered_indices]
# print(max(thetas))
# log_filtered_joint_llr=np.log(filtered_joint_llr)
# #log_filtered_joint_llr=np.log(joint_llr)
# bins = np.linspace(-0.03, 0.03, 30)  
# plt.hist(log_filtered_joint_llr, bins=bins, histtype='step', color='teal', density=True,linewidth=2)
# plt.hist(log_filtered_joint_llr, bins=bins, color='teal', density=True, alpha=0.2)
# plt.plot(-0.5,0.5,label = "Signal Only", linewidth = 2, color = "teal")
# thetas = np.load('/lstore/titan/martafsilva/master_thesis/master_thesis_output/detector_level_pythia_and_delphes/training_samples/alices_alices_gaussian_prior_0_0.4_10000_thetas_10e7_samples_CP_even/theta0_train_ratio_wh_signalWithBSMAndBackgrounds_CP_even_0.npy') 
# joint_llr = np.load('/lstore/titan/martafsilva/master_thesis/master_thesis_output/detector_level_pythia_and_delphes/training_samples/alices_alices_gaussian_prior_0_0.4_10000_thetas_10e7_samples_CP_even/r_xz_train_ratio_wh_signalWithBSMAndBackgrounds_CP_even_0.npy') 

# filtered_indices =  np.where((thetas >= -0.1) & (thetas <= 0.1))

# filtered_joint_llr = joint_llr[filtered_indices]
# thetas = thetas[filtered_indices]
# #print(max(thetas))
# log_filtered_joint_llr=np.log(filtered_joint_llr)
# #log_filtered_joint_llr=np.log(joint_llr)
# bins = np.linspace(-0.03, 0.03, 30)  
# plt.hist(log_filtered_joint_llr, bins=bins, histtype='step', color='mediumblue', density=True,linewidth=2)
# plt.hist(log_filtered_joint_llr, bins=bins, color='mediumblue', density=True, alpha=0.2)
# print(np.where(log_filtered_joint_llr!=0.00000))
# plt.plot(-0.5,0.5,label = "Signal + Backgrounds", linewidth = 2, color = "mediumblue")
# plt.xlim(-0.03, 0.03)
# #plt.yscale("log")
# plt.xlabel(r'$\log r(x,z|\theta_0,\theta_1)$', size=14)
# #plt.xlabel(r'$\hat{r}(x|\theta_0,\theta_1)$', size=14)
# plt.ylabel('Normalized distribution', size=14)
# plt.legend(frameon=False, fontsize=11)
# plt.savefig("joint_llr_hist_cp_even.pdf", dpi=600,bbox_inches='tight')