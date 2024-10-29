import numpy as np
import logging
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import os

from madminer.ml import  Ensemble
from madminer.sampling import SampleAugmenter

logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.DEBUG
)

# Output of all other modules (e.g. matplotlib)
for key in logging.Logger.manager.loggerDict:
    if "madminer" not in key:
        logging.getLogger(key).setLevel(logging.WARNING)

# Choose the GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "0")
#os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "0")


model_path = "/lstore/titan/martafsilva/master_thesis/master_thesis_output/parton_level_validation_simplified_process/all/models/flat_prior_m1.2_p1.2/kinematic_only/alices_hidden_[50]_relu_alpha_5_epochs_50_bs_128/alices_ensemble_ud_wph_mu_smeftsim_lhe/"
filename = "/lstore/titan/martafsilva/master_thesis/master_thesis_output/parton_level_validation_simplified_process/all/ud_wph_mu_smeftsim_lhe.h5"

alices = Ensemble()
alices.load(model_path)
theta_each = np.linspace(-1.2,1.2,25)
theta_grid = np.array([theta_each]).T

sa = SampleAugmenter(filename, include_nuisance_parameters=True)

start_event_test, end_event_test, correction_factor_test = sa._train_validation_test_split('test',validation_split=0.2,test_split=0.2)

x,weights=sa.weighted_events(theta='sm',start_event=start_event_test,end_event=end_event_test)

weights*=correction_factor_test # scale the events by the correction factor

joint_likelihood_ratio = np.load("/lstore/titan/martafsilva/master_thesis/master_thesis_output/parton_level_validation_simplified_process/all/training_samples/alices_flat_prior_m1.2_p1.2/r_xz_train_ratio_ud_wph_mu_smeftsim_lhe_0.npy")[:200]
joint_score = np.load("/lstore/titan/martafsilva/master_thesis/master_thesis_output/parton_level_validation_simplified_process/all/training_samples/alices_flat_prior_m1.2_p1.2/t_xz_train_ratio_ud_wph_mu_smeftsim_lhe_0.npy")[:200]
thetas = np.load("/lstore/titan/martafsilva/master_thesis/master_thesis_output/parton_level_validation_simplified_process/all/training_samples/alices_flat_prior_m1.2_p1.2/theta0_train_ratio_ud_wph_mu_smeftsim_lhe_0.npy")[:200]

joint_likelihood_ratio_log = np.squeeze(np.log(joint_likelihood_ratio))
joint_score = np.squeeze(joint_score)


log_r_hat, _=alices.evaluate_log_likelihood_ratio(x=x, theta = theta_grid,test_all_combinations=True)
print(log_r_hat.shape)

t_th0 = -2*joint_score
x_t_th0 = np.ones_like(t_th0)
lengths = (t_th0**2 + x_t_th0**2)**0.5
t_th0 /= lengths
x_t_th0 /= lengths

plt.quiver(thetas, -2*joint_likelihood_ratio_log, x_t_th0, t_th0, 
           scale=3.5, units='inches', angles='xy',
           alpha=.5, color='c', label = r"$r(x,z|\theta)$ (position) + $t(x,z|\theta)$ (slope)")


plt.plot(theta_each,-2*np.average(log_r_hat,axis=1), ls='-', color= '#CC002E', label = r"$\hat{r}(x|\theta)$ (ALICES)")
plt.ylim(-0.3,1)
plt.xlim(-1.3,1.3)
plt.xlabel(r'$\theta = c_{H\tildeW}$')
plt.ylabel('Log Likelihood Ratio')
plt.legend()
plt.savefig("test.png")