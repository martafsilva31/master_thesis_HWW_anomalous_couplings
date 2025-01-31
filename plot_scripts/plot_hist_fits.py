import numpy as np
import logging
from matplotlib import pyplot as plt
from madminer.ml import ParameterizedRatioEstimator
from madminer.limits import AsymptoticLimits

filename = "/lstore/titan/martafsilva/master_thesis/master_thesis_output/detector_level_pythia_and_delphes/wh_signalWithBSMAndBackgrounds.h5"
limits_file=AsymptoticLimits(filename)

parameter_grid,p_values,index_best_point,llr_kin,llr_rate,(histos, observed, observed_weights)=limits_file.expected_limits(
    mode='rate',theta_true=[0.0], 
    luminosity=300*1000.0,
    return_asimov=True,test_split=1.0,n_histo_toys=None,
    grid_ranges=[(-1.2,1.2)],grid_resolutions=[303])

grid_points = np.linspace(-1.2,1.2,303)

rescaled_log_r = llr_kin+llr_rate
rescaled_log_r = -2.0*(rescaled_log_r[:] - rescaled_log_r[index_best_point])

plt.plot(grid_points,rescaled_log_r, label = 'Rate', color = 'gray')

parameter_grid,p_values,index_best_point,llr_kin,llr_rate,(histos, observed, observed_weights)=limits_file.expected_limits(
    mode='histo',theta_true=[0.0], 
    hist_vars=['ql_cos_deltaPlus'],
    luminosity=300*1000.0,
    return_asimov=True,test_split=1.0,n_histo_toys=None,
    grid_ranges=[(-1.2,1.2)],grid_resolutions=[303])

grid_points = np.linspace(-1.2,1.2,303)

rescaled_log_r = llr_kin+llr_rate
rescaled_log_r = -2.0*(rescaled_log_r[:] - rescaled_log_r[index_best_point])

plt.plot(grid_points,rescaled_log_r, label = r'$Q_{\ell} \cos \delta^+$', color = "#33BBEE")

parameter_grid,p_values,index_best_point,llr_kin,llr_rate,(histos, observed, observed_weights)=limits_file.expected_limits(
    mode='histo',theta_true=[0.0], 
    hist_vars=['mt_tot'],
    luminosity=300*1000.0,
    return_asimov=True,test_split=1.0,n_histo_toys=None,
    grid_ranges=[(-1.2,1.2)],grid_resolutions=[303])

grid_points = np.linspace(-1.2,1.2,303)

rescaled_log_r = llr_kin+llr_rate
rescaled_log_r = -2.0*(rescaled_log_r[:] - rescaled_log_r[index_best_point])

plt.plot(grid_points,rescaled_log_r, label = r'$m_T^{\ell \nu b \bar{b}}$', color = "#B759F8")

parameter_grid,p_values,index_best_point,llr_kin,llr_rate,(histos, observed, observed_weights)=limits_file.expected_limits(
    mode='histo',theta_true=[0.0], 
    hist_vars=['pt_w'],
    luminosity=300*1000.0,
    return_asimov=True,test_split=1.0,n_histo_toys=None,
    grid_ranges=[(-1.2,1.2)],grid_resolutions=[303])

grid_points = np.linspace(-1.2,1.2,303)

rescaled_log_r = llr_kin+llr_rate
rescaled_log_r = -2.0*(rescaled_log_r[:] - rescaled_log_r[index_best_point])

plt.plot(grid_points,rescaled_log_r, label = r'$p_T^W$', color = "darkgreen")


plt.axhline(y=1.0, color='gray', linestyle='-.', linewidth=1.2)
plt.axhline(y=3.84, color='gray', linestyle='--', linewidth=1.2)
plt.text(-0.05, 1.0, '68% CL', color='gray', verticalalignment='bottom', horizontalalignment='left', fontsize=11)
plt.text(-0.05, 3.84, '95% CL', color='gray', verticalalignment='bottom', horizontalalignment='left', fontsize=11)

plt.xlabel(r"$c_{H\tilde{W}}$",size=14)
plt.ylabel(r"$q(\theta)$",size=14) 
plt.ylim(0,10)
plt.xlim(-0.45,0.45)
plt.axhline(y=3.84,lw = 1.5, linestyle='-.',color='grey',label='95%CL')
plt.axhline(y=1.0,lw=1.5, linestyle=':',color='grey',label='68%CL')
plt.legend(title=r"$\textbf{Q_{\ell} \cos \delta^+}$",frameon=False, fontsize=11)
plt.savefig("hist_fits_CP_odd.pdf", dpi=600,bbox_inches='tight')





