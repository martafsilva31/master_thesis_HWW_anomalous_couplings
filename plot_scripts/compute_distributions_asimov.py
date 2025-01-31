import numpy as np
import matplotlib.pyplot as plt
from madminer.sampling import SampleAugmenter

# Define your luminosity value
luminosity = 300000 # Example luminosity value in fb^-1

# Your existing code for loading data
sa_sig = SampleAugmenter('/lstore/titan/martafsilva/master_thesis/master_thesis_output/2D_high_pT/wh_signalWithBSM_2D_high_pT.h5')
x_weighted_sig, weights_sig = sa_sig.weighted_events(theta=0, generated_close_to=None)

sa_sig_asimov = SampleAugmenter('/lstore/titan/martafsilva/master_thesis/master_thesis_output/2D_high_pT/wh_signalWithBSMAndBackgrounds_2D_high_pT.h5')
x_weighted_sig_asimov, weights_sig_asimov = sa_sig_asimov.weighted_events(theta=0, generated_close_to=0)

x_sec, x_sec_uncertainties = sa_sig_asimov.xsecs(thetas=[0])
sa_bkgs = SampleAugmenter('/lstore/titan/martafsilva/master_thesis/master_thesis_output/2D_high_pT/wh_backgroundOnly_2D_high_pT.h5')
x_weighted_bkg, weights_bkg = sa_bkgs.weighted_events(theta=0, generated_close_to=None)


observable_index = 50
xmin = np.min(x_weighted_sig_asimov[:,observable_index])
xmax = np.max(x_weighted_sig_asimov[:,observable_index])
# xmin = 0
# xmax=1400


bins_ptw = np.linspace(xmin,xmax,25)
# Print debug information

#bins_ptw = [150., 250., 400., 600., xmax]
#bins_ptw = [0., 400. ,800., xmax]

# Compute and scale histograms
histo_bkg, bin_edges = np.histogram(
    x_weighted_bkg[:,observable_index], 
    bins=bins_ptw, 
    weights=weights_bkg, 
    range=(xmin,xmax),
    density=False
)
histo_sig, _ = np.histogram(
    x_weighted_sig[:,observable_index], 
    bins=bins_ptw, 
    weights=weights_sig, 
    range=(xmin,xmax),
    density=False
)

# Scale the histograms by the luminosity
histo_bkg *= luminosity
histo_sig *= luminosity
print(np.sum(histo_bkg))
print(np.sum(histo_sig))

# Compute and scale Asimov histogram
asimov_histo, _ = np.histogram(
    x_weighted_sig_asimov[:, observable_index], 
    bins=bins_ptw, 
    weights=weights_sig_asimov, 
    density=False,
    range=(xmin,xmax)
)
asimov_histo *= luminosity

# Compute Asimov bin centers for error bars
asimov_bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Calculate uncertainties for each bin
asimov_errors = np.zeros_like(asimov_histo)
for i in range(len(asimov_histo)):
    bin_mask = (x_weighted_sig_asimov[:, observable_index] >= bin_edges[i]) & (x_weighted_sig_asimov[:, observable_index] < bin_edges[i + 1])
    asimov_errors[i] = np.sqrt(np.sum(weights_sig_asimov[bin_mask] ** 2))

# Scale uncertainties by the square root of luminosity
asimov_errors *= np.sqrt(luminosity)

# Plot stacked histograms
fig = plt.figure(figsize=(8,6))
ax1 = plt.subplot(111)
fig.subplots_adjust(left=0.1667, right=0.8333, bottom=0.17, top=0.97)

ax1.set_ylabel('Number of events', color='black', fontsize=14)
#ax1.set_xlabel(r'$m_T^{\ell \nu b \bar{b}} \ (GeV)$', color='black', fontsize=14)
#ax1.set_xlabel(r'$p_T^W \ (GeV)$', color='black', fontsize=14)
ax1.set_xlabel(r'$Q_{\ell} \cos \delta^+$', color='black', fontsize=14)

#Plot histograms
ax1.hist(
    [bins_ptw[:-1], bins_ptw[:-1]],  # Bin edges
    weights=[histo_bkg, histo_sig],
    bins=bins_ptw,
    histtype='stepfilled',
    stacked=True,
    color=['teal', 'indigo'],
    alpha=0.7,
    label=["Background Only", "Signal + Backgrounds"]
)

# Plot the Asimov data errors (no histogram, just error bars)
plt.errorbar(
    asimov_bin_centers, asimov_histo, 
    yerr=x_sec_uncertainties*np.sqrt(luminosity), 
    fmt='o', ms=4, color='black', label='Asimov data'
)
# luminosity_info = r"$pp \rightarrow WH \rightarrow \ell \nu b \bar{b}$" + "\n" + r"$\mathcal{L} = 300\, \mathrm{fb}^{-1}$"  
# plt.text(1, 9, luminosity_info, fontsize=12, bbox=dict(facecolor='white', edgecolor='none', alpha=0.5), horizontalalignment='center')

#ax1.set_ylim(0,2800)
ax1.set_xlim(xmin,xmax)
#ax1.set_yscale("log")

# Add legend and save figure
ax1.legend(frameon=False, loc = "upper right",fontsize=12)
plot_dir = '/lstore/titan/martafsilva/master_thesis/master_thesis_output/2D_high_pT/plots/asimov_distributions'
fig.savefig(f'{plot_dir}/ql_cos_delta_plus.pdf', dpi=600, bbox_inches='tight')

