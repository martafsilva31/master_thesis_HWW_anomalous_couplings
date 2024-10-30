# Searching for HWW anomalous couplings with simulation-based inference
 This is the code of my master thesis, where the technique ALICES (Approximate likelihood with improved cross-entropy estimator and score) is benchmarked against SALLY (Score Approximates Likelihood LocallY) and histograms of kinematic and angular observables.

This work uses version 0.9.3 of MadMiner (https://github.com/madminer-tool/madminer) and requires version >= 3.3.1 of Madgraph if generating samples.

Everytime you want to generate samples, either locally or in a batch system, you should set up the download and/or linkage to the SMEFTsim model and restrict card used, by running the following command:
> source setup_SMEFTsim_model.sh

To run the analysis chain the programs must be ran in a certain order:

_1_setup.py_: defines the Wilson coefficient and morphing setup and creates setup file.

_2a_gen_signal.py_: generate signal samples or prepare files to run sample generation on batch systems.

_2b_gen_background.py_: generate background samples or prepare files to run sample generation on batch systems.

_3_delphes_analysis.py_: defined the detector-level observables and performs the analysis with Delphes.

_4_analysis_sample_combiner.py_: combines the analysed samples.

_5a_alices_training.py_: performs data augmentation and training for ALICES (Approximate likelihood with improved cross-entropy estimator and score).

_5b_sally_training.py_: performs data augmentation and training for SALLY (Score Approximates Likelihood LocallY).

_6_evaluate_limits.py_: compute limits with the full likelihood ratio (in the asymptotic limit) for ALICES, SALLY and 1D/2D histograms.

All of the scripts have an argument parser describing their API, just type _script_name_ -h to see the list of available options. Each scenario has a corresponding _config file_ that defines the specific parameters and settings.

This work builds on the paper Simulation-based inference in the search for CP violation in leptonic WH production (https://arxiv.org/abs/2308.02882), and the code for that study (https://github.com/rbarrue/sally_cpv_wh) was used as a starting basis. 
