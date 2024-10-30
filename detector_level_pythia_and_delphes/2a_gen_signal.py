
"""
gen_signal.py

Generates WH signal events WH(->l v b b~), divided by W decay channel and charge 

Can use different morphing setups (uncomment/ change the bsm benchmarks depending on the setup being used (CP-odd, CP-even, 2D) and use the corresponding config file).

- sample contains weights for different benchmarks (from MG reweighting)

Can also generate events at the BSM benchmarks to populate regions of phase space not well populated by the SM sample
- smaller number than for SM point, 1/5 for each charge+flavour combination
- reweighted to other benchmarks (inc. SM point)

Marta Silva (LIP/IST/CERN-ATLAS), 02/04/2024
"""


from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import os
import argparse 
import yaml
import math
from madminer.core import MadMiner
from madminer.lhe import LHEReader

# MadMiner output
logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.DEBUG #INFO 
)

# Output of all other modules (e.g. matplotlib)
for key in logging.Logger.manager.loggerDict:
    if "madminer" not in key:
        logging.getLogger(key).setLevel(logging.WARNING)

def gen_signal(main_dir, setup_file, do_pythia, pythia_card, auto_widths, prepare_scripts, generate_BSM, mg_dir, launch_SLURM_jobs, cards_folder_name):
    """
    Generates WH signal events WH(->l v b b~), divided by W decay channel and charge.
    """

    # Load morphing setup file
    miner = MadMiner()
    miner.load(f'{main_dir}/{setup_file}.h5')
    lhe = LHEReader(f'{main_dir}/{setup_file}.h5')

    # auto width calculation
    # NB: Madgraph+SMEFTsim include terms up to quadratic order in the automatic width calculation, even when the ME^2 is truncated at the SM+interference term
    if auto_widths:
        param_card_template_file=f'{cards_folder_name}/param_card_template_SMEFTsim3_MwScheme_autoWidths.dat'
    else:
        param_card_template_file=f'{cards_folder_name}/param_card_template_SMEFTsim3_MwScheme.dat'

    # LIP specifics
    init_command="export LD_LIBRARY_PATH=/cvmfs/sw.el7/gcc63/madgraph/3.3.1/b01/HEPTools/pythia8/lib:$LD_LIBRARY_PATH; module load gcc63/madgraph/3.3.1; module unload gcc63/pythia/8.2.40",
    
    #init_command= "module load gcc63/madgraph/3.3.1"


    channels = ['wph_mu', 'wph_e', 'wmh_mu', 'wmh_e']


    factor=3*math.ceil(args.nevents/1e6) # to have as many signal events as you have total background events ()

    # SM samples with MG (re)weights of BSM benchmarks
    for channel in channels:
        miner.run_multiple(
            mg_directory=mg_dir,
            log_directory=f'{main_dir}/logs/{channel}_smeftsim_SM',
            mg_process_directory=f'{main_dir}/signal_samples/{channel}_smeftsim_SM',
            proc_card_file=f'{cards_folder_name}/signal_processes/proc_card_{channel}_smeftsim.dat',
            param_card_template_file=param_card_template_file,
            pythia8_card_file=pythia_card if do_pythia else None,
            sample_benchmarks =['sm'],
            is_background = not args.reweight,
            run_card_files=[f'{cards_folder_name}/run_card_250k_WHMadminerCuts.dat' for _ in range(factor)],
            initial_command=init_command if init_command != '' else None,
            only_prepare_script=prepare_scripts
        )

    # BSM samples with MG (re)weights of other benchmarks (inc. SM)
    if generate_BSM:
        for channel in channels:

    ################################# Uncomment this for CP-even BSM generation: just need to change the name of the bsm points #############################

        #     miner.run_multiple(
        #         mg_directory=mg_dir,
        #         log_directory=f'{main_dir}/logs/{channel}_smeftsim_morphing_basis_vector_1',
        #         mg_process_directory=f'{main_dir}/signal_samples/{channel}_smeftsim_morphing_basis_vector_1',
        #         proc_card_file=f'{cards_folder_name}/signal_processes/proc_card_{channel}_smeftsim.dat',
        #         param_card_template_file=param_card_template_file,
        #         pythia8_card_file=pythia_card if do_pythia else None,
        #         sample_benchmarks=['morphing_basis_vector_1'],
        #         is_background = not args.reweight,
        #         run_card_files=[f'{cards_folder_name}/run_card_50k_WHMadminerCuts.dat' for _ in range(factor)],
        #         initial_command=init_command if init_command != '' else None,
        #         only_prepare_script=prepare_scripts
        # )



        #     miner.run_multiple(
        #         mg_directory=mg_dir,
        #         log_directory=f'{main_dir}/logs/{channel}_smeftsim_morphing_basis_vector_2',
        #         mg_process_directory=f'{main_dir}/signal_samples/{channel}_smeftsim_morphing_basis_vector_2',
        #         proc_card_file=f'{cards_folder_name}/signal_processes/proc_card_{channel}_smeftsim.dat',
        #         param_card_template_file=param_card_template_file,
        #         pythia8_card_file=pythia_card if do_pythia else None,
        #         sample_benchmarks=['morphing_basis_vector_2'],
        #         is_background = not args.reweight,
        #         run_card_files=[f'{cards_folder_name}/run_card_50k_WHMadminerCuts.dat' for _ in range(factor)],
        #         initial_command=init_command if init_command != '' else None,
        #         only_prepare_script=prepare_scripts
        #     )

            miner.run_multiple(
                mg_directory=mg_dir,
                log_directory=f'{main_dir}/logs/{channel}_smeftsim_bench_1',
                mg_process_directory=f'{main_dir}/signal_samples/{channel}_smeftsim_bench_1',
                proc_card_file=f'{cards_folder_name}/signal_processes/proc_card_{channel}_smeftsim.dat',
                param_card_template_file=param_card_template_file,
                pythia8_card_file=pythia_card if do_pythia else None,
                sample_benchmarks=['bench_1'],
                is_background = not args.reweight,
                run_card_files=[f'{cards_folder_name}/run_card_50k_WHMadminerCuts.dat' for _ in range(factor)],
                initial_command=init_command if init_command != '' else None,
                only_prepare_script=prepare_scripts
        )

            miner.run_multiple(
                mg_directory=mg_dir,
                log_directory=f'{main_dir}/logs/{channel}_smeftsim_bench_2',
                mg_process_directory=f'{main_dir}/signal_samples/{channel}_smeftsim_bench_2',
                proc_card_file=f'{cards_folder_name}/signal_processes/proc_card_{channel}_smeftsim.dat',
                param_card_template_file=param_card_template_file,
                pythia8_card_file=pythia_card if do_pythia else None,
                sample_benchmarks=['bench_2'],
                is_background = not args.reweight,
                run_card_files=[f'{cards_folder_name}/run_card_50k_WHMadminerCuts.dat' for _ in range(factor)],
                initial_command=init_command if init_command != '' else None,
                only_prepare_script=prepare_scripts
            )

            miner.run_multiple(
                mg_directory=mg_dir,
                log_directory=f'{main_dir}/logs/{channel}_smeftsim_bench_3',
                mg_process_directory=f'{main_dir}/signal_samples/{channel}_smeftsim_bench_3',
                proc_card_file=f'{cards_folder_name}/signal_processes/proc_card_{channel}_smeftsim.dat',
                param_card_template_file=param_card_template_file,
                pythia8_card_file=pythia_card if do_pythia else None,
                sample_benchmarks=['bench_3'],
                is_background = not args.reweight,
                run_card_files=[f'{cards_folder_name}/run_card_50k_WHMadminerCuts.dat' for _ in range(factor)],
                initial_command=init_command if init_command != '' else None,
                only_prepare_script=prepare_scripts
            )

            miner.run_multiple(
                mg_directory=mg_dir,
                log_directory=f'{main_dir}/logs/{channel}_smeftsim_bench_4',
                mg_process_directory=f'{main_dir}/signal_samples/{channel}_smeftsim_bench_4',
                proc_card_file=f'{cards_folder_name}/signal_processes/proc_card_{channel}_smeftsim.dat',
                param_card_template_file=param_card_template_file,
                pythia8_card_file=pythia_card if do_pythia else None,
                sample_benchmarks=['bench_4'],
                is_background = not args.reweight,
                run_card_files=[f'{cards_folder_name}/run_card_50k_WHMadminerCuts.dat' for _ in range(factor)],
                initial_command=init_command if init_command != '' else None,
                only_prepare_script=prepare_scripts
            )
            miner.run_multiple(
                mg_directory=mg_dir,
                log_directory=f'{main_dir}/logs/{channel}_smeftsim_bench_5',
                mg_process_directory=f'{main_dir}/signal_samples/{channel}_smeftsim_bench_5',
                proc_card_file=f'{cards_folder_name}/signal_processes/proc_card_{channel}_smeftsim.dat',
                param_card_template_file=param_card_template_file,
                pythia8_card_file=pythia_card if do_pythia else None,
                sample_benchmarks=['bench_5'],
                is_background = not args.reweight,
                run_card_files=[f'{cards_folder_name}/run_card_50k_WHMadminerCuts.dat' for _ in range(factor)],
                initial_command=init_command if init_command != '' else None,
                only_prepare_script=prepare_scripts
            )


    # launch gen jobs to SLURM # LIP specifics
    if args.launch_SLURM_jobs and args.prepare_scripts:
        logging.info("Launching SLURM generation jobs")
        cmd=f'find {main_dir}/signal_samples/*/madminer -name "run.sh" -exec sbatch -p lipq --mem=4G {{}} \;'
        os.popen(cmd)
        print(cmd)



    os.remove('/tmp/generate.mg5')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generates WH signal events WH(->l v b b~), divided by W decay channel and charge.',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config_file', help='Path to the YAML configuration file', default='config_2D.yaml')

    parser.add_argument('--do_pythia',help='whether or not to run Pythia after Madgraph',action='store_true',default=False)

    parser.add_argument('--auto_widths',help='Use parameter card with automatic width calculation',action='store_true',default=False)

    parser.add_argument('--prepare_scripts',help='Prepares only run scripts to e.g. submit to a batch system separately',action='store_true',default=False)

    parser.add_argument('--generate_BSM',help='Generate additional events at the BSM benchmarks',action='store_true',default=False)

    parser.add_argument('--launch_SLURM_jobs',help='If SLURM jobs are to be launched immediately after preparation of scripts',action="store_true",default=False)
    
    parser.add_argument('--reweight',help='if running reweighting alongside generation (doesnt work on multi-core mode)',action='store_true',default=False)

    parser.add_argument('--nevents',help='number of total hard scattering events to generate (Madgraph-level)',type=float,default=10e6)


    args=parser.parse_args()

    # Read configuration parameters from the YAML file
    with open(args.config_file, 'r') as config_file:
        config = yaml.safe_load(config_file)

        main_dir = config['main_dir']
        setup_file = config['setup_file']
        pythia_card = config['pythia_card']
        mg_dir = config['mg_dir']
        cards_folder_name = config['cards_folder_name']

    # Generate signal
    gen_signal(main_dir, setup_file, args.do_pythia, pythia_card, args.auto_widths, args.prepare_scripts, args.generate_BSM, mg_dir, args.launch_SLURM_jobs, cards_folder_name)

